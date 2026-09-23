import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, Twist
from moveit_msgs.srv import GetPositionIK, GetCartesianPath
from moveit_msgs.msg import PositionIKRequest
from moveit_msgs.msg import (MotionPlanRequest,Constraints,PositionConstraint,OrientationConstraint,BoundingVolume,)
from rclpy.action import ActionClient
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from action_msgs.msg import GoalStatus
from shape_msgs.msg import SolidPrimitive
import tf2_ros
import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
    qx = (math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2)
          - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2))
    qy = (math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
          + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2))
    qz = (math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
          - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2))
    qw = (math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2)
          + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2))
    return [qx, qy, qz, qw]

def quaternion_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w, 0],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w, 0],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y, 0],
        [                0,                 0,                 0,   1],
    ])

def matrix_to_quaternion(M):
    trace = M[0, 0] + M[1, 1] + M[2, 2]
    if trace > 0:
        S = math.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (M[2, 1] - M[1, 2]) / S
        y = (M[0, 2] - M[2, 0]) / S
        z = (M[1, 0] - M[0, 1]) / S
    elif (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
        S = math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        w = (M[2, 1] - M[1, 2]) / S
        x = 0.25 * S
        y = (M[0, 1] + M[1, 0]) / S
        z = (M[0, 2] + M[2, 0]) / S
    elif M[1, 1] > M[2, 2]:
        S = math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        w = (M[0, 2] - M[2, 0]) / S
        x = (M[0, 1] + M[1, 0]) / S
        y = 0.25 * S
        z = (M[1, 2] + M[2, 1]) / S
    else:
        S = math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        w = (M[1, 0] - M[0, 1]) / S
        x = (M[0, 2] + M[2, 0]) / S
        y = (M[1, 2] + M[2, 1]) / S
        z = 0.25 * S
    return [x, y, z, w]

def translation_matrix(t):
    M = np.eye(4)
    M[0:3, 3] = t
    return M

class ArcWaypointNode(Node):
    def __init__(self):
        super().__init__('arc_waypoint_node')

        self.declare_parameter('planning_frame', 'world')
        self.declare_parameter('toilet_frame', 'toilet')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('end_effector_frame', 'gripper_base_link')
        self.declare_parameter('move_group', 'xarm7_arm')

        self.declare_parameter('base_target_x', 0.90)
        self.declare_parameter('base_target_y', 0.0)
        self.declare_parameter('base_target_yaw', 3.14159)
        self.declare_parameter('min_safe_base_x', 0.90)
        self.declare_parameter('max_base_speed', 0.04)
        self.declare_parameter('max_base_turn_speed', 0.10)
        self.declare_parameter('nav_xy_tol', 0.04)
        self.declare_parameter('nav_yaw_tol', 0.04)

        # Fixed target for lid grasp in world frame
        self.declare_parameter('grasp_target_x', -0.03986) #-0.02086
        self.declare_parameter('grasp_target_y', 0.00357)
        self.declare_parameter('grasp_target_z', 0.44867)
        # Assuming typical downward grasping approach
        self.declare_parameter('grasp_target_roll', 0.0)
        self.declare_parameter('grasp_target_pitch', 1.5708)
        self.declare_parameter('grasp_target_yaw', 3.14159)
        
        self.declare_parameter('arm_speed_scale', 8.0)
        self.declare_parameter('speed_scale', 5.0)  # for the arc opening move

        # ── Lid arc parameters ─────────────────────────────────────────────────
        self.declare_parameter('theta_start', 0.01)
        self.declare_parameter('theta_end', 1.62)   
        self.declare_parameter('num_waypoints', 20)
        self.declare_parameter('tilt_fraction', 0.4)

        # Hinge transform relative to toilet (from URDF)
        self.hinge_origin_xyz = [-0.07218, 0.0, 0.42234]
        self.hinge_origin_rpy = [1.5708, 0.0, 0.0]

        self.state = 'NAVIGATING_AND_CHECKING_REACH'
        self.is_checking_reachability = False
        self.current_goal_handle = None
        self.arc_waypoints = []
        self.current_arc_index = 0
        self.arc_chunk_end_index = 0


        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.waypoint_pub = self.create_publisher(PoseArray, 'lid_arc_waypoints', 10)
        
        self.compute_ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.compute_path_client = self.create_client(GetCartesianPath, 'compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')

        self.get_logger().info("Arc Waypoint Node initialised.")
        self.timer = self.create_timer(0.1, self.state_machine_tick)

    def get_transform_matrix(self, frame_id, child_frame_id):
        try:
            t = self.tf_buffer.lookup_transform(frame_id, child_frame_id, rclpy.time.Time())
            q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            tr = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            return np.dot(translation_matrix(tr), quaternion_matrix(q))
        except Exception as e:
            return None

    def state_machine_tick(self):
        if self.state == 'NAVIGATING_AND_CHECKING_REACH':
            self.navigate_and_check()
        elif self.state == 'FINAL_PLANNING':
            self.perform_final_plan()
        elif self.state == 'GENERATING_ARC':
            self.generate_and_start_arc()
        elif self.state == 'PLANNING_ARC_CHUNK':
            self.plan_arc_chunk()
        elif self.state == 'WAITING_FOR_ARC_PATH':
            pass
        elif self.state == 'ADJUSTING_BASE':
            self.adjust_base_for_reach()
        elif self.state == 'EXECUTING_ARM':
            pass
        elif self.state == 'DONE':
            self.get_logger().info("Grasp completed or aborted. Node idle.", once=True)
            self.timer.cancel()

    def navigate_and_check(self):
        planning_frame = self.get_parameter('planning_frame').value
        toilet_frame = self.get_parameter('toilet_frame').value
        base_frame = self.get_parameter('base_frame').value

        T_world_toilet = self.get_transform_matrix(planning_frame, toilet_frame)
        T_world_base = self.get_transform_matrix(planning_frame, base_frame)

        if T_world_toilet is None or T_world_base is None:
            return

        tx = self.get_parameter('base_target_x').value
        ty = self.get_parameter('base_target_y').value
        tyaw = self.get_parameter('base_target_yaw').value
        min_safe_x = self.get_parameter('min_safe_base_x').value

        tx = max(tx, min_safe_x)

        q_target = euler_to_quaternion(0.0, 0.0, tyaw)
        T_toilet_target = np.dot(translation_matrix([tx, ty, 0.0]), quaternion_matrix(q_target))
        T_world_target = np.dot(T_world_toilet, T_toilet_target)

        T_base_target = np.dot(np.linalg.inv(T_world_base), T_world_target)
        err_x = T_base_target[0, 3]
        err_y = T_base_target[1, 3]
        err_yaw = math.atan2(T_base_target[1, 0], T_base_target[0, 0])
        dist = math.sqrt(err_x**2 + err_y**2)

        T_toilet_base = np.dot(np.linalg.inv(T_world_toilet), T_world_base)
        curr_base_x = T_toilet_base[0, 3]

        if curr_base_x < min_safe_x - 0.02:
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().error(f"SAFETY STOP! Base too close to toilet.")
            self.state = 'DONE'
            return

        tol_xy = self.get_parameter('nav_xy_tol').value
        tol_yaw = self.get_parameter('nav_yaw_tol').value
        max_spd = self.get_parameter('max_base_speed').value
        max_trn = self.get_parameter('max_base_turn_speed').value

        if dist > tol_xy or abs(err_yaw) > tol_yaw:
            msg = Twist()
            msg.linear.x = max(-max_spd, min(max_spd, err_x * 0.4))
            msg.angular.z = max(-max_trn, min(max_trn, err_yaw * 0.8))
            self.cmd_vel_pub.publish(msg)
        else:
            self.cmd_vel_pub.publish(Twist())
            if not self.is_checking_reachability:
                self.get_logger().error("Reached navigation target but arm is still not reachable. Aborting.")
                self.state = 'DONE'
            return

        if not self.is_checking_reachability:
            grasp_pose = self.compute_grasp_pose()
            self.is_checking_reachability = True
            self._check_ik_feasibility(grasp_pose)

    def compute_grasp_pose(self, extra_x_offset=0.0):
        gx = self.get_parameter('grasp_target_x').value
        gy = self.get_parameter('grasp_target_y').value
        gz = self.get_parameter('grasp_target_z').value
        
        r = self.get_parameter('grasp_target_roll').value
        p = self.get_parameter('grasp_target_pitch').value
        y = self.get_parameter('grasp_target_yaw').value
        
        q = euler_to_quaternion(r, p, y)
        
        # Calculate target pose of the FINGERTIP in world frame
        T_world_fingertip = np.dot(translation_matrix([gx, gy, gz]), quaternion_matrix(q))
        
        # The fingertip is at +0.060m along the local Z axis of gripper_base_link.
        # To find the required gripper_base_link pose, we move -0.060m along the local Z axis.
        T_fingertip_offset = translation_matrix([0.0, 0.0, -0.060])
        
        T_world_gripper_base_link = np.dot(T_world_fingertip, T_fingertip_offset)
        
        pose = Pose()
        # Move backward along the world X direction for the pre-grasp pose
        pose.position.x = T_world_gripper_base_link[0, 3] + extra_x_offset
        pose.position.y = T_world_gripper_base_link[1, 3]
        pose.position.z = T_world_gripper_base_link[2, 3]
        
        # The orientation is unchanged
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

    def _check_ik_feasibility(self, target_pose):
        if not self.compute_ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("compute_ik server not available!")
            self.is_checking_reachability = False
            return

        req = PositionIKRequest()
        req.group_name = self.get_parameter('move_group').value
        req.pose_stamped.header.frame_id = self.get_parameter('planning_frame').value
        req.pose_stamped.pose = target_pose
        req.timeout.sec = 0
        req.timeout.nanosec = 100000000  # 0.1s
        # Disable collision checking because the final grasp pose is inside the gap
        req.avoid_collisions = False

        req_msg = GetPositionIK.Request()
        req_msg.ik_request = req

        future = self.compute_ik_client.call_async(req_msg)
        future.add_done_callback(self._ik_response_cb)

    def _ik_response_cb(self, future):
        self.is_checking_reachability = False
        try:
            res = future.result()
            if res.error_code.val == 1:
                self.get_logger().info("Reachability confirmed! Stopping base and transitioning to FINAL_PLANNING.")
                self.cmd_vel_pub.publish(Twist())
                self.state = 'FINAL_PLANNING'
        except Exception as e:
            self.get_logger().warn(f"IK check failed: {e}")

    def perform_final_plan(self):
        if self.is_checking_reachability:
            return
        
        self.is_checking_reachability = True
        
        # Plan to a pre-grasp pose pulled back by 15 cm along the world X axis
        pre_grasp_pose = self.compute_grasp_pose(extra_x_offset=0.15)
        # Store the actual grasp pose for the Cartesian approach later
        self.final_grasp_pose = self.compute_grasp_pose(extra_x_offset=0.0)
        
        speed = self.get_parameter('arm_speed_scale').value
        self._plan_freespace_move(pre_grasp_pose, speed)

    def _plan_freespace_move(self, target_pose, speed_scale):
        if not self.move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("move_action server not available! Aborting.")
            self.state = 'DONE'
            return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.get_parameter('move_group').value

        scaling = max(0.01, min(1.0, 1.0 / speed_scale))
        req.max_velocity_scaling_factor = scaling
        req.max_acceleration_scaling_factor = scaling
        req.num_planning_attempts = 3
        req.allowed_planning_time = 5.0

        pc = PositionConstraint()
        pc.header.frame_id = self.get_parameter('planning_frame').value
        pc.link_name = self.get_parameter('end_effector_frame').value
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]

        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv.primitive_poses.append(target_pose)
        pc.constraint_region = bv
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = self.get_parameter('planning_frame').value
        oc.link_name = self.get_parameter('end_effector_frame').value
        oc.orientation = target_pose.orientation
        oc.absolute_x_axis_tolerance = 0.05
        oc.absolute_y_axis_tolerance = 0.05
        oc.absolute_z_axis_tolerance = 0.05
        oc.weight = 1.0

        c = Constraints()
        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        req.goal_constraints.append(c)

        goal_msg.request = req
        goal_msg.planning_options.plan_only = True

        future = self.move_group_client.send_goal_async(goal_msg)
        future.add_done_callback(self._freespace_goal_response_cb)

    def _freespace_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("MoveGroup goal rejected.")
            self.state = 'DONE'
            return
            
        goal_handle.get_result_async().add_done_callback(self._freespace_result_cb)

    def _freespace_result_cb(self, future):
        result = future.result().result
        self.is_checking_reachability = False
        
        if result.error_code.val != 1:
            self.get_logger().warn(f"Final plan failed (error code: {result.error_code.val}).")
            self.state = 'NAVIGATING_AND_CHECKING_REACH'
            return

        self.get_logger().info("Final freespace plan succeeded. Executing arm pre-grasp approach...")
        self.state = 'EXECUTING_PRE_GRASP_ARM'

        if not self.execute_client.wait_for_server(timeout_sec=5.0):
            self.state = 'DONE'
            return

        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = result.planned_trajectory

        future_goal = self.execute_client.send_goal_async(goal_msg)
        future_goal.add_done_callback(self._arm_goal_response_cb)

    def _arm_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.state = 'DONE'
            return
        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self._arm_result_cb)

    def _arm_result_cb(self, future):
        result = future.result().result
        status = future.result().status
        self.current_goal_handle = None

        if status == GoalStatus.STATUS_CANCELED:
            self.state = 'DONE'
            return

        if result.error_code.val != 1:
            self.state = 'DONE'
            return

        if self.state == 'EXECUTING_PRE_GRASP_ARM':
            self.get_logger().info("Pre-grasp reached. Planning Cartesian approach to the lid...")
            self.state = 'PLANNING_GRASP_APPROACH'
            self._plan_cartesian_approach(self.final_grasp_pose)
        elif self.state == 'EXECUTING_GRASP_APPROACH_ARM':
            self.get_logger().info("Grasp approach complete. Generating arc waypoints...")
            self.state = 'GENERATING_ARC'
        elif self.state == 'EXECUTING_ARC_CHUNK':
            self.get_logger().info("Arc chunk execution complete.")
            self.current_arc_index = self.arc_chunk_end_index
            if self.current_arc_index >= len(self.arc_waypoints) - 1:
                self.get_logger().info("Lid opening fully completed!")
                self.state = 'DONE'
            else:
                self.get_logger().info("Arc chunk done, planning next chunk...")
                self.state = 'PLANNING_ARC_CHUNK'
        else:
            self.get_logger().info("MoveIt execution complete.")
            self.state = 'DONE'

    def _plan_cartesian_approach(self, target_pose):
        if not self.compute_path_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("compute_cartesian_path not available! Aborting.")
            self.state = 'DONE'
            return

        req = GetCartesianPath.Request()
        req.header.frame_id = self.get_parameter('planning_frame').value
        req.group_name      = self.get_parameter('move_group').value
        req.waypoints       = [target_pose]
        req.max_step        = 0.01
        req.jump_threshold  = 0.0
        # Disable collision checking because the insertion pushes into the gap
        req.avoid_collisions = False

        future = self.compute_path_client.call_async(req)
        future.add_done_callback(self._cartesian_path_computed_cb)

    def _cartesian_path_computed_cb(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"compute_cartesian_path call failed: {e}")
            self.state = 'DONE'
            return

        if response.fraction < 0.1:
            self.get_logger().error(f"Cartesian approach failed (fraction: {response.fraction:.2f}). Aborting.")
            self.state = 'DONE'
            return

        if response.fraction < 0.9:
            self.get_logger().warn(f"Cartesian approach is partial (fraction: {response.fraction:.2f}). Executing what was found.")
        else:
            self.get_logger().info("Cartesian approach fully planned. Scaling speed and executing...")
        
        # Scale down trajectory speed (higher arm_speed_scale -> slower motion)
        speed_scale = self.get_parameter('arm_speed_scale').value
        for pt in response.solution.joint_trajectory.points:
            total_time = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            total_time *= speed_scale
            pt.time_from_start.sec    = int(total_time)
            pt.time_from_start.nanosec = int((total_time - int(total_time)) * 1e9)
            pt.velocities    = [v / speed_scale for v in pt.velocities]
            pt.accelerations = [a / (speed_scale ** 2) for a in pt.accelerations]

        self.state = 'EXECUTING_GRASP_APPROACH_ARM'
        
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = response.solution
        future_goal = self.execute_client.send_goal_async(goal_msg)
        future_goal.add_done_callback(self._arm_goal_response_cb)

    def generate_waypoints(self):
        planning_frame  = self.get_parameter('planning_frame').value
        toilet_frame    = self.get_parameter('toilet_frame').value
        ee_frame        = self.get_parameter('end_effector_frame').value

        T_world_toilet  = self.get_transform_matrix(planning_frame, toilet_frame)
        if T_world_toilet is None:
            return None

        q_hinge      = euler_to_quaternion(*self.hinge_origin_rpy)
        T_toilet_hinge = np.dot(translation_matrix(self.hinge_origin_xyz), quaternion_matrix(q_hinge))
        T_world_hinge = np.dot(T_world_toilet, T_toilet_hinge)

        T_world_gripper = self.get_transform_matrix(planning_frame, ee_frame)
        if T_world_gripper is None:
            self.get_logger().error(f"Could not get TF for {ee_frame}.")
            return None

        # Capture insertion orientation once and lock it for every waypoint
        q_locked = matrix_to_quaternion(T_world_gripper)

        T_hinge_gripper_initial = np.dot(np.linalg.inv(T_world_hinge), T_world_gripper)

        theta_start   = self.get_parameter('theta_start').value
        theta_end     = self.get_parameter('theta_end').value
        num_waypoints = self.get_parameter('num_waypoints').value

        thetas    = np.linspace(theta_start, theta_end, num_waypoints)
        waypoints = []

        for theta in thetas:
            q_rot_pos     = euler_to_quaternion(0, 0, theta)
            T_hinge_rot_pos = quaternion_matrix(q_rot_pos)

            T_world_pos = np.dot(T_world_hinge, np.dot(T_hinge_rot_pos, T_hinge_gripper_initial))

            pose = Pose()
            pose.position.x = T_world_pos[0, 3]
            pose.position.y = T_world_pos[1, 3]
            pose.position.z = T_world_pos[2, 3]

            pose.orientation.x = q_locked[0]
            pose.orientation.y = q_locked[1]
            pose.orientation.z = q_locked[2]
            pose.orientation.w = q_locked[3]
            waypoints.append(pose)

        pose_array = PoseArray()
        pose_array.header.frame_id = planning_frame
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.poses = waypoints
        self.waypoint_pub.publish(pose_array)

        return waypoints

    def generate_and_start_arc(self):
        waypoints = self.generate_waypoints()
        if waypoints is None or len(waypoints) < 2:
            self.get_logger().error("Failed to generate arc waypoints.")
            self.state = 'DONE'
            return

        self.arc_waypoints = waypoints
        self.current_arc_index = 0
        self.state = 'PLANNING_ARC_CHUNK'

    def plan_arc_chunk(self):
        remaining_waypoints = self.arc_waypoints[self.current_arc_index:]
        if not remaining_waypoints:
            self.state = 'DONE'
            return
            
        req = GetCartesianPath.Request()
        req.header.frame_id = self.get_parameter('planning_frame').value
        req.group_name      = self.get_parameter('move_group').value
        req.waypoints       = remaining_waypoints
        req.max_step        = 0.01
        req.jump_threshold  = 0.0
        req.avoid_collisions = False

        self.state = 'WAITING_FOR_ARC_PATH'
        future = self.compute_path_client.call_async(req)
        future.add_done_callback(self._arc_chunk_computed_cb)

    def _arc_chunk_computed_cb(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Arc chunk computation failed: {e}")
            self.state = 'DONE'
            return

        num_waypoints = len(self.arc_waypoints[self.current_arc_index:])
        reached_points = int(response.fraction * num_waypoints)
        
        if reached_points < 1:
            self.get_logger().warn("Next waypoint unreachable. Moving base forward...")
            self.state = 'ADJUSTING_BASE'
            return
            
        self.arc_chunk_end_index = self.current_arc_index + reached_points
        self.get_logger().info(f"Arc chunk planned: {reached_points}/{num_waypoints} waypoints. Executing...")
        
        speed_scale = self.get_parameter('speed_scale').value
        for pt in response.solution.joint_trajectory.points:
            total_time = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            total_time *= speed_scale
            pt.time_from_start.sec    = int(total_time)
            pt.time_from_start.nanosec = int((total_time - int(total_time)) * 1e9)
            pt.velocities    = [v / speed_scale for v in pt.velocities]
            pt.accelerations = [a / (speed_scale ** 2) for a in pt.accelerations]
            
        self.state = 'EXECUTING_ARC_CHUNK'
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = response.solution
        future_goal = self.execute_client.send_goal_async(goal_msg)
        future_goal.add_done_callback(self._arm_goal_response_cb)

    def adjust_base_for_reach(self):
        if not hasattr(self, 'base_adjust_start_time'):
            self.base_adjust_start_time = self.get_clock().now()
            
        elapsed = (self.get_clock().now() - self.base_adjust_start_time).nanoseconds / 1e9
        if elapsed < 1.0:
            # Move base forward slowly (2 cm/s for 1 sec = 2 cm)
            twist = Twist()
            twist.linear.x = 0.02
            self.cmd_vel_pub.publish(twist)
        else:
            self.cmd_vel_pub.publish(Twist()) # stop
            del self.base_adjust_start_time
            self.get_logger().info("Base adjustment complete. Re-planning arc chunk...")
            self.state = 'PLANNING_ARC_CHUNK'


def main(args=None):
    rclpy.init(args=args)
    node = ArcWaypointNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
