#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, Twist, WrenchStamped
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import RobotTrajectory, MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from rclpy.action import ActionClient
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from action_msgs.msg import GoalStatus
from shape_msgs.msg import SolidPrimitive
import tf2_ros
import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def quaternion_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w, 0],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w, 0],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y, 0],
        [                0,                 0,                 0, 1]
    ])

def translation_matrix(t):
    M = np.eye(4)
    M[0:3, 3] = t
    return M

def matrix_to_quaternion(M):
    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]

class LidSeatManipulationNode(Node):
    def __init__(self):
        super().__init__('lid_seat_manipulation_node')

        # Frames
        self.declare_parameter('planning_frame', 'world')
        self.declare_parameter('toilet_frame', 'toilet')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('end_effector_frame', 'link7')
        self.declare_parameter('move_group', 'xarm7_arm')

        # Base Navigation Parameters
        self.declare_parameter('base_target_x', 0.90) # Increased to 0.90m for better clearance
        self.declare_parameter('base_target_y', 0.0)
        self.declare_parameter('base_target_yaw', 3.14159) # Facing toilet
        self.declare_parameter('min_safe_base_x', 0.70) # Minimum safe approach
        self.declare_parameter('max_base_speed', 0.06)
        self.declare_parameter('max_base_turn_speed', 0.20)
        self.declare_parameter('nav_xy_tol', 0.04)
        self.declare_parameter('nav_yaw_tol', 0.04)

        # Gap Grasp Point Parameters (Relative to toilet frame)
        self.declare_parameter('gap_x', 0.25)
        self.declare_parameter('gap_y', 0.0)
        self.declare_parameter('gap_z', 0.395) # Lowered to slide under the lid
        # Orientation of the blade entering the gap (horizontal blade, X-axis points DOWN to avoid kinematic singularities)
        self.declare_parameter('gap_roll', 0.0)
        self.declare_parameter('gap_pitch', 1.5708)
        self.declare_parameter('gap_yaw', 3.14159)
        self.declare_parameter('gap_approach_offset', 0.20) # 20cm pull-back for initial approach to avoid singularities

        # Gripper Physical Offsets (to move the mathematical tip to the physical blade edge)
        self.declare_parameter('gripper_tip_offset', 0.02)
        self.declare_parameter('gripper_blade_offset_x', -0.050)
        self.declare_parameter('gripper_blade_offset_y', 0.006)
        self.declare_parameter('gripper_blade_offset_z', 0.02)
        self.declare_parameter('gripper_yaw_offset', 3.14159)

        # Lid Hinge Parameters (from URDF)
        self.hinge_origin_xyz = [-0.07218, 0.0, 0.42234]
        self.hinge_origin_rpy = [1.5708, 0.0, 0.0]
        self.declare_parameter('theta_end', 1.62) # Just below 1.63 upper limit
        self.declare_parameter('num_arc_waypoints', 25)

        # Navigation Parameters
        self.declare_parameter('target_base_distance', 1.15) # Increased for better arm clearance
        self.declare_parameter('min_base_distance', 0.65)

        # Speeds
        self.declare_parameter('arm_speed_scale', 12.0)
        self.declare_parameter('open_speed_scale', 20.0)

        # Force Control Parameters
        self.declare_parameter('force_topic', '/force_torque_sensor_broadcaster/wrench')
        self.declare_parameter('max_force_limit', 50.0) # Increased limit to account for inertial spikes

        # State Machine
        self.state = 'NAVIGATING'
        self.current_goal_handle = None
        self.baseline_force_mag = 0.0
        self.current_force_mag = 0.0
        
        self.partial_retries = 0

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ROS interfaces
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.waypoint_pub = self.create_publisher(PoseArray, 'lid_waypoints', 10)
        self.wrench_sub = self.create_subscription(WrenchStamped, self.get_parameter('force_topic').value, self.wrench_callback, 10)
        
        self.compute_path_client = self.create_client(GetCartesianPath, 'compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')

        self.get_logger().info("Lid Seat Manipulation Node initialized. Starting state machine...")
        self.timer = self.create_timer(0.1, self.state_machine_tick)

    def get_transform_matrix(self, frame_id, child_frame_id):
        try:
            t = self.tf_buffer.lookup_transform(frame_id, child_frame_id, rclpy.time.Time())
            q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            tr = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            return np.dot(translation_matrix(tr), quaternion_matrix(q))
        except Exception as e:
            return None

    def wrench_callback(self, msg):
        # Using magnitude of force for omnidirectional safety during lid manipulation
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        self.current_force_mag = math.sqrt(fx**2 + fy**2 + fz**2)
        
        if self.state in ['ENTERING_GAP', 'OPENING_LID']:
            if hasattr(self, 'state_start_time'):
                elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
                if elapsed < 1.0:
                    return # Ignore inertial spikes at the beginning of the motion
            
            delta_force = abs(self.current_force_mag - getattr(self, 'baseline_force_mag', 0.0))
            max_f = self.get_parameter('max_force_limit').value
            
            if delta_force > max_f:
                self.get_logger().warn(f"FORCE LIMIT EXCEEDED! (Delta: {delta_force:.2f} N > {max_f} N). Cancelling trajectory!")
                self.state = 'FORCE_ABORT'
                if self.current_goal_handle is not None:
                    self.current_goal_handle.cancel_goal_async()
                    self.current_goal_handle = None
                
                self.state = 'DONE'
                self.get_logger().info("Aborted due to force threshold. Operation stopped safely.")
                self.timer.cancel()

    def state_machine_tick(self):
        if self.state == 'NAVIGATING':
            self.navigate_to_toilet()
        elif self.state == 'VERIFYING_REACH':
            self.verify_reachability()
        elif self.state == 'ADJUSTING_BASE':
            self.adjust_base_pose()
        elif self.state == 'APPROACHING':
            pass
        elif self.state == 'ENTERING_GAP':
            pass
        elif self.state == 'OPENING_LID':
            pass
        elif self.state == 'DONE':
            self.get_logger().info("Lid Manipulation completed or aborted safely.")
            self.timer.cancel()
            
    def navigate_to_toilet(self):
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
        if tx < min_safe_x:
            tx = min_safe_x

        q_target = euler_to_quaternion(0, 0, tyaw)
        T_toilet_target = np.dot(translation_matrix([tx, ty, 0]), quaternion_matrix(q_target))
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
            self.get_logger().error(f"SAFETY STOP! Base too close: {curr_base_x:.3f}m < {min_safe_x:.3f}m.")
            self.state = 'VERIFYING_REACH'
            return

        tol_xy = self.get_parameter('nav_xy_tol').value
        tol_yaw = self.get_parameter('nav_yaw_tol').value
        max_speed = self.get_parameter('max_base_speed').value
        max_turn = self.get_parameter('max_base_turn_speed').value

        msg = Twist()
        if dist > tol_xy or abs(err_yaw) > tol_yaw:
            msg.linear.x = max(-max_speed, min(max_speed, err_x * 0.4))
            msg.angular.z = max(-max_turn, min(max_turn, err_yaw * 0.8))
            self.cmd_vel_pub.publish(msg)
        else:
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info("Reached base pose. Transitioning to VERIFYING_REACH.")
            self.state = 'VERIFYING_REACH'

    def adjust_base_pose(self):
        current_tx = self.get_parameter('base_target_x').value
        min_safe_x = self.get_parameter('min_safe_base_x').value
        
        if current_tx - 0.05 >= min_safe_x:
            new_tx = current_tx - 0.05
            self.get_logger().warn(f"Adjusting base target closer to toilet: {current_tx:.2f}m -> {new_tx:.2f}m")
            self.set_parameters([rclpy.parameter.Parameter('base_target_x', rclpy.Parameter.Type.DOUBLE, new_tx)])
            self.state = 'NAVIGATING'
        else:
            self.get_logger().error("Cannot reach lid gap, and cannot move base any closer due to safety limit. Aborting!")
            self.state = 'DONE'

    def get_target_gripper_pose(self, x_offset):
        planning_frame = self.get_parameter('planning_frame').value
        toilet_frame = self.get_parameter('toilet_frame').value
        T_world_toilet = self.get_transform_matrix(planning_frame, toilet_frame)
        if T_world_toilet is None:
            return None

        # Desired gap pose for the BLADE edge
        bx = self.get_parameter('gap_x').value + x_offset
        by = self.get_parameter('gap_y').value
        bz = self.get_parameter('gap_z').value
        r = self.get_parameter('gap_roll').value
        p = self.get_parameter('gap_pitch').value
        y = self.get_parameter('gap_yaw').value

        target_tip_point = [bx, by, bz]
        q_toilet_gripper = euler_to_quaternion(r, p, y)
        T_toilet_gripper_tip = np.dot(translation_matrix(target_tip_point), quaternion_matrix(q_toilet_gripper))

        # Shift back by blade offset vector to find gripper_base_link pose
        blade_ox = self.get_parameter('gripper_blade_offset_x').value
        blade_oy = self.get_parameter('gripper_blade_offset_y').value
        blade_oz = self.get_parameter('gripper_blade_offset_z').value
        if blade_oz == 0.0 and self.has_parameter('gripper_tip_offset'):
            blade_oz = self.get_parameter('gripper_tip_offset').value

        T_tip_gripperbase = translation_matrix([-blade_ox, -blade_oy, -blade_oz])
        # Apply yaw offset for wrist (e.g. 180 deg to flip blade)
        yaw_offset = self.get_parameter('gripper_yaw_offset').value
        T_yaw = quaternion_matrix(euler_to_quaternion(0, 0, yaw_offset))
        
        T_toilet_gripperbase = np.dot(T_toilet_gripper_tip, np.dot(T_yaw, T_tip_gripperbase))

        link7_frame = self.get_parameter('end_effector_frame').value
        T_link7_gripperbase = self.get_transform_matrix(link7_frame, 'gripper_base_link')
        if T_link7_gripperbase is None:
            q_mount = euler_to_quaternion(0.0, 0.0, 3.1708)
            T_link7_gripperbase = quaternion_matrix(q_mount)

        T_gripperbase_link7 = np.linalg.inv(T_link7_gripperbase)
        T_toilet_link7 = np.dot(T_toilet_gripperbase, T_gripperbase_link7)
        T_world_link7 = np.dot(T_world_toilet, T_toilet_link7)

        pose = Pose()
        pose.position.x = T_world_link7[0, 3]
        pose.position.y = T_world_link7[1, 3]
        pose.position.z = T_world_link7[2, 3]
        
        q_final = matrix_to_quaternion(T_world_link7)
        pose.orientation.x = q_final[0]
        pose.orientation.y = q_final[1]
        pose.orientation.z = q_final[2]
        pose.orientation.w = q_final[3]
        
        return pose

    def verify_reachability(self):
        # Plan directly to the exact lid grasp point (x_offset=0.0).
        # MoveIt's OMPL planner will find a safe, collision-free path — no manual
        # approach offset is needed. When the arm reaches this pose, it stops there.
        grasp_pose = self.get_target_gripper_pose(0.0)
        
        if grasp_pose is None:
            return

        self.get_logger().info(
            f"Planning OMPL freespace path directly to lid grasp point "
            f"({grasp_pose.position.x:.3f}, {grasp_pose.position.y:.3f}, "
            f"{grasp_pose.position.z:.3f})..."
        )
        self.state = 'WAITING_PLAN'
        speed = self.get_parameter('arm_speed_scale').value
        # 'ENTERING_GAP' is used directly — the freespace planner gets the arm
        # to the grasp point and the robot halts there before opening the lid.
        self.plan_freespace_move(grasp_pose, next_state='ENTERING_GAP', speed_scale=speed)

    def plan_freespace_move(self, target_pose, next_state, speed_scale):
        if not self.move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("move_action server not available!")
            self.state = 'DONE'
            return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.get_parameter('move_group').value
        # Map speed scale directly (larger speed_scale means slower movement)
        # e.g., speed_scale=12.0 -> scaling=1.0/12.0 = 0.083 (8.3% max speed)
        scaling = 1.0 / speed_scale if speed_scale > 0 else 1.0
        scaling = max(0.01, min(1.0, scaling))
        req.max_velocity_scaling_factor = scaling
        req.max_acceleration_scaling_factor = scaling
        req.num_planning_attempts = 3
        req.allowed_planning_time = 5.0
        
        c = Constraints()
        
        pc = PositionConstraint()
        pc.header.frame_id = self.get_parameter('planning_frame').value
        pc.link_name = self.get_parameter('end_effector_frame').value
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0
        
        s = SolidPrimitive()
        s.type = SolidPrimitive.SPHERE
        s.dimensions = [0.01] # 1cm tolerance
        
        bv = BoundingVolume()
        bv.primitives.append(s)
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
        
        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        req.goal_constraints.append(c)
        
        goal_msg.request = req
        goal_msg.planning_options.plan_only = True
        
        self.get_logger().info(f"Sending OMPL freespace plan request for {next_state}...")
        future = self.move_group_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f: self.freespace_goal_response_cb(f, next_state, speed_scale))

    def freespace_goal_response_cb(self, future, next_state, speed_scale):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('MoveGroup goal rejected')
            self.state = 'ADJUSTING_BASE'
            return
            
        goal_handle.get_result_async().add_done_callback(lambda f: self.freespace_result_cb(f, next_state, speed_scale))

    def freespace_result_cb(self, future, next_state, speed_scale):
        result = future.result().result
        
        if result.error_code.val != 1:
            self.get_logger().error(f"Freespace planning failed with error code: {result.error_code.val}")
            self.state = 'ADJUSTING_BASE'
            return
            
        self.get_logger().info(f"Freespace plan generated successfully. Executing {next_state}...")
        
        if not self.execute_client.wait_for_server(timeout_sec=5.0):
            self.state = 'DONE'
            return
            
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = result.planned_trajectory
        
        # Record force baseline before the guarded gap move
        if next_state == 'ENTERING_GAP':
            self.baseline_force_mag = getattr(self, 'current_force_mag', 0.0)
            self.state_start_time = self.get_clock().now()
        
        # current_fraction = 1.0 because OMPL always plans the full path or fails
        self.current_fraction = 1.0
        self.state = next_state
        future_goal = self.execute_client.send_goal_async(goal_msg)
        future_goal.add_done_callback(lambda f: self.goal_response_callback(f, next_state))

    def plan_cartesian_move(self, waypoints, next_state, speed_scale, check_reach=False):
        if not self.compute_path_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("compute_cartesian_path service not available!")
            self.state = 'DONE'
            return

        req = GetCartesianPath.Request()
        req.header.frame_id = self.get_parameter('planning_frame').value
        req.group_name = self.get_parameter('move_group').value
        req.waypoints = waypoints
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False

        pose_array = PoseArray()
        pose_array.header.frame_id = req.header.frame_id
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.poses = waypoints
        self.waypoint_pub.publish(pose_array)

        future = self.compute_path_client.call_async(req)
        future.add_done_callback(lambda f: self.path_computed_cb(f, next_state, speed_scale, check_reach))

    def path_computed_cb(self, future, next_state, speed_scale, check_reach):
        try:
            response = future.result()
            
            if check_reach and response.fraction < 0.98:
                self.get_logger().error(f"Reachability failed (fraction: {response.fraction:.4f}). Arm cannot reach point!")
                self.state = 'ADJUSTING_BASE'
                return

            if response.fraction < 0.20:
                self.get_logger().error(f"Path planning failed completely (fraction: {response.fraction:.4f}).")
                self.state = 'DONE'
                return

            self.current_fraction = response.fraction
            if response.fraction < 0.98:
                self.get_logger().warn(f"Partial path planned ({response.fraction*100:.1f}%). Executing segment...")
            else:
                self.get_logger().info(f"Full path planned successfully ({response.fraction*100:.1f}%). Executing {next_state}...")
            
            for pt in response.solution.joint_trajectory.points:
                sec = pt.time_from_start.sec
                nanosec = pt.time_from_start.nanosec
                total_time = sec + nanosec * 1e-9
                total_time *= speed_scale
                pt.time_from_start.sec = int(total_time)
                pt.time_from_start.nanosec = int((total_time - int(total_time)) * 1e9)
                pt.velocities = [v / speed_scale for v in pt.velocities]
                pt.accelerations = [a / (speed_scale * speed_scale) for a in pt.accelerations]

            if not self.execute_client.wait_for_server(timeout_sec=5.0):
                self.state = 'DONE'
                return

            goal_msg = ExecuteTrajectory.Goal()
            goal_msg.trajectory = response.solution
            
            # Record force baseline before guarded moves
            if next_state in ['ENTERING_GAP', 'OPENING_LID']:
                self.baseline_force_mag = getattr(self, 'current_force_mag', 0.0)
                self.state_start_time = self.get_clock().now()

            self.state = next_state
            future_goal = self.execute_client.send_goal_async(goal_msg)
            future_goal.add_done_callback(lambda f: self.goal_response_callback(f, next_state))
            
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            self.state = 'DONE'

    def goal_response_callback(self, future, next_state):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.state = 'DONE'
            return

        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(lambda f: self.get_result_callback(f, next_state))

    def get_result_callback(self, future, next_state):
        if self.state in ['DONE', 'FORCE_ABORT']:
            return

        result = future.result().result
        status = future.result().status
        self.current_goal_handle = None
        
        if status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn('Trajectory canceled (likely force limit).')
            return

        if result.error_code.val != 1:
            self.get_logger().error(f'Trajectory execution failed (error {result.error_code.val})')
            self.state = 'DONE'
            return

        self.get_logger().info(f'Finished execution for {next_state}.')
        
        if next_state == 'ENTERING_GAP':
            # Gripper has reached the lid grasp point — robot stops here.
            # Now trigger the arc-based lid opening motion.
            self.get_logger().info("Gripper confirmed at lid grasp point. Starting lid opening...")
            self.trigger_lid_opening()
        elif next_state == 'OPENING_LID':
            self.state = 'DONE'

    def trigger_lid_opening(self):
        self.get_logger().info("Transitioning to OPENING_LID...")
        
        planning_frame = self.get_parameter('planning_frame').value
        toilet_frame = self.get_parameter('toilet_frame').value
        end_effector = self.get_parameter('end_effector_frame').value
        
        T_world_toilet = self.get_transform_matrix(planning_frame, toilet_frame)
        T_world_gripper_current = self.get_transform_matrix(planning_frame, end_effector)
        
        if T_world_toilet is None or T_world_gripper_current is None:
            self.get_logger().error("TF failure before opening lid!")
            self.state = 'DONE'
            return

        q_hinge = euler_to_quaternion(*self.hinge_origin_rpy)
        T_toilet_hinge = np.dot(translation_matrix(self.hinge_origin_xyz), quaternion_matrix(q_hinge))
        T_world_hinge = np.dot(T_world_toilet, T_toilet_hinge)

        # Calculate relationship between hinge and current gripper pose
        T_world_hinge_inv = np.linalg.inv(T_world_hinge)
        T_hinge_gripper_initial = np.dot(T_world_hinge_inv, T_world_gripper_current)

        theta_end = self.get_parameter('theta_end').value
        num_wp = self.get_parameter('num_arc_waypoints').value
        thetas = np.linspace(0.0, theta_end, num_wp)
        
        waypoints = []
        for theta in thetas[1:]: # Skip 0.0 (current pose)
            q_rot = euler_to_quaternion(0, 0, theta)
            T_hinge_rot = quaternion_matrix(q_rot)

            T_world_gripper_new = np.dot(T_world_hinge, np.dot(T_hinge_rot, T_hinge_gripper_initial))

            pose = Pose()
            pose.position.x = T_world_gripper_new[0, 3]
            pose.position.y = T_world_gripper_new[1, 3]
            pose.position.z = T_world_gripper_new[2, 3]

            q_final = matrix_to_quaternion(T_world_gripper_new)
            pose.orientation.x = q_final[0]
            pose.orientation.y = q_final[1]
            pose.orientation.z = q_final[2]
            pose.orientation.w = q_final[3]
            waypoints.append(pose)

        speed = self.get_parameter('open_speed_scale').value
        self.plan_cartesian_move(waypoints, 'OPENING_LID', speed)

def main(args=None):
    rclpy.init(args=args)
    node = LidSeatManipulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()