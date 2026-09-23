#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, Twist, WrenchStamped
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import RobotTrajectory
from rclpy.action import ActionClient
from moveit_msgs.action import ExecuteTrajectory
from action_msgs.msg import GoalStatus
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

class FlushActuationNode(Node):
    def __init__(self):
        super().__init__('flush_actuation_node')

        # Frames
        self.declare_parameter('planning_frame', 'world')
        self.declare_parameter('toilet_frame', 'toilet')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('end_effector_frame', 'link7')
        self.declare_parameter('move_group', 'xarm7_arm')

        # Base Navigation Parameters (Relative to toilet)
        self.declare_parameter('base_target_x', 0.73) # 73cm in front of toilet (optimal reach + chassis safety)
        self.declare_parameter('base_target_y', 0.0)
        self.declare_parameter('base_target_yaw', 3.14159) # Facing the toilet
        self.declare_parameter('min_safe_base_x', 0.67) # Minimum safe distance for base_link to avoid collision
        self.declare_parameter('max_base_speed', 0.06) # Reduced linear speed (0.06 m/s)
        self.declare_parameter('max_base_turn_speed', 0.20) # Reduced angular speed (0.20 rad/s)
        self.declare_parameter('nav_xy_tol', 0.04) # 4cm tolerance
        self.declare_parameter('nav_yaw_tol', 0.04) # ~2.3 degrees

        # Flush Button Parameters (Exact URDF values from wsingle4.urdf.xacro for large_button)
        self.declare_parameter('flush_btn_x', -0.19316)
        self.declare_parameter('flush_btn_y', 0.0037)
        self.declare_parameter('flush_btn_z', 0.8665) # x=-0.1932, y=0.0037, z=0.8665
        
        # Arm Motion Parameters (Slower speeds)
        self.declare_parameter('approach_z_offset', 0.15) # 15cm above button
        self.declare_parameter('press_depth', 0.005) # Reduced to 1cm press for a lighter touch
        self.declare_parameter('gripper_tip_offset', 0.02) # 2cm beyond link7
        self.declare_parameter('gripper_blade_offset_x', -0.050) # Negated to shift link7 to the opposite side, centering blade tip over button
        self.declare_parameter('gripper_blade_offset_y', 0.006) # Y offset of blade tip relative to cylinder center
        self.declare_parameter('gripper_blade_offset_z', 0.02) # Z length/extension of blade tip
        self.declare_parameter('gripper_yaw_offset', 3.14159) # Extra yaw rotation for wrist (e.g. 3.14159 to flip blade 180 degrees)
        self.declare_parameter('arm_speed_scale', 12.0) # Slower arm motion (12x slower)
        self.declare_parameter('press_speed_scale', 20.0) # Ultra slow for button pressing (25x slower)

        # Force Control Parameters
        self.declare_parameter('force_topic', '/force_torque_sensor_broadcaster/wrench')
        self.declare_parameter('max_press_force', 10.0) # Newtons (Z axis)

        # State Machine
        self.state = 'NAVIGATING'
        self.current_goal_handle = None

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers / Subscribers / Clients
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.waypoint_pub = self.create_publisher(PoseArray, 'flush_waypoints', 10)
        self.wrench_sub = self.create_subscription(WrenchStamped, self.get_parameter('force_topic').value, self.wrench_callback, 10)
        
        self.compute_path_client = self.create_client(GetCartesianPath, 'compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')

        self.get_logger().info("Flush Actuation Node initialized with updated safety & speed parameters. Starting state machine...")
        
        # Main control loop (runs at 10Hz)
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
        self.current_force_z = msg.wrench.force.z
        if self.state == 'PRESSING':
            delta_force = abs(self.current_force_z - getattr(self, 'baseline_force_z', 0.0))
            max_f = self.get_parameter('max_press_force').value
            
            if delta_force > max_f:
                self.get_logger().warn(f"FORCE LIMIT EXCEEDED! (Delta: {delta_force:.2f} N > {max_f} N). Cancelling press trajectory!")
                self.state = 'FORCE_ABORT'
                if self.current_goal_handle is not None:
                    self.current_goal_handle.cancel_goal_async()
                    self.current_goal_handle = None
                
                self.state = 'RETRACTING'
                self.get_logger().info("Transitioning to RETRACTING state...")
                self.timer.reset()

    def state_machine_tick(self):
        if self.state == 'NAVIGATING':
            self.navigate_to_toilet()
        elif self.state == 'VERIFYING_REACH':
            self.verify_reachability()
        elif self.state == 'APPROACHING':
            pass
        elif self.state == 'CONTACTING':
            pass
        elif self.state == 'PRESSING':
            pass
        elif self.state == 'RETRACTING':
            pass
        elif self.state == 'DONE':
            self.get_logger().info("Flush operation completed successfully!")
            self.timer.cancel()
            
    def navigate_to_toilet(self):
        planning_frame = self.get_parameter('planning_frame').value
        toilet_frame = self.get_parameter('toilet_frame').value
        base_frame = self.get_parameter('base_frame').value

        T_world_toilet = self.get_transform_matrix(planning_frame, toilet_frame)
        T_world_base = self.get_transform_matrix(planning_frame, base_frame)

        if T_world_toilet is None or T_world_base is None:
            return # Wait for TF

        # Target Pose in Toilet Frame
        tx = self.get_parameter('base_target_x').value
        ty = self.get_parameter('base_target_y').value
        tyaw = self.get_parameter('base_target_yaw').value
        
        min_safe_x = self.get_parameter('min_safe_base_x').value
        if tx < min_safe_x:
            self.get_logger().warn(f"Requested target X ({tx:.2f}m) is below min safe distance ({min_safe_x:.2f}m). Clamping to safe limit.")
            tx = min_safe_x

        q_target = euler_to_quaternion(0, 0, tyaw)
        T_toilet_target = np.dot(translation_matrix([tx, ty, 0]), quaternion_matrix(q_target))
        T_world_target = np.dot(T_world_toilet, T_toilet_target)

        # Calculate error in Base Frame
        T_base_target = np.dot(np.linalg.inv(T_world_base), T_world_target)
        
        err_x = T_base_target[0, 3]
        err_y = T_base_target[1, 3]
        err_yaw = math.atan2(T_base_target[1, 0], T_base_target[0, 0])

        dist = math.sqrt(err_x**2 + err_y**2)
        
        # Check current position of base_link in toilet frame to prevent chassis collision
        T_toilet_base = np.dot(np.linalg.inv(T_world_toilet), T_world_base)
        curr_base_x_in_toilet = T_toilet_base[0, 3]

        if curr_base_x_in_toilet < min_safe_x - 0.02:
            self.cmd_vel_pub.publish(Twist()) # Hard emergency stop
            self.get_logger().error(f"SAFETY STOP! Mobile base is too close to toilet ({curr_base_x_in_toilet:.3f}m < {min_safe_x:.3f}m). Halting base!")
            self.state = 'VERIFYING_REACH'
            return

        tol_xy = self.get_parameter('nav_xy_tol').value
        tol_yaw = self.get_parameter('nav_yaw_tol').value

        max_speed = self.get_parameter('max_base_speed').value
        max_turn = self.get_parameter('max_base_turn_speed').value

        msg = Twist()
        if dist > tol_xy or abs(err_yaw) > tol_yaw:
            # Proportional controller capped at reduced max speeds
            msg.linear.x = max(-max_speed, min(max_speed, err_x * 0.4))
            msg.angular.z = max(-max_turn, min(max_turn, err_yaw * 0.8))
            self.cmd_vel_pub.publish(msg)
        else:
            self.cmd_vel_pub.publish(Twist()) # Stop
            self.get_logger().info("Reached safe base pose with full collision margin. Transitioning to VERIFYING_REACH.")
            self.state = 'VERIFYING_REACH'

    def get_target_arm_pose(self, z_offset):
        planning_frame = self.get_parameter('planning_frame').value
        toilet_frame = self.get_parameter('toilet_frame').value
        T_world_toilet = self.get_transform_matrix(planning_frame, toilet_frame)
        if T_world_toilet is None:
            return None

        # Check if button frame exists in TF tree
        button_frames = ['large_button', 'small_button', 'flush_button']
        T_toilet_button = None
        for b_frame in button_frames:
            T_toilet_button = self.get_transform_matrix(toilet_frame, b_frame)
            if T_toilet_button is not None:
                bx = T_toilet_button[0, 3]
                by = T_toilet_button[1, 3]
                bz = T_toilet_button[2, 3]
                self.get_logger().info(f"Using dynamic TF for button ({b_frame}): x={bx:.4f}, y={by:.4f}, z={bz:.4f}")
                break

        if T_toilet_button is None:
            bx = self.get_parameter('flush_btn_x').value
            by = self.get_parameter('flush_btn_y').value
            bz = self.get_parameter('flush_btn_z').value

        # The actual target point for the physical tip of the gripper
        target_tip_point = [bx, by, bz + z_offset]

        # Desired Orientation: Gripper points straight down (-Z in toilet frame) with optional extra yaw rotation
        yaw_offset = self.get_parameter('gripper_yaw_offset').value
        q_toilet_gripper = euler_to_quaternion(0, math.pi, yaw_offset)
        T_toilet_gripper_tip = np.dot(translation_matrix(target_tip_point), quaternion_matrix(q_toilet_gripper))

        # Shift back by 3D blade edge offset vector along gripper_base_link local axes
        blade_ox = self.get_parameter('gripper_blade_offset_x').value
        blade_oy = self.get_parameter('gripper_blade_offset_y').value
        blade_oz = self.get_parameter('gripper_blade_offset_z').value
        if blade_oz == 0.0 and self.has_parameter('gripper_tip_offset'):
            blade_oz = self.get_parameter('gripper_tip_offset').value

        T_tip_gripperbase = translation_matrix([-blade_ox, -blade_oy, -blade_oz])
        T_toilet_gripperbase = np.dot(T_toilet_gripper_tip, T_tip_gripperbase)

        # Account for gripper_mount_joint: parent=link7, child=gripper_base_link (rpy="0 0 3.1708", xyz="0 0 0")
        link7_frame = self.get_parameter('end_effector_frame').value # link7
        T_link7_gripperbase = self.get_transform_matrix(link7_frame, 'gripper_base_link')
        if T_link7_gripperbase is None:
            # Fallback to URDF mount joint definition: rpy="0 0 3.1708"
            q_mount = euler_to_quaternion(0.0, 0.0, 3.1708)
            T_link7_gripperbase = quaternion_matrix(q_mount)

        # Calculate target link7 pose: T_toilet_link7 = T_toilet_gripperbase * inv(T_link7_gripperbase)
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
        approach_z = self.get_parameter('approach_z_offset').value
        approach_pose = self.get_target_arm_pose(approach_z)
        
        if approach_pose is None:
            return

        self.get_logger().info("Planning Cartesian path to Approach Pose...")
        
        planning_frame = self.get_parameter('planning_frame').value
        end_effector = self.get_parameter('end_effector_frame').value
        T_curr = self.get_transform_matrix(planning_frame, end_effector)
        if T_curr is None:
            return
            
        start_pose = Pose()
        start_pose.position.x = T_curr[0, 3]
        start_pose.position.y = T_curr[1, 3]
        start_pose.position.z = T_curr[2, 3]
        q_start = matrix_to_quaternion(T_curr)
        start_pose.orientation.x = q_start[0]
        start_pose.orientation.y = q_start[1]
        start_pose.orientation.z = q_start[2]
        start_pose.orientation.w = q_start[3]
        
        waypoints = [start_pose, approach_pose]
        
        self.state = 'WAITING_PLAN'
        self.plan_cartesian_move(waypoints, next_state='APPROACHING', speed_scale=self.get_parameter('arm_speed_scale').value)

    def plan_cartesian_move(self, waypoints, next_state, speed_scale):
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
        req.avoid_collisions = False # MoveIt static planning scene contains toilet tank; set False to allow blade contact

        pose_array = PoseArray()
        pose_array.header.frame_id = req.header.frame_id
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.poses = waypoints
        self.waypoint_pub.publish(pose_array)

        future = self.compute_path_client.call_async(req)
        future.add_done_callback(lambda f: self.path_computed_cb(f, next_state, speed_scale))

    def path_computed_cb(self, future, next_state, speed_scale):
        try:
            response = future.result()
            if response.fraction < 0.20:
                self.get_logger().error(f"Path planning failed (fraction: {response.fraction:.4f}). Arm cannot reach point!")
                
                # If we failed to verify reachability, adjust target base pose safely
                if next_state == 'APPROACHING':
                    current_tx = self.get_parameter('base_target_x').value
                    min_safe_x = self.get_parameter('min_safe_base_x').value
                    
                    if current_tx - 0.03 >= min_safe_x:
                        new_tx = current_tx - 0.03
                        self.get_logger().warn(f"Safely adjusting base target closer to toilet: {current_tx:.2f}m -> {new_tx:.2f}m (min safe limit: {min_safe_x:.2f}m)")
                        self.set_parameters([rclpy.parameter.Parameter('base_target_x', rclpy.Parameter.Type.DOUBLE, new_tx)])
                        self.state = 'NAVIGATING'
                        return
                    else:
                        self.get_logger().error(f"Cannot adjust base any closer without violating minimum safety distance ({min_safe_x:.2f}m). Aborting operation for safety.")
                        self.state = 'DONE'
                        return

                self.state = 'DONE' # Abort for other stages
                return

            self.current_fraction = response.fraction
            if response.fraction < 0.98:
                self.get_logger().warn(f"Partial path planned ({response.fraction*100:.1f}%). Executing segment to unfold arm towards button...")
            else:
                self.get_logger().info(f"Full path planned successfully ({response.fraction*100:.1f}%). Executing...")
            
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
                self.get_logger().error("ExecuteTrajectory action server not available!")
                self.state = 'DONE'
                return

            goal_msg = ExecuteTrajectory.Goal()
            goal_msg.trajectory = response.solution
            
            self.state = next_state
            future_goal = self.execute_client.send_goal_async(goal_msg)
            future_goal.add_done_callback(lambda f: self.goal_response_callback(f, next_state))
            
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            self.state = 'DONE'

    def goal_response_callback(self, future, next_state):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server.')
            self.state = 'DONE'
            return

        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(lambda f: self.get_result_callback(f, next_state))

    def get_result_callback(self, future, next_state):
        result = future.result().result
        status = future.result().status
        self.current_goal_handle = None
        
        if status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn('Trajectory was canceled (likely due to force limit).')
            if self.state == 'RETRACTING':
                self.trigger_retract()
            return

        if result.error_code.val != 1:
            self.get_logger().error(f'Trajectory execution failed with error code: {result.error_code.val}')
            self.state = 'DONE'
            return

        self.get_logger().info(f'Finished execution for {next_state}.')
        
        if next_state == 'APPROACHING':
            if hasattr(self, 'current_fraction') and self.current_fraction < 0.98:
                if not hasattr(self, 'partial_retries'):
                    self.partial_retries = 0
                self.partial_retries += 1
                
                if self.partial_retries >= 3:
                    self.get_logger().error('Stuck in partial path loop. Arm cannot fully reach target. Aborting!')
                    self.state = 'DONE'
                    return
                
                self.get_logger().info(f'Arm reached {self.current_fraction*100:.1f}% of approach path. Re-verifying remaining approach distance...')
                self.state = 'VERIFYING_REACH'
            else:
                self.partial_retries = 0
                self.trigger_contact()
        elif next_state == 'CONTACTING':
            self.trigger_press()
        elif next_state == 'PRESSING':
            self.trigger_retract()
        elif next_state == 'RETRACTING':
            self.state = 'DONE'

    def trigger_contact(self):
        self.get_logger().info("Transitioning to CONTACTING...")
        contact_pose = self.get_target_arm_pose(0.0)
        speed = self.get_parameter('arm_speed_scale').value
        self.plan_cartesian_move([contact_pose], 'CONTACTING', speed)

    def trigger_press(self):
        self.get_logger().info("Transitioning to PRESSING...")
        self.baseline_force_z = getattr(self, 'current_force_z', 0.0)
        press_z = -self.get_parameter('press_depth').value
        press_pose = self.get_target_arm_pose(press_z)
        speed = self.get_parameter('press_speed_scale').value
        self.plan_cartesian_move([press_pose], 'PRESSING', speed)

    def trigger_retract(self):
        self.get_logger().info("Transitioning to RETRACTING...")
        approach_z = self.get_parameter('approach_z_offset').value
        retract_pose = self.get_target_arm_pose(approach_z)
        speed = self.get_parameter('arm_speed_scale').value
        self.plan_cartesian_move([retract_pose], 'RETRACTING', speed)


def main(args=None):
    rclpy.init(args=args)
    node = FlushActuationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
