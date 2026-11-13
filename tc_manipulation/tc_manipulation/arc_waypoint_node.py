#!/usr/bin/env python3
"""
node.py
=======
Autonomous Toilet Lid Manipulation Node.

Sequence:
  1. Closed-loop base navigation + MoveIt IK reachability check.
  2. Arm pre-grasp approach.
  3. Cartesian insertion of the gripper at the lid grasp point.
  4. NEW: settle + verify the gripper really is at the grasp point
          (corrective re-insertion if it is off).
  5. Lift the lid along a circular arc about the hinge.
     FIXED: the whole gripper pose (position AND orientation) now rotates
     rigidly with the lid about the hinge axis, so the contact point on the
     lid does not slide (previously orientation was frozen -> lid slid off
     the blade -> "miss / slip").
  6. Retract along the tool axis (disengages the blade cleanly).
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, PoseArray, Twist
from moveit_msgs.srv import GetPositionIK, GetCartesianPath
from moveit_msgs.msg import (
    PositionIKRequest,
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    JointConstraint,
)
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from action_msgs.msg import GoalStatus
from shape_msgs.msg import SolidPrimitive
import tf2_ros


# ─────────────────────────────────────────────────────────────────────────────
# Matrix & Quaternion Utilities
# ─────────────────────────────────────────────────────────────────────────────

def euler_to_quaternion(roll, pitch, yaw):
    """Convert RPY angles (radians) to quaternion [x, y, z, w]."""
    qx = (math.sin(roll / 2.0) * math.cos(pitch / 2.0) * math.cos(yaw / 2.0)
          - math.cos(roll / 2.0) * math.sin(pitch / 2.0) * math.sin(yaw / 2.0))
    qy = (math.cos(roll / 2.0) * math.sin(pitch / 2.0) * math.cos(yaw / 2.0)
          + math.sin(roll / 2.0) * math.cos(pitch / 2.0) * math.sin(yaw / 2.0))
    qz = (math.cos(roll / 2.0) * math.cos(pitch / 2.0) * math.sin(yaw / 2.0)
          - math.sin(roll / 2.0) * math.sin(pitch / 2.0) * math.cos(yaw / 2.0))
    qw = (math.cos(roll / 2.0) * math.cos(pitch / 2.0) * math.cos(yaw / 2.0)
          + math.sin(roll / 2.0) * math.sin(pitch / 2.0) * math.sin(yaw / 2.0))
    return [qx, qy, qz, qw]


def quaternion_matrix(q):
    """Convert quaternion [x, y, z, w] to 4x4 homogeneous rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),       2.0 * (x*z + y*w),       0.0],
        [2.0 * (x*y + z*w),       1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w),       0.0],
        [2.0 * (x*z - y*w),       2.0 * (y*z + x*w),       1.0 - 2.0 * (x*x + y*y), 0.0],
        [0.0,                     0.0,                     0.0,                     1.0],
    ], dtype=np.float64)


def translation_matrix(t):
    """Convert 3D translation [x, y, z] to 4x4 homogeneous matrix."""
    m = np.eye(4, dtype=np.float64)
    m[0:3, 3] = t
    return m


def matrix_to_quaternion(M):
    """Convert rotation submatrix of 4x4 homogeneous matrix to quaternion [x, y, z, w]."""
    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return [qx, qy, qz, qw]


def matrix_to_pose(M):
    """Convert 4x4 homogeneous matrix to geometry_msgs.msg.Pose."""
    p = Pose()
    p.position.x = float(M[0, 3])
    p.position.y = float(M[1, 3])
    p.position.z = float(M[2, 3])
    q = matrix_to_quaternion(M)
    p.orientation.x = q[0]
    p.orientation.y = q[1]
    p.orientation.z = q[2]
    p.orientation.w = q[3]
    return p


def scale_trajectory(traj, scale):
    """Stretch a RobotTrajectory in time by `scale` (>1 = slower)."""
    for pt in traj.joint_trajectory.points:
        t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
        t *= scale
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
        pt.velocities = [v / scale for v in pt.velocities]
        pt.accelerations = [a / (scale ** 2) for a in pt.accelerations]


# ─────────────────────────────────────────────────────────────────────────────
# Main Node Class
# ─────────────────────────────────────────────────────────────────────────────

class ToiletLidNode(Node):
    def __init__(self):
        super().__init__('toilet_lid_node')

        # ── Frame Parameters ──────────────────────────────────────────────────
        self.declare_parameter('planning_frame', 'world')
        self.declare_parameter('toilet_frame', 'toilet')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('gripper_frame', 'gripper_base_link')
        self.declare_parameter('move_group', 'xarm7_arm')

        # ── Mobile Base Navigation Parameters ──────────────────────────────────
        self.declare_parameter('base_target_x', 1.00)
        self.declare_parameter('base_target_y', 0.0)
        self.declare_parameter('base_target_yaw', 3.14159)
        self.declare_parameter('min_safe_base_x', 0.85)
        self.declare_parameter('max_base_speed', 0.06)
        self.declare_parameter('max_base_turn_speed', 0.15)
        self.declare_parameter('nav_xy_tol', 0.04)
        self.declare_parameter('nav_yaw_tol', 0.04)
        # Stop driving once the grasp point is within this distance of the arm shoulder.
        # Adjusted from 0.55 to 0.42 to ensure the robot gets close enough to reach the 92 deg full lift!
        self.declare_parameter('max_arm_reach', 0.42)
        # Names of the arm joints (used to build the joint-space pre-grasp goal)
        self.declare_parameter('arm_joint_names', ['joint1', 'joint2', 'joint3', 'joint4',
                                                   'joint5', 'joint6', 'joint7'])
        # Pre-grasp standoffs (m) tried in order until IK + planning succeed
        self.declare_parameter('pre_grasp_standoffs', [0.15, 0.10, 0.05])

        self.declare_parameter('use_toilet_relative_grasp', True)
        self.declare_parameter('grasp_target_x_toilet', 0.36287)
        self.declare_parameter('grasp_target_y_toilet', 0.0007090354338288307)
        self.declare_parameter('grasp_target_z_toilet', 0.36162)

        self.declare_parameter('grasp_target_x_world', -0.035139404237270355)
        self.declare_parameter('grasp_target_y_world', 0.002155768685042858)
        self.declare_parameter('grasp_target_z_world', 0.36162)

        self.declare_parameter('grasp_roll', 0.0)
        self.declare_parameter('grasp_pitch', 1.5708)
        self.declare_parameter('grasp_yaw', 3.14159)

        # Fingertip offset relative to gripper_base_link (local Z).
        # >>> CALIBRATE THIS: if it is wrong the blade stops short of / past the lid.
        self.declare_parameter('fingertip_offset_z', -0.060)

        self.declare_parameter('pre_grasp_x_offset', 0.15)
        self.declare_parameter('pre_grasp_offset_x', 0.15)  # alias
        self.declare_parameter('post_lift_retract_x', 0.15)
        # Retract along -Z of the tool (pull the blade straight out) instead of world X
        self.declare_parameter('retract_along_tool', True)

        # ── Speed / Scaling Parameters ─────────────────────────────────────────
        self.declare_parameter('arm_speed_scale', 6.0)
        self.declare_parameter('speed_scale', 5.0)

        # ── Grasp verification ─────────────────────────────────────────────────
        self.declare_parameter('settle_time', 1.0)          # s to let the arm stop
        self.declare_parameter('grasp_pos_tol', 0.035)      # m, max error before lifting (35 mm for sim contact/sag)
        self.declare_parameter('max_grasp_corrections', 2)  # re-insert attempts

        # ── Lid Arc Geometry Parameters ────────────────────────────────────────
        self.declare_parameter('theta_start', 0.01)
        self.declare_parameter('theta_end', 1.62)
        self.declare_parameter('num_waypoints', 30)
        # Orientation follow factor along the arc, tried in order.
        # 1.0 = gripper rotates exactly with the lid (no slipping).
        # Lower values only used if the wrist cannot reach 1.0 (more slip risk).
        self.declare_parameter('tilt_candidates', [1.0, 0.85, 0.70, 0.55])
        self.declare_parameter('arc_min_fraction', 0.95)    # accept path if >= this
        self.declare_parameter('cartesian_jump_threshold', 0.0)  # 0.0 disables jump cutoff

        # ── Staged Lifting & Deep Insertion Parameters ─────────────────────────
        self.declare_parameter('use_staged_lid_lift', True)
        self.declare_parameter('stage1_waypoints', 5)             # First N waypoints to crack open lid
        self.declare_parameter('deep_insert_distance', 0.05)       # m (5 cm) deeper into lid along blade / tool Z
        self.declare_parameter('deep_insert_along_tool', True)     # True = along tool +Z; False = world -X
        self.declare_parameter('deep_insert_speed_scale', 4.0)

        # Hinge location from URDF relative to toilet frame
        self.hinge_origin_xyz = [-0.07218, 0.0, 0.42234]
        self.hinge_origin_rpy = [1.5708, 0.0, 0.0]

        # ── State Machine Variables ────────────────────────────────────────────
        self.state = 'NAVIGATING_AND_CHECKING_REACH'
        self.is_checking_reachability = False
        self.reachability_confirmed = False
        self.current_goal_handle = None
        self.arc_waypoints = []
        self.settle_start = None
        self.grasp_corrections = 0
        self.arc_tilt_index = 0
        self.arc_best = None  # (fraction, solution, tilt)
        self.pre_grasp_standoffs = []
        self.pre_grasp_idx = 0

        # ── TF2 Buffer & Listener ──────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── ROS Publishers & Action/Service Clients ───────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.waypoint_pub = self.create_publisher(PoseArray, 'lid_arc_waypoints', 10)

        self.compute_ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.compute_path_client = self.create_client(GetCartesianPath, 'compute_cartesian_path')
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        self.execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')

        self.get_logger().info("Toilet Lid Node initialized.")
        self.timer = self.create_timer(0.1, self.control_loop)

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_param(self, name, default=None):
        try:
            return self.get_parameter(name).value
        except Exception:
            return default

    def get_transform_matrix(self, target_frame, source_frame):
        """Lookup transform and return 4x4 matrix (None on failure)."""
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
            q = [t.transform.rotation.x, t.transform.rotation.y,
                 t.transform.rotation.z, t.transform.rotation.w]
            tr = [t.transform.translation.x, t.transform.translation.y,
                  t.transform.translation.z]
            return np.dot(translation_matrix(tr), quaternion_matrix(q))
        except Exception:
            return None

    def get_gripper_matrix(self):
        planning_frame = self.get_param('planning_frame', 'world')
        gripper_frame = self.get_param('gripper_frame', 'gripper_base_link')
        T = self.get_transform_matrix(planning_frame, gripper_frame)
        if T is None:
            T = self.get_transform_matrix(planning_frame, 'gripper')
        if T is None:
            T = self.get_transform_matrix(planning_frame, 'link7')
        return T

    def get_current_poses(self):
        planning_frame = self.get_param('planning_frame', 'world')
        toilet_frame = self.get_param('toilet_frame', 'toilet')
        base_frame = self.get_param('base_frame', 'base_link')

        T_world_toilet = self.get_transform_matrix(planning_frame, toilet_frame)
        if T_world_toilet is None:
            T_world_toilet = self.get_transform_matrix(planning_frame, 'toilet_base_link')

        T_world_base = self.get_transform_matrix(planning_frame, base_frame)
        if T_world_base is None:
            T_world_base = self.get_transform_matrix(planning_frame, 'base_footprint')

        return T_world_toilet, T_world_base, self.get_gripper_matrix()

    def make_cartesian_request(self, waypoints):
        req = GetCartesianPath.Request()
        req.header.frame_id = self.get_param('planning_frame', 'world')
        req.group_name = self.get_param('move_group', 'xarm7_arm')
        req.link_name = self.get_param('gripper_frame', 'gripper_base_link')
        req.waypoints = waypoints
        req.max_step = 0.01
        req.jump_threshold = float(self.get_param('cartesian_jump_threshold', 5.0))
        req.avoid_collisions = False
        return req

    # ─────────────────────────────────────────────────────────────────────────
    # Grasp Pose & Base Target Calculation
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_grasp_matrix(self, extra_x_offset=0.0):
        """4x4 grasp matrix of gripper_base_link in world frame."""
        planning_frame = self.get_param('planning_frame', 'world')
        toilet_frame = self.get_param('toilet_frame', 'toilet')
        T_world_toilet = self.get_transform_matrix(planning_frame, toilet_frame)
        if T_world_toilet is None:
            T_world_toilet = self.get_transform_matrix(planning_frame, 'toilet_base_link')

        roll = self.get_param('grasp_roll', 0.0)
        pitch = self.get_param('grasp_pitch', 1.5708)
        yaw = self.get_param('grasp_yaw', 3.14159)
        q_rot = euler_to_quaternion(roll, pitch, yaw)

        use_toilet = self.get_param('use_toilet_relative_grasp', True)
        if use_toilet and T_world_toilet is not None:
            # gx = self.get_param('grasp_target_x_toilet', 0.37287)
            # gy = self.get_param('grasp_target_y_toilet', 0.00357)
            # gz = self.get_param('grasp_target_z_toilet', 0.44867)

            gx = self.get_param('grasp_target_x_toilet', 0.36287)
            gy = self.get_param('grasp_target_y_toilet', 0.0007090354338288307)
            gz = self.get_param('grasp_target_z_toilet', 0.36162)
            
            T_toilet_fingertip = np.dot(translation_matrix([gx, gy, gz]), quaternion_matrix(q_rot))
            T_world_fingertip = np.dot(T_world_toilet, T_toilet_fingertip)
        else:
            # gx = self.get_param('grasp_target_x_world', -0.0427)
            # gy = self.get_param('grasp_target_y_world', 0.00357)
            # gz = self.get_param('grasp_target_z_world', 0.44867)

            gx = self.get_param('grasp_target_x_world',-0.035139404237270355)
            gy = self.get_param('grasp_target_y_world', 0.002155768685042858)
            gz = self.get_param('grasp_target_z_world', 0.36162)
            T_world_fingertip = np.dot(translation_matrix([gx, gy, gz]), quaternion_matrix(q_rot))

        ft_offset_z = self.get_param('fingertip_offset_z', -0.060)
        T_world_gripper = np.dot(T_world_fingertip, translation_matrix([0.0, 0.0, ft_offset_z]))
        T_world_gripper[0, 3] += extra_x_offset
        return T_world_gripper

    def calculate_grasp_pose(self, extra_x_offset=0.0):
        return matrix_to_pose(self.calculate_grasp_matrix(extra_x_offset=extra_x_offset))

    def calculate_open_pose(self):
        """Pose of the gripper when the lid is at full open (theta = theta_end)."""
        planning_frame = self.get_param('planning_frame', 'world')
        toilet_frame = self.get_param('toilet_frame', 'toilet')
        T_world_toilet = self.get_transform_matrix(planning_frame, toilet_frame)
        if T_world_toilet is None:
            T_world_toilet = self.get_transform_matrix(planning_frame, 'toilet_base_link')
        if T_world_toilet is None:
            return None

        q_hinge = euler_to_quaternion(*self.hinge_origin_rpy)
        T_toilet_hinge = np.dot(translation_matrix(self.hinge_origin_xyz), quaternion_matrix(q_hinge))
        T_world_hinge = np.dot(T_world_toilet, T_toilet_hinge)

        T_world_grasp = self.calculate_grasp_matrix(extra_x_offset=0.0)
        T_hinge_grip_init = np.dot(np.linalg.inv(T_world_hinge), T_world_grasp)

        theta_end = float(self.get_param('theta_end', 1.62))
        tilt = 0.85
        d_deep = float(self.get_param('deep_insert_distance', 0.05))

        T_pos = np.dot(T_world_hinge, np.dot(
            quaternion_matrix(euler_to_quaternion(0.0, 0.0, theta_end)), T_hinge_grip_init))
        T_ori = np.dot(T_world_hinge, np.dot(
            quaternion_matrix(euler_to_quaternion(0.0, 0.0, theta_end * tilt)), T_hinge_grip_init))

        T_open = np.copy(T_ori)
        T_open[0:3, 3] = T_pos[0:3, 3]
        T_open = np.dot(T_open, translation_matrix([0.0, 0.0, d_deep]))
        return matrix_to_pose(T_open)

    # ─────────────────────────────────────────────────────────────────────────
    # State Machine Loop
    # ─────────────────────────────────────────────────────────────────────────

    def control_loop(self):
        if self.state == 'NAVIGATING_AND_CHECKING_REACH':
            self.step_navigating_and_checking()
        elif self.state == 'PLANNING_PRE_GRASP':
            self.step_planning_pre_grasp()
        elif self.state == 'EXECUTING_PRE_GRASP':
            pass
        elif self.state == 'POSITIONING_GRIPPER':
            self.step_positioning_gripper()
        elif self.state == 'EXECUTING_GRIPPER_POSITION':
            pass
        elif self.state == 'SETTLING_AND_VERIFYING':
            self.step_settling_and_verifying()
        elif self.state == 'GENERATING_STAGE1_ARC':
            self.step_generating_stage1_arc()
        elif self.state == 'EXECUTING_STAGE1_ARC':
            pass
        elif self.state == 'INSERTING_DEEP_GRASP':
            self.step_inserting_deep_grasp()
        elif self.state == 'EXECUTING_DEEP_INSERT':
            pass
        elif self.state == 'GENERATING_STAGE2_ARC':
            self.step_generating_stage2_arc()
        elif self.state == 'EXECUTING_STAGE2_ARC':
            pass
        elif self.state == 'GENERATING_ARC_AND_LIFTING':
            self.step_generating_arc_and_lifting()
        elif self.state == 'EXECUTING_LID_ARC':
            pass
        elif self.state == 'RETRACTING_FROM_GRASP':
            self.step_retracting_from_grasp()
        elif self.state == 'DONE':
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info("Lid opening sequence complete. Node is idle.", once=True)
            self.timer.cancel()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Navigate Mobile Base & Check Reachability Simultaneously
    # ─────────────────────────────────────────────────────────────────────────

    def step_navigating_and_checking(self):
        T_world_toilet, T_world_base, T_world_gripper = self.get_current_poses()

        if T_world_toilet is None or T_world_base is None:
            self.get_logger().info("Waiting for TF transforms (world -> toilet, world -> base_link)...", throttle_duration_sec=3.0)
            return

        tx = self.get_param('base_target_x', 1.00)
        ty = self.get_param('base_target_y', 0.0)
        tyaw = self.get_param('base_target_yaw', 3.14159)
        min_safe_x = self.get_param('min_safe_base_x', 0.85)
        tx = max(tx, min_safe_x)

        q_target = euler_to_quaternion(0.0, 0.0, tyaw)
        T_toilet_target = np.dot(translation_matrix([tx, ty, 0.0]), quaternion_matrix(q_target))
        T_world_target = np.dot(T_world_toilet, T_toilet_target)

        T_base_target = np.dot(np.linalg.inv(T_world_base), T_world_target)
        err_x = T_base_target[0, 3]
        err_y = T_base_target[1, 3]
        err_yaw = math.atan2(T_base_target[1, 0], T_base_target[0, 0])
        dist = math.hypot(err_x, err_y)

        T_toilet_base = np.dot(np.linalg.inv(T_world_toilet), T_world_base)
        curr_base_dist_x = T_toilet_base[0, 3]
        if curr_base_dist_x < (min_safe_x - 0.03):
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().warn(f"Safety boundary reached ({curr_base_dist_x:.3f}m). Stopping base navigation.")
            self.reachability_confirmed = True
            self.state = 'PLANNING_PRE_GRASP'
            return

        T_world_grasp = self.calculate_grasp_matrix(extra_x_offset=0.0)
        T_base_grasp = np.dot(np.linalg.inv(T_world_base), T_world_grasp)
        arm_dx = T_base_grasp[0, 3] - 0.23492
        arm_dy = T_base_grasp[1, 3]
        arm_dz = T_base_grasp[2, 3] - 0.44766
        dist_to_arm = math.sqrt(arm_dx**2 + arm_dy**2 + arm_dz**2)

        if T_world_gripper is not None:
            gp = T_world_gripper[0:3, 3]
            bp = T_world_base[0:3, 3]
            self.get_logger().info(
                f"[POS STATUS] Gripper: ({gp[0]:.3f}, {gp[1]:.3f}, {gp[2]:.3f}) | "
                f"Base: ({bp[0]:.3f}, {bp[1]:.3f}) | Dist to staging: {dist:.2f}m | Arm-to-grasp: {dist_to_arm:.2f}m",
                throttle_duration_sec=2.0
            )

        tol_xy = self.get_param('nav_xy_tol', 0.04)
        tol_yaw = self.get_param('nav_yaw_tol', 0.04)
        max_spd = self.get_param('max_base_speed', 0.06)
        max_trn = self.get_param('max_base_turn_speed', 0.15)

        if dist <= tol_xy and abs(err_yaw) <= tol_yaw:
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info("Target navigation coordinates reached. Stopping base.")
            self.reachability_confirmed = True
            self.state = 'PLANNING_PRE_GRASP'
            return

        cmd = Twist()
        cmd.linear.x = float(max(-max_spd, min(max_spd, err_x * 0.4)))
        cmd.angular.z = float(max(-max_trn, min(max_trn, err_yaw * 0.8)))
        self.cmd_vel_pub.publish(cmd)

        max_arm_reach = self.get_param('max_arm_reach', 0.42)
        if dist_to_arm <= max_arm_reach and not self.is_checking_reachability:
            target_open_pose = self.calculate_open_pose()
            target_grasp_pose = matrix_to_pose(T_world_grasp)
            self.is_checking_reachability = True
            self.check_moveit_reachability(target_open_pose, target_grasp_pose, dist_to_arm)

    def check_moveit_reachability(self, target_open_pose, target_grasp_pose, current_arm_dist):
        if not self.compute_ik_client.service_is_ready():
            self.is_checking_reachability = False
            return

        # Check full open pose (reaching 92.8 deg full lift is the limiting physical reach constraint)
        test_pose = target_open_pose if target_open_pose is not None else target_grasp_pose

        req = PositionIKRequest()
        req.group_name = self.get_param('move_group', 'xarm7_arm')
        req.ik_link_name = self.get_param('gripper_frame', 'gripper_base_link')
        req.pose_stamped.header.frame_id = self.get_param('planning_frame', 'world')
        req.pose_stamped.pose = test_pose
        req.timeout.sec = 0
        req.timeout.nanosec = 80000000
        req.avoid_collisions = False

        msg = GetPositionIK.Request()
        msg.ik_request = req

        future = self.compute_ik_client.call_async(msg)
        future.add_done_callback(lambda f: self.ik_check_callback(f, current_arm_dist))

    def ik_check_callback(self, future, arm_dist_when_called):
        self.is_checking_reachability = False
        try:
            res = future.result()
            if (self.state == 'NAVIGATING_AND_CHECKING_REACH'
                    and res.error_code.val == 1
                    and arm_dist_when_called <= self.get_param('max_arm_reach', 0.42)):
                self.get_logger().info(
                    f"MoveIt reachability CONFIRMED for full lid opening arc (arm dist={arm_dist_when_called:.2f}m)!"
                )
                self.cmd_vel_pub.publish(Twist())
                self.reachability_confirmed = True
                self.state = 'PLANNING_PRE_GRASP'
        except Exception as e:
            self.get_logger().warn(f"Reachability check exception: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Plan and Move Arm to Pre-Grasp Pose
    # ─────────────────────────────────────────────────────────────────────────

    def step_planning_pre_grasp(self):
        self.cmd_vel_pub.publish(Twist())
        self.pre_grasp_standoffs = [float(x) for x in
                                    self.get_param('pre_grasp_standoffs', [0.15, 0.10, 0.05])]
        self.pre_grasp_idx = 0
        self.state = 'EXECUTING_PRE_GRASP'  # blocks re-entry while async calls run
        self.try_pre_grasp_ik()

    def try_pre_grasp_ik(self):
        """Solve IK for the pre-grasp pose (collision-aware), then plan to that joint goal."""
        if self.pre_grasp_idx >= len(self.pre_grasp_standoffs):
            self.get_logger().error(
                "Pre-grasp failed for every standoff. Base is probably too far/too close "
                "or the pose collides. Check the move_group terminal for the reason.")
            self.state = 'DONE'
            return
        if not self.compute_ik_client.service_is_ready():
            self.get_logger().error("compute_ik service not available!")
            self.state = 'DONE'
            return

        standoff = self.pre_grasp_standoffs[self.pre_grasp_idx]
        pose = self.calculate_grasp_pose(extra_x_offset=standoff)
        self.get_logger().info(
            f"Pre-grasp IK (standoff={standoff:.2f} m): "
            f"x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}")

        req = PositionIKRequest()
        req.group_name = self.get_param('move_group', 'xarm7_arm')
        req.ik_link_name = self.get_param('gripper_frame', 'gripper_base_link')
        req.pose_stamped.header.frame_id = self.get_param('planning_frame', 'world')
        req.pose_stamped.pose = pose
        req.avoid_collisions = True
        req.timeout.sec = 0
        req.timeout.nanosec = 500000000  # 0.5 s

        msg = GetPositionIK.Request()
        msg.ik_request = req
        future = self.compute_ik_client.call_async(msg)
        future.add_done_callback(lambda f, s=standoff: self.pre_grasp_ik_cb(f, s))

    def pre_grasp_ik_cb(self, future, standoff):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Pre-grasp IK exception: {e}")
            self.state = 'DONE'
            return

        if res.error_code.val != 1:
            self.get_logger().warn(
                f"Pre-grasp IK failed for standoff {standoff:.2f} m (code {res.error_code.val}). Trying next.")
            self.pre_grasp_idx += 1
            self.try_pre_grasp_ik()
            return

        names = list(self.get_param('arm_joint_names', []))
        js = res.solution.joint_state
        c = Constraints()
        for n, p in zip(js.name, js.position):
            if n in names:
                jc = JointConstraint()
                jc.joint_name = n
                jc.position = float(p)
                jc.tolerance_above = 0.005
                jc.tolerance_below = 0.005
                jc.weight = 1.0
                c.joint_constraints.append(jc)
        if not c.joint_constraints:
            self.get_logger().error(
                f"None of arm_joint_names {names} found in IK solution {list(js.name)}. "
                f"Set the 'arm_joint_names' parameter.")
            self.state = 'DONE'
            return

        self.send_pre_grasp_plan(c)

    def send_pre_grasp_plan(self, constraints):
        if not self.move_group_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("MoveGroup action server not available! Aborting.")
            self.state = 'DONE'
            return

        speed_scale = self.get_param('arm_speed_scale', 6.0)
        scaling = max(0.01, min(1.0, 1.0 / speed_scale))

        req = MotionPlanRequest()
        req.group_name = self.get_param('move_group', 'xarm7_arm')
        req.max_velocity_scaling_factor = scaling
        req.max_acceleration_scaling_factor = scaling
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.goal_constraints.append(constraints)

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        future = self.move_group_client.send_goal_async(goal)
        future.add_done_callback(self.pre_grasp_goal_cb)

    def pre_grasp_goal_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Pre-grasp MoveGroup goal rejected.")
            self.state = 'DONE'
            return
        goal_handle.get_result_async().add_done_callback(self.pre_grasp_plan_result_cb)

    def pre_grasp_plan_result_cb(self, future):
        result = future.result().result
        if result.error_code.val != 1:
            self.get_logger().warn(
                f"Pre-grasp planning failed (code: {result.error_code.val}). Trying next standoff.")
            self.pre_grasp_idx += 1
            self.try_pre_grasp_ik()
            return

        self.get_logger().info("Pre-grasp plan succeeded. Executing trajectory...")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = result.planned_trajectory
        exec_future = self.execute_client.send_goal_async(goal)
        exec_future.add_done_callback(self.pre_grasp_exec_cb)

    def pre_grasp_exec_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Execute pre-grasp trajectory rejected.")
            self.state = 'DONE'
            return
        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.pre_grasp_done_cb)

    def pre_grasp_done_cb(self, future):
        self.current_goal_handle = None
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_CANCELED or result.error_code.val != 1:
            self.get_logger().error(f"Pre-grasp trajectory execution failed (code: {result.error_code.val}).")
            self.state = 'DONE'
            return

        self.get_logger().info("Arm reached pre-grasp position. Transitioning to POSITIONING_GRIPPER.")
        self.state = 'POSITIONING_GRIPPER'

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Insert Gripper at Final Grasp Point (Cartesian Approach)
    # ─────────────────────────────────────────────────────────────────────────

    def step_positioning_gripper(self):
        final_grasp_pose = self.calculate_grasp_pose(extra_x_offset=0.0)
        self.get_logger().info(
            f"Positioning gripper at grasping point: "
            f"x={final_grasp_pose.position.x:.3f}, y={final_grasp_pose.position.y:.3f}, z={final_grasp_pose.position.z:.3f}"
        )

        if not self.compute_path_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("compute_cartesian_path service not available!")
            self.state = 'DONE'
            return

        # Approach must be a clean straight line: reject jumps strictly
        req = self.make_cartesian_request([final_grasp_pose])

        self.state = 'EXECUTING_GRIPPER_POSITION'
        future = self.compute_path_client.call_async(req)
        future.add_done_callback(self.gripper_approach_computed_cb)

    def gripper_approach_computed_cb(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Cartesian approach failed with exception: {e}")
            self.state = 'DONE'
            return

        # Executing a partial insertion would stop short of the lid -> require ~full path
        if res.fraction < 0.95:
            self.get_logger().error(f"Cartesian approach fraction too low ({res.fraction:.2f}). Aborting.")
            self.state = 'DONE'
            return

        self.get_logger().info(f"Cartesian approach planned ({res.fraction*100:.1f}%). Executing...")
        scale_trajectory(res.solution, self.get_param('arm_speed_scale', 6.0))

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        exec_future = self.execute_client.send_goal_async(goal)
        exec_future.add_done_callback(self.gripper_approach_exec_cb)

    def gripper_approach_exec_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Cartesian approach execution rejected.")
            self.state = 'DONE'
            return
        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.gripper_approach_done_cb)

    def gripper_approach_done_cb(self, future):
        self.current_goal_handle = None
        result = future.result().result
        if result.error_code.val != 1:
            self.get_logger().error(f"Cartesian approach execution failed (code: {result.error_code.val}).")
            self.state = 'DONE'
            return

        self.get_logger().info("Insertion move finished. Settling and verifying grasp position...")
        self.settle_start = self.get_clock().now()
        self.state = 'SETTLING_AND_VERIFYING'

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3b (NEW): Settle & verify the gripper is really at the grasp point
    # ─────────────────────────────────────────────────────────────────────────

    def step_settling_and_verifying(self):
        if self.settle_start is None:
            self.settle_start = self.get_clock().now()
        elapsed = (self.get_clock().now() - self.settle_start).nanoseconds * 1e-9
        if elapsed < self.get_param('settle_time', 1.0):
            return

        T_cur = self.get_gripper_matrix()
        T_tgt = self.calculate_grasp_matrix(extra_x_offset=0.0)
        if T_cur is None or T_tgt is None:
            return  # wait for TF

        diff = T_cur[0:3, 3] - T_tgt[0:3, 3]
        err = float(np.linalg.norm(diff))
        tol = self.get_param('grasp_pos_tol', 0.035)

        self.get_logger().info(
            f"Grasp position check: cur=({T_cur[0,3]:.3f}, {T_cur[1,3]:.3f}, {T_cur[2,3]:.3f}) | "
            f"tgt=({T_tgt[0,3]:.3f}, {T_tgt[1,3]:.3f}, {T_tgt[2,3]:.3f}) | "
            f"err={err*1000:.1f} mm (dx={diff[0]*1000:.1f}, dy={diff[1]*1000:.1f}, dz={diff[2]*1000:.1f}) mm"
        )

        if err > tol:
            if self.grasp_corrections < self.get_param('max_grasp_corrections', 2):
                self.grasp_corrections += 1
                self.get_logger().warn(
                    f"Grasp position error {err*1000:.1f} mm > {tol*1000:.1f} mm. "
                    f"Correcting ({self.grasp_corrections}).")
                self.state = 'POSITIONING_GRIPPER'
                return

            if err < 0.050:
                self.get_logger().warn(
                    f"Gripper is {err*1000:.1f} mm from target, but within grasp envelope (<50 mm). "
                    f"Proceeding to lift lid from current gripper pose.")
            else:
                self.get_logger().error(
                    f"Gripper is {err*1000:.1f} mm from the grasp point after corrections. "
                    f"Lifting now would miss the lid -> aborting. Check fingertip_offset_z / toilet TF.")
                self.state = 'DONE'
                return

        if self.get_param('use_staged_lid_lift', True):
            self.get_logger().info(f"Grasp position verified (error {err*1000:.1f} mm <= {tol*1000:.1f} mm).")
            self.get_logger().info("Starting Stage 1: Cracking open lid (first 5 waypoints)...")
            self.arc_tilt_index = 0
            self.arc_best = None
            self.state = 'GENERATING_STAGE1_ARC'
        else:
            self.get_logger().info(f"Grasp position verified (error {err*1000:.1f} mm <= {tol*1000:.1f} mm). Lifting lid.")
            self.arc_tilt_index = 0
            self.arc_best = None
            self.state = 'GENERATING_ARC_AND_LIFTING'

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Arc Waypoint Generation (Common Geometry)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_arc_waypoints(self, tilt, theta_array=None, T_gripper_ref=None, theta_ref=0.0):
        """
        Waypoints for the gripper rotating about the lid hinge.

        Position : always on the exact circle about the hinge.
        Orientation: rotated about the hinge by theta * tilt.
            tilt = 1.0 -> gripper turns rigidly with the lid (contact point
                          stays fixed on the lid = NO SLIDING).
            tilt < 1.0 -> fallback if the wrist can't follow; the lid will
                          slide along the blade by (1 - tilt) * theta.
        """
        planning_frame = self.get_param('planning_frame', 'world')
        toilet_frame = self.get_param('toilet_frame', 'toilet')

        T_world_toilet = self.get_transform_matrix(planning_frame, toilet_frame)
        if T_world_toilet is None:
            T_world_toilet = self.get_transform_matrix(planning_frame, 'toilet_base_link')
        if T_world_toilet is None:
            self.get_logger().error("Could not obtain toilet transform for arc calculation!")
            return None

        q_hinge = euler_to_quaternion(*self.hinge_origin_rpy)
        T_toilet_hinge = np.dot(translation_matrix(self.hinge_origin_xyz), quaternion_matrix(q_hinge))
        T_world_hinge = np.dot(T_world_toilet, T_toilet_hinge)

        if T_gripper_ref is None:
            T_gripper_ref = self.get_gripper_matrix()
        if T_gripper_ref is None:
            self.get_logger().error("Could not obtain gripper transform for arc calculation!")
            return None

        R_ref = quaternion_matrix(euler_to_quaternion(0.0, 0.0, theta_ref))
        T_hinge_grip_ref = np.dot(np.linalg.inv(T_world_hinge), T_gripper_ref)
        T_hinge_at_zero = np.dot(np.linalg.inv(R_ref), T_hinge_grip_ref)

        if theta_array is None:
            theta_start = self.get_param('theta_start', 0.01)
            theta_end = self.get_param('theta_end', 1.62)
            num_waypoints = int(self.get_param('num_waypoints', 30))
            theta_array = np.linspace(theta_start, theta_end, num_waypoints)

        waypoints = []
        prev_q = None
        for theta in theta_array:
            # Full rotation -> position on circle
            T_pos = np.dot(T_world_hinge, np.dot(
                quaternion_matrix(euler_to_quaternion(0.0, 0.0, theta)), T_hinge_at_zero))

            # Orientation rotation relative to reference:
            d_theta = theta - theta_ref
            R_delta_ori = quaternion_matrix(euler_to_quaternion(0.0, 0.0, d_theta * tilt))
            T_ori = np.dot(T_world_hinge, np.dot(R_delta_ori, T_hinge_grip_ref))

            q = matrix_to_quaternion(T_ori)
            if prev_q is not None and np.dot(q, prev_q) < 0.0:
                q = [-c for c in q]  # keep quaternion continuous
            prev_q = q

            wp = Pose()
            wp.position.x = float(T_pos[0, 3])
            wp.position.y = float(T_pos[1, 3])
            wp.position.z = float(T_pos[2, 3])
            wp.orientation.x, wp.orientation.y, wp.orientation.z, wp.orientation.w = q
            waypoints.append(wp)

        pose_array = PoseArray()
        pose_array.header.frame_id = planning_frame
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.poses = waypoints
        self.waypoint_pub.publish(pose_array)

        self.get_logger().info(
            f"Generated {len(waypoints)} arc waypoints (tilt={tilt:.2f}, theta [{theta_array[0]:.2f} -> {theta_array[-1]:.2f}] rad)."
        )
        return waypoints

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: Crack Open Lid (First 5 Waypoints)
    # ─────────────────────────────────────────────────────────────────────────

    def step_generating_stage1_arc(self):
        candidates = list(self.get_param('tilt_candidates', [1.0]))
        if self.arc_tilt_index >= len(candidates):
            self.get_logger().error("No arc tilt candidates left for Stage 1. Aborting.")
            self.state = 'DONE'
            return
        tilt = float(candidates[self.arc_tilt_index])

        theta_start = self.get_param('theta_start', 0.01)
        theta_end = self.get_param('theta_end', 1.62)
        num_waypoints = int(self.get_param('num_waypoints', 30))
        all_thetas = np.linspace(theta_start, theta_end, num_waypoints)

        n_stage1 = int(self.get_param('stage1_waypoints', 5))
        n_stage1 = max(2, min(n_stage1, len(all_thetas)))
        stage1_thetas = all_thetas[:n_stage1]

        waypoints = self.generate_arc_waypoints(tilt, theta_array=stage1_thetas, theta_ref=0.0)
        if not waypoints or len(waypoints) < 2:
            self.get_logger().error("Stage 1 arc waypoint generation failed! Aborting.")
            self.state = 'DONE'
            return

        if not self.compute_path_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("compute_cartesian_path service not available!")
            self.state = 'DONE'
            return

        req = self.make_cartesian_request(waypoints)
        self.state = 'EXECUTING_STAGE1_ARC'
        future = self.compute_path_client.call_async(req)
        future.add_done_callback(lambda f, t=tilt: self.stage1_trajectory_computed_cb(f, t))

    def stage1_trajectory_computed_cb(self, future, tilt):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Stage 1 Cartesian path computation failed: {e}")
            self.state = 'DONE'
            return

        frac = res.fraction
        self.get_logger().info(f"Stage 1 Arc (tilt={tilt:.2f}) fraction = {frac*100:.1f}%")
        if self.arc_best is None or frac > self.arc_best[0]:
            self.arc_best = (frac, res.solution, tilt)

        candidates = list(self.get_param('tilt_candidates', [1.0]))
        min_frac = self.get_param('arc_min_fraction', 0.95)

        if frac < min_frac and self.arc_tilt_index + 1 < len(candidates):
            self.arc_tilt_index += 1
            self.state = 'GENERATING_STAGE1_ARC'
            return

        best_frac, best_sol, best_tilt = self.arc_best
        if best_frac < 0.2:
            self.get_logger().error(f"Stage 1 best fraction only {best_frac*100:.1f}%. Aborting.")
            self.state = 'DONE'
            return

        self.get_logger().info(f"Executing Stage 1 lift (crack open lid, 5 waypoints, fraction={best_frac*100:.1f}%)...")
        scale_trajectory(best_sol, self.get_param('speed_scale', 5.0))

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = best_sol
        exec_future = self.execute_client.send_goal_async(goal)
        exec_future.add_done_callback(self.stage1_exec_cb)

    def stage1_exec_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Stage 1 trajectory goal rejected.")
            self.state = 'DONE'
            return
        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.stage1_done_cb)

    def stage1_done_cb(self, future):
        self.current_goal_handle = None
        result = future.result().result
        if result.error_code.val != 1:
            self.get_logger().warn(f"Stage 1 execution finished with code: {result.error_code.val}")

        self.get_logger().info("Stage 1 complete! Lid is cracked open. Transitioning to INSERTING_DEEP_GRASP.")
        self.state = 'INSERTING_DEEP_GRASP'

    # ─────────────────────────────────────────────────────────────────────────
    # Deep Insertion: Push Gripper Deeper into the Opened Lid Gap
    # ─────────────────────────────────────────────────────────────────────────

    def step_inserting_deep_grasp(self):
        T_cur = self.get_gripper_matrix()
        if T_cur is None:
            return

        d_deep = float(self.get_param('deep_insert_distance', 0.05))
        along_tool = self.get_param('deep_insert_along_tool', True)

        if along_tool:
            # Move along tool +Z (forward along spatula blade, pointing towards hinge into the lid gap)
            T_deep = np.dot(T_cur, translation_matrix([0.0, 0.0, d_deep]))
            self.get_logger().info(
                f"Inserting gripper {d_deep*1000:.1f} mm deeper along blade (+Z) into lid gap for full grasp..."
            )
        else:
            # Move along world -X into the toilet
            T_deep = np.copy(T_cur)
            T_deep[0, 3] -= d_deep
            self.get_logger().info(
                f"Inserting gripper {d_deep*1000:.1f} mm deeper along world -X into lid gap for full grasp..."
            )

        deep_pose = matrix_to_pose(T_deep)

        if not self.compute_path_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("compute_cartesian_path service not available!")
            self.state = 'DONE'
            return

        req = self.make_cartesian_request([deep_pose])
        self.state = 'EXECUTING_DEEP_INSERT'
        future = self.compute_path_client.call_async(req)
        future.add_done_callback(self.deep_insert_computed_cb)

    def deep_insert_computed_cb(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Deep insertion Cartesian path failed: {e}")
            self.state = 'DONE'
            return

        frac = res.fraction
        self.get_logger().info(f"Deep insertion approach fraction = {frac*100:.1f}%")
        if frac < 0.5:
            self.get_logger().warn(
                f"Deep insertion fraction low ({frac*100:.1f}%). Proceeding directly to Stage 2 from current pose."
            )
            self.arc_tilt_index = 0
            self.arc_best = None
            self.state = 'GENERATING_STAGE2_ARC'
            return

        speed_scale = self.get_param('deep_insert_speed_scale', 4.0)
        scale_trajectory(res.solution, speed_scale)

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        exec_future = self.execute_client.send_goal_async(goal)
        exec_future.add_done_callback(self.deep_insert_exec_cb)

    def deep_insert_exec_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Deep insertion trajectory goal rejected.")
            self.state = 'DONE'
            return
        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.deep_insert_done_cb)

    def deep_insert_done_cb(self, future):
        self.current_goal_handle = None
        result = future.result().result
        if result.error_code.val != 1:
            self.get_logger().warn(f"Deep insertion finished with code: {result.error_code.val}")

        self.get_logger().info("Deep insertion complete! Full gripper blade is firmly engaged under the lid.")
        self.get_logger().info("Transitioning to GENERATING_STAGE2_ARC (lifting lid to fully open position)...")
        self.arc_tilt_index = 0
        self.arc_best = None
        self.state = 'GENERATING_STAGE2_ARC'

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: Lift Lid Along Arc to Full Open Position
    # ─────────────────────────────────────────────────────────────────────────

    def step_generating_stage2_arc(self):
        T_deep_gripper = self.get_gripper_matrix()
        if T_deep_gripper is None:
            return

        candidates = list(self.get_param('tilt_candidates', [1.0]))
        if self.arc_tilt_index >= len(candidates):
            self.get_logger().error("No arc tilt candidates left for Stage 2. Aborting.")
            self.state = 'DONE'
            return
        tilt = float(candidates[self.arc_tilt_index])

        theta_start = self.get_param('theta_start', 0.01)
        theta_end = self.get_param('theta_end', 1.62)
        num_waypoints = int(self.get_param('num_waypoints', 30))
        all_thetas = np.linspace(theta_start, theta_end, num_waypoints)

        n_stage1 = int(self.get_param('stage1_waypoints', 5))
        n_stage1 = max(2, min(n_stage1, len(all_thetas)))
        theta_ref = all_thetas[n_stage1 - 1]
        stage2_thetas = all_thetas[n_stage1 - 1:]

        waypoints = self.generate_arc_waypoints(
            tilt, theta_array=stage2_thetas, T_gripper_ref=T_deep_gripper, theta_ref=theta_ref
        )
        if not waypoints or len(waypoints) < 2:
            self.get_logger().error("Stage 2 arc waypoint generation failed! Aborting.")
            self.state = 'DONE'
            return

        if not self.compute_path_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("compute_cartesian_path service not available!")
            self.state = 'DONE'
            return

        req = self.make_cartesian_request(waypoints)
        self.state = 'EXECUTING_STAGE2_ARC'
        future = self.compute_path_client.call_async(req)
        future.add_done_callback(lambda f, t=tilt: self.stage2_trajectory_computed_cb(f, t))

    def stage2_trajectory_computed_cb(self, future, tilt):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Stage 2 Cartesian path computation failed: {e}")
            self.state = 'DONE'
            return

        frac = res.fraction
        self.get_logger().info(f"Stage 2 Arc (tilt={tilt:.2f}) fraction = {frac*100:.1f}%")
        if self.arc_best is None or frac > self.arc_best[0]:
            self.arc_best = (frac, res.solution, tilt)

        candidates = list(self.get_param('tilt_candidates', [1.0]))
        min_frac = self.get_param('arc_min_fraction', 0.95)

        if frac < min_frac and self.arc_tilt_index + 1 < len(candidates):
            self.arc_tilt_index += 1
            self.state = 'GENERATING_STAGE2_ARC'
            return

        best_frac, best_sol, best_tilt = self.arc_best
        if best_frac < 0.2:
            self.get_logger().error(f"Stage 2 best arc fraction only {best_frac*100:.1f}%. Aborting.")
            self.state = 'DONE'
            return
        if best_tilt < 1.0:
            self.get_logger().warn(
                f"Using tilt={best_tilt:.2f}: blade has deep engagement so lid will remain secure."
            )
 
        self.get_logger().info(f"Lifting lid to full open (tilt={best_tilt:.2f}, fraction={best_frac*100:.1f}%)...")
        scale_trajectory(best_sol, self.get_param('speed_scale', 5.0))

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = best_sol
        exec_future = self.execute_client.send_goal_async(goal)
        exec_future.add_done_callback(self.stage2_exec_cb)

    def stage2_exec_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Stage 2 trajectory goal rejected.")  
            self.state = 'DONE'
            return
        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.stage2_done_cb)

    def stage2_done_cb(self, future):
        self.current_goal_handle = None
        result = future.result().result
        if result.error_code.val != 1:
            self.get_logger().warn(f"Stage 2 execution finished with code: {result.error_code.val}")

        self.get_logger().info("Lid lifting complete (Stage 2 finished)! Transitioning to RETRACTING_FROM_GRASP.")
        self.state = 'RETRACTING_FROM_GRASP'

    # ─────────────────────────────────────────────────────────────────────────
    # Single-Stage Fallback: Full Arc Lifting
    # ─────────────────────────────────────────────────────────────────────────

    def step_generating_arc_and_lifting(self):
        candidates = list(self.get_param('tilt_candidates', [1.0]))
        if self.arc_tilt_index >= len(candidates):
            self.get_logger().error("No arc tilt candidates left. Aborting.")
            self.state = 'DONE'
            return
        tilt = float(candidates[self.arc_tilt_index])

        waypoints = self.generate_arc_waypoints(tilt)
        if not waypoints or len(waypoints) < 2:
            self.get_logger().error("Arc waypoint generation failed! Aborting.")
            self.state = 'DONE'
            return
        self.arc_waypoints = waypoints

        if not self.compute_path_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("compute_cartesian_path service not available!")
            self.state = 'DONE'
            return

        req = self.make_cartesian_request(waypoints)
        self.state = 'EXECUTING_LID_ARC'  # blocks re-entry while the async call runs
        future = self.compute_path_client.call_async(req)
        future.add_done_callback(lambda f, t=tilt: self.arc_trajectory_computed_cb(f, t))

    def arc_trajectory_computed_cb(self, future, tilt):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Arc Cartesian path computation failed: {e}")
            self.state = 'DONE'
            return

        frac = res.fraction
        self.get_logger().info(f"Arc (tilt={tilt:.2f}) fraction = {frac*100:.1f}%")
        if self.arc_best is None or frac > self.arc_best[0]:
            self.arc_best = (frac, res.solution, tilt)

        candidates = list(self.get_param('tilt_candidates', [1.0]))
        min_frac = self.get_param('arc_min_fraction', 0.95)

        # Wrist can't follow the lid fully -> try the next (less rigid) candidate
        if frac < min_frac and self.arc_tilt_index + 1 < len(candidates):
            self.arc_tilt_index += 1
            self.state = 'GENERATING_ARC_AND_LIFTING'
            return

        best_frac, best_sol, best_tilt = self.arc_best
        if best_frac < 0.2:
            self.get_logger().error(f"Best arc fraction only {best_frac*100:.1f}%. Aborting.")
            self.state = 'DONE'
            return
        if best_tilt < 1.0:
            self.get_logger().warn(
                f"Using tilt={best_tilt:.2f}: lid may slide on the blade. "
                f"Move the base so the wrist has more range (joint6 limits).")

        self.get_logger().info(f"Lifting lid (tilt={best_tilt:.2f}, fraction={best_frac*100:.1f}%)...")
        scale_trajectory(best_sol, self.get_param('speed_scale', 5.0))

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = best_sol
        exec_future = self.execute_client.send_goal_async(goal)
        exec_future.add_done_callback(self.arc_exec_cb)

    def arc_exec_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Arc execution trajectory goal rejected.")
            self.state = 'DONE'
            return
        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.arc_done_cb)

    def arc_done_cb(self, future):
        self.current_goal_handle = None
        result = future.result().result
        if result.error_code.val != 1:
            self.get_logger().warn(f"Arc execution finished with code: {result.error_code.val}")

        self.get_logger().info("Lid lifting complete! Transitioning to RETRACTING_FROM_GRASP.")
        self.state = 'RETRACTING_FROM_GRASP'

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Retract
    # ─────────────────────────────────────────────────────────────────────────

    def step_retracting_from_grasp(self):
        T_world_gripper = self.get_gripper_matrix()
        if T_world_gripper is None:
            self.get_logger().info("Lid opening sequence complete (no retract TF). Done.")
            self.state = 'DONE'
            return

        d = self.get_param('post_lift_retract_x', 0.15)
        if self.get_param('retract_along_tool', True):
            # Pull the blade straight out along its own axis
            T_ret = np.dot(T_world_gripper, translation_matrix([0.0, 0.0, -d]))
            retract_pose = matrix_to_pose(T_ret)
            self.get_logger().info(f"Retracting {d:.2f} m along tool -Z...")
        else:
            retract_pose = matrix_to_pose(T_world_gripper)
            retract_pose.position.x += d
            self.get_logger().info(f"Retracting {d:.2f} m along world +X...")

        if not self.compute_path_client.wait_for_service(timeout_sec=2.0):
            self.state = 'DONE'
            return

        req = self.make_cartesian_request([retract_pose])
        self.state = 'DONE'
        future = self.compute_path_client.call_async(req)
        future.add_done_callback(self.retract_computed_cb)

    def retract_computed_cb(self, future):
        try:
            res = future.result()
            if res.fraction > 0.3:
                scale_trajectory(res.solution, self.get_param('arm_speed_scale', 6.0))
                goal = ExecuteTrajectory.Goal()
                goal.trajectory = res.solution
                self.execute_client.send_goal_async(goal)
                self.get_logger().info("Retract trajectory sent to controller.")
        except Exception as e:
            self.get_logger().warn(f"Retract exception: {e}")
        self.get_logger().info("Task complete! Toilet lid is open and arm cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ToiletLidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()