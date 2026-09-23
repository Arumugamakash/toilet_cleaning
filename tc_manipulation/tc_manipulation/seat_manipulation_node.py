#!/usr/bin/env python3
"""
seat_manipulation_node.py
=========================
Simultaneous navigation and MoveIt reachability check for toilet seat/lid manipulation.

State machine
-------------
  APPROACHING         - Drive the mobile base using cmd_vel. Continuously check MoveIt reachability for grasp target.
  PLANNING_REACHED    - Reachability found. Stop base. Perform final MoveGroup plan.
  EXECUTING_GRASP     - Execute the arm motion to the grasp pose.
  OPENING_LID         - Execute arc-waypoint Cartesian trajectory to open the lid.
  DONE                - Node idle; process completed or aborted.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from rclpy.action import ActionClient
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from action_msgs.msg import GoalStatus
from shape_msgs.msg import SolidPrimitive

# Geometry Helpers
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

class SeatManipulationNode(Node):
    def __init__(self):
        super().__init__('seat_manipulation_node')

        self.declare_parameter('planning_frame', 'world')
        self.declare_parameter('move_group', 'xarm7')
        self.declare_parameter('ee_link', 'gripper_base_link')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        
        self.state = 'APPROACHING'

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 10
        )

        # Action and Service Clients
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        self.compute_path_client = self.create_client(
            GetCartesianPath, 'compute_cartesian_path'
        )
        self.execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')

        # Control variables
        self.checking_reachability = False
        
        # Define exact target grasp pose for gripper finger in world
        self.target_finger_pose = Pose()
        self.target_finger_pose.position.x = -0.02692192
        self.target_finger_pose.position.y = 0.006337
        self.target_finger_pose.position.z = 0.03983926
        # Downward pointing (pitch = pi/2)
        qx, qy, qz, qw = euler_to_quaternion(0.0, math.pi / 2.0, 0.0)
        self.target_finger_pose.orientation.x = qx
        self.target_finger_pose.orientation.y = qy
        self.target_finger_pose.orientation.z = qz
        self.target_finger_pose.orientation.w = qw

        # Subtract the offset for gripper_base_link in global coordinates
        # Offset user provided: x=0.064, y=0.033, z=-0.020
        self.target_base_pose = Pose()
        self.target_base_pose.position.x = self.target_finger_pose.position.x - 0.064
        self.target_base_pose.position.y = self.target_finger_pose.position.y - 0.033
        self.target_base_pose.position.z = self.target_finger_pose.position.z - (-0.020)
        self.target_base_pose.orientation = self.target_finger_pose.orientation

        self.get_logger().info("Seat Manipulation Node starting (APPROACHING)...")
        
        # Fast timer for base control
        self.base_timer = self.create_timer(0.1, self.base_control_tick)
        
        # Slower timer for moveit reachability checks
        self.reachability_timer = self.create_timer(1.5, self.check_reachability)

    def base_control_tick(self):
        """Controls base movement while in APPROACHING state."""
        if self.state == 'APPROACHING':
            # Publish forward velocity
            twist = Twist()
            twist.linear.x = 0.05  # 5 cm/s
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
        elif self.state == 'PLANNING_REACHED':
            # Stop the base
            twist = Twist()
            self.cmd_vel_pub.publish(twist)

    def check_reachability(self):
        """Periodically requests a plan_only MoveGroup action to check if the target is reachable."""
        if self.state != 'APPROACHING':
            return
            
        if self.checking_reachability:
            return # Wait until current check finishes
            
        self.get_logger().info("Checking MoveIt reachability for target pose...")
        self.checking_reachability = True
        
        goal_msg = self.build_move_group_goal(self.target_base_pose, plan_only=True)
        
        if not self.move_group_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("MoveGroup action server not available.")
            self.checking_reachability = False
            return
            
        future = self.move_group_client.send_goal_async(goal_msg)
        future.add_done_callback(self.reachability_goal_response_cb)

    def reachability_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Reachability check goal rejected.")
            self.checking_reachability = False
            return
            
        self.get_logger().debug("Reachability check goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.reachability_result_cb)

    def reachability_result_cb(self, future):
        result = future.result().result
        error_code = result.error_code.val
        
        if error_code == 1: # SUCCESS
            self.get_logger().info("Path reachable! Stopping base and switching to PLANNING_REACHED state.")
            self.state = 'PLANNING_REACHED'
            self.plan_final_trajectory()
        else:
            self.get_logger().info(f"Target not reachable yet (error_code={error_code}). Continuing base movement.")
            
        self.checking_reachability = False

    def plan_final_trajectory(self):
        """Plans and executes the final trajectory once base has stopped."""
        # Stop base completely
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        
        self.get_logger().info("Sending final MoveGroup goal to move arm...")
        goal_msg = self.build_move_group_goal(self.target_base_pose, plan_only=False)
        
        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("MoveGroup action server not available.")
            self.state = 'DONE'
            return
            
        future = self.move_group_client.send_goal_async(goal_msg)
        future.add_done_callback(self.final_goal_response_cb)

    def final_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Final MoveGroup goal rejected. Aborting.")
            self.state = 'DONE'
            return
            
        self.get_logger().info("Final MoveGroup goal accepted. Executing...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.final_result_cb)

    def final_result_cb(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            self.get_logger().info("Arm successfully reached the grasp pose.")
            self.state = 'OPENING_LID'
            self.open_lid()
        else:
            self.get_logger().error(f"Arm failed to reach grasp pose. Error code: {result.error_code.val}")
            self.state = 'DONE'

    def open_lid(self):
        """Computes and executes arc trajectory to open the lid."""
        self.get_logger().info("Starting lid opening process (computing arc path)...")
        
        if not self.compute_path_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("compute_cartesian_path service not available!")
            self.state = 'DONE'
            return
            
        waypoints = self.generate_lid_arc_waypoints()
        
        req = GetCartesianPath.Request()
        req.header.frame_id = self.get_parameter('planning_frame').value
        req.group_name = self.get_parameter('move_group').value
        req.waypoints = waypoints
        req.max_step = 0.01
        req.jump_threshold = 0.0
        
        future = self.compute_path_client.call_async(req)
        future.add_done_callback(self.arc_path_response_cb)

    def arc_path_response_cb(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"compute_cartesian_path call failed: {e}")
            self.state = 'DONE'
            return

        if response.fraction < 0.05:
            self.get_logger().error("Arc path planning failed (fraction < 5%). Aborting.")
            self.state = 'DONE'
            return
            
        self.get_logger().info(f"Arc path planned. Fraction achieved: {response.fraction:.4f}. Executing...")
        
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = response.solution
        
        if not self.execute_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("execute_trajectory server not available!")
            self.state = 'DONE'
            return
            
        exec_future = self.execute_client.send_goal_async(goal_msg)
        exec_future.add_done_callback(self.execute_arc_response_cb)

    def execute_arc_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Execute arc goal rejected.")
            self.state = 'DONE'
            return
            
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.execute_arc_result_cb)

    def execute_arc_result_cb(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Lid opening arc successfully completed!")
        else:
            self.get_logger().error(f"Lid opening arc execution failed with status: {status}")
        self.state = 'DONE'

    def generate_lid_arc_waypoints(self):
        """Generates waypoints for a simple circular arc to open the lid."""
        waypoints = []
        
        # Hinge assumption (can be tuned for the real toilet)
        hinge_offset_x = 0.20
        hinge_offset_z = 0.0
        
        center_x = self.target_base_pose.position.x + hinge_offset_x
        center_z = self.target_base_pose.position.z + hinge_offset_z
        
        radius = math.hypot(hinge_offset_x, hinge_offset_z)
        start_angle = math.atan2(-hinge_offset_z, -hinge_offset_x)
        end_angle = start_angle + math.radians(75) # Open 75 degrees
        
        num_points = 15
        angles = np.linspace(start_angle, end_angle, num_points)
        
        for idx, ang in enumerate(angles):
            p = Pose()
            p.position.x = center_x + radius * math.cos(ang)
            p.position.y = self.target_base_pose.position.y
            p.position.z = center_z + radius * math.sin(ang)
            
            # Interpolate orientation (pitching up)
            pitch = math.pi / 2.0 - (ang - start_angle)
            qx, qy, qz, qw = euler_to_quaternion(0.0, pitch, 0.0)
            p.orientation.x = qx
            p.orientation.y = qy
            p.orientation.z = qz
            p.orientation.w = qw
            
            waypoints.append(p)
            
        return waypoints

    def build_move_group_goal(self, target_pose, plan_only=False):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.get_parameter('move_group').value
        goal.request.num_planning_attempts = 3
        goal.request.allowed_planning_time = 2.0
        goal.request.max_velocity_scaling_factor = 0.2
        goal.request.max_acceleration_scaling_factor = 0.2
        
        goal.request.start_state.is_diff = True
        goal.request.workspace_parameters.header.frame_id = self.get_parameter('planning_frame').value
        goal.request.workspace_parameters.min_corner.x = -2.0
        goal.request.workspace_parameters.min_corner.y = -2.0
        goal.request.workspace_parameters.min_corner.z = -2.0
        goal.request.workspace_parameters.max_corner.x = 2.0
        goal.request.workspace_parameters.max_corner.y = 2.0
        goal.request.workspace_parameters.max_corner.z = 2.0
        
        # Set target constraint
        constraint = Constraints()
        constraint.name = "target_grasp_pose"
        
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.get_parameter('planning_frame').value
        pos_constraint.link_name = self.get_parameter('ee_link').value
        pos_constraint.weight = 1.0
        
        bv = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01] # 1cm tolerance
        bv.primitives.append(sphere)
        bv.primitive_poses.append(target_pose)
        pos_constraint.constraint_region = bv
        
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = self.get_parameter('planning_frame').value
        ori_constraint.link_name = self.get_parameter('ee_link').value
        ori_constraint.orientation = target_pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.05
        ori_constraint.absolute_y_axis_tolerance = 0.05
        ori_constraint.absolute_z_axis_tolerance = 0.05
        ori_constraint.weight = 1.0
        
        constraint.position_constraints.append(pos_constraint)
        constraint.orientation_constraints.append(ori_constraint)
        
        goal.request.goal_constraints.append(constraint)
        goal.planning_options.plan_only = plan_only
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        
        return goal

def main(args=None):
    rclpy.init(args=args)
    node = SeatManipulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
