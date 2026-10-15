import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    RegisterEventHandler
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource


# Build MoveIt configs to pass parameters directly to RViz

def generate_launch_description():
    pkg_path = get_package_share_directory("tc_manipulation")
    mesh_file_path = '/home/arz--1016/arm_basics/my_project/src'
    world_path =os.path.join(pkg_path, 'worlds', 'toilet_model.sdf')
    # world_path =os.path.join(pkg_path, 'worlds', 'empty.sdf')
    # 1. Main Robot (TBot + xArm7)
    xacro_file = os.path.join(pkg_path, 'urdf', 'main.urdf.xacro')
    robot_description_content = xacro.process_file(xacro_file).toxml()
    robot_description = {"robot_description": robot_description_content}

    # 2. Toilet Robot Description & State Publisher (Namespaced)
    toilet_xacro_file = os.path.join(pkg_path, 'urdf', 'wsingle4.urdf.xacro')
    toilet_description_content = xacro.process_file(toilet_xacro_file).toxml()
    toilet_description = {"robot_description": toilet_description_content}

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    toilet_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="toilet_state_publisher",
        namespace="toilet",
        output="screen",
        parameters=[toilet_description, {"use_sim_time": True}],
    )
    # 3. Gazebo Simulator
    open_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={
            # "gz_args": "-r empty.sdf",
            'gz_args':f'-r {world_path}',
            "on_exit_shutdown": "true",
        }.items(),
    )

    # 4. Spawn TBot + xArm7 manipulator
    spawn_robot_node = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-name", "toilet_manipulation",
            "-allow_renaming", "true",
            "-x", "0.9", "-y", "0.0", "-z", "0.3",
            "-Y", "3.14159"
        ],
        parameters=[{"use_sim_time": True}],
    )

    # 5. Spawn Toilet model (reads fromnamespaced /toilet/robot_description)
    spawn_toilet_node = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "/toilet/robot_description",
            "-name", "toilet",
            "-allow_renaming", "true",
            "-x", "0.0", "-y", "0.0", "-z", "0.0",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # 6. Static TF Publishers establishing global 'world' frame hierarchy
    static_tf_toilet = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_toilet",
        arguments=[
            "--x", "0.0",
            "--y", "0.0",
            "--z", "0.0",
            "--yaw", "0.0",
            "--pitch", "0.0",
            "--roll", "0.0",
            "--frame-id", "world",
            "--child-frame-id", "toilet_base_link"
        ],
        parameters=[{"use_sim_time": True}],
    )

    static_tf_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_odom",
        arguments=[
            "--x", "0.9",
            "--y", "0.0",
            "--z", "0.035",
            "--yaw", "3.14159",
            "--pitch", "0.0",
            "--roll", "0.0",
            "--frame-id", "world",
            "--child-frame-id", "odom"
        ],
        parameters=[{"use_sim_time": True}],
    )

    # 7. Topic Bridge (bridging clock, camera, cmd_vel/odom, TF, and joint states)
    bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        arguments=[
            # clock
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            # camera topics
            "/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            # cmd_vel and odom
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            # tf
            "/model/toilet_manipulation/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/world/empty/model/toilet_manipulation/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
        ],
        remappings=[
            ("/model/toilet_manipulation/tf", "/tf"),
            ("/world/empty/model/toilet_manipulation/joint_state", "/joint_states"),
        ],
        
    )

    # 11. MoveIt Launch and RViz
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tc_manipulation_movit_config"), "launch", "move_group.launch.py")
        ),
        launch_arguments={"use_sim_time": "true"}.items()
    )

    moveit_rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tc_manipulation_movit_config"), "launch", "moveit_rviz.launch.py")
        ),
        launch_arguments={"use_sim_time": "true"}.items()
    )

    # 9. Controller Spawners (TBot + xArm7)
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    ft_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["force_torque_sensor_broadcaster"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    admittance_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["admittance_controller"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    # 10. Namespaced Spawners (Toilet)
    toilet_jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/toilet/controller_manager"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    toilet_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["toilet_controller", "-c", "/toilet/controller_manager"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    # Event handlers / Delay sequences
    delay_jsb_after_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot_node,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    delay_arm_controller_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner, ft_broadcaster_spawner, admittance_controller_spawner],
        )
    )

    delay_moveit_after_arm_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[move_group_launch, moveit_rviz_launch],
        )
    )

    delay_toilet_jsb_after_spawn_toilet = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_toilet_node,
            on_exit=[toilet_jsb_spawner],
        )
    )

    delay_toilet_controller_after_toilet_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=toilet_jsb_spawner,
            on_exit=[toilet_controller_spawner],
        )
    )

    launchDescriptionObject = LaunchDescription()
    launchDescriptionObject.add_action(SetParameter(name='use_sim_time', value=True))
    launchDescriptionObject.add_action(
        SetEnvironmentVariable(name="GZ_SIM_RESOURCE_PATH", value=mesh_file_path)
    )

    launchDescriptionObject.add_action(open_gazebo)
    launchDescriptionObject.add_action(robot_state_publisher)
    launchDescriptionObject.add_action(toilet_state_publisher)
    launchDescriptionObject.add_action(spawn_robot_node) 
    launchDescriptionObject.add_action(spawn_toilet_node)
    launchDescriptionObject.add_action(static_tf_toilet)
    launchDescriptionObject.add_action(bridge_node)
    launchDescriptionObject.add_action(static_tf_odom)
    # Event handlers
    launchDescriptionObject.add_action(delay_jsb_after_spawn)
    launchDescriptionObject.add_action(delay_arm_controller_after_jsb)
    launchDescriptionObject.add_action(delay_toilet_jsb_after_spawn_toilet)
    launchDescriptionObject.add_action(delay_toilet_controller_after_toilet_jsb)
    launchDescriptionObject.add_action(delay_moveit_after_arm_controller)
    
    return launchDescriptionObject