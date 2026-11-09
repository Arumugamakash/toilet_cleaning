# TC Manipulation: Autonomous Mobile Manipulation for Restroom Maintenance

[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy_Jalisco-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange.svg)](https://gazebosim.org/home)
[![MoveIt 2](https://img.shields.io/badge/MoveIt-2-green.svg)](https://moveit.picknik.ai/jazzy/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

An autonomous robotic mobile manipulation system designed for restroom maintenance tasks (toilet lid manipulation and flush actuation). The system integrates a custom 4-wheel differential-drive mobile base (**TBot**), a 7-DOF robotic arm (**UFactory xArm7**) equipped with a custom contact tool and wrist-mounted RGB-D camera, simulated in **Gazebo Harmonic** with kinematic planning powered by **MoveIt 2** and control through **ros2_control**.

---

## Table of Contents
- [System Overview](#system-overview)
- [Repository Structure](#repository-structure)
- [Package Overview: `tc_manipulation`](#package-overview-tc_manipulation)
- [Package Overview: `tc_manipulation_movit_config`](#package-overview-tc_manipulation_movit_config)
- [Dataset Pipeline: `dataset`](#dataset-pipeline-dataset)
- [System Requirements & Prerequisites](#system-requirements--prerequisites)
- [Setup & Build Instructions](#setup--build-instructions)
- [Portability & Environment Configuration](#portability--environment-configuration)
- [Step-by-Step Running Guide](#step-by-step-running-guide)
- [Detailed Node Explanations](#detailed-node-explanations)
- [Key Topics, Services, and Actions](#key-topics-services-and-actions)
- [Troubleshooting](#troubleshooting)
- [License & Maintainer](#license--maintainer)

---

## System Overview

```
                      +---------------------------------------+
                      |          Gazebo Harmonic Sim          |
                      |   - Toilet World (SDF)                |
                      |   - TBot Diff-Drive Plugin            |
                      |   - Wrist RGB-D Camera Sensor         |
                      |   - Force/Torque Sensor               |
                      +-------------------+-------------------+
                                          |
                                    ros_gz_bridge
                                          |
                      +-------------------+-------------------+
                      |               ROS 2 Core              |
                      +---------+-------------------+---------+
                                |                   |
           +--------------------+----+         +----+--------------------+
           |       ros2_control      |         |        MoveIt 2         |
           | - joint_state_broadc... |         | - move_group            |
           | - arm_controller (JTC)  |         | - OMPL / KDL Planners   |
           | - toilet_controller     |         | - Cartesian Path Service|
           +------------+------------+         +------------+------------+
                        |                                   |
           +------------+-----------------------------------+------------+
           |                    Autonomous Nodes                         |
           |  - flush_actuation_node (Base Nav + Arm Cartesian Press)   |
           |  - arc_waypoint_node (Base Nav + Arc Waypoint Lid Opening)  |
           |  - dataset_collector_node (Synchronized RGB-D Recording)   |
           |  - clean_and_split_dataset (Hash Dedup, Quality, Split)     |
           +-------------------------------------------------------------+
```

### Key Hardware & Simulation Components
- **Mobile Base (`tc_base_v2`)**: 4-wheel differential-drive chassis (`base_link`), wheel separation 0.62 m, wheel radius 0.075 m. Controlled via `/cmd_vel` using Gazebo's native `DiffDrive` system plugin.
- **Manipulator (`xArm7`)**: 7-DOF arm mounted at `xyz="0.23492 0.0 0.44766"` with pitch `-0.69813 rad` relative to `base_link`. Controlled via `arm_controller` (JointTrajectoryController).
- **Tool End-Effector**: Custom blade/pusher tip with tool center point (TCP) force/torque sensor (`tcp_fts_sensor`).
- **Sensor**: Wrist-mounted RGB-D camera publishing `/camera/image`, `/camera/depth_image`, `/camera/camera_info`, and `/camera/points`.
- **Target Model (`wsingle4`)**: Articulated restroom fixture featuring revolute joints (`lid_joint`, `seat_joint`) and prismatic button actuators (`large_button_axis`, `small_button_axis`).

---

## Repository Structure

```
my_project/src/
├── dataset/                                # Computer vision dataset directory
│   └── lid/
│       ├── rgb/                            # Raw RGB captures (.png)
│       ├── depth/                          # Raw metric 16-bit depth captures (.png)
│       ├── dataset_split/                  # Cleaned, curated, and split dataset
│       │   ├── train/ (rgb/, depth/)       # Training split (70%)
│       │   ├── val/   (rgb/, depth/)       # Validation split (20%)
│       │   ├── test/  (rgb/, depth/)       # Testing split (10%)
│       │   ├── data.yaml                   # YOLO-compatible dataset configuration
│       │   └── cleaning_report.txt         # Audit log of cleaned/removed frames
├── tc_manipulation/                        # Core ROS 2 Python package
│   ├── config/                             # Controller and hardware configurations
│   │   ├── arm_controller.yaml             # ros2_control JTC, FT sensor, admittance config
│   │   ├── diff_drive_controller.yaml      # Mobile base ros2_control configuration
│   │   └── toilet_controller.yaml          # Toilet fixture joint controllers
│   ├── launch/
│   │   └── display_launch.py               # Master simulation and bringup launch file
│   ├── meshes/                             # STL collision and visual meshes
│   │   ├── mobile_robot/                   # TBot chassis and wheel STLs
│   │   ├── toilet/                         # Toilet body, lid, seat, and button STLs
│   │   └── xarm7/                          # xArm7 links (1-7) and gripper meshes
│   ├── tc_manipulation/                    # Python executable nodes
│   │   ├── arc_waypoint_node.py            # Hinge-constrained arc trajectory planner for lid
│   │   ├── flush_actuation_node.py         # Multi-stage navigation and force-guided flush
│   │   ├── dataset_collector_node.py       # Live RGB-D capture GUI node
│   │   └── clean_and_split_dataset.py      # Perceptual hash dedup & dataset splitter
│   ├── urdf/                               # Robot and world Xacro descriptions
│   │   ├── main.urdf.xacro                 # Top-level combined robot (TBot + xArm7)
│   │   ├── tc_base_v2.urdf.xacro           # Mobile base kinematics & Gazebo plugins
│   │   ├── xarm7.urdf.xacro                # 7-DOF arm, sensors, and ros2_control hardware
│   │   └── wsingle4.urdf.xacro             # Toilet model with articulated joints
│   ├── worlds/
│   │   ├── empty.sdf                       # Basic Gazebo simulation world
│   │   └── toilet_model.sdf                # Gazebo environment containing restroom scene
│   ├── package.xml
│   └── setup.py
└── tc_manipulation_movit_config/           # MoveIt 2 configuration package
    ├── config/
    │   ├── combined_robot.srdf             # Semantic Robot Description (groups, poses, collisions)
    │   ├── combined_robot.urdf.xacro       # MoveIt URDF wrapper
    │   ├── initial_positions.yaml          # Default joint states (initial_pose, stow poses)
    │   ├── joint_limits.yaml               # Velocity and acceleration constraints
    │   ├── kinematics.yaml                 # Kinematic solver plugins (KDL)
    │   ├── moveit_controllers.yaml         # MoveIt-to-ROS 2 controller mapping
    │   └── moveit.rviz                     # Preconfigured RViz display setup
    ├── launch/
    │   ├── demo.launch.py                  # Standalone MoveIt test demo
    │   ├── move_group.launch.py            # MoveIt move_group action and planning pipeline
    │   └── moveit_rviz.launch.py           # RViz visualization with MotionPlanning plugin
    ├── CMakeLists.txt
    └── package.xml
```

---

## Package Overview: `tc_manipulation`

This is an `ament_python` package providing the core robotics logic:
1. **Model Description & Simulation (`urdf/`, `worlds/`)**: Combines the mobile platform and manipulator with dynamic Gazebo plugins for physics simulation, wheel odometry, and wrist RGB-D sensing.
2. **Controller Orchestration (`config/`, `launch/`)**: Launches `gz_ros2_control` systems, spawns joint broadcasters and trajectory controllers, and connects ROS 2 communication to Gazebo via `ros_gz_bridge`.
3. **Task Automation (`tc_manipulation/`)**:
   - Closed-loop proportional mobile navigation to approach target fixtures.
   - Kinematic reachability validation and Cartesian path execution via MoveIt 2.
   - Force-torque feedback monitoring to avoid hardware damage during contact tasks.
   - Synchronized computer vision data acquisition and curation.

---

## Package Overview: `tc_manipulation_movit_config`

Generated using the MoveIt Setup Assistant and tailored for the combined robot:
- **Planning Group**: `xarm7_arm` (Chain from `link_base` to `link7`, joints `joint1` through `joint7`).
- **Kinematic Solver**: `kdl_kinematics_plugin/KDLKinematicsPlugin` with search resolution `0.005 rad` and timeout `0.005 s`.
- **Predefined States**: `initial_pose`, `pose1`, `pose2`, `pose3`, `pose4`.
- **Controllers**: Configured for `arm_controller` through `FollowJointTrajectory`.

---

## Dataset Pipeline: `dataset`

Contains RGB and metric depth images collected from the wrist camera to train vision detection models (e.g., toilet lid / button pose estimation):
- **Raw Storage**: `src/dataset/lid/rgb` and `src/dataset/lid/depth`.
- **Cleaned & Split**: `src/dataset/lid/dataset_split/` containing balanced `train`, `val`, and `test` splits.
- **Labeling Format**: Compatible with YOLO vision training pipelines (`data.yaml`).

---

## System Requirements & Prerequisites

- **Operating System**: Ubuntu 24.04 LTS (Noble Numbat) or Ubuntu 22.04 LTS
- **ROS 2 Distribution**: ROS 2 Jazzy Jalisco (or Humble Hawksbill)
- **Gazebo Simulator**: Gazebo Harmonic (`ros_gz` packages)
- **Motion Planning**: MoveIt 2 for ROS 2 Jazzy
- **Python Libraries**:
  - `numpy`
  - `opencv-python` (`cv2`)
  - `setuptools`

### Install Required APT Packages
```bash
sudo apt update && sudo apt install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-joint-state-broadcaster \
    ros-jazzy-joint-trajectory-controller \
    ros-jazzy-force-torque-sensor-broadcaster \
    ros-jazzy-admittance-controller \
    ros-jazzy-moveit \
    ros-jazzy-moveit-ros-move-group \
    ros-jazzy-moveit-ros-visualization \
    ros-jazzy-moveit-planners-ompl \
    ros-jazzy-cv-bridge
```

---

## Setup & Build Instructions

### 1. Create Workspace and Clone Repository
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Arumugamakash/my_project.git .
```

### 2. Install ROS Dependencies via `rosdep`
```bash
cd ~/ros2_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

### 3. Install Python Dependencies
```bash
pip install opencv-python numpy
```

### 4. Build Packages
```bash
cd ~/ros2_ws
colcon build --symlink-install
```

### 5. Source the Workspace
```bash
source ~/ros2_ws/install/setup.bash
```
*(Tip: Add `source ~/ros2_ws/install/setup.bash` to your `~/.bashrc`)*

---

## Portability & Environment Configuration

> [!IMPORTANT]
> **Check Hardcoded Paths Before Running on a New Machine!**
> The current scripts reference the original developer path (`/home/arz--1016/arm_basics/my_project/src`). If your username or workspace directory differs, update or verify the following locations:
> 
> 1. **`tc_manipulation/launch/display_launch.py` (Line 19)**:
>    ```python
>    # Set mesh_file_path dynamically using your actual workspace src path:
>    mesh_file_path = os.path.expanduser('~/ros2_ws/src')
>    ```
>    This ensures Gazebo resolves `package://tc_manipulation/meshes/...` properly via `GZ_SIM_RESOURCE_PATH`.
>
> 2. **`tc_manipulation/tc_manipulation/dataset_collector_node.py` (Line 36)**:
>    ```python
>    self.base_dir = os.path.expanduser('~/ros2_ws/src/dataset/lid')
>    ```
>
> 3. **`tc_manipulation/tc_manipulation/clean_and_split_dataset.py` (Lines 35-37)**:
>    ```python
>    RGB_DIR = os.path.expanduser('~/ros2_ws/src/dataset/lid/rgb')
>    DEPTH_DIR = os.path.expanduser('~/ros2_ws/src/dataset/lid/depth')
>    OUTPUT_DIR = os.path.expanduser('~/ros2_ws/src/dataset/lid/dataset_split')
>    ```

---

## Step-by-Step Running Guide

### Step 1: Launch the Main Simulation, Controllers, and MoveIt
In your first terminal:
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch tc_manipulation display_launch.py
```
**What to expect**:
- Gazebo Harmonic opens with the toilet fixture model.
- The mobile manipulator (TBot + xArm7) spawns facing the toilet at `x=2.3m, yaw=3.14 rad`.
- The controller manager starts up: `joint_state_broadcaster`, `arm_controller`, `force_torque_sensor_broadcaster`, and namespaced toilet controllers.
- RViz launches displaying the MoveIt Motion Planning panel, robot model, and point cloud stream.

---

### Step 2: Choose an Autonomous Task

#### Option A: Run the Flush Actuation Task
In a second terminal:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run tc_manipulation flush_actuation
```
- The robot navigates autonomously to `x=0.73m` in front of the toilet.
- TF reachability to the flush button is checked.
- The xArm7 arm computes a Cartesian trajectory above the flush button.
- The end-effector presses down until force feedback detects contact or press depth is reached, then safely retracts.

#### Option B: Run the Toilet Lid Opening Task
In a second terminal:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run tc_manipulation arc_waypoint_node
```
- The robot navigates towards the toilet lid grasping zone.
- It calculates an arc trajectory centered on the lid's physical hinge axis (`[-0.07218, 0.0, 0.42234]`).
- The arm grasps and follows the circular arc to smoothly raise the lid from 0° to ~92°.

#### Option C: Collect RGB-D Vision Dataset
In a second terminal:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run tc_manipulation dataset_collector_node
```
- A GUI window showing the live RGB stream and JET-colorized normalized depth stream appears.
- Press **`s`** to save a synchronized RGB + 16-bit metric depth image pair.
- Press **`q`** or **`ESC`** to exit.

#### Option D: Clean, Deduplicate, and Split Dataset
In a second terminal:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run tc_manipulation clean_split_dataset
```
- Filters out duplicate/stationary frames using perceptual image hashing.
- Discards corrupted depth frames (>35% zero pixels) and blurry images (Laplacian variance < 40).
- Generates a 70% Train / 20% Val / 10% Test split with a full summary report in `cleaning_report.txt`.

---

## Detailed Node Explanations

### 1. `display_launch.py` (Master Orchestrator)
- **`robot_state_publisher`**: Parses `main.urdf.xacro` and publishes `/robot_description` and forward kinematics to `/tf`.
- **`toilet_state_publisher`**: Namespaced under `/toilet`, publishes the toilet model kinematics to `/toilet/robot_description`.
- **`gz_sim` & `create` (Spawn nodes)**: Starts the simulation engine and spawns both robot entities with specific coordinates.
- **`static_transform_publisher`**: Publishes constant transforms `world -> toilet_base_link` and `world -> odom`.
- **`ros_gz_bridge`**: Bridges bidirectional messaging between Gazebo and ROS 2:
  - Clock: `/clock`
  - Cameras: `/camera/image`, `/camera/depth_image`, `/camera/camera_info`, `/camera/points`
  - Control & Odometry: `/cmd_vel`, `/odom`
  - Kinematics: `/tf`, `/joint_states`
- **Controller Spawners**: Sequential event-driven startup of `joint_state_broadcaster`, `arm_controller`, `ft_broadcaster`, and `toilet_controller`.
- **MoveIt 2 Launchers**: Integrates `move_group.launch.py` and `moveit_rviz.launch.py`.

### 2. `flush_actuation_node.py` (`flush_actuation`)
- **State Machine Architecture**:
  - `NAVIGATING`: Proportional velocity control on `/cmd_vel` to reach target standoff (`base_target_x = 0.73 m`, `yaw = 3.14 rad`).
  - `VERIFYING_REACH`: Validates button coordinates (`x=-0.1932, y=0.0037, z=0.8665` in toilet frame) via TF2 buffer against arm reachable workspace.
  - `APPROACHING`: Plans and executes MoveIt Cartesian path positioning the blade tool 15 cm directly above the button.
  - `PRESSING`: Low-speed vertical trajectory with real-time force monitoring (`/force_torque_sensor_broadcaster/wrench`). If $\Delta F_z > 10.0\text{ N}$, execution cancels instantly (`FORCE_ABORT`) to prevent structural overload.
  - `RETRACTING`: Returns arm to safe pre-press position.

### 3. `arc_waypoint_node.py` (`arc_waypoint_node`)
- **Hinge-Constrained Manipulation**:
  - Solves the constrained kinematics of rotating a lid around a fixed mechanical hinge axis (`[-0.07218, 0.0, 0.42234]`).
  - Computes 20 angular waypoints from $\theta_{\text{start}} = 0.01\text{ rad}$ to $\theta_{\text{end}} = 1.62\text{ rad}$ (~93°).
  - Dynamically recalculates tool tip orientation at each point along the arc tangent using 3D rotation matrices and quaternions.
  - Submits Cartesian trajectory requests to MoveIt's `compute_cartesian_path` service and coordinates base adjustments if joint limits are approached.

### 4. `dataset_collector_node.py` (`dataset_collector_node`)
- Subscribes to `/camera/image` (BGR8) and `/camera/depth_image` (32FC1).
- Converts ROS Image messages into OpenCV format using `CvBridge`.
- Renders an 8-bit normalized JET-colormap depth preview for visualization while preserving exact millimeter float depth values for storage.
- Listens for keyboard triggers (`s`) to store frame-synchronized datasets.

### 5. `clean_and_split_dataset.py` (`clean_split_dataset`)
- **Verification**: Validates matching pairs between RGB and Depth channels.
- **Perceptual Hash Deduplication**: 256-bit average hash calculation with Hamming distance $\le 4$ and mean pixel difference $\le 2.0$ over a sliding window of 8 frames.
- **Quality Filtering**: Laplacian variance threshold ($< 40$) for blur; pixel zero-ratio threshold ($> 35\%$) for depth drops.
- **Partitioning**: Generates random seeded 70/20/10 train/validation/test split folders ready for YOLOv8/v11 training.

---

## Key Topics, Services, and Actions

### Published / Subscribed Topics
| Topic Name | Type | Direction | Description |
|------------|------|-----------|-------------|
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Published | Mobile base velocity control commands |
| `/odom` | `nav_msgs/msg/Odometry` | Subscribed | Mobile base wheel odometry |
| `/camera/image` | `sensor_msgs/msg/Image` | Subscribed | Raw RGB stream from wrist camera |
| `/camera/depth_image` | `sensor_msgs/msg/Image` | Subscribed | 32-bit floating point depth in meters |
| `/camera/points` | `sensor_msgs/msg/PointCloud2` | Subscribed | 3D point cloud stream in RViz |
| `/joint_states` | `sensor_msgs/msg/JointState` | Subscribed | Combined robot joint positions & velocities |
| `/force_torque_sensor_broadcaster/wrench` | `geometry_msgs/msg/WrenchStamped` | Subscribed | End-effector contact forces for compliant pressing |

### Services and Actions
| Interface Name | Type | Purpose |
|----------------|------|---------|
| `/compute_cartesian_path` | `moveit_msgs/srv/GetCartesianPath` | Computes smooth linear/arc end-effector trajectories |
| `/compute_ik` | `moveit_msgs/srv/GetPositionIK` | Verifies kinematic reachability for candidate waypoints |
| `/execute_trajectory` | `moveit_msgs/action/ExecuteTrajectory` | Action server executing validated trajectories on the arm |

---

## Troubleshooting

1. **Meshes not rendering in Gazebo (Invisible Robot/Toilet)**:
   - Ensure `GZ_SIM_RESOURCE_PATH` contains the directory with your packages.
   - Run: `export GZ_SIM_RESOURCE_PATH=~/ros2_ws/src:$GZ_SIM_RESOURCE_PATH`
2. **MoveIt Fails with "Transform error: lookupTransform..."**:
   - Check if `use_sim_time` is enabled on all nodes.
   - Verify that `display_launch.py` is actively running and publishing `/clock`.
3. **Arm Controllers fail to spawn**:
   - Check controller status: `ros2 control list_controllers`.
   - Ensure Gazebo has finished loading before controller spawners execute (event handlers handle this automatically, but on slow CPUs launch timeouts may need to be adjusted).
4. **Camera images are black or not updating**:
   - Confirm that the `ros_gz_bridge` is running: `ros2 topic hz /camera/image`.

---

## License & Maintainer

- **Developer / Maintainer**: Arumugam Akash ([arumugamakashn@gmail.com](mailto:arumugamakashn@gmail.com))
- **License**: BSD-3-Clause

