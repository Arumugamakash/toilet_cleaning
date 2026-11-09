import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class DatasetCollector(Node):
    def __init__(self):
        super().__init__('dataset_collector')

        # ---- Topic Parameters ----
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('depth_topic', '/camera/depth_image')
        rgb_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value

        # Initialize cv_bridge
        self.bridge = CvBridge()

        # ---- Subscriptions (RGB + Depth, separate callbacks) ----
        self.rgb_subscription = self.create_subscription(
            Image,
            rgb_topic,
            self.image_callback,
            10)

        self.depth_subscription = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10)

        # ---- Save Directories (RGB and Depth kept separate, same base folder) ----
        self.base_dir = '/home/arz--1016/arm_basics/my_project/src/dataset/lid'
        self.rgb_dir = os.path.join(self.base_dir, 'rgb_img')
        self.depth_dir = os.path.join(self.base_dir, 'depth_img')
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        self.image_count = 0

        # Latest received frames (updated continuously by callbacks)
        self.latest_rgb = None
        self.latest_depth_raw = None      # raw float32 depth (meters), for saving
        self.latest_depth_vis = None      # normalized 8-bit version, for on-screen preview only

        self.received_first_rgb = False
        self.received_first_depth = False

        # Timer to handle OpenCV UI (same as before)
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info(f"Dataset collector initialized.")
        self.get_logger().info(f"RGB topic:   '{rgb_topic}'")
        self.get_logger().info(f"Depth topic: '{depth_topic}'")
        self.get_logger().info(f"RGB images will be saved to:   '{self.rgb_dir}'")
        self.get_logger().info(f"Depth images will be saved to: '{self.depth_dir}'")
        self.get_logger().info("Press 's' or 'S' to save an RGB+Depth pair. Press 'q' or 'ESC' to quit.")

    # ------------------------------------------------------------------
    # RGB Callback
    # ------------------------------------------------------------------
    def image_callback(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if not self.received_first_rgb:
                self.get_logger().info("Received first RGB image!")
                self.received_first_rgb = True
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")

    # ------------------------------------------------------------------
    # Depth Callback
    # Gazebo depth_camera publishes 32-bit float depth in METERS (encoding '32FC1').
    # We keep the raw float array (for saving) and also build an 8-bit
    # normalized version (for on-screen preview only, NOT for saving).
    # ------------------------------------------------------------------
    def depth_callback(self, msg):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.latest_depth_raw = depth_raw

            # Build a preview-only visualization (8-bit, normalized, colorized)
            depth_clean = np.nan_to_num(depth_raw, nan=0.0, posinf=0.0, neginf=0.0)
            if depth_clean.max() > 0:
                depth_norm = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
            else:
                depth_norm = depth_clean
            depth_8u = depth_norm.astype(np.uint8)
            self.latest_depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

            if not self.received_first_depth:
                self.get_logger().info("Received first Depth image!")
                self.received_first_depth = True
        except Exception as e:
            self.get_logger().error(f"Failed to convert Depth image: {e}")

    # ------------------------------------------------------------------
    # Timer: display RGB + Depth preview side-by-side, handle key presses
    # ------------------------------------------------------------------
    def timer_callback(self):
        if self.latest_rgb is None:
            return  # wait until at least RGB has arrived

        # Build display frame: RGB alongside Depth preview (if available)
        if self.latest_depth_vis is not None:
            depth_resized = cv2.resize(
                self.latest_depth_vis,
                (self.latest_rgb.shape[1], self.latest_rgb.shape[0])
            )
            display = np.hstack((self.latest_rgb, depth_resized))
        else:
            display = self.latest_rgb

        cv2.imshow("Dataset Collector - RGB (left) | Depth (right)", display)
        key = cv2.waitKey(1) & 0xFF

        # ---- Save Pair on 's' ----
        if key == ord('s') or key == ord('S'):
            if self.latest_depth_raw is None:
                self.get_logger().warn("No depth frame received yet — skipping save (RGB+Depth must be paired).")
                return

            # Find next free index so RGB and Depth always share the SAME number
            filename = f"dataset_img_{self.image_count:04d}.png"
            rgb_path = os.path.join(self.rgb_dir, filename)
            depth_path = os.path.join(self.depth_dir, filename)
            while os.path.exists(rgb_path) or os.path.exists(depth_path):
                self.image_count += 1
                filename = f"dataset_img_{self.image_count:04d}.png"
                rgb_path = os.path.join(self.rgb_dir, filename)
                depth_path = os.path.join(self.depth_dir, filename)

            # --- Save RGB (standard 8-bit PNG) ---
            cv2.imwrite(rgb_path, self.latest_rgb)

            # --- Save Depth (16-bit PNG, millimeters, preserves precision) ---
            depth_clean = np.nan_to_num(self.latest_depth_raw, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = (depth_clean * 1000.0)                  # meters -> millimeters
            depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
            cv2.imwrite(depth_path, depth_mm)

            self.get_logger().info(f"Saved pair #{self.image_count:04d}:")
            self.get_logger().info(f"   RGB:   {rgb_path}")
            self.get_logger().info(f"   Depth: {depth_path}  (16-bit PNG, millimeters)")
            self.image_count += 1

        # ---- Quit on 'q' / ESC ----
        elif key == ord('q') or key == 27:
            self.get_logger().info("Quit key pressed. Shutting down...")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()