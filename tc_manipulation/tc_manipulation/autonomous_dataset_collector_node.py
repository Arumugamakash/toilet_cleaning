#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class AutonomousDatasetCollector(Node):
    """
    Autonomous Dataset Collector Node
    ---------------------------------
    Automatically collects synchronized RGB and Depth image pairs at a 
    set interval or based on scene motion. Fully compatible with the 
    clean_and_split_dataset pipeline.
    """

    def __init__(self):
        super().__init__('autonomous_dataset_collector')

        # -----------------------------
        # ROS Parameters
        # -----------------------------
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('depth_topic', '/camera/depth_image')
        self.declare_parameter('save_dir', '/home/arz--1016/arm_basics/my_project/src/dataset/lid')
        self.declare_parameter('interval', 0.5)            # Auto-save interval in seconds
        self.declare_parameter('max_images', 100)          # Stop after saving N images (0 = infinite)
        self.declare_parameter('min_difference', 2.0)      # Mean pixel difference required to save (avoids duplicate static frames)
        self.declare_parameter('auto_start', True)         # Start recording immediately
        self.declare_parameter('show_gui', True)           # Show preview window

        rgb_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.base_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.interval = self.get_parameter('interval').get_parameter_value().double_value
        self.max_images = self.get_parameter('max_images').get_parameter_value().integer_value
        self.min_difference = self.get_parameter('min_difference').get_parameter_value().double_value
        self.is_recording = self.get_parameter('auto_start').get_parameter_value().bool_value
        self.show_gui = self.get_parameter('show_gui').get_parameter_value().bool_value

        # Directories
        self.rgb_dir = os.path.join(self.base_dir, 'rgb')
        self.depth_dir = os.path.join(self.base_dir, 'depth')
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        # Bridge
        self.bridge = CvBridge()

        # State buffers
        self.latest_rgb = None
        self.latest_depth_raw = None
        self.latest_depth_vis = None
        self.last_saved_rgb = None
        self.received_first_rgb = False
        self.received_first_depth = False

        # Find existing image count to avoid overwriting previous datasets
        self.image_count = self._get_next_index()

        # Subscriptions
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)

        # Timers: one for display (20 Hz) and one for autonomous capture
        self.display_timer = self.create_timer(0.05, self.display_callback)
        self.capture_timer = self.create_timer(self.interval, self.auto_capture_callback)

        self.get_logger().info('Autonomous Dataset Collector Initialized.')
        self.get_logger().info(f'RGB Topic:        {rgb_topic}')
        self.get_logger().info(f'Depth Topic:      {depth_topic}')
        self.get_logger().info(f'Save Directory:   {self.base_dir}')
        self.get_logger().info(f'Auto Interval:    {self.interval}s')
        self.get_logger().info(f'Target Max Count: {self.max_images} (0 = unlimited)')
        self.get_logger().info(f'Initial Index:    {self.image_count:04d}')
        self.get_logger().info("Controls: [SPACE] Pause/Resume, [S] Force Save, [Q/ESC] Quit")

    def _get_next_index(self):
        """Finds the next available image number so existing data is preserved."""
        existing_indices = []
        for d in [self.rgb_dir, self.depth_dir]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.startswith('dataset_img_') and f.endswith('.png'):
                        try:
                            num = int(f.replace('dataset_img_', '').replace('.png', ''))
                            existing_indices.append(num)
                        except ValueError:
                            pass
        return max(existing_indices) + 1 if existing_indices else 0

    def image_callback(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if not self.received_first_rgb:
                self.get_logger().info('First RGB frame received.')
                self.received_first_rgb = True
        except Exception as e:
            self.get_logger().error(f'RGB conversion error: {e}')

    def depth_callback(self, msg: Image):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.latest_depth_raw = depth_raw

            # Visual preview representation (8-bit colormap)
            depth_clean = np.nan_to_num(depth_raw, nan=0.0, posinf=0.0, neginf=0.0)
            if depth_clean.max() > 0:
                depth_norm = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
            else:
                depth_norm = depth_clean
            self.latest_depth_vis = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)

            if not self.received_first_depth:
                self.get_logger().info('First Depth frame received.')
                self.received_first_depth = True
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    def auto_capture_callback(self):
        """Called periodically to autonomously save frames if recording is active."""
        if not self.is_recording:
            return

        # Check limit
        if self.max_images > 0 and self.image_count >= self.max_images:
            self.get_logger().info(f'Reached target image count ({self.max_images}). Auto recording stopped.')
            self.is_recording = False
            return

        if self.latest_rgb is None or self.latest_depth_raw is None:
            return

        # Motion / difference check: avoid saving duplicate static frames
        if self.min_difference > 0 and self.last_saved_rgb is not None:
            diff = cv2.absdiff(self.latest_rgb, self.last_saved_rgb)
            mean_diff = float(np.mean(diff))
            if mean_diff < self.min_difference:
                # Scene has not moved enough, skip this tick
                return

        self._save_current_pair(tag='AUTO')

    def _save_current_pair(self, tag='AUTO'):
        if self.latest_rgb is None or self.latest_depth_raw is None:
            self.get_logger().warn('Cannot save: waiting for both RGB and Depth frames.')
            return

        filename = f'dataset_img_{self.image_count:04d}.png'
        rgb_path = os.path.join(self.rgb_dir, filename)
        depth_path = os.path.join(self.depth_dir, filename)

        # Double check to prevent accidental overwrite
        while os.path.exists(rgb_path) or os.path.exists(depth_path):
            self.image_count += 1
            filename = f'dataset_img_{self.image_count:04d}.png'
            rgb_path = os.path.join(self.rgb_dir, filename)
            depth_path = os.path.join(self.depth_dir, filename)

        # Save RGB (8-bit)
        cv2.imwrite(rgb_path, self.latest_rgb)

        # Save Depth (16-bit uint16 in millimeters for full precision)
        depth_clean = np.nan_to_num(self.latest_depth_raw, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_clean * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(depth_path, depth_mm)

        self.last_saved_rgb = self.latest_rgb.copy()

        self.get_logger().info(
            f'[{tag}] Saved pair #{self.image_count:04d} -> {filename}'
        )
        self.image_count += 1

    def display_callback(self):
        """Displays preview window and listens for keyboard interactions."""
        if not self.show_gui or self.latest_rgb is None:
            return

        # Stack RGB and Depth side-by-side
        if self.latest_depth_vis is not None:
            depth_resized = cv2.resize(
                self.latest_depth_vis,
                (self.latest_rgb.shape[1], self.latest_rgb.shape[0])
            )
            display = np.hstack((self.latest_rgb, depth_resized))
        else:
            display = self.latest_rgb.copy()

        # Add HUD overlay bar
        h, w = display.shape[:2]
        hud = display.copy()
        cv2.rectangle(hud, (0, 0), (w, 40), (0, 0, 0), -1)
        display = cv2.addWeighted(hud, 0.6, display, 0.4, 0)

        # Status text
        status_text = 'REC [ON]' if self.is_recording else 'PAUSED'
        status_color = (0, 255, 0) if self.is_recording else (0, 165, 255)
        cv2.putText(display, status_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        info_text = f'Saved: {self.image_count}'
        if self.max_images > 0:
            info_text += f' / {self.max_images}'
        info_text += f' | Int: {self.interval}s | [SPACE] Pause | [S] Save | [Q] Quit'
        cv2.putText(display, info_text, (130, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        try:
            cv2.imshow('Autonomous Dataset Collector', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Toggle pause / record
                self.is_recording = not self.is_recording
                state = 'RESUMED' if self.is_recording else 'PAUSED'
                self.get_logger().info(f'Autonomous collection {state}')

            elif key == ord('s') or key == ord('S'):  # Force single save
                self._save_current_pair(tag='MANUAL')

            elif key == ord('q') or key == 27:  # Quit
                self.get_logger().info('Quit signal received. Closing node...')
                rclpy.shutdown()

        except cv2.error as e:
            self.get_logger().warn(f'GUI error: {e}. Disabling preview window.')
            self.show_gui = False


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousDatasetCollector()

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
