#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
import cv2


class LidFindNode(Node):

    def __init__(self):
        super().__init__('lid_find_node')

        # -----------------------------
        # ROS Parameters
        # -----------------------------
        self.declare_parameter(
            'model_path',
            '/home/arz--1016/Downloads/runs-20261015T142858Z-1-001/runs/segment/train/weights/best.pt'
        )
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('show_gui', True)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.show_gui = self.get_parameter('show_gui').get_parameter_value().bool_value

        # Check if weights file exists
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model weights file NOT found at: {model_path}')
            raise FileNotFoundError(f'Model file not found: {model_path}')

        # -----------------------------
        # YOLO model
        # -----------------------------
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(model_path)

        # -----------------------------
        # ROS image converter
        # -----------------------------
        self.bridge = CvBridge()

        # -----------------------------
        # Camera RGB image subscription
        # -----------------------------
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.get_logger().info(
            f'Lid Find Node Started | Topic: {image_topic} | Conf: {self.conf_threshold} | GUI: {self.show_gui}'
        )

    def image_callback(self, msg: Image):
        # Convert ROS Image → OpenCV image
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        # -----------------------------
        # YOLO inference
        # -----------------------------
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        # -----------------------------
        # Process results & extract lid
        # -----------------------------
        display_frame = frame
        for result in results:
            # result.plot() automatically overlays segmentation masks and bounding boxes
            if self.show_gui:
                display_frame = result.plot()

            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names.get(class_id, str(class_id))

                # Filter for 'lid' or 'lid_1'
                if 'lid' not in class_name.lower():
                    continue

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                self.get_logger().info(
                    f'LID FOUND | confidence={confidence:.2f} | bbox=({x1},{y1})-({x2},{y2})'
                )

        # -----------------------------
        # Display image (if enabled)
        # -----------------------------
        if self.show_gui:
            try:
                cv2.imshow('YOLO Lid Detection', display_frame)
                cv2.waitKey(1)
            except cv2.error as e:
                self.get_logger().warn(f'Display window error: {e}. Disabling GUI.')
                self.show_gui = False


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = LidFindNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error running LidFindNode: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()