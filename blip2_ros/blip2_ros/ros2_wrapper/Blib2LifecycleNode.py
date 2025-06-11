from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import State
from lifecycle_msgs.msg import Transition
import rclpy
from rclpy.lifecycle import TransitionCallbackReturn
from rcl_interfaces.msg import ParameterDescriptor
import os
from ament_index_python.packages import get_package_share_directory
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch


class Blip2LifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('Blip2_lifecycle_node')
        self.get_logger().info("Initializing Blip2 Lifecycle Node...")

        self.get_logger().info("Finished initializing Blip2LifecycleNode.")


    def _declare_parameters(self):
        self.declare_parameter('config_name', 'focall_unicl_lang_demo.yaml', ParameterDescriptor(description='Config file name'))

    def on_configure(self, state):
        if True:
            return TransitionCallbackReturn.SUCCESS
        return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State):
        try:
            self.image_sub = self.create_subscription(Image, '/rgb', self.image_callback, 10)

            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Activation failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: State):
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State):
        return TransitionCallbackReturn.SUCCESS

    def image_callback(self, msg: Image):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image processing failed: {e}')
