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
import cv2
import logging

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from vlm_interface.srv import SemanticSimilarity

from blip2_ros.ros2_wrapper.utils import ros2_image_to_pil


class VLMBaseLifecycleNode(LifecycleNode):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.rgb_image = None

    def on_configure(self, state: State):
        try:
            self.model = self.load_model()
            if self.model:
                self.get_logger().info("Model loaded successfully.")
                return TransitionCallbackReturn.SUCCESS
            else:
                self.get_logger().error("Model loading failed.")
                return TransitionCallbackReturn.FAILURE
        except Exception as e:
            self.get_logger().error(f"Configuration failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State):
        try:
            self.image_sub = self.create_subscription(Image, '/rgb', self.image_callback, 10)
            self.create_services()
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Activation failed: {e}")
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
            self.get_logger().error(f"Image callback failed: {e}")

    # Diese Methoden implementieren Kindklassen:
    def load_model(self):
        raise NotImplementedError

    def create_services(self):
        raise NotImplementedError


class Blip2Node(VLMBaseLifecycleNode):
    def __init__(self):
        super().__init__('blip2_node')

    def load_model(self):
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            "blip_image_text_matching", "large", device=self.device, is_eval=True
        )
        return self.model

    def create_services(self):
        self.semantic_similarity_srv = self.create_service(
            SemanticSimilarity, 'semantic_similarity', self.semantic_similarity
        )

    def semantic_similarity(self, request, response):
        pil_image = ros2_image_to_pil(request.image, logger=self.get_logger())
        if pil_image is None:
            response.score = float('nan')
            return response
        img = self.vis_processors["eval"](pil_image).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](request.query)
        with torch.inference_mode():
            response.score = self.model({"image": img, "text_input": txt}, match_head="itc").item()
        return response
