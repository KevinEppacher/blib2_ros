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


class Blip2LifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('Blip2_lifecycle_node')
        self.get_logger().info("Initializing Blip2 Lifecycle Node...")

        rclpy.logging.set_logger_level(self.get_logger().name, logging.DEBUG)

        self.bridge = CvBridge()

        self.get_logger().info("Finished initializing Blip2LifecycleNode.")


    def _declare_parameters(self):
        self.declare_parameter('config_name', 'focall_unicl_lang_demo.yaml', ParameterDescriptor(description='Config file name'))

    def on_configure(self, state):
        self.get_logger().info("Configuring Blip2 Lifecycle Node...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=self.device, is_eval=True)

        if self.model:
            self.get_logger().info("Model loaded successfully.")
            return TransitionCallbackReturn.SUCCESS
        self.get_logger().error("Failed to load model.")
        return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State):
        self.get_logger().info("Activating Blip2 Lifecycle Node...")

        try:
            self.image_sub = self.create_subscription(Image, '/rgb', self.image_callback, 10)

            self.semantic_similarity_srv = self.create_service(SemanticSimilarity, 'semantic_similarity', self.semantic_similarity)

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

    def semantic_similarity(self, request, response):
        try:
            self.get_logger().debug(f"Received request: query='{request.query}'")
            
            pil_image = ros2_image_to_pil(request.image, logger=self.get_logger())
            if pil_image is None:
                self.get_logger().warn("PIL image conversion failed.")
                response.score = float('nan')
                return response
            
            self.get_logger().debug(f"PIL image size: {pil_image.size}")
            
            img = self.vis_processors["eval"](pil_image).unsqueeze(0).to(self.device)
            txt = self.text_processors["eval"](request.query)

            self.get_logger().debug(f"Text input: {txt}")

            with torch.inference_mode():
                cosine = self.model({"image": img, "text_input": txt}, match_head="itc").item()

            self.get_logger().debug(f"Similarity score: {cosine}")

            response.score = cosine
            return response
        except Exception as e:
            self.get_logger().error(f'Semantic similarity computation failed: {e}')
            response.score = float('nan')
            return response
