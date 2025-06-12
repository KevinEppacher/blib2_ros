import rclpy
from blip2_ros.ros2_wrapper.blip2_ros import Blip2Node

def main(args=None):
    rclpy.init(args=args)
    node = Blip2Node()
    rclpy.spin(node)
    rclpy.shutdown()
