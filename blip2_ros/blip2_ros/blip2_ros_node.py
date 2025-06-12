import rclpy
from blip2_ros.ros2_wrapper.Blib2LifecycleNode import Blip2Node

def main(args=None):
    rclpy.init(args=args)
    node = Blip2Node()
    rclpy.spin(node)
    rclpy.shutdown()
