import rclpy
from blib2_ros.ros2_wrapper.Blib2LifecycleNode import Blib2LifecycleNode

def main(args=None):
    rclpy.init(args=args)
    node = Blib2LifecycleNode()
    rclpy.spin(node)
    rclpy.shutdown()
