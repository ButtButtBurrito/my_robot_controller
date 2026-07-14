#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__("first_node")
       # self.get_logger().info("ROS2 butts")
        self.counter = 0
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        self.get_logger().info("Hello " + str(self.counter))
        self.counter += 1
def main(args=None):
    rclpy.init(args=args)

    #You write everything here until before shutdown
    node = MyNode()
    rclpy.spin(node) #node keep running until shutdown (ctrl +c)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
