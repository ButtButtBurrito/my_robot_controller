#!/usr/bin/env python3
import rclpy 
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist

class MyDrawingNode(Node):

    def __init__(self):
        super().__init__("my_drawing")
        #create a publisher
        self.cmd_vel_pub_ = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)
        self.timer = self.create_timer(0.1, self.send_velocity_command) 
        self.step = 0
        self.step_time = 0.0
        self.get_logger().info("My drawing node has been started")

    def send_velocity_command(self):
        self.step_time += 0.1
        msg = Twist()
        duration_turn = 0.9
        duration_straight = 2.0

        if self.step == 0:
            # move straight for 2 seconds
            msg.linear.x = 1.0
            msg.angular.z = 0.0
            duration = duration_straight
        elif self.step == 1:
            # turn 90 degrees for 1 second
            msg.linear.x = 0.0
            msg.angular.z = math.pi / 2
            duration = duration_turn  
            
        elif self.step == 2:
            msg.linear.x = 1.0
            msg.angular.z = 0.0
            duration = duration_straight

        elif self.step == 3:
            msg.linear.x = 0.0
            msg.angular.z = math.pi / 2
            duration = duration_turn

        elif self.step == 4:
            msg.linear.x = 1.0
            msg.angular.z = 0.0
            duration = duration_straight

        elif self.step == 5:
            msg.linear.x = 0.0
            msg.angular.z = math.pi / 2
            duration = duration_turn
        
        elif self.step == 6:
            msg.linear.x = 1.0
            msg.angular.z = 0.0
            duration = 1.0

        elif self.step == 7:
            msg.linear.x = 2.0
            msg.angular.z = 1.0
            duration = 4.0
        
        elif self.step == 8:
            msg.linear.x = 1.0
            msg.angular.z = 0.0
            duration = 2.0

       

        if self.step_time > duration:
            self.step += 1
            self.step_time = 0.0
            
        self.cmd_vel_pub_.publish(msg) #publishes the message to the topic
      

def main(args=None):
    rclpy.init(args=args)
    node = MyDrawingNode()
    rclpy.spin(node) #node keep running until shutdown (ctrl +c)
    rclpy.shutdown()

