#!/usr/bin/env python3
"""
arm_hold.py
-----------
Publishes the Piper home pose at 50 Hz and does nothing else.
Run this whenever you launch arm_code but aren't running your
actual controller yet — keeps the model visible in RViz.

Replace (Ctrl+C) with your actual script when ready to move.

Usage:
    python3 arm_hold.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import threading

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

# Safe mid-range pose — arm visible and not jammed against any limit
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

TOPIC   = '/control/joint_states'
RATE_HZ = 50


class ArmHold(Node):
    def __init__(self):
        super().__init__('arm_hold')
        self.pub = self.create_publisher(JointState, TOPIC, 10)
        self.create_timer(1.0 / RATE_HZ, self._publish)
        threading.Thread(target=rclpy.spin, args=(self,), daemon=True).start()
        self.get_logger().info(
            'Holding arm at home pose. '
            'Ctrl+C and run your actual script when ready to move.'
        )

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = JOINT_NAMES
        msg.position = HOME
        msg.velocity = [0.0] * len(JOINT_NAMES)
        msg.effort   = [0.0] * len(JOINT_NAMES)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ArmHold()
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()