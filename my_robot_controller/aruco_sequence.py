#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import Trigger
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange

class ArucoSequence(Node):
    def __init__(self):
        super().__init__('aruco_sequence')
        
    # --- Parameters ---
        speed_descriptor = ParameterDescriptor(
            description='Time in seconds to complete the move (Lower = Faster)',
            floating_point_range=[FloatingPointRange(from_value=0.5, to_value=10.0, step=0.1)]
        )
        self.declare_parameter('move_duration', 4.0, speed_descriptor)

        # --- Action Client ---
        self._action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/arm_controller/follow_joint_trajectory'
        )
        
        # Piper Standard Joint Names
        self.joint_names = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6"
        ]

        # --- Pre-determined Poses (Radians) ---
        # TODO: Update with actual ArUco poses for your Piper setup (radians)
        self.poses = [
            [0.0,  0.0,   0.0,  0.0,  0.0,  0.0],   # Pose 1: Home
            [0.3, -0.5,   0.8, -0.3,  0.5,  0.0],   # Pose 2: placeholder
            [-0.3, -0.5,  0.8,  0.3,  0.5,  0.0],   # Pose 3: placeholder
            [0.0, -0.8,   1.0,  0.0,  0.8,  0.0],   # Pose 4: placeholder
        ]
        self.current_pose_index = -1

        # --- Custom Sequence Queue ---
        from collections import deque
        self.duration_queue = deque()
        from std_msgs.msg import String
        self.duration_sub = self.create_subscription(String, '/aruco_sequence/durations', self.duration_callback, 10)

        # --- Service Server ---
        self.srv_next = self.create_service(Trigger, '/aruco_sequence/next_pose', self.next_pose_callback)
        self.srv_reset = self.create_service(Trigger, '/aruco_sequence/reset', self.reset_callback)

        self.get_logger().info("Aruco Sequence Node Ready. Call /aruco_sequence/next_pose to start.")

    def duration_callback(self, msg):
        try:
            # Parse string "1, 2, 1.5" -> [1.0, 2.0, 1.5]
            durations = [float(x.strip()) for x in msg.data.split(',')]
            self.duration_queue.extend(durations)
            self.get_logger().info(f"Received duration sequence: {durations}. Queue size: {len(self.duration_queue)}")
        except ValueError:
            self.get_logger().error(f"Invalid duration format: {msg.data}")

    def send_goal(self, joint_angles):
        if not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Action server not available!")
            return False

        if self.duration_queue:
            duration_sec = self.duration_queue.popleft()
            self.get_logger().info(f"Using queued duration: {duration_sec}s")
        else:
            duration_sec = self.get_parameter('move_duration').value
        
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in joint_angles]
        
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        
        goal_msg.trajectory.points = [point]
        
        self.get_logger().info(f"Moving to Pose {self.current_pose_index + 1} in {duration_sec}s...")
        self._action_client.send_goal_async(goal_msg)
        return True

    def next_pose_callback(self, request, response):
        if self.current_pose_index + 1 >= len(self.poses):
            response.success = False
            response.message = "No more poses in sequence. Call /aruco_sequence/reset to restart."
            return response

        self.current_pose_index += 1
        target_pose = self.poses[self.current_pose_index]
        
        success = self.send_goal(target_pose)
        
        if success:
            response.success = True
            response.message = f"Moving to Pose {self.current_pose_index + 1}/{len(self.poses)}"
        else:
            response.success = False
            response.message = "Failed to send goal (Action Server unavailable?)"
            self.current_pose_index -= 1 # Revert index if failed
            
        return response

    def reset_callback(self, request, response):
        self.current_pose_index = -1
        self.duration_queue.clear()
        response.success = True
        response.message = "Sequence reset to beginning. Duration queue cleared."
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ArucoSequence()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()