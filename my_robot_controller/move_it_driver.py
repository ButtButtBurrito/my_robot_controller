#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState

class MoveItServiceBridge(Node):

    def __init__(self):
        super().__init__('moveit_service_bridge')
        
        # 1. Connect to MoveIt's Planning Service
        self.get_logger().info("Connecting to MoveIt Planning Service...")
        self.cli = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for MoveIt service to come online...')
            
        # 2. Setup a publisher to physically drive the visualizer joints
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.get_logger().info("Connected successfully! Ready to plan.")

    def move_to_cartesian(self, x: float, y: float, z: float):
        """Asks MoveIt to plan a path to X, Y, Z and moves the robot arm."""
        self.get_logger().info(f"Planning route to: X={x}, Y={y}, Z={z}")
        
        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = "agx_arm" # Check if your group is called "arm" or "piper_arm"
        req.motion_plan_request.num_planning_attempts = 5
        req.motion_plan_request.allowed_planning_time = 2.0
        
        # Define Goal Constraints
        goal_constraints = Constraints()
        
        # A. Position Constraint
        pos_const = PositionConstraint()
        pos_const.header.frame_id = "base_link"
        pos_const.link_name = "link6" # Your end-effector tip link
        
        # Define a small tolerance bounding box around target coordinate
        bbox = SolidPrimitive()
        bbox.type = SolidPrimitive.BOX
        bbox.dimensions = [0.01, 0.01, 0.01] # 1 cm tolerance window
        pos_const.constraint_region.primitives.append(bbox)
        
        target_pose = PoseStamped()
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        pos_const.constraint_region.primitive_poses.append(target_pose.pose)
        pos_const.weight = 1.0
        
        # B. Orientation Constraint (Keep gripper facing down)
        ori_const = OrientationConstraint()
        ori_const.header.frame_id = "base_link"
        ori_const.link_name = "link6"
        ori_const.orientation.w = 1.0 # Downward/neutral orientation
        ori_const.absolute_x_axis_tolerance = 0.1
        ori_const.absolute_y_axis_tolerance = 0.1
        ori_const.absolute_z_axis_tolerance = 0.1
        ori_const.weight = 1.0
        
        goal_constraints.position_constraints.append(pos_const)
        goal_constraints.orientation_constraints.append(ori_const)
        req.motion_plan_request.goal_constraints.append(goal_constraints)
        
        # Send Request to MoveIt
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response and len(response.motion_plan_response.trajectory.joint_trajectory.points) > 0:
            self.get_logger().info("IK Path found! Executing trajectory...")
            self.execute_trajectory(response.motion_plan_response.trajectory.joint_trajectory)
        else:
            self.get_logger().error("MoveIt couldn't resolve the math. Coordinate is out of reach!")

    def execute_trajectory(self, trajectory: JointTrajectory):
        """Plays back the path points smoothly in the RViz visualizer."""
        import time
        for point in trajectory.points:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = trajectory.joint_names
            msg.position = point.positions
            self.joint_pub.publish(msg)
            time.sleep(0.08) # Smooth step interval playback speed

def main():
    rclpy.init()
    bridge = MoveItServiceBridge()

    # Define your Cartesian Star (X, Y, Z in meters from base)
    star_vertices = [
        (0.5,  0.00,  0.50),  # Vertex 1: Top Tip
        (0.5,  0.00,  0.80),  # Vertex 1: Top Tip
        (0.5,  0.00,  0.30),  # Vertex 1: Top Tip
        (0.5,  0.00,  0.90),  # Vertex 1: Top Tip
    
    
    ]

    try:
        for pt in star_vertices:
            bridge.move_to_cartesian(pt[0], pt[1], pt[2])
            import time
            time.sleep(1.0) # Pause at each star corner briefly
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()