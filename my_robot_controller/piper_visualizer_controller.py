#!/usr/bin/env python3
"""
piper_visualizer_controller.py  (corrected from URDF)
------------------------------------------------------


Controls the Piper arm visualizer in RViz2 by publishing joint states.


CURRENT SETUP (what ros2 node/topic list shows):
 joint_state_publisher  →  /control/joint_states  →  robot_state_publisher  →  RViz2


HOW THIS SCRIPT FITS IN:
 This script REPLACES the joint_state_publisher as the source of joint positions.
 When you run this, your published positions drive the 3D model in RViz.


STEP 1 — Find your joint names (run this in a separate terminal):
 ros2 topic echo /control/joint_states --once
 Copy the list under "name:" and paste into JOINT_NAMES below.


STEP 2 — Stop the existing joint_state_publisher so it doesn't conflict:
 In the terminal where you launched the visualizer, Ctrl+C and relaunch
 without the joint_state_publisher, OR just run both and see which wins.


STEP 3 — Run this script:
 python3 piper_visualizer_controller.py

Joint limits extracted directly from piper_description.urdf:

  joint1  revolute  base → link1    axis Z   [-2.618,  2.618] rad  (±150°)
  joint2  revolute  link1 → link2   axis Z   [ 0.000,  3.142] rad  (0° to +180°)  ← NO negative!
  joint3  revolute  link2 → link3   axis Z   [-2.967,  0.000] rad  (-170° to 0°)  ← NO positive!
  joint4  revolute  link3 → link4   axis Z   [-1.745,  1.745] rad  (±100°)
  joint5  revolute  link4 → link5   axis Z   [-1.222,  1.222] rad  (±70°)
  joint6  revolute  link5 → link6   axis Z   [-2.094,  2.094] rad  (±120°)

WHY IT WAS GLITCHING:
  The previous file used symmetric ±45° demos for every joint.
  joint2 cannot go negative → model snapped to limit.
  joint3 cannot go positive → model snapped to limit.
  Snapping to a limit causes the visual spasm you saw.

HOME POSE (safe midpoints, not all-zeros):
  All-zeros puts joint2 AT its lower limit (0) and joint3 AT its upper
  limit (0). That is a degenerate configuration. The home pose below
  puts each joint at the centre of its actual travel range.

HOW TO RUN:
  python3 piper_visualizer_controller.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import time
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


# ── Joint names from URDF (order must match) ─────────────────────────
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

# ── Exact limits from URDF <limit lower="..." upper="..."> ───────────
JOINT_LIMITS = {
    'joint1': (-2.6179938,  2.6179938),
    'joint2': ( 0.0,        3.1415926),   # ← ONE-SIDED: only 0 → +180°
    'joint3': (-2.9670597,  0.0      ),   # ← ONE-SIDED: only -170° → 0
    'joint4': (-1.7453292,  1.7453292),
    'joint5': (-1.2217304,  1.2217304),
    'joint6': (-2.0943951,  2.0943951),
}

# ── Safe home pose: midpoint of each joint's travel range ────────────
# Calculated as (lower + upper) / 2 for each joint
HOME_POSE = [
    0.0,      # joint1:  mid of (-2.618, 2.618)
    1.5708,   # joint2:  mid of (0, π)        → arm tilted 90° from base
    -1.4835,  # joint3:  mid of (-2.967, 0)   → natural elbow bend
    0.0,      # joint4:  mid of (-1.745, 1.745)
    0.0,      # joint5:  mid of (-1.222, 1.222)
    0.0,      # joint6:  mid of (-2.094, 2.094)
]

TOPIC          = '/control/joint_states'
PUBLISH_HZ     = 50
SMOOTHING      = 0.08   # 0.01 = very slow creep, 1.0 = instant snap


class PiperVisualizerController(Node):

    def __init__(self):
        super().__init__('piper_visualizer_controller')
        self.pub = self.create_publisher(JointState, TOPIC, 10)
        self.create_timer(1.0 / PUBLISH_HZ, self._tick)

        self.marker_pub = self.create_publisher(Marker, '/drawing_pen', 10)
        self.line_strip = Marker()
        self._init_pen()

        # Add this inside your __init__ method
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        

        # Start at home pose so the arm isn't in a degenerate position
        self.current  = list(HOME_POSE)
        self.target   = list(HOME_POSE)

        self.get_logger().info(
            f'Publishing to {TOPIC} at {PUBLISH_HZ} Hz\n'
            f'Joints: {JOINT_NAMES}\n'
            f'Home:   {[f"{v:.3f}" for v in HOME_POSE]}'
        )

    # ── Internal timer callback ───────────────────────────────────────

    def _tick(self):
        for i in range(len(JOINT_NAMES)):
            self.current[i] += (self.target[i] - self.current[i]) * SMOOTHING
        self._publish(self.current)

        # ADD THIS LINE: Automatically draw wherever the gripper moved this tick!
        self.drop_ink_from_gripper()

    def _publish(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = JOINT_NAMES
        msg.position = [float(p) for p in positions]
        msg.velocity = [0.0] * len(JOINT_NAMES)
        msg.effort   = [0.0] * len(JOINT_NAMES)
        self.pub.publish(msg)

    def _spin(self, seconds: float):
        """Let ROS2 run (executing _tick) for N seconds."""
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    # ── Public control methods ────────────────────────────────────────

    def move_to(self, positions: list, wait: float = 0.0):
        """
        Set target joint configuration. Positions are in radians.
        Values outside URDF limits are clamped automatically.

        Example:
            ctrl.move_to([0.0, 1.0, -1.0, 0.0, 0.5, 0.0], wait=2.5)
        """
        if len(positions) != len(JOINT_NAMES):
            self.get_logger().error(
                f'Expected {len(JOINT_NAMES)} values, got {len(positions)}'
            )
            return

        clamped = []
        for name, pos in zip(JOINT_NAMES, positions):
            lo, hi = JOINT_LIMITS[name]
            safe = max(lo, min(hi, pos))
            if abs(safe - pos) > 0.001:
                self.get_logger().warn(
                    f'{name}: {math.degrees(pos):.1f}° out of range '
                    f'[{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°] '
                    f'→ clamped to {math.degrees(safe):.1f}°'
                )
            clamped.append(safe)

        self.target = clamped
        self.get_logger().info(
            'Target: ' + '  '.join(
                f'{n}={math.degrees(v):.1f}°'
                for n, v in zip(JOINT_NAMES, clamped)
            )
        )
        if wait > 0:
            self._spin(wait)

    def move_joint(self, index: int, angle_rad: float, wait: float = 0.0):
        """
        Move one joint, leave the rest at their current targets.

        index:     0=joint1 (base), 1=joint2, 2=joint3, 3=joint4,
                   4=joint5, 5=joint6 (wrist)
        angle_rad: desired angle in radians (use math.radians() to convert
                   from degrees)

        Example — rotate base 60°:
            ctrl.move_joint(0, math.radians(60), wait=2.0)
        """
        new = list(self.target)
        new[index] = angle_rad
        self.move_to(new, wait=wait)

    def go_home(self, wait: float = 2.5):
        """Move to the safe home pose (midpoint of every joint's range)."""
        self.get_logger().info('Going home...')
        self.move_to(HOME_POSE, wait=wait)

    def _init_pen(self):
        """Sets up the virtual drawing pen settings."""
        self.line_strip.header.frame_id = "base_link"  # Or your robot's base frame name
        self.line_strip.type = Marker.LINE_STRIP
        self.line_strip.action = Marker.ADD
        self.line_strip.scale.x = 0.005  # Thickness of the line (5 millimeters)
    
        # Color of the line (Red=1.0, Green=0.0, Blue=0.0, Alpha/Opacity=1.0)
        self.line_strip.color.r = 1.0
        self.line_strip.color.a = 1.0

    def draw_point(self, x, y, z):  
        """Call this function to drop ink at a specific 3D coordinate."""
        self.line_strip.header.stamp = self.get_clock().now().to_msg()
    
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
    
        self.line_strip.points.append(p)
        self.marker_pub.publish(self.line_strip)

    def drop_ink_from_gripper(self):
        """Looks up the live 3D position of the gripper and draws a point."""
        try:
            # Look up the transform from base to the very tip of the arm
            # NOTE: If your URDF uses 'link6' or 'wrist_link', change 'link6' to match your gripper link name!
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'base_link',   # Target frame (where the ink stays)
                'link6',       # Source frame (the pen tip moving)
                now
            )
        
            # Extract the live X, Y, Z coordinates
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
        
            # Draw it!
            self.draw_point(x, y, z)
        
        except TransformException as ex:
            # It takes a split second for TF to warm up on startup, so we ignore initial misses
            pass

    


# ── Demo sequences ────────────────────────────────────────────────────

def demo_each_joint(ctrl: PiperVisualizerController):
    """
    Move each joint independently so you can see what it controls.
    Uses the CORRECT range for each joint — no out-of-bounds moves.
    """
    ctrl.get_logger().info('=== Demo: each joint one by one ===')
    ctrl.go_home(wait=2.0)

    # joint1 — base rotation: full symmetric range
    ctrl.get_logger().info('joint1 (base rotation)...')
    ctrl.move_joint(0, math.radians( 80), wait=2.5)
    ctrl.move_joint(0, math.radians(-80), wait=2.5)
    ctrl.go_home(wait=2.0)

    # joint2 — shoulder: range is 0 → +180° only, so stay positive
    ctrl.get_logger().info('joint2 (shoulder, 0 to 180°)...')
    ctrl.move_joint(1, math.radians( 30), wait=2.5)   # near lower limit
    ctrl.move_joint(1, math.radians(160), wait=2.5)   # near upper limit
    ctrl.go_home(wait=2.0)

    # joint3 — elbow: range is -170° → 0° only, so stay negative
    ctrl.get_logger().info('joint3 (elbow, -170° to 0°)...')
    ctrl.move_joint(2, math.radians( -20), wait=2.5)  # near upper limit
    ctrl.move_joint(2, math.radians(-150), wait=2.5)  # near lower limit
    ctrl.go_home(wait=2.0)

    # joint4 — forearm roll: symmetric ±100°
    ctrl.get_logger().info('joint4 (forearm roll)...')
    ctrl.move_joint(3, math.radians( 80), wait=2.5)
    ctrl.move_joint(3, math.radians(-80), wait=2.5)
    ctrl.go_home(wait=2.0)

    # joint5 — wrist pitch: symmetric ±70°
    ctrl.get_logger().info('joint5 (wrist pitch)...')
    ctrl.move_joint(4, math.radians( 60), wait=2.5)
    ctrl.move_joint(4, math.radians(-60), wait=2.5)
    ctrl.go_home(wait=2.0)

    # joint6 — wrist roll: symmetric ±120°
    ctrl.get_logger().info('joint6 (wrist roll)...')
    ctrl.move_joint(5, math.radians( 100), wait=2.5)
    ctrl.move_joint(5, math.radians(-100), wait=2.5)
    ctrl.go_home(wait=2.0)


def demo_reach_and_fold(ctrl: PiperVisualizerController):
    """
    Extend the arm outward then fold it back.
    All positions are within the correct URDF ranges.
    """
    ctrl.get_logger().info('=== Demo: reach and fold ===')
    ctrl.go_home(wait=2.0)

    # Extend forward: shoulder up (joint2 smaller = more upright),
    # elbow less bent (joint3 closer to 0)
    ctrl.move_to(
        [0.0,   0.5,  -0.3,  0.0,  0.3,  0.0],
        wait=3.0
    )

    # Fold back: shoulder forward (joint2 larger), elbow more bent (joint3 more negative)
    ctrl.move_to(
        [0.0,   2.5,  -2.3,  0.0, -0.5,  0.0],
        wait=3.0
    )

    ctrl.go_home(wait=2.5)


def demo_wave(ctrl: PiperVisualizerController):
    """Rotate the base left and right while wrist tilts."""
    ctrl.get_logger().info('=== Demo: wave ===')
    ctrl.go_home(wait=2.0)
    for angle in [70, -70, 40, -40, 0]:
        ctrl.move_joint(0, math.radians(angle), wait=2.0)
    ctrl.go_home(wait=2.0)

def demo_draw_star_keyframes(ctrl: PiperVisualizerController):
    ctrl.get_logger().info('=== Drawing Star via Keyframes ===')
    ctrl.go_home(wait=2.0)
    
    # Write your points using standard degrees inside math.radians()
    # Star 11 points
    star_points_degrees = [
        [0.0, 50.0, -50.0, 0.0, 20.0, 0.0],    # Vertex 1 A
        [11.0, 47.0, -39.0, -42.0, 17.0, 45.0],   # Vertex 2 B
        [40.0, 53.0, -42.0, -74.0, 41.0, 83.0],   # Vertex 3 C
        [25.0, 46.0, -28.0, -81.0, 25.0, 88.0], # Vertex 4 D
        [53.0, 56.0, -16.0, 81.0, -52.0, -55.0],    # Vertex 5 E
        [0.0, 43.0, -18.0, 0.0, -5.0, 1.0],    # Vertex 6 F
        [-53.0, 60.0, -7.0, -72.0, -54.0, 42.0], # Vertex 7 G
        [-24.0, 45.0, -21.0, -84.0, -24.0, 77.0],    # Vertex 8 H
        [-37.0, 50.0, -35.0, 78.0, 37.0, -86.0],    # Vertex 9 I
        [-8.0, 45.0, -34.0, 43.0, 12.0, -44.0], # Vertex 10 J
        [0.0, 50.0, -50.0, 0.0, 20.0, 0.0]      # Vertex 11 A,K
    ]
    #star 6 points
    # star_points_degrees = [
    #       [0.0, 60.0, -65.0, 0.0 ,0.0 ,0.0],
    #       [20.0,64.0,-41.0,-42.0,-30.0,-40.0],
    #       [-30.0,65.0,-60.0,-75.0,-30.0,75.0],
    #       [45.0, 70.0, -60.0, 80.0, -45.0,-80.0],
    #       [-20.0, 70.0, -30.0, -30.0, -45.0, 30.0],
    #       [0.0, 60.0, -65.0, 0.0 ,0.0 ,0.0]
    #   ]
    
    # Convert them to radians on the fly as the loop runs
    for pt_deg in star_points_degrees:
        pt_rad = [math.radians(angle) for angle in pt_deg]
        ctrl.move_to(pt_rad, wait=1.5)
        
    ctrl.go_home(wait=2.5)




# ── Main ──────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    ctrl = PiperVisualizerController()

    
    try:
        demo_draw_star_keyframes(ctrl)
        # Recommended: run this first to see each joint move independently
        #demo_each_joint(ctrl)
        # Then try these if you like:
        # demo_wave(ctrl)
        # demo_reach_and_fold(ctrl)

        # Or write your own (all values are in radians):
        # ctrl.go_home(wait=2.0)
        # ctrl.move_to([0.0, 1.0, -1.5, 0.3, -0.5, 1.0], wait=3.0)
        # ctrl.move_joint(0, math.radians(45), wait=2.0)
        # ctrl.go_home(wait=2.0)

        ctrl.get_logger().info('Done.')

    except KeyboardInterrupt:
        ctrl.get_logger().info('Stopped.')
    finally:
        ctrl.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()