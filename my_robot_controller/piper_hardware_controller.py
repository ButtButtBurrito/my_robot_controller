"""
piper_hardware_controller.py
-----------------------------
Controls the REAL Piper arm hardware via the JointTrajectoryController.

Confirmed from your setup:
  Controller:  arm_controller  (JointTrajectoryController) — active
  Command topic: /arm_controller/joint_trajectory
  Planning group: "arm"
  Action servers: /move_action, /execute_trajectory

HOW IT DIFFERS FROM piper_visualizer_controller.py:
  Visualizer → publishes JointState  (fakes encoder feedback, only tricks RViz)
  This file  → publishes JointTrajectory (real hardware command interface)

LAUNCH BEFORE RUNNING:
  Terminal 1: bash ~/agx_arm_ws/src/agx_arm_ros/scripts/can_activate.sh
  Terminal 2: ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
                can_port:=can0 arm_type:=piper effector_type:=none \
                tcp_offset:='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
  Terminal 3: ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py \
                can_port:=can0 arm_type:=piper effector_type:=agx_gripper
  Terminal 4: python3 piper_hardware_controller.py

JOINT LIMITS (from your URDF):
  joint1  -150° to +150°   base rotation      (symmetric)
  joint2     0° to +180°   shoulder           (NO negative)
  joint3  -170° to    0°   elbow              (NO positive)
  joint4  -100° to +100°   forearm roll       (symmetric)
  joint5   -70° to  +70°   wrist pitch        (symmetric)
  joint6  -120° to +120°   wrist roll         (symmetric)
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
import math
import time
import threading


JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

JOINT_LIMITS = {
    'joint1': (-2.6179938,  2.6179938),
    'joint2': ( 0.0,        3.1415926),  # shoulder: 0 → +180° only
    'joint3': (-2.9670597,  0.0      ),  # elbow:  -170° → 0° only
    'joint4': (-1.7453292,  1.7453292),
    'joint5': (-1.2217304,  1.2217304),
    'joint6': (-2.0943951,  2.0943951),
}

# Safe mid-range home pose (midpoint of each joint's travel)
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Confirmed from: ros2 control list_controllers
COMMAND_TOPIC = '/arm_controller/joint_trajectory'

# How long between moves if no duration specified (seconds)
DEFAULT_MOVE_DURATION = 3.0


class PiperHardwareController(Node):
    """
    Controls the real Piper arm by publishing JointTrajectory messages
    to the arm_controller (JointTrajectoryController).

    KEY CONCEPT — JointTrajectory vs JointState:
      JointState    = encoder readings coming OUT of the robot (read-only)
      JointTrajectory = position commands going INTO the controller (read-write)

    The controller handles smooth interpolation between waypoints.
    You specify WHERE and HOW LONG — the controller handles the HOW.
    """

    def __init__(self):
        super().__init__('piper_hardware_controller')

        self.pub = self.create_publisher(
            JointTrajectory,
            COMMAND_TOPIC,
            10
        )

        # Track real joint positions from encoder feedback
        self.current_positions = list(HOME)
        self.feedback_received = False

        # Subscribe to real joint state feedback from the hardware
        self.create_subscription(
            JointState,
            'feedback/joint_states',   # real encoder data from joint_state_broadcaster
            self._feedback_callback,
            10
        )

        # Spin in background so callbacks fire without blocking main thread
        self._executor = rclpy.executors.MultiThreadedExecutor(2)
        self._executor.add_node(self)
        self._thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._thread.start()

        self.get_logger().info(
            f'Hardware controller ready.\n'
            f'Publishing to: {COMMAND_TOPIC}\n'
            f'Waiting for encoder feedback on feedback/joint_states...'
        )

        # Wait until we receive real joint positions from hardware
        timeout = time.time() + 5.0
        while not self.feedback_received and time.time() < timeout:
            time.sleep(0.1)

        if self.feedback_received:
            self.get_logger().info(
                f'Hardware connected. Current joints: '
                f'{[f"{math.degrees(p):.1f}°" for p in self.current_positions]}'
            )
        else:
            self.get_logger().warn(
                'No joint feedback received — is the hardware driver running?'
            )

    # ── Feedback callback ─────────────────────────────────────────────

    def _feedback_callback(self, msg: JointState):
        """Receives real encoder positions from the hardware."""
        if msg.position:
            self.current_positions = list(msg.position)
            self.feedback_received = True

    # ── Core command method ───────────────────────────────────────────

    def move_to(self, positions: list, duration_sec: float = DEFAULT_MOVE_DURATION):
        """
        Move all joints to the target positions over duration_sec seconds.

        The controller smoothly interpolates from current position to target.
        A longer duration = slower, smoother movement.
        A shorter duration = faster movement (don't go too fast on hardware).

        positions:    list of 6 floats in radians
        duration_sec: how long to take to reach the target (default 3.0s)

        Example:
            ctrl.move_to([0.0, 1.0, -1.5, 0.3, 0.0, 0.5], duration_sec=4.0)
        """
        if len(positions) != len(JOINT_NAMES):
            self.get_logger().error(
                f'Expected {len(JOINT_NAMES)} positions, got {len(positions)}'
            )
            return

        # Clamp to URDF limits — protects hardware from out-of-range commands
        clamped = []
        for name, val in zip(JOINT_NAMES, positions):
            lo, hi = JOINT_LIMITS[name]
            safe = max(lo, min(hi, val))
            if abs(safe - val) > 0.001:
                self.get_logger().warn(
                    f'{name}: {math.degrees(val):.1f}° clamped to '
                    f'{math.degrees(safe):.1f}° '
                    f'(limit: [{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°])'
                )
            clamped.append(safe)

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in clamped]
        point.velocities = [0.0] * len(JOINT_NAMES)  # zero velocity at target = stop smoothly
        point.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec % 1) * 1e9)
        )

        msg.points = [point]
        self.pub.publish(msg)

        self.get_logger().info(
            f'Command sent ({duration_sec:.1f}s): ' +
            '  '.join(f'{n}={math.degrees(v):+.1f}°'
                      for n, v in zip(JOINT_NAMES, clamped))
        )

    def move_trajectory(self, waypoints: list):
        """
        Execute a multi-point trajectory smoothly in one command.

        waypoints: list of (positions, time_from_start_sec) tuples
                   times are cumulative from when the trajectory starts

        Example — wave motion as one smooth trajectory:
            ctrl.move_trajectory([
                ([0.5, 1.5, -1.5, 0.0, 0.0, 0.0], 2.0),   # point 1 at t=2s
                ([-0.5, 1.5, -1.5, 0.0, 0.0, 0.0], 4.0),  # point 2 at t=4s
                ([0.0, 1.5708, -1.4835, 0.0, 0.0, 0.0], 6.0), # home at t=6s
            ])
        """
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = JOINT_NAMES

        for positions, t in waypoints:
            # Clamp each waypoint
            clamped = []
            for name, val in zip(JOINT_NAMES, positions):
                lo, hi = JOINT_LIMITS[name]
                clamped.append(max(lo, min(hi, val)))

            point = JointTrajectoryPoint()
            point.positions = [float(p) for p in clamped]
            point.velocities = [0.0] * len(JOINT_NAMES)
            point.time_from_start = Duration(
                sec=int(t),
                nanosec=int((t % 1) * 1e9)
            )
            msg.points.append(point)

        self.pub.publish(msg)
        total = waypoints[-1][1]
        self.get_logger().info(
            f'Trajectory sent: {len(waypoints)} waypoints over {total:.1f}s'
        )

    def move_joint(self, index: int, angle_rad: float,
                   duration_sec: float = DEFAULT_MOVE_DURATION):
        """
        Move one joint, hold all others at current target.

        index: 0=joint1 (base) ... 5=joint6 (wrist)

        Example:
            ctrl.move_joint(0, math.radians(45), duration_sec=2.0)
        """
        new = list(self.current_positions)
        new[index] = angle_rad
        self.move_to(new, duration_sec=duration_sec)

    def go_home(self, duration_sec: float = 4.0):
        """Move to safe mid-range home pose."""
        self.get_logger().info('Going to home pose...')
        self.move_to(HOME, duration_sec=duration_sec)

    def get_joint_positions(self) -> list:
        """Return current joint positions in radians (from encoder feedback)."""
        return list(self.current_positions)

    def get_joint_positions_deg(self) -> list:
        """Return current joint positions in degrees."""
        return [math.degrees(p) for p in self.current_positions]


# ── Demo sequences ────────────────────────────────────────────────────

def demo_go_home(ctrl: PiperHardwareController):
    """Safest first move — go to home pose from wherever the arm is."""
    print('\n=== Moving to home pose ===')
    ctrl.go_home(duration_sec=3.0)     # slow on first real move
    time.sleep(6.0)                    # wait for motion to complete


def demo_each_joint(ctrl: PiperHardwareController):
    """Move each joint individually so you can feel what each one does."""
    print('\n=== Each joint one by one ===')
    ctrl.go_home(duration_sec=5.0)
    time.sleep(6.0)

    moves = [
        (0, 'base rotation',   40,  -40, 3.0),
        (1, 'shoulder',        30,  160, 3.0),   # stays positive
        (2, 'elbow',          -20, -140, 3.0),   # stays negative
        (3, 'forearm roll',    60,  -60, 3.0),
        (4, 'wrist pitch',     50,  -50, 3.0),
        (5, 'wrist roll',      80,  -80, 3.0),
    ]

    for idx, label, a, b, dur in moves:
        print(f'\n  joint{idx+1} — {label}')
        ctrl.move_joint(idx, math.radians(a), duration_sec=dur)
        time.sleep(dur + 1.0)
        ctrl.move_joint(idx, math.radians(b), duration_sec=dur)
        time.sleep(dur + 1.0)
        ctrl.go_home(duration_sec=3.0)
        time.sleep(4.0)


def demo_smooth_wave(ctrl: PiperHardwareController):
    """
    Smooth wave using multi-point trajectory — one command, no gaps.
    The controller executes all waypoints as one continuous motion.
    """
    print('\n=== Smooth wave trajectory ===')
    ctrl.go_home(duration_sec=4.0)
    time.sleep(5.0)

    ctrl.move_trajectory([
        # (joint positions,                                cumulative_time_sec)
        ([0.7,  1.5708, -1.4835, 0.0,  0.0, 0.0],  2.5),  # base left
        ([-0.7, 1.5708, -1.4835, 0.0,  0.0, 0.0],  5.0),  # base right
        ([0.0,  1.5708, -1.4835, 0.0,  0.0, 0.0],  7.0),  # centre
    ])
    time.sleep(8.0)


# ── Main ──────────────────────────────────────────────────────────────
def user_custom_sequence(ctrl: PiperHardwareController):
    """
    GUIDE: Place your own custom movements in this function.
    
    1. JOINT ANGLES (Radians vs Degrees):
       You can command the joints using a list of 6 numbers.
       It's easiest to write the angles in degrees and convert them using math.radians().
       Example:
         my_joints_deg = [45, 90, -45, 0, 30, 0] # Base=45°, Shoulder=90°, Elbow=-45°, etc.
         my_joints_rad = [math.radians(angle) for angle in my_joints_deg]
         ctrl.move_to(my_joints_rad, duration_sec=4.0)
         time.sleep(5.0) # Always sleep for duration_sec + safety buffer to let the arm finish moving.
    2. CARTESIAN COORDINATES (X, Y, Z):
       Note: This driver runs the joint trajectory controller directly (low-level).
       If you want to command Cartesian positions (e.g., X=0.2m, Y=0.1m, Z=0.3m),
       you should use MoveIt (e.g., move_it_driver.py) or publish to MoveIt's topics:
       - '/control/move_p' (Point-to-point Cartesian motion)
       - '/control/move_l' (Linear Cartesian motion)
    """
    print('\n=== Running Custom User Sequence ===')
    # Example Move 1: Base turn 30 degrees, lift shoulder slightly
    # Keep joints within limits! Refer to JOINT_LIMITS above.
    move1_deg = [0.0, 20.0, -42.0, 0.0, 25.0, 0.0]
    move1_rad = [math.radians(x) for x in move1_deg]
    move2_deg = [65.0, 6.0, -45.0, -75.0, 70.0, 45.0]
    move2_rad = [math.radians(x) for x in move2_deg]
    move3_deg = [-65.0, 6.0, -48.0, 75.0, 70.0, -45.0]
    move3_rad = [math.radians(x) for x in move3_deg]
    move4_deg = [-60.0, 18.0, -8.0, 98.0, 61.0, -98.0]
    move4_rad = [math.radians(x) for x in move4_deg]
    move5_deg = [70.0, 13.0, -7.0, 84.0, -70.0, -87.0]
    move5_rad = [math.radians(x) for x in move5_deg]
    print(f"Moving to: {move1_deg}")
    ctrl.move_to(move1_rad, duration_sec=1.0)
    time.sleep(5.0) # Wait for movement to finish
    ctrl.move_to(move2_rad, duration_sec=1.0)
    time.sleep(5.0) # Wait for movement to finish
    ctrl.move_to(move3_rad, duration_sec=1.0)
    time.sleep(5.0) # Wait for movement to finish
    ctrl.move_to(move4_rad, duration_sec=1.0)
    time.sleep(5.0) # Wait for movement to finish
    ctrl.move_to(move5_rad, duration_sec=1.0)
    time.sleep(5.0) # Wait for movement to finish
    # Example Move 2: Return to home pose
    print("Returning home...")
    ctrl.go_home(duration_sec=4.0)
    time.sleep(5.0)

def main():
    rclpy.init()
    ctrl = PiperHardwareController()

    # Print where the arm currently is before doing anything
    print(f'\nCurrent joint positions: {ctrl.get_joint_positions_deg()}')
    print('Starting in 2 seconds — STAND CLEAR of the robot.\n')
    time.sleep(2.0)

    try:
        # ALWAYS start with go_home on real hardware
        demo_go_home(ctrl)
  # HOW TO RUN YOUR OWN MOVES:
        # 1. Edit the `user_custom_sequence(ctrl)` function above with your angles.
        # 2. Uncomment the line below to execute it:
        # ----------------------------------------------------
        user_custom_sequence(ctrl)
        demo_go_home(ctrl)
        user_custom_sequence(ctrl)
        demo_go_home(ctrl)

        # Demos:

        # Then uncomment what you want to test:
        # demo_each_joint(ctrl)
        # demo_smooth_wave(ctrl)

   

        print('\nDone.')

    except KeyboardInterrupt:
        print('\nStopped by user.')
        # On Ctrl+C, go home safely
        print('Returning to home pose...')
        ctrl.go_home(duration_sec=5.0)
        time.sleep(6.0)

    finally:
        ctrl.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
