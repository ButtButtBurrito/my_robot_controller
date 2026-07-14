#!/usr/bin/env python3
"""
camera_to_arm_bridge.py
------------------------
Skeleton node: RealSense camera -> 3D target point -> Piper arm command.

THIS IS A SKELETON. It will run and print diagnostics, but the actual
object-detection logic and the camera->robot calibration are placeholders
you need to fill in. They are marked clearly with "TODO" below.

ARCHITECTURE (matches your current setup):

  /camera/color/image_raw            ─┐
  /camera/aligned_depth_to_color/      ├─> this node ─> target_xyz (robot frame)
      image_raw                      ─┤                      │
  /camera/aligned_depth_to_color/      │                      ▼
      camera_info                    ─┘            piper_hardware_controller.py
                                                    (joint_trajectory)  OR
                                                    MoveIt Cartesian topics
                                                    (/control/move_p, /control/move_l)

WHY "ALIGNED" DEPTH:
  Use the *aligned* depth stream (aligned_depth_to_color), not raw depth.
  Aligned depth guarantees pixel (u,v) in the depth image lines up exactly
  with pixel (u,v) in the color image. Without alignment, the two cameras
  inside the RealSense are physically offset and your math will be wrong.

BEFORE RUNNING (once you have the camera):
  Terminal 1: ros2 launch realsense2_camera rs_launch.py \
                align_depth.enable:=true
  Terminal 2: python3 camera_to_arm_bridge.py

VERIFY TOPIC NAMES FIRST (they vary slightly by realsense2_camera version):
  ros2 topic list | grep camera
  Update COLOR_TOPIC / DEPTH_TOPIC / CAMERA_INFO_TOPIC below to match.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import numpy as np
import math


# ── Topics — CONFIRMED from `ros2 topic list` on this system ────────────
# Note the doubled "camera" — that's just how this realsense2_camera
# version names its topics by default, not a typo or an error.
COLOR_TOPIC        = '/camera/camera/color/image_raw'
DEPTH_TOPIC        = '/camera/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC  = '/camera/camera/aligned_depth_to_color/camera_info'

# ── Cartesian command topic — CONFIRMED from agx_arm_ctrl_single_node.py ─
#   self.create_subscription(PoseStamped, "control/move_p", ...)
# Use move_p for point-to-point moves, move_l for straight-line moves.
# Leading slash vs no slash depends on whether you launch with a namespace —
# if publishing fails silently, check `ros2 topic list` for the exact form.
MOVE_P_TOPIC = '/control/move_p'
MOVE_L_TOPIC = '/control/move_l'

# Orientation to send with every move — straight down, gripper facing the table.
# This is a quaternion [x, y, z, w]. (0,0,0,1) = no rotation from the default
# end-effector orientation defined in your URDF/SRDF. Adjust once you know
# what orientation your gripper needs for picking.
DEFAULT_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)

# ── TODO #1: CAMERA-TO-ROBOT CALIBRATION ─────────────────────────────────
# This is the single most important number you don't have yet.
# It describes where the camera physically sits relative to the robot's
# base_link. Until this is measured/calibrated, every 3D point this node
# produces will be in the WRONG coordinate frame for the arm.
#
# Two ways to get this:
#   A) Measure with a ruler/CAD model — rough but works for a fixed mount.
#      Fill in CAMERA_OFFSET_FROM_BASE below (metres, robot base frame).
#   B) Proper way — eye-to-hand calibration using a checkerboard or
#      ArUco marker, then publish a static_transform_publisher and use
#      tf2 to do this lookup automatically instead of a hardcoded offset.
#      (Ask me for this when you're ready — it's a separate, important step.)
#
# For now this uses a placeholder offset. REPLACE before trusting any
# coordinate this node outputs.
CAMERA_OFFSET_FROM_BASE = {
    'x': 0.0,   # metres, camera position relative to robot base_link
    'y': 0.0,
    'z': 0.0,
}


class CameraToArmBridge(Node):
    """
    Subscribes to RealSense color + depth + camera_info.
    Converts a detected pixel into a 3D point in the camera frame,
    then offsets it into the robot's base frame (using the placeholder
    above), and prints/forwards the result.
    """

    def __init__(self):
        super().__init__('camera_to_arm_bridge')

        self.latest_color = None
        self.latest_depth = None
        self.intrinsics   = None   # fx, fy, cx, cy from CameraInfo

        self.create_subscription(Image, COLOR_TOPIC, self._color_cb, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self._depth_cb, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._info_cb, 10)

        # Publisher for sending Cartesian targets to the real/simulated arm.
        # Confirmed message type + topic from agx_arm_ctrl_single_node.py.
        self.move_p_pub = self.create_publisher(PoseStamped, MOVE_P_TOPIC, 1)
        self.move_l_pub = self.create_publisher(PoseStamped, MOVE_L_TOPIC, 1)

        # Run detection+conversion at 5 Hz — plenty for most pick/place tasks
        self.create_timer(0.2, self._process)

        self.get_logger().info(
            f'Bridge node started.\n'
            f'  Color topic: {COLOR_TOPIC}\n'
            f'  Depth topic: {DEPTH_TOPIC}\n'
            f'  Camera info: {CAMERA_INFO_TOPIC}\n'
            f'Waiting for frames...'
        )

    # ── Subscriber callbacks ────────────────────────────────────────────

    def _color_cb(self, msg: Image):
        self.latest_color = self._image_to_numpy(msg)

    def _depth_cb(self, msg: Image):
        # Depth images from RealSense are typically 16-bit unsigned,
        # values in millimetres. Confirm with: ros2 topic info <depth_topic>
        # and check msg.encoding (commonly '16UC1').
        self.latest_depth = self._image_to_numpy(msg)

    def _info_cb(self, msg: CameraInfo):
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1] (row-major 3x3 intrinsic matrix)
        if self.intrinsics is None:
            self.intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
            }
            self.get_logger().info(f'Camera intrinsics received: {self.intrinsics}')

    @staticmethod
    def _image_to_numpy(msg: Image) -> np.ndarray:
        """Convert a ROS2 Image message into a numpy array."""
        dtype = np.uint8 if msg.encoding in ('rgb8', 'bgr8') else np.uint16
        arr = np.frombuffer(msg.data, dtype=dtype)
        if msg.encoding in ('rgb8', 'bgr8'):
            return arr.reshape(msg.height, msg.width, 3)
        return arr.reshape(msg.height, msg.width)

    # ── Main processing loop ────────────────────────────────────────────

    def _process(self):
        if self.latest_color is None or self.latest_depth is None or self.intrinsics is None:
            return  # still waiting for first frames

        # ── TODO #2: OBJECT DETECTION ────────────────────────────────────
        # Replace this with your actual detection logic — color thresholding,
        # a trained model, ArUco marker detection, etc. It must return a
        # pixel coordinate (u, v) of the target point in the COLOR image.
        pixel = self._detect_object_placeholder(self.latest_color)
        if pixel is None:
            return  # nothing detected this frame

        u, v = pixel

        # Look up the depth (distance from camera) at that exact pixel
        depth_mm = self._sample_depth(self.latest_depth, u, v)
        if depth_mm is None or depth_mm == 0:
            self.get_logger().warn(f'No valid depth at pixel ({u},{v})')
            return

        depth_m = depth_mm / 1000.0  # mm -> metres

        # Deproject 2D pixel + depth into a 3D point in the CAMERA's frame
        cam_xyz = self._deproject(u, v, depth_m)

        # Convert camera-frame point into robot base-frame point
        # (this is only as accurate as CAMERA_OFFSET_FROM_BASE above)
        robot_xyz = self._camera_to_robot_frame(cam_xyz)

        self.get_logger().info(
            f'Detected pixel ({u},{v}) -> camera frame {cam_xyz} '
            f'-> robot frame {robot_xyz}'
        )

        # ── Send to arm ───────────────────────────────────────────────
        # Uncomment once you've confirmed CAMERA_OFFSET_FROM_BASE is
        # correct and you're ready to actually move the hardware:
        #
        # self._send_to_arm(robot_xyz)
        #
        # Left commented out by default — this loop runs at 5 Hz, so
        # uncommenting this means the arm gets a NEW move command five
        # times a second for as long as the placeholder detector keeps
        # "detecting" something. Add your own logic (e.g. only send once
        # per detected object, or only when the target is stable across
        # several frames) before enabling this for real.

    # ── Geometry helpers ─────────────────────────────────────────────────

    def _sample_depth(self, depth_img: np.ndarray, u: int, v: int):
        """Read depth value at pixel (u,v), with a small averaging window
        to reduce noise from a single bad pixel."""
        h, w = depth_img.shape
        if not (0 <= u < w and 0 <= v < h):
            return None
        window = depth_img[max(0, v-2):v+3, max(0, u-2):u+3]
        valid = window[window > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _deproject(self, u: int, v: int, depth_m: float) -> dict:
        """
        Convert pixel (u,v) + depth into a 3D point in the CAMERA's
        own coordinate frame (standard pinhole camera model).

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        """
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']

        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m

        return {'x': x, 'y': y, 'z': z}

    def _camera_to_robot_frame(self, cam_xyz: dict) -> dict:
        """
        Shift a camera-frame point into the robot's base_link frame.

        THIS IS A PLACEHOLDER. A simple offset only works if the camera
        is mounted with NO rotation relative to the robot base (rare).
        If the camera is tilted or rotated at all, you need a full
        rotation matrix here, not just addition. Ask for the proper
        tf2-based version once you have the camera mounted and can
        measure/calibrate the real transform.
        """
        return {
            'x': cam_xyz['x'] + CAMERA_OFFSET_FROM_BASE['x'],
            'y': cam_xyz['y'] + CAMERA_OFFSET_FROM_BASE['y'],
            'z': cam_xyz['z'] + CAMERA_OFFSET_FROM_BASE['z'],
        }

    # ── Placeholder detection (REPLACE THIS) ────────────────────────────

    def _detect_object_placeholder(self, color_img: np.ndarray):
        """
        REPLACE THIS FUNCTION with real detection logic.

        Returns (u, v) pixel coordinates of the target, or None if
        nothing was detected this frame.

        For now it just returns the centre of the image every frame,
        purely so you can confirm the geometry math is wired up
        correctly before adding real detection.
        """
        h, w = color_img.shape[:2]
        return (w // 2, h // 2)

    # ── Sending the result to the arm (fill in once verified) ───────────

    def _send_to_arm(self, robot_xyz: dict, linear: bool = False):
        """
        Publish a Cartesian target to the real agx_arm_ctrl driver node.

        CONFIRMED from agx_arm_ctrl_single_node.py:
            self.create_subscription(PoseStamped, "control/move_p", ...)
            self.create_subscription(PoseStamped, "control/move_l", ...)

        robot_xyz: dict with 'x', 'y', 'z' in METRES, in the robot's
                   base_link frame (NOT the camera frame — make sure
                   _camera_to_robot_frame() has already been applied).
        linear:    False -> move_p (point-to-point, fastest path)
                   True  -> move_l (straight-line path, e.g. for a
                            controlled final approach onto an object)

        SAFETY NOTE: This publishes immediately with no plan preview.
        Unlike dragging in MoveIt's RViz GUI, there is no Execute button
        to double check the move first. Test with the arm raised away
        from obstacles, and consider clamping robot_xyz to a known-safe
        workspace box before calling this in a real detection loop.
        """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.pose.position.x = float(robot_xyz['x'])
        msg.pose.position.y = float(robot_xyz['y'])
        msg.pose.position.z = float(robot_xyz['z'])

        msg.pose.orientation.x = DEFAULT_QUAT_XYZW[0]
        msg.pose.orientation.y = DEFAULT_QUAT_XYZW[1]
        msg.pose.orientation.z = DEFAULT_QUAT_XYZW[2]
        msg.pose.orientation.w = DEFAULT_QUAT_XYZW[3]

        if linear:
            self.move_l_pub.publish(msg)
            self.get_logger().info(f'move_l -> x={msg.pose.position.x:.3f} '
                                    f'y={msg.pose.position.y:.3f} '
                                    f'z={msg.pose.position.z:.3f}')
        else:
            self.move_p_pub.publish(msg)
            self.get_logger().info(f'move_p -> x={msg.pose.position.x:.3f} '
                                    f'y={msg.pose.position.y:.3f} '
                                    f'z={msg.pose.position.z:.3f}')


def main():
    rclpy.init()
    node = CameraToArmBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()