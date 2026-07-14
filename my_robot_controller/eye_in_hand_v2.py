#!/usr/bin/env python3
"""
Eye-in-Hand v2 — AGX Piper / RealSense D435 (headless)

Runs once per launch: waits for the ArUco marker, searches for a *reachable*
camera pose APPROACH_DIST_M metres from the marker with the optical axis
aimed at it (marker stays centred in frame), publishes RViz preview markers,
pauses APPROACH_CONFIRM_DELAY_S seconds so the target can be checked in RViz,
then plans and executes the move. After the move it verifies the marker is
still detected and reports how far off-centre it sits in the image.

Orientation search: a rigid "camera straight down, roll 0" target is NOT
reachable for this arm (proven 2026-07-03 — IK fails 100%). Instead we sweep
look-at candidates (tilt from vertical × azimuth × roll about the optical
axis) and take the first one /compute_ik accepts. Any of these keeps the
marker centred; roll/tilt only change how the image is rotated/skewed.

Calibration is loaded from ~/.ros2/easy_handeye2/calibrations/piper_eih.calib
at startup (no more hardcoded-constant sync step); baked-in constants are the
fallback if the file is missing/unreadable.

Designed to run under agx_arm_gui's Custom Script Runner (Run Script / Stop
Script buttons) — not as a standalone interactive terminal program. Nothing
here reads stdin; all status is written to stdout, which the GUI streams
into its script log panel.

Deadman safety is owned entirely by agx_arm_gui: releasing F5 there kills
this process (SIGINT → SIGTERM → SIGKILL on the whole process group) and
sends the arm home independently, regardless of what this script is doing.
This script has no deadman logic of its own — do not run it outside the GUI
without an equivalent external safety layer.
"""

import threading
import time
from pathlib import Path

import numpy as np
import yaml
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints, JointConstraint, MoveItErrorCodes,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
import tf2_geometry_msgs  # noqa: F401  (registers TF↔geometry_msgs conversions)



# ═══════════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Approach distance: marker surface to camera lens, along the optical axis.
# The gripper IS modelled in the URDF so MoveIt will plan around it,
# but verify in RViz that the gripper tip clears the table at this value.
APPROACH_DIST_M = 0.15          # metres marker → camera  ← tune here

# Camera centering correction (metres, world / base_link frame).
# If the marker appears off-centre after the arm moves, tweak these:
#   upper-right drift → increase X_CORR_M and/or Y_CORR_M
#   lower-left  drift → decrease (go negative)
# Increment in 0.01 m steps and re-run until marker is centred.
X_CORR_M = 0.00    # + = right in image  / world +X
Y_CORR_M = 0.00    # + = up in image     / world +Y

# Robot speed — keep low (0.05–0.10) until you have confirmed the arm moves
# in the correct direction. These scale MoveIt's joint-speed limits.
# Hard per-joint limits live in joint_limits.yaml and are unaffected here.
MAX_VELOCITY_SCALE  = 0.10      # 0.0 – 1.0
MAX_ACCEL_SCALE     = 0.10      # 0.0 – 1.0

# How long to wait for the marker to appear before giving up (seconds).
MARKER_WAIT_TIMEOUT_S = 30.0

# Pause between publishing the preview (RViz markers + IK solution) and
# actually executing the move, so there's time to glance at RViz.
APPROACH_CONFIRM_DELAY_S = 4.0

MARKER_CACHE_EXPIRY = 5.0       # seconds to accept a cached marker TF

MOVE_GROUP    = 'arm'
EFFECTOR_LINK = 'link6'         # calibration was recorded relative to link6

# Look-at candidate sweep. Ordered by preference: straight-down first, then
# increasing tilt. Roll is free (rotates the image only); tilt keeps the
# marker centred but views it obliquely. 90° roll is what makes straight-down
# reachable on this arm, hence it comes right after 0.
SWEEP_TILT_DEG = [0, 10, 20, 30, 45]
SWEEP_AZIM_DEG = [0, 90, 180, 270]
SWEEP_ROLL_DEG = [0, 90, 180, 270]
SWEEP_IK_TIMEOUT_S = 0.5        # per-candidate /compute_ik budget

# RealSense D435 colour FOV is ~69°×42°; warn if the marker ends up more
# than this many degrees off the optical axis after the move (≈ vertical
# half-FOV minus margin).
IN_FRAME_WARN_DEG = 18.0


# ═══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION — link6 → camera_color_optical_frame
#  Loaded from CALIB_FILE at import; constants below are the fallback only.
# ═══════════════════════════════════════════════════════════════════════════════

CALIB_FILE = Path.home() / '.ros2/easy_handeye2/calibrations/piper_eih.calib'

_FALLBACK_CAM_T = np.array([-0.05593476016675718,
                            -0.031925813829647605,
                             0.044513090224353104])
_FALLBACK_CAM_Q = np.array([-0.10433398988297468,   # x
                             0.08134981927603824,    # y
                            -0.5141169607716367,     # z
                             0.8474552354583642])    # w  (xyzw)


def _load_calibration() -> tuple[np.ndarray, np.ndarray, str]:
    """Return (CAM_T, CAM_Q, source_description) from the calib file, or the
    baked-in fallback if the file is missing/unreadable."""
    try:
        data = yaml.safe_load(CALIB_FILE.read_text())
        tr = data['transform']['translation']
        ro = data['transform']['rotation']
        t = np.array([tr['x'], tr['y'], tr['z']])
        q = np.array([ro['x'], ro['y'], ro['z'], ro['w']])
        age_d = (time.time() - CALIB_FILE.stat().st_mtime) / 86400.0
        return t, q, f'{CALIB_FILE} (saved {age_d:.1f} days ago)'
    except Exception as e:
        return _FALLBACK_CAM_T, _FALLBACK_CAM_Q, f'FALLBACK constants ({e})'


CAM_T, CAM_Q, CALIB_SOURCE = _load_calibration()


# ═══════════════════════════════════════════════════════════════════════════════
#  Look-at target computation
# ═══════════════════════════════════════════════════════════════════════════════

def lookat_candidates():
    """
    Yield (tilt_deg, azim_deg, roll_deg, R_world_cam) camera orientations
    whose +Z (optical axis) points at the marker, ordered by preference.

    R_world_cam = Rz(az)·Ry(tilt)·Rz(−az) · Rx(180°) · Rz(roll)
      Rx(180°)          — straight down, image x = world +X (roll 0 reference)
      Rz(az)Ry(t)Rz(−az)— tilt the view axis 'tilt' degrees from vertical,
                          leaning toward world azimuth 'az'
      Rz(roll)          — spin about the optical axis (marker stays centred)
    """
    for tilt in SWEEP_TILT_DEG:
        azims = [0] if tilt == 0 else SWEEP_AZIM_DEG
        for az in azims:
            world_tilt = (Rotation.from_euler('z', az, degrees=True) *
                          Rotation.from_euler('y', tilt, degrees=True) *
                          Rotation.from_euler('z', -az, degrees=True))
            base = world_tilt * Rotation.from_euler('x', 180, degrees=True)
            for roll in SWEEP_ROLL_DEG:
                yield tilt, az, roll, base * Rotation.from_euler('z', roll, degrees=True)


def camera_to_link6(cam_pos: np.ndarray, R_world_cam: Rotation
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Convert a desired camera pose to the link6 pose that realises it,
    using the hand-eye calibration. Returns (position, quaternion_xyzw)."""
    R_link6_cam   = Rotation.from_quat(CAM_Q)
    R_world_link6 = R_world_cam * R_link6_cam.inv()
    link6_pos     = cam_pos - R_world_link6.apply(CAM_T)
    return link6_pos, R_world_link6.as_quat()


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _call_service(node: Node, cli, data: bool = True, timeout: float = 3.0) -> bool:
    """Call a SetBool service safely from a non-spin thread."""
    if not cli.wait_for_service(timeout_sec=timeout):
        node.get_logger().warn(f'Service not available: {cli.srv_name}')
        return False
    future = cli.call_async(SetBool.Request(data=data))
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout=timeout):
        node.get_logger().warn(f'Service timed out: {cli.srv_name}')
        return False
    res = future.result()
    node.get_logger().info(f'{cli.srv_name}(data={data}): {res.message}')
    return res.success


def _make_pose(pos: np.ndarray, quat: np.ndarray) -> Pose:
    p = Pose()
    p.position.x    = float(pos[0]);  p.position.y    = float(pos[1])
    p.position.z    = float(pos[2])
    p.orientation.x = float(quat[0]); p.orientation.y = float(quat[1])
    p.orientation.z = float(quat[2]); p.orientation.w = float(quat[3])
    return p


# ═══════════════════════════════════════════════════════════════════════════════
#  Node
# ═══════════════════════════════════════════════════════════════════════════════

class EyeInHandNode(Node):

    def __init__(self):
        super().__init__('eye_in_hand_v2')

        # Static TF: link6 → camera_color_optical_frame (from calibration)
        self._cam_broadcaster = StaticTransformBroadcaster(self)
        self._publish_camera_tf()

        # spin_thread=False: we drive rclpy.spin() manually below so it doesn't
        # conflict with the TransformListener's own spin attempt
        self._tf  = Buffer(node=self)
        self._tfl = TransformListener(self._tf, self, spin_thread=False)

        self._plan_client    = ActionClient(self, MoveGroup,              '/move_action')
        self._execute_client = ActionClient(self, ExecuteTrajectory,      '/execute_trajectory')
        self._gate_cli       = self.create_client(SetBool, '/control_enable')

        # Primary marker update: subscribe to the raw ArUco transform topic.
        # This fires whenever ArUco detects the marker and does NOT require
        # marker_frame to be in the TF tree — we chain the transforms ourselves.
        self.create_subscription(TransformStamped, '/aruco_single/transform',
                                 self._on_aruco_transform, 10)
        # Keepalive: some aruco_ros builds gate TF publishing on pose subscribers.
        self.create_subscription(PoseStamped, '/aruco_single/pose', lambda _: None, 10)

        # /feedback/joint_states = actual encoder feedback (NOT /joint_states,
        # which is read-only on this robot). Used to seed IK and bias OMPL
        # toward the current configuration — see _call_ik / _plan_joint_goal.
        self._current_joint_state = None   # type: JointState | None
        self.create_subscription(JointState, '/feedback/joint_states',
                                 self._on_joint_state, 10)

        self._last_marker     = None   # (x, y, z, monotonic_time) in base_link
        self._last_marker_cam = None   # (np.array xyz in camera frame, monotonic_time)

        self._viz_pub    = self.create_publisher(MarkerArray, '/eye_in_hand/viz', 10)
        self._marker_pub = self.create_publisher(Marker, '/eye_in_hand/marker_live', 10)
        self._ik_cli  = self.create_client(GetPositionIK, '/compute_ik')
        self._cartesian_cli = self.create_client(GetCartesianPath, '/compute_cartesian_path')

        self.create_timer(1.0, self._status_tick)
        self._log(f'Ready. Calibration: {CALIB_SOURCE}')
        self._log(f'  CAM_T=[{CAM_T[0]:+.4f}, {CAM_T[1]:+.4f}, {CAM_T[2]:+.4f}]  '
                  f'CAM_Q=[{CAM_Q[0]:+.4f}, {CAM_Q[1]:+.4f}, {CAM_Q[2]:+.4f}, {CAM_Q[3]:+.4f}]')

    # ── Camera TF ─────────────────────────────────────────────────────────────

    def _publish_camera_tf(self):
        t = TransformStamped()
        t.header.stamp            = self.get_clock().now().to_msg()
        t.header.frame_id         = 'link6'
        t.child_frame_id          = 'camera_color_optical_frame'
        t.transform.translation.x = float(CAM_T[0])
        t.transform.translation.y = float(CAM_T[1])
        t.transform.translation.z = float(CAM_T[2])
        t.transform.rotation.x    = float(CAM_Q[0])
        t.transform.rotation.y    = float(CAM_Q[1])
        t.transform.rotation.z    = float(CAM_Q[2])
        t.transform.rotation.w    = float(CAM_Q[3])
        self._cam_broadcaster.sendTransform(t)

    # ── Marker subscription ───────────────────────────────────────────────────

    def _on_aruco_transform(self, msg: TransformStamped):
        """
        Fires every time ArUco detects the marker.
        msg.header.frame_id  = camera_color_optical_frame
        msg.child_frame_id   = marker_frame

        Marker position in base_link is chained MANUALLY:
          base_link → link6 (robot TF) ∘ calibration ∘ camera → marker (this msg).
        Never look up base_link → camera_color_optical_frame in TF: the camera
        frame has competing static parents (the RealSense driver publishes its
        own camera_color_frame → camera_color_optical_frame edge), and which
        parent a listener ends up with is a latched-message race. Manual
        chaining is immune to that.
        """
        t_c_m = np.array([
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z,
        ])
        self._last_marker_cam = (t_c_m, time.monotonic())
        try:
            tf_b_l6 = self._tf.lookup_transform(
                'base_link', EFFECTOR_LINK,
                Time(), Duration(seconds=0.1))

            R_b_l6 = Rotation.from_quat([
                tf_b_l6.transform.rotation.x,
                tf_b_l6.transform.rotation.y,
                tf_b_l6.transform.rotation.z,
                tf_b_l6.transform.rotation.w,
            ])
            t_b_l6 = np.array([
                tf_b_l6.transform.translation.x,
                tf_b_l6.transform.translation.y,
                tf_b_l6.transform.translation.z,
            ])
            R_b_c = R_b_l6 * Rotation.from_quat(CAM_Q)
            t_b_c = t_b_l6 + R_b_l6.apply(CAM_T)
            t_b_m = t_b_c + R_b_c.apply(t_c_m)
            self._last_marker = (float(t_b_m[0]), float(t_b_m[1]), float(t_b_m[2]),
                                 time.monotonic())

            # Live green sphere in base_link frame — visible in RViz at all times.
            # Lifetime 1.5 s: disappears automatically if ArUco stops detecting.
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns = 'eye_in_hand_live'; m.id = 0
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = float(t_b_m[0])
            m.pose.position.y = float(t_b_m[1])
            m.pose.position.z = float(t_b_m[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 0.9
            m.lifetime.nanosec = int(1.5e9)
            self._marker_pub.publish(m)
        except Exception:
            pass

    def _on_joint_state(self, msg: JointState):
        self._current_joint_state = msg

    # ── Status / logging ──────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = 'INFO'):
        """Print to stdout (captured by the GUI's script log panel) and to the
        ROS logger."""
        ts   = time.strftime('%H:%M:%S')
        print(f'[{ts}] {level}: {msg}', flush=True)
        if level == 'WARN':
            self.get_logger().warn(msg)
        elif level == 'ERROR':
            self.get_logger().error(msg)
        else:
            self.get_logger().info(msg)

    def _status_tick(self):
        """1 Hz heads-up while waiting: report if the marker stream is absent.
        (The old TF fallback through marker_frame was removed — that path runs
        through the contested camera TF edge and can silently give a position
        chained through the WRONG parent.)"""
        if self._last_marker is None:
            self._log('Waiting: no marker detection yet on /aruco_single/transform...')

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _publish_viz(self, marker_xyz: np.ndarray, cam_pos: np.ndarray,
                     R_world_cam: Rotation,
                     link6_pos: np.ndarray, link6_quat: np.ndarray,
                     label: str):
        """Publish MarkerArray to /eye_in_hand/viz for RViz inspection:
        marker sphere, link6 target arrow, optical-axis arrow, camera frustum,
        and a text label with the chosen sweep candidate."""
        now = self.get_clock().now().to_msg()
        markers = []

        def _base(m, ns_id, mtype):
            m.header.frame_id = 'base_link'; m.header.stamp = now
            m.ns = 'eye_in_hand'; m.id = ns_id
            m.type = mtype; m.action = Marker.ADD
            m.lifetime.sec = 30
            return m

        # Green sphere — detected ArUco marker position
        sp = _base(Marker(), 0, Marker.SPHERE)
        sp.pose.position.x = float(marker_xyz[0])
        sp.pose.position.y = float(marker_xyz[1])
        sp.pose.position.z = float(marker_xyz[2])
        sp.pose.orientation.w = 1.0
        sp.scale.x = sp.scale.y = sp.scale.z = 0.05
        sp.color.r = 0.0; sp.color.g = 1.0; sp.color.b = 0.0; sp.color.a = 0.8
        markers.append(sp)

        # Orange arrow — link6 target pose (arrow shaft = link6 x-axis direction)
        ar = _base(Marker(), 1, Marker.ARROW)
        ar.pose.position.x = float(link6_pos[0])
        ar.pose.position.y = float(link6_pos[1])
        ar.pose.position.z = float(link6_pos[2])
        ar.pose.orientation.x = float(link6_quat[0])
        ar.pose.orientation.y = float(link6_quat[1])
        ar.pose.orientation.z = float(link6_quat[2])
        ar.pose.orientation.w = float(link6_quat[3])
        ar.scale.x = 0.12; ar.scale.y = 0.02; ar.scale.z = 0.02
        ar.color.r = 1.0; ar.color.g = 0.4; ar.color.b = 0.0; ar.color.a = 0.9
        markers.append(ar)

        def _pt(v):
            p = Point(); p.x = float(v[0]); p.y = float(v[1]); p.z = float(v[2])
            return p

        # Magenta arrow — camera optical axis: camera position → marker
        ax = _base(Marker(), 2, Marker.ARROW)
        ax.points = [_pt(cam_pos), _pt(marker_xyz)]
        ax.scale.x = 0.008; ax.scale.y = 0.02; ax.scale.z = 0.03
        ax.color.r = 1.0; ax.color.g = 0.0; ax.color.b = 1.0; ax.color.a = 0.9
        markers.append(ax)

        # Cyan frustum — D435 colour FOV (~69°×42°) rays from the camera pose,
        # length = approach distance, ending in a rectangle around the marker.
        fr = _base(Marker(), 3, Marker.LINE_LIST)
        fr.pose.orientation.w = 1.0
        fr.scale.x = 0.003
        fr.color.r = 0.0; fr.color.g = 0.8; fr.color.b = 1.0; fr.color.a = 0.7
        dist = float(np.linalg.norm(marker_xyz - cam_pos))
        hx = dist * np.tan(np.radians(69.0 / 2))   # image x half-extent
        hy = dist * np.tan(np.radians(42.0 / 2))   # image y half-extent
        corners_cam = [np.array([sx * hx, sy * hy, dist])
                       for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        corners = [cam_pos + R_world_cam.apply(c) for c in corners_cam]
        pts = []
        for i in range(4):
            pts += [_pt(cam_pos), _pt(corners[i])]                 # rays
            pts += [_pt(corners[i]), _pt(corners[(i + 1) % 4])]    # rectangle
        fr.points = pts
        markers.append(fr)

        # Text — which sweep candidate was chosen
        tx = _base(Marker(), 4, Marker.TEXT_VIEW_FACING)
        tx.pose.position.x = float(cam_pos[0])
        tx.pose.position.y = float(cam_pos[1])
        tx.pose.position.z = float(cam_pos[2]) + 0.06
        tx.pose.orientation.w = 1.0
        tx.scale.z = 0.025
        tx.color.r = 1.0; tx.color.g = 1.0; tx.color.b = 1.0; tx.color.a = 0.9
        tx.text = label
        markers.append(tx)

        self._viz_pub.publish(MarkerArray(markers=markers))
        self._log('[VIZ] Published to /eye_in_hand/viz — add MarkerArray in RViz')

    # ── IK query ──────────────────────────────────────────────────────────────

    _JOINT_LIMITS = {
        'joint1': (-2.618,  2.618),
        'joint2': ( 0.0,    3.14),
        'joint3': (-2.967,  0.0),
        'joint4': (-2.217,  2.217),
        'joint5': (-1.562,  1.562),
        'joint6': (-2.094,  2.094),
    }

    def _call_ik(self, pos: np.ndarray, quat: np.ndarray,
                 timeout_s: float = SWEEP_IK_TIMEOUT_S) -> dict | None:
        """
        Call /compute_ik for a link6 pose. Returns {joint_name: position}
        for the solution, or None if unreachable / service problem.
        """
        if not self._ik_cli.wait_for_service(timeout_sec=5.0):
            self._log('[IK] /compute_ik service not available.', 'ERROR')
            return None

        req = GetPositionIK.Request()
        req.ik_request.group_name                    = MOVE_GROUP
        req.ik_request.ik_link_name                  = EFFECTOR_LINK
        if self._current_joint_state is not None:
            req.ik_request.robot_state.joint_state = self._current_joint_state
        else:
            req.ik_request.robot_state.is_diff = True
        req.ik_request.avoid_collisions              = True
        req.ik_request.pose_stamped.header.frame_id  = 'base_link'
        req.ik_request.pose_stamped.header.stamp     = self.get_clock().now().to_msg()
        req.ik_request.pose_stamped.pose             = _make_pose(pos, quat)
        req.ik_request.timeout.sec                   = 0
        req.ik_request.timeout.nanosec               = int(timeout_s * 1e9)

        future = self._ik_cli.call_async(req)
        done   = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=timeout_s + 3.0):
            self._log('[IK] Service timed out.', 'ERROR')
            return None

        resp = future.result()
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        return dict(zip(resp.solution.joint_state.name,
                        resp.solution.joint_state.position))

    def _find_reachable_camera_pose(self, marker_xyz: np.ndarray):
        """
        Sweep look-at candidates and return the first reachable one as
        (cam_pos, R_world_cam, link6_pos, link6_quat, ik_solution, label),
        or None if every candidate fails IK.
        """
        self._log(f'[SWEEP] Searching reachable camera pose {APPROACH_DIST_M:.2f} m '
                  'from marker, optical axis on marker...')
        n_tried = 0
        for tilt, az, roll, R_world_cam in lookat_candidates():
            optical_axis = R_world_cam.apply([0.0, 0.0, 1.0])
            cam_pos = (marker_xyz + np.array([X_CORR_M, Y_CORR_M, 0.0])
                       - optical_axis * APPROACH_DIST_M)
            link6_pos, link6_quat = camera_to_link6(cam_pos, R_world_cam)
            n_tried += 1
            sol = self._call_ik(link6_pos, link6_quat)
            if sol is None:
                continue
            label = f'tilt {tilt}°  az {az}°  roll {roll}°'
            self._log(f'[SWEEP] Reachable candidate #{n_tried}: {label}')
            all_ok = True
            for name, val in sorted(sol.items()):
                lo, hi = self._JOINT_LIMITS.get(name, (-1e9, 1e9))
                ok = lo <= val <= hi
                all_ok = all_ok and ok
                flag = '' if ok else '  ← OUT OF LIMITS'
                self._log(f'     {name}: {val:+.3f} rad ({np.degrees(val):+.1f}°){flag}')
            self._log(f'[SWEEP] All joints within limits? {"YES" if all_ok else "NO — check above"}')
            return cam_pos, R_world_cam, link6_pos, link6_quat, sol, label
        self._log(f'[SWEEP] All {n_tried} candidates unreachable. The hand-eye '
                  'calibration is the prime suspect — redo it and confirm '
                  f'{CALIB_FILE} gets a fresh mtime after saving.', 'ERROR')
        return None

    # ── Marker lookup ─────────────────────────────────────────────────────────

    def _get_marker_xyz(self) -> np.ndarray | None:
        """Return cached marker position in base_link, or None if too stale / never seen.
        _last_marker is kept fresh by _on_aruco_transform (event-driven) and
        _status_tick (1 Hz TF fallback)."""
        if self._last_marker is None:
            return None
        age = time.monotonic() - self._last_marker[3]
        if age > MARKER_CACHE_EXPIRY:
            return None
        if age > 1.0:
            self._log(f'Using cached marker position ({age:.1f}s old).', 'WARN')
        return np.array(self._last_marker[:3])

    # ── Main sequence ─────────────────────────────────────────────────────────

    def run_once(self) -> bool:
        """
        Wait for the marker, find a reachable look-at camera pose, preview it,
        pause for visual inspection in RViz, then plan and execute. Finally
        verify the marker is still in frame. Returns True on success.
        """
        self._log(f'Waiting for marker (timeout {MARKER_WAIT_TIMEOUT_S:.0f}s)...')
        deadline = time.monotonic() + MARKER_WAIT_TIMEOUT_S
        marker_xyz = None
        while rclpy.ok() and time.monotonic() < deadline:
            marker_xyz = self._get_marker_xyz()
            if marker_xyz is not None:
                break
            time.sleep(0.5)

        if marker_xyz is None:
            self._log('Marker never seen — aborting.', 'ERROR')
            return False

        self._log(f'[VIZ] Marker at: x={marker_xyz[0]:+.3f}  y={marker_xyz[1]:+.3f}  '
                  f'z={marker_xyz[2]:+.3f}')

        found = self._find_reachable_camera_pose(marker_xyz)
        if found is None:
            return False
        cam_pos, R_world_cam, link6_pos, link6_quat, ik_solution, label = found

        euler_deg = Rotation.from_quat(link6_quat).as_euler('xyz', degrees=True)
        self._log(f'[VIZ] Camera target:          x={cam_pos[0]:+.3f}  y={cam_pos[1]:+.3f}  '
                  f'z={cam_pos[2]:+.3f}  ({label})')
        self._log(f'[VIZ] link6 target:           x={link6_pos[0]:+.3f}  y={link6_pos[1]:+.3f}  '
                  f'z={link6_pos[2]:+.3f}')
        self._log(f'[VIZ] link6 orientation (euler XYZ deg): '
                  f'[{euler_deg[0]:.1f}, {euler_deg[1]:.1f}, {euler_deg[2]:.1f}]')

        self._publish_viz(marker_xyz, cam_pos, R_world_cam, link6_pos, link6_quat, label)

        self._log(f'[PREVIEW] Executing in {APPROACH_CONFIRM_DELAY_S:.0f}s — check RViz now.')
        time.sleep(APPROACH_CONFIRM_DELAY_S)

        trajectory, fraction = self._plan_cartesian(link6_pos, link6_quat, 'MARKER')
        if trajectory is None:
            self._log(f'[MARKER] Cartesian path incomplete (fraction={fraction:.2f}) '
                      '— falling back to joint-space goal from the IK solution.', 'WARN')
            trajectory = self._plan_joint_goal(ik_solution, 'MARKER')
            if trajectory is None:
                return False

        _call_service(self, self._gate_cli, data=True)

        ok = self._execute_trajectory(trajectory, 'MARKER')
        if not ok:
            self._log('[MARKER] Execution failed.', 'ERROR')
            return False
        self._log('[MARKER] Move complete.')
        self._verify_marker_in_frame()
        return True

    def _verify_marker_in_frame(self):
        """After the move: confirm ArUco still sees the marker and report how
        far off the optical axis it sits. This is the ground-truth check of
        the whole pipeline (and of the calibration quality)."""
        self._log('[VERIFY] Checking marker is still in frame...')
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            if (self._last_marker_cam is not None
                    and time.monotonic() - self._last_marker_cam[1] < 1.0):
                t_c_m, _ = self._last_marker_cam
                dist = float(np.linalg.norm(t_c_m))
                off_deg = float(np.degrees(
                    np.arctan2(np.hypot(t_c_m[0], t_c_m[1]), t_c_m[2])))
                self._log(f'[VERIFY] Marker in frame: {dist:.3f} m from camera '
                          f'(target {APPROACH_DIST_M:.2f}), {off_deg:.1f}° off centre.')
                if abs(dist - APPROACH_DIST_M) > 0.03 or off_deg > IN_FRAME_WARN_DEG:
                    self._log('[VERIFY] Off target — hand-eye calibration is '
                              'likely stale; consider recalibrating, or trim '
                              'X_CORR_M/Y_CORR_M for small centring drift.', 'WARN')
                return
            time.sleep(0.2)
        self._log('[VERIFY] Marker NOT detected after move — it is out of frame '
                  'or occluded. The hand-eye calibration is likely wrong; redo it.',
                  'ERROR')

    # ── Planning / execution pipeline ─────────────────────────────────────────

    def _plan_cartesian(self, pos: np.ndarray, quat: np.ndarray, label: str):
        """
        Request a straight-line Cartesian path from the current link6 pose to
        the target pose via /compute_cartesian_path. Forces the end-effector
        to travel in a line — eliminates the sideways swerve OMPL can produce.
        Returns (trajectory, fraction). trajectory is None if the service was
        unreachable or fraction < 0.9 (caller should fall back).
        """
        if not self._cartesian_cli.wait_for_service(timeout_sec=5.0):
            self._log(f'[{label}] /compute_cartesian_path not available.', 'ERROR')
            return None, 0.0

        try:
            tf_b_l6 = self._tf.lookup_transform(
                'base_link', EFFECTOR_LINK, Time(), Duration(seconds=0.1))
        except Exception as e:
            self._log(f'[{label}] Could not look up current {EFFECTOR_LINK} pose: {e}', 'ERROR')
            return None, 0.0

        current_pose = Pose()
        current_pose.position.x = tf_b_l6.transform.translation.x
        current_pose.position.y = tf_b_l6.transform.translation.y
        current_pose.position.z = tf_b_l6.transform.translation.z
        current_pose.orientation = tf_b_l6.transform.rotation

        req = GetCartesianPath.Request()
        req.header.frame_id  = 'base_link'
        req.header.stamp     = self.get_clock().now().to_msg()
        if self._current_joint_state is not None:
            req.start_state.joint_state = self._current_joint_state
        req.group_name        = MOVE_GROUP
        req.link_name         = EFFECTOR_LINK
        req.waypoints         = [current_pose, _make_pose(pos, quat)]
        req.max_step          = 0.01
        req.jump_threshold    = 0.0
        req.avoid_collisions  = True

        self._log(f'[{label}] Requesting Cartesian path...')

        future = self._cartesian_cli.call_async(req)
        done   = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=15.0):
            self._log(f'[{label}] Cartesian path service timed out.', 'ERROR')
            return None, 0.0

        resp = future.result()
        self._log(f'[{label}] Cartesian path fraction: {resp.fraction:.2f}')
        if resp.fraction < 0.9:
            return None, resp.fraction
        pts = len(resp.solution.joint_trajectory.points)
        self._log(f'[{label}] Cartesian plan OK ({pts} waypoints).')
        return resp.solution, resp.fraction

    def _plan_joint_goal(self, ik_solution: dict, label: str):
        """
        Plan to the already-IK-validated joint configuration. Joint-space
        goals are far more reliable than pose goals: the target is a single
        known-feasible point, so the planner only has to connect to it.
        Returns trajectory or None.
        """
        if not self._plan_client.wait_for_server(timeout_sec=5.0):
            self._log(f'[{label}] /move_action not available.', 'ERROR')
            return None

        constraints = Constraints()
        for name in self._JOINT_LIMITS:            # the 6 arm joints only
            if name not in ik_solution:
                continue
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(ik_solution[name])
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)

        goal = MoveGroup.Goal()
        goal.request.group_name                      = MOVE_GROUP
        goal.request.allowed_planning_time           = 10.0
        goal.request.num_planning_attempts           = 10
        goal.request.max_velocity_scaling_factor     = MAX_VELOCITY_SCALE
        goal.request.max_acceleration_scaling_factor = MAX_ACCEL_SCALE
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = True

        return self._send_plan_goal(goal, label)

    def _send_plan_goal(self, goal: MoveGroup.Goal, label: str):
        """Send a MoveGroup plan goal and block until done. Returns trajectory or None."""
        done   = threading.Event()
        result = [None]
        handle_box = [None]   # store handle so we can cancel on timeout

        def on_result(f):
            result[0] = f.result()
            done.set()

        def on_response(f):
            handle = f.result()
            if not handle.accepted:
                self._log(f'[{label}] Plan goal rejected (server busy — wait a moment).', 'ERROR')
                done.set()
                return
            handle_box[0] = handle
            self._log(f'[{label}] Planning...')
            handle.get_result_async().add_done_callback(on_result)

        self._plan_client.send_goal_async(goal).add_done_callback(on_response)
        done.wait(timeout=30.0)

        if result[0] is None:
            # Cancel the hanging goal so the server is free for the next attempt
            if handle_box[0] is not None:
                handle_box[0].cancel_goal_async()
            self._log(f'[{label}] Planning timed out — goal cancelled, try again.', 'ERROR')
            return None
        code = result[0].result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self._log(f'[{label}] Planning failed (code {code}).', 'ERROR')
            return None
        pts = len(result[0].result.planned_trajectory.joint_trajectory.points)
        self._log(f'[{label}] Plan OK ({pts} waypoints).')
        return result[0].result.planned_trajectory

    def _execute_trajectory(self, trajectory, label: str) -> bool:
        """Send ExecuteTrajectory and block until done. Returns True on success."""
        if not self._execute_client.wait_for_server(timeout_sec=5.0):
            self._log(f'[{label}] /execute_trajectory not available.', 'ERROR')
            return False

        done   = threading.Event()
        result = [None]

        def on_result(f):
            result[0] = f.result()
            done.set()

        def on_response(f):
            handle = f.result()
            if not handle.accepted:
                self._log(f'[{label}] Execute goal rejected.', 'ERROR')
                done.set()
                return
            self._log(f'[{label}] Executing...')
            handle.get_result_async().add_done_callback(on_result)

        self._execute_client.send_goal_async(
            ExecuteTrajectory.Goal(trajectory=trajectory)
        ).add_done_callback(on_response)

        done.wait(timeout=60.0)
        if result[0] is None:
            return False
        return result[0].result.error_code.val == MoveItErrorCodes.SUCCESS


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node = EyeInHandNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run_once()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
