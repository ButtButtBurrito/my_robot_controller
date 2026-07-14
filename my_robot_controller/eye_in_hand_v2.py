#!/usr/bin/env python3
"""
Eye-in-Hand v2 — AGX Piper / RealSense D435 (headless)

Runs once per launch: waits for the ArUco marker, computes the link6 pose
that places the calibrated camera directly above it facing straight down,
publishes RViz preview markers and the IK solution, pauses
APPROACH_CONFIRM_DELAY_S seconds so the target can be checked in RViz, then
plans and executes the move. Exits after one attempt.

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

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    BoundingVolume, Constraints, JointConstraint,
    MoveItErrorCodes, OrientationConstraint, PositionConstraint,
)
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
import tf2_geometry_msgs  # noqa: F401  (registers TF↔geometry_msgs conversions)



# ═══════════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Hover height: distance from marker surface to the camera lens.
# The gripper IS modelled in the URDF so MoveIt will plan around it,
# but verify in RViz that the gripper tip clears the table at this value.
# Increase this first if you see near-miss warnings in the planning output.
APPROACH_HEIGHT_M = 0.20        # metres above marker  ← tune here

# Camera centering correction (metres, world / base_link frame).
# If the marker appears off-centre after the arm moves, tweak these:
#   upper-right drift → increase X_CORR_M and/or Y_CORR_M
#   lower-left  drift → decrease (go negative)
#   left/right only   → adjust X_CORR_M alone
#   up/down only      → adjust Y_CORR_M alone
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
# actually executing the move, so there's time to glance at RViz. Replaces
# the old "press Enter again to confirm" step, which required a keyboard
# that this headless script no longer has access to.
APPROACH_CONFIRM_DELAY_S = 4.0

MARKER_CACHE_EXPIRY = 5.0       # seconds to accept a cached marker TF

MOVE_GROUP    = 'arm'
EFFECTOR_LINK = 'link6'         # calibration was recorded relative to link6


# ═══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION — ~/.ros2/easy_handeye2/calibrations/piper_eih.calib
#  Transform:  link6  →  camera_color_optical_frame
# ═══════════════════════════════════════════════════════════════════════════════

CAM_T = np.array([-0.05593476016675718,
                  -0.031925813829647605,
                   0.044513090224353104])

CAM_Q = np.array([-0.10433398988297468,   # x
                   0.08134981927603824,    # y
                  -0.5141169607716367,     # z
                   0.8474552354583642])    # w   (scipy/ROS xyzw convention)


# ═══════════════════════════════════════════════════════════════════════════════
#  IK target computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_link6_target(
    marker_xyz: np.ndarray,
    approach_h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (link6_position, link6_quaternion_xyzw) that places the calibrated
    camera approach_h metres above marker_xyz with camera +Z pointing down.

    Derivation
    ----------
    Desired camera orientation: camera +Z → world −Z  (looking straight down)
      R_world_cam = Rx(180°)   →  [1 0 0 / 0 -1 0 / 0 0 -1]

    Calibration gives R_link6_cam (rotation from link6 frame to camera frame):
      R_world_link6 = R_world_cam × inv(R_link6_cam)

    Camera origin in world:
      p_cam = marker_xyz + [0, 0, approach_h]

    Link6 origin in world (camera is at CAM_T offset from link6 in link6 frame):
      p_link6 = p_cam − R_world_link6 × CAM_T
    """
    R_world_cam   = Rotation.from_euler('x', 180, degrees=True)
    R_link6_cam   = Rotation.from_quat(CAM_Q)          # scipy takes xyzw
    R_world_link6 = R_world_cam * R_link6_cam.inv()

    cam_pos   = marker_xyz + np.array([X_CORR_M, Y_CORR_M, approach_h])
    link6_pos = cam_pos - R_world_link6.apply(CAM_T)

    return link6_pos, R_world_link6.as_quat()           # xyzw


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
        self._enable_cli     = self.create_client(SetBool, '/enable_agx_arm')
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
        # toward the current configuration — see _call_ik / _plan_pose.
        self._current_joint_state = None   # type: JointState | None
        self.create_subscription(JointState, '/feedback/joint_states',
                                 self._on_joint_state, 10)

        self._last_marker = None   # (x, y, z, monotonic_time)

        self._viz_pub    = self.create_publisher(MarkerArray, '/eye_in_hand/viz', 10)
        self._marker_pub = self.create_publisher(Marker, '/eye_in_hand/marker_live', 10)
        self._ik_cli  = self.create_client(GetPositionIK, '/compute_ik')

        self.create_timer(1.0, self._status_tick)
        self._log('Ready.')

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
        We look up base_link → camera_color_optical_frame (always present via our
        static TF) and chain it with the camera→marker translation to get the
        marker position in base_link — no marker_frame TF entry needed.
        """
        try:
            tf_b_c = self._tf.lookup_transform(
                'base_link', msg.header.frame_id,
                Time(), Duration(seconds=0.1))

            R_b_c = Rotation.from_quat([
                tf_b_c.transform.rotation.x,
                tf_b_c.transform.rotation.y,
                tf_b_c.transform.rotation.z,
                tf_b_c.transform.rotation.w,
            ])
            t_b_c = np.array([
                tf_b_c.transform.translation.x,
                tf_b_c.transform.translation.y,
                tf_b_c.transform.translation.z,
            ])
            t_c_m = np.array([
                msg.transform.translation.x,
                msg.transform.translation.y,
                msg.transform.translation.z,
            ])
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
        """1 Hz TF fallback for _last_marker, in case /aruco_single/transform
        callbacks are sparse."""
        try:
            tf = self._tf.lookup_transform(
                'base_link', 'marker_frame', Time(), Duration(seconds=0.3))
            t = tf.transform.translation
            self._last_marker = (t.x, t.y, t.z, time.monotonic())
        except Exception:
            pass

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _publish_viz(self, marker_xyz: np.ndarray,
                     link6_pos: np.ndarray, link6_quat: np.ndarray):
        """Publish MarkerArray to /eye_in_hand/viz for RViz inspection."""
        now = self.get_clock().now().to_msg()
        markers = []

        # Green sphere — detected ArUco marker position
        sp = Marker()
        sp.header.frame_id = 'base_link'; sp.header.stamp = now
        sp.ns = 'eye_in_hand'; sp.id = 0
        sp.type = Marker.SPHERE; sp.action = Marker.ADD
        sp.pose.position.x = float(marker_xyz[0])
        sp.pose.position.y = float(marker_xyz[1])
        sp.pose.position.z = float(marker_xyz[2])
        sp.pose.orientation.w = 1.0
        sp.scale.x = sp.scale.y = sp.scale.z = 0.05
        sp.color.r = 0.0; sp.color.g = 1.0; sp.color.b = 0.0; sp.color.a = 0.8
        sp.lifetime.sec = 30
        markers.append(sp)

        # Orange arrow — link6 target pose (arrow shaft = link6 x-axis direction)
        ar = Marker()
        ar.header.frame_id = 'base_link'; ar.header.stamp = now
        ar.ns = 'eye_in_hand'; ar.id = 1
        ar.type = Marker.ARROW; ar.action = Marker.ADD
        ar.pose.position.x = float(link6_pos[0])
        ar.pose.position.y = float(link6_pos[1])
        ar.pose.position.z = float(link6_pos[2])
        ar.pose.orientation.x = float(link6_quat[0])
        ar.pose.orientation.y = float(link6_quat[1])
        ar.pose.orientation.z = float(link6_quat[2])
        ar.pose.orientation.w = float(link6_quat[3])
        ar.scale.x = 0.12; ar.scale.y = 0.02; ar.scale.z = 0.02
        ar.color.r = 1.0; ar.color.g = 0.4; ar.color.b = 0.0; ar.color.a = 0.9
        ar.lifetime.sec = 30
        markers.append(ar)

        # Cyan line — offset from marker to link6 target (shows CAM_T contribution)
        ln = Marker()
        ln.header.frame_id = 'base_link'; ln.header.stamp = now
        ln.ns = 'eye_in_hand'; ln.id = 2
        ln.type = Marker.LINE_STRIP; ln.action = Marker.ADD
        ln.pose.orientation.w = 1.0
        ln.scale.x = 0.005
        ln.color.r = 0.0; ln.color.g = 0.8; ln.color.b = 1.0; ln.color.a = 0.7
        ln.lifetime.sec = 30
        p1 = Point()
        p1.x = float(marker_xyz[0]); p1.y = float(marker_xyz[1]); p1.z = float(marker_xyz[2])
        p2 = Point()
        p2.x = float(link6_pos[0]);  p2.y = float(link6_pos[1]);  p2.z = float(link6_pos[2])
        ln.points = [p1, p2]
        markers.append(ln)

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

    def _call_ik(self, pos: np.ndarray, quat: np.ndarray) -> bool:
        """
        Call /compute_ik, log the joint solution verbosely.
        Returns True if a valid IK solution was found.
        """
        if not self._ik_cli.wait_for_service(timeout_sec=5.0):
            self._log('[IK] /compute_ik service not available.', 'ERROR')
            return False

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
        req.ik_request.timeout.sec                   = 5

        self._log('[IK] Requesting IK for link6 target pose...')

        future = self._ik_cli.call_async(req)
        done   = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=12.0):
            self._log('[IK] Service timed out.', 'ERROR')
            return False

        resp = future.result()
        code = resp.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self._log(f'[IK] No solution found (error code {code}).', 'ERROR')
            return False

        js     = resp.solution.joint_state
        all_ok = True
        self._log('[IK] Solution found:')
        for name, val in zip(js.name, js.position):
            deg    = float(np.degrees(val))
            lo, hi = self._JOINT_LIMITS.get(name, (-1e9, 1e9))
            ok     = lo <= val <= hi
            if not ok:
                all_ok = False
            flag = '' if ok else '  ← OUT OF LIMITS'
            self._log(f'     {name}: {val:+.3f} rad ({deg:+.1f}°){flag}')
        self._log(f'[IK] All joints within limits? {"YES" if all_ok else "NO — check above"}')
        return True

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
        Wait for the marker, preview the IK target, pause for visual
        inspection in RViz, then plan and execute. Returns True on success.
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

        link6_pos, link6_quat = compute_link6_target(marker_xyz, APPROACH_HEIGHT_M)

        euler_deg = Rotation.from_quat(link6_quat).as_euler('xyz', degrees=True)
        self._log(f'[VIZ] Marker at:              x={marker_xyz[0]:+.3f}  y={marker_xyz[1]:+.3f}  z={marker_xyz[2]:+.3f}')
        self._log(f'[VIZ] link6 target:           x={link6_pos[0]:+.3f}  y={link6_pos[1]:+.3f}  z={link6_pos[2]:+.3f}')
        self._log(f'[VIZ] CAM_T offset (link6 frame): [{CAM_T[0]:+.3f}, {CAM_T[1]:+.3f}, {CAM_T[2]:+.3f}]')
        self._log(f'[VIZ] Camera desired orientation (euler XYZ deg): [{euler_deg[0]:.1f}, {euler_deg[1]:.1f}, {euler_deg[2]:.1f}]')

        self._publish_viz(marker_xyz, link6_pos, link6_quat)

        ik_ok = self._call_ik(link6_pos, link6_quat)
        if not ik_ok:
            # KDL strict IK failed — OMPL planner uses wider tolerance and may
            # still succeed.
            self._log('[IK] Strict IK failed — OMPL planner will try with tolerance.')

        self._log(f'[PREVIEW] Executing in {APPROACH_CONFIRM_DELAY_S:.0f}s — check RViz now.')
        time.sleep(APPROACH_CONFIRM_DELAY_S)

        trajectory = self._plan_pose(link6_pos, link6_quat, 'MARKER')
        if trajectory is None:
            return False

        _call_service(self, self._enable_cli, data=True)
        _call_service(self, self._gate_cli,   data=True)

        ok = self._execute_trajectory(trajectory, 'MARKER')
        if ok:
            self._log('[MARKER] Complete.')
        else:
            self._log('[MARKER] Execution failed.', 'ERROR')
        return ok

    # ── Planning / execution pipeline ─────────────────────────────────────────

    def _plan_pose(self, pos: np.ndarray, quat: np.ndarray, label: str):
        """Build and send a pose MoveGroup goal (plan only). Returns trajectory or None."""
        if not self._plan_client.wait_for_server(timeout_sec=5.0):
            self._log(f'[{label}] /move_action not available.', 'ERROR')
            return None

        # Position: 2 cm sphere around target.
        region = BoundingVolume()
        region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.02]))
        region.primitive_poses.append(_make_pose(pos, np.array([0, 0, 0, 1])))

        pc = PositionConstraint()
        pc.header.frame_id   = 'base_link'
        pc.link_name         = EFFECTOR_LINK
        pc.constraint_region = region
        pc.weight            = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id    = 'base_link'
        oc.link_name          = EFFECTOR_LINK
        oc.orientation.x      = float(quat[0])
        oc.orientation.y      = float(quat[1])
        oc.orientation.z      = float(quat[2])
        oc.orientation.w      = float(quat[3])
        oc.absolute_x_axis_tolerance = 0.35   # ±20°
        oc.absolute_y_axis_tolerance = 0.35   # ±20°
        oc.absolute_z_axis_tolerance = 0.35   # ±20°
        oc.weight             = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)

        # Soft-bias OMPL toward the arm's current configuration so it doesn't
        # pick a valid-but-distant solution (elbow flip, joint wraparound).
        # Wide ±90° tolerance keeps this from blocking the move; low 0.5 weight
        # keeps it subordinate to the position/orientation goal (weight 1.0).
        if self._current_joint_state is not None:
            for name, val in zip(self._current_joint_state.name,
                                 self._current_joint_state.position):
                jc = JointConstraint()
                jc.joint_name      = name
                jc.position        = float(val)
                jc.tolerance_above = 1.57
                jc.tolerance_below = 1.57
                jc.weight          = 0.5
                constraints.joint_constraints.append(jc)

        goal = MoveGroup.Goal()
        goal.request.group_name                      = MOVE_GROUP
        goal.request.allowed_planning_time           = 30.0
        goal.request.num_planning_attempts           = 50
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
        done.wait(timeout=45.0)   # extra buffer beyond MoveIt's own 30 s

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
