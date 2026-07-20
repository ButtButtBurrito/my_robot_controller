#!/usr/bin/env python3
"""
eih_core — shared eye-in-hand foundation for the AGX Piper + RealSense D435.

Extracted from the validated eye_in_hand_v2.py (2026-07-03). Everything here
was proven live read-only before extraction:

- Calibration is loaded from easy_handeye2's .calib file, never hardcoded.
- Marker/world math NEVER routes through TF lookups of
  camera_color_optical_frame (contested static parents — latched race, see
  session notes 2026-07-03). All chains go base_link → link6 (uncontested)
  and are composed manually with the calibration.
- Targets are found with a look-at candidate sweep validated by /compute_ik,
  not a single rigid orientation ("straight down roll 0" is NOT reachable on
  this arm; roll 90° is).
- Planning prefers a Cartesian straight line, falls back to a joint-space
  goal built from the already-validated IK solution.

Scripts (eye_in_hand_v2, pick_and_place) subclass EihBaseNode and add their
own sequence logic on top.
"""

import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Point, Pose, TransformStamped
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

MOVE_GROUP    = 'arm'
EFFECTOR_LINK = 'link6'   # calibration and all tool offsets are relative to link6

# Gripper geometry (piper_with_gripper_description.xacro): the prismatic
# finger joints mount 0.1358 m from link6 along +Z; the usable fingertip
# grasp point sits a few cm beyond that. Verify in RViz and tune.
GRIPPER_FINGER_MOUNT_Z = 0.1358
GRIPPER_MAX_WIDTH_M    = 0.10     # 2 × 0.05 prismatic travel

JOINT_LIMITS = {
    'joint1': (-2.618,  2.618),
    'joint2': ( 0.0,    3.14),
    'joint3': (-2.967,  0.0),
    'joint4': (-2.217,  2.217),
    'joint5': (-1.562,  1.562),
    'joint6': (-2.094,  2.094),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Calibration
# ═══════════════════════════════════════════════════════════════════════════════

CALIB_FILE = Path.home() / '.ros2/easy_handeye2/calibrations/piper_eih.calib'

_FALLBACK_CAM_T = np.array([-0.05593476016675718,
                            -0.031925813829647605,
                             0.044513090224353104])
_FALLBACK_CAM_Q = np.array([-0.10433398988297468, 0.08134981927603824,
                            -0.5141169607716367,  0.8474552354583642])


def load_calibration() -> tuple[np.ndarray, np.ndarray, str]:
    """Return (CAM_T, CAM_Q(xyzw), source_description) — file first, fallback
    constants only if the file is missing/unreadable."""
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


CAM_T, CAM_Q, CALIB_SOURCE = load_calibration()
R_LINK6_CAM = Rotation.from_quat(CAM_Q)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool frames + look-at sweep
# ═══════════════════════════════════════════════════════════════════════════════

class ToolFrame:
    """A frame rigidly attached to link6 whose pose we want to command.
    camera → hand-eye calibration; TCP → gripper fingertip midpoint."""

    def __init__(self, name: str, t_link6: np.ndarray, R_link6: Rotation):
        self.name = name
        self.t = np.asarray(t_link6, dtype=float)
        self.R = R_link6

    def to_link6(self, tool_pos: np.ndarray, R_world_tool: Rotation
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Desired world pose of this tool frame → required link6 pose."""
        R_world_link6 = R_world_tool * self.R.inv()
        link6_pos = np.asarray(tool_pos) - R_world_link6.apply(self.t)
        return link6_pos, R_world_link6.as_quat()


CAMERA_TOOL = ToolFrame('camera', CAM_T, R_LINK6_CAM)


def make_tcp_tool(tcp_z: float) -> ToolFrame:
    """Gripper TCP: tcp_z metres from link6 along +Z, axes aligned with link6
    (TCP +Z = approach direction = link6 +Z)."""
    return ToolFrame('tcp', np.array([0.0, 0.0, tcp_z]), Rotation.identity())


def lookat_candidates(tilts=(0, 10, 20, 30, 45),
                      azims=(0, 90, 180, 270),
                      rolls=(0, 90, 180, 270)):
    """
    Yield (tilt, az, roll, R_world_tool) with tool +Z pointing at the target,
    ordered by preference (straight-down first).

    R = Rz(az)·Ry(tilt)·Rz(−az) · Rx(180°) · Rz(roll)
      Rx(180°)           — straight down (tool +Z = world −Z), roll-0 reference
      Rz(az)Ry(t)Rz(−az) — tilt the axis 'tilt'° from vertical toward azimuth
      Rz(roll)           — spin about the tool axis (target stays on-axis)
    """
    for tilt in tilts:
        for az in ([0] if tilt == 0 else azims):
            world_tilt = (Rotation.from_euler('z', az, degrees=True) *
                          Rotation.from_euler('y', tilt, degrees=True) *
                          Rotation.from_euler('z', -az, degrees=True))
            base = world_tilt * Rotation.from_euler('x', 180, degrees=True)
            for roll in rolls:
                yield tilt, az, roll, base * Rotation.from_euler('z', roll, degrees=True)


def make_pose(pos: np.ndarray, quat: np.ndarray) -> Pose:
    p = Pose()
    p.position.x    = float(pos[0]);  p.position.y    = float(pos[1])
    p.position.z    = float(pos[2])
    p.orientation.x = float(quat[0]); p.orientation.y = float(quat[1])
    p.orientation.z = float(quat[2]); p.orientation.w = float(quat[3])
    return p


def pt(v) -> Point:
    p = Point(); p.x = float(v[0]); p.y = float(v[1]); p.z = float(v[2])
    return p


# ═══════════════════════════════════════════════════════════════════════════════
#  Base node
# ═══════════════════════════════════════════════════════════════════════════════

class EihBaseNode(Node):
    """Plumbing shared by all eye-in-hand scripts: TF, IK, planning,
    execution, gripper, logging, viz helpers."""

    def __init__(self, node_name: str,
                 max_velocity_scale: float = 0.10,
                 max_accel_scale: float = 0.10):
        super().__init__(node_name)
        self.max_velocity_scale = max_velocity_scale
        self.max_accel_scale    = max_accel_scale

        # Static TF (for RViz display only — code never looks this edge up)
        self._cam_broadcaster = StaticTransformBroadcaster(self)
        self._publish_camera_tf()

        self._tf  = Buffer(node=self)
        self._tfl = TransformListener(self._tf, self, spin_thread=False)

        self._plan_client    = ActionClient(self, MoveGroup,         '/move_action')
        self._execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self._gate_cli       = self.create_client(SetBool, '/control_enable')
        self._ik_cli         = self.create_client(GetPositionIK, '/compute_ik')
        self._cartesian_cli  = self.create_client(GetCartesianPath, '/compute_cartesian_path')

        # Gripper: the arm driver dispatches gripper.move(width) for any
        # JointState named 'gripper' on control/joint_states (same path the
        # GUI's Gripper Control panel uses).
        self._gripper_pub = self.create_publisher(JointState, 'control/joint_states', 10)

        self._current_joint_state = None   # /feedback/joint_states (encoders)
        self.create_subscription(JointState, '/feedback/joint_states',
                                 self._on_joint_state, 10)

        self._log(f'Calibration: {CALIB_SOURCE}')
        self._log(f'  CAM_T=[{CAM_T[0]:+.4f}, {CAM_T[1]:+.4f}, {CAM_T[2]:+.4f}]  '
                  f'CAM_Q=[{CAM_Q[0]:+.4f}, {CAM_Q[1]:+.4f}, {CAM_Q[2]:+.4f}, {CAM_Q[3]:+.4f}]')

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = 'INFO'):
        ts = time.strftime('%H:%M:%S')
        print(f'[{ts}] {level}: {msg}', flush=True)
        if level == 'WARN':
            self.get_logger().warn(msg)
        elif level == 'ERROR':
            self.get_logger().error(msg)
        else:
            self.get_logger().info(msg)

    # ── TF / frames ───────────────────────────────────────────────────────────

    def _publish_camera_tf(self):
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = EFFECTOR_LINK
        t.child_frame_id  = 'camera_color_optical_frame'
        t.transform.translation.x = float(CAM_T[0])
        t.transform.translation.y = float(CAM_T[1])
        t.transform.translation.z = float(CAM_T[2])
        t.transform.rotation.x = float(CAM_Q[0])
        t.transform.rotation.y = float(CAM_Q[1])
        t.transform.rotation.z = float(CAM_Q[2])
        t.transform.rotation.w = float(CAM_Q[3])
        self._cam_broadcaster.sendTransform(t)

    def _on_joint_state(self, msg: JointState):
        self._current_joint_state = msg

    def link6_in_base(self) -> tuple[np.ndarray, Rotation] | None:
        """Current link6 pose in base_link from robot TF (uncontested edge)."""
        try:
            tf = self._tf.lookup_transform('base_link', EFFECTOR_LINK,
                                           Time(), Duration(seconds=0.1))
        except Exception:
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        return (np.array([t.x, t.y, t.z]),
                Rotation.from_quat([q.x, q.y, q.z, q.w]))

    def cam_point_to_base(self, t_cam: np.ndarray) -> np.ndarray | None:
        """Point in camera_color_optical_frame → base_link, chained manually
        via base→link6 TF ∘ calibration (never the contested TF edge)."""
        l6 = self.link6_in_base()
        if l6 is None:
            return None
        t_b_l6, R_b_l6 = l6
        R_b_c = R_b_l6 * R_LINK6_CAM
        t_b_c = t_b_l6 + R_b_l6.apply(CAM_T)
        return t_b_c + R_b_c.apply(np.asarray(t_cam))

    # ── Services / IK ─────────────────────────────────────────────────────────

    def call_setbool(self, cli, data: bool = True, timeout: float = 3.0) -> bool:
        if not cli.wait_for_service(timeout_sec=timeout):
            self._log(f'Service not available: {cli.srv_name}', 'WARN')
            return False
        future = cli.call_async(SetBool.Request(data=data))
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=timeout):
            self._log(f'Service timed out: {cli.srv_name}', 'WARN')
            return False
        res = future.result()
        self._log(f'{cli.srv_name}(data={data}): {res.message}')
        return res.success

    def enable_control_gate(self) -> bool:
        return self.call_setbool(self._gate_cli, data=True)

    def call_ik(self, pos: np.ndarray, quat: np.ndarray,
                timeout_s: float = 0.5) -> dict | None:
        """IK for a link6 pose. Returns {joint: position} or None."""
        if not self._ik_cli.wait_for_service(timeout_sec=5.0):
            self._log('[IK] /compute_ik service not available.', 'ERROR')
            return None
        req = GetPositionIK.Request()
        req.ik_request.group_name   = MOVE_GROUP
        req.ik_request.ik_link_name = EFFECTOR_LINK
        if self._current_joint_state is not None:
            req.ik_request.robot_state.joint_state = self._current_joint_state
        else:
            req.ik_request.robot_state.is_diff = True
        req.ik_request.avoid_collisions             = True
        req.ik_request.pose_stamped.header.frame_id = 'base_link'
        req.ik_request.pose_stamped.header.stamp    = self.get_clock().now().to_msg()
        req.ik_request.pose_stamped.pose            = make_pose(pos, quat)
        req.ik_request.timeout.sec     = 0
        req.ik_request.timeout.nanosec = int(timeout_s * 1e9)

        future = self._ik_cli.call_async(req)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=timeout_s + 3.0):
            self._log('[IK] Service timed out.', 'ERROR')
            return None
        resp = future.result()
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        return dict(zip(resp.solution.joint_state.name,
                        resp.solution.joint_state.position))

    def find_reachable_tool_pose(self, tool: ToolFrame, target_xyz: np.ndarray,
                                 dist: float, label: str = 'SWEEP',
                                 tilts=(0, 10, 20, 30, 45)):
        """
        Sweep look-at candidates for `tool` aimed at target_xyz from `dist`
        metres away. Returns (tool_pos, R_world_tool, link6_pos, link6_quat,
        ik_solution, desc) or None.
        """
        self._log(f'[{label}] Searching reachable {tool.name} pose {dist:.3f} m '
                  f'from ({target_xyz[0]:+.3f}, {target_xyz[1]:+.3f}, {target_xyz[2]:+.3f})...')
        n = 0
        for tilt, az, roll, R_world_tool in lookat_candidates(tilts=tilts):
            axis = R_world_tool.apply([0.0, 0.0, 1.0])
            tool_pos = np.asarray(target_xyz) - axis * dist
            link6_pos, link6_quat = tool.to_link6(tool_pos, R_world_tool)
            n += 1
            sol = self.call_ik(link6_pos, link6_quat)
            if sol is None:
                continue
            desc = f'tilt {tilt}°  az {az}°  roll {roll}°'
            self._log(f'[{label}] Reachable candidate #{n}: {desc}')
            bad = [j for j, v in sol.items()
                   if not (JOINT_LIMITS.get(j, (-9e9, 9e9))[0] <= v
                           <= JOINT_LIMITS.get(j, (-9e9, 9e9))[1])]
            if bad:
                self._log(f'[{label}] Solution outside limits ({bad}) — continuing sweep.', 'WARN')
                continue
            return tool_pos, R_world_tool, link6_pos, link6_quat, sol, desc
        self._log(f'[{label}] All {n} candidates unreachable.', 'ERROR')
        return None

    # ── Planning / execution ──────────────────────────────────────────────────

    def plan_cartesian(self, pos: np.ndarray, quat: np.ndarray, label: str,
                       avoid_collisions: bool = True):
        """Straight-line Cartesian path current link6 pose → target.
        Returns (trajectory|None, fraction)."""
        if not self._cartesian_cli.wait_for_service(timeout_sec=5.0):
            self._log(f'[{label}] /compute_cartesian_path not available.', 'ERROR')
            return None, 0.0
        l6 = self.link6_in_base()
        if l6 is None:
            self._log(f'[{label}] Could not look up current {EFFECTOR_LINK} pose.', 'ERROR')
            return None, 0.0
        t_b_l6, R_b_l6 = l6
        current = make_pose(t_b_l6, R_b_l6.as_quat())

        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_link'
        req.header.stamp    = self.get_clock().now().to_msg()
        if self._current_joint_state is not None:
            req.start_state.joint_state = self._current_joint_state
        req.group_name       = MOVE_GROUP
        req.link_name        = EFFECTOR_LINK
        req.waypoints        = [current, make_pose(pos, quat)]
        req.max_step         = 0.01
        req.jump_threshold   = 0.0
        req.avoid_collisions = avoid_collisions

        future = self._cartesian_cli.call_async(req)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=15.0):
            self._log(f'[{label}] Cartesian path service timed out.', 'ERROR')
            return None, 0.0
        resp = future.result()
        self._log(f'[{label}] Cartesian path fraction: {resp.fraction:.2f}')
        if resp.fraction < 0.9:
            return None, resp.fraction
        self._log(f'[{label}] Cartesian plan OK '
                  f'({len(resp.solution.joint_trajectory.points)} waypoints).')
        return resp.solution, resp.fraction

    def plan_joint_goal(self, ik_solution: dict, label: str):
        """Plan to an IK-validated joint configuration (reliable: the goal is
        a single known-feasible point). Returns trajectory or None."""
        if not self._plan_client.wait_for_server(timeout_sec=5.0):
            self._log(f'[{label}] /move_action not available.', 'ERROR')
            return None
        constraints = Constraints()
        for name in JOINT_LIMITS:                      # the 6 arm joints only
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
        goal.request.max_velocity_scaling_factor     = self.max_velocity_scale
        goal.request.max_acceleration_scaling_factor = self.max_accel_scale
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = True
        return self._send_plan_goal(goal, label)

    def _send_plan_goal(self, goal: MoveGroup.Goal, label: str):
        done = threading.Event()
        result = [None]
        handle_box = [None]

        def on_result(f):
            result[0] = f.result()
            done.set()

        def on_response(f):
            handle = f.result()
            if not handle.accepted:
                self._log(f'[{label}] Plan goal rejected (server busy).', 'ERROR')
                done.set()
                return
            handle_box[0] = handle
            self._log(f'[{label}] Planning...')
            handle.get_result_async().add_done_callback(on_result)

        self._plan_client.send_goal_async(goal).add_done_callback(on_response)
        done.wait(timeout=30.0)
        if result[0] is None:
            if handle_box[0] is not None:
                handle_box[0].cancel_goal_async()
            self._log(f'[{label}] Planning timed out — goal cancelled.', 'ERROR')
            return None
        code = result[0].result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self._log(f'[{label}] Planning failed (code {code}).', 'ERROR')
            return None
        pts = len(result[0].result.planned_trajectory.joint_trajectory.points)
        self._log(f'[{label}] Plan OK ({pts} waypoints).')
        return result[0].result.planned_trajectory

    def plan_to_link6_pose(self, link6_pos, link6_quat, ik_solution, label: str,
                           avoid_collisions: bool = True):
        """Cartesian first, joint-space fallback. Returns trajectory or None."""
        traj, fraction = self.plan_cartesian(link6_pos, link6_quat, label,
                                             avoid_collisions=avoid_collisions)
        if traj is not None:
            return traj
        self._log(f'[{label}] Cartesian incomplete (fraction={fraction:.2f}) — '
                  'falling back to joint-space goal.', 'WARN')
        return self.plan_joint_goal(ik_solution, label)

    def execute_trajectory(self, trajectory, label: str) -> bool:
        if not self._execute_client.wait_for_server(timeout_sec=5.0):
            self._log(f'[{label}] /execute_trajectory not available.', 'ERROR')
            return False
        done = threading.Event()
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
            self._active_exec_handle = handle
            self._log(f'[{label}] Executing...')
            handle.get_result_async().add_done_callback(on_result)

        self._execute_client.send_goal_async(
            ExecuteTrajectory.Goal(trajectory=trajectory)
        ).add_done_callback(on_response)
        done.wait(timeout=60.0)
        self._active_exec_handle = None
        if result[0] is None:
            return False
        return result[0].result.error_code.val == MoveItErrorCodes.SUCCESS

    def cancel_active_execution(self):
        """Best-effort cancel of an in-flight ExecuteTrajectory goal (used by
        the skill executor's `stop`). Mid-trajectory emergency stop remains
        the GUI deadman's job — this only asks MoveIt to abort cleanly."""
        handle = getattr(self, '_active_exec_handle', None)
        if handle is not None:
            self._log('[STOP] Cancelling active trajectory execution...', 'WARN')
            handle.cancel_goal_async()

    def move_to(self, link6_pos, link6_quat, ik_solution, label: str,
                avoid_collisions: bool = True) -> bool:
        """Plan + execute a link6 target. Returns True on success."""
        traj = self.plan_to_link6_pose(link6_pos, link6_quat, ik_solution,
                                       label, avoid_collisions=avoid_collisions)
        if traj is None:
            return False
        return self.execute_trajectory(traj, label)

    # ── Gripper ───────────────────────────────────────────────────────────────

    def set_gripper_width(self, width_m: float, settle_s: float = 1.5):
        """Command gripper opening (0.0 closed … 0.10 open) via the driver's
        control/joint_states 'gripper' dispatch. Blocks settle_s for motion."""
        width_m = float(np.clip(width_m, 0.0, GRIPPER_MAX_WIDTH_M))
        msg = JointState()
        msg.name = ['gripper']
        msg.position = [width_m]
        self._log(f'[GRIPPER] width ← {width_m:.3f} m')
        # The driver's control/joint_states subscription is depth-1 and the
        # MoveIt follow stream floods it at 200 Hz, so a single publish is
        # routinely dropped — repeat at 10 Hz through the settle window.
        deadline = time.monotonic() + max(settle_s, 0.5)
        while time.monotonic() < deadline:
            msg.header.stamp = self.get_clock().now().to_msg()
            self._gripper_pub.publish(msg)
            time.sleep(0.1)

    # ── Viz helpers (callers own ns/id assignment and publishing) ────────────

    def viz_marker(self, ns: str, mid: int, mtype: int,
                   lifetime_s: int = 0) -> Marker:
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns = ns; m.id = mid
        m.type = mtype; m.action = Marker.ADD
        m.lifetime.sec = lifetime_s
        m.pose.orientation.w = 1.0
        return m

    def viz_sphere(self, ns, mid, xyz, rgba, diam=0.05, lifetime_s=0) -> Marker:
        m = self.viz_marker(ns, mid, Marker.SPHERE, lifetime_s)
        m.pose.position = pt(xyz)
        m.scale.x = m.scale.y = m.scale.z = float(diam)
        m.color.r, m.color.g, m.color.b, m.color.a = [float(v) for v in rgba]
        return m

    def viz_cube(self, ns, mid, center_xyz, dims_xyz, rgba, lifetime_s=0) -> Marker:
        m = self.viz_marker(ns, mid, Marker.CUBE, lifetime_s)
        m.pose.position = pt(center_xyz)
        m.scale.x, m.scale.y, m.scale.z = [float(v) for v in dims_xyz]
        m.color.r, m.color.g, m.color.b, m.color.a = [float(v) for v in rgba]
        return m

    def viz_text(self, ns, mid, xyz, text, rgba=(1, 1, 1, 0.9),
                 height=0.03, lifetime_s=0) -> Marker:
        m = self.viz_marker(ns, mid, Marker.TEXT_VIEW_FACING, lifetime_s)
        m.pose.position = pt(xyz)
        m.scale.z = float(height)
        m.color.r, m.color.g, m.color.b, m.color.a = [float(v) for v in rgba]
        m.text = text
        return m

    def viz_arrow_between(self, ns, mid, p_from, p_to, rgba,
                          shaft=0.008, lifetime_s=0) -> Marker:
        m = self.viz_marker(ns, mid, Marker.ARROW, lifetime_s)
        m.points = [pt(p_from), pt(p_to)]
        m.scale.x = shaft; m.scale.y = shaft * 2.5; m.scale.z = shaft * 4
        m.color.r, m.color.g, m.color.b, m.color.a = [float(v) for v in rgba]
        return m


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-marker tracking (aruco_ros marker_publisher)
# ═══════════════════════════════════════════════════════════════════════════════

class TrackedMarker:
    def __init__(self, window: int):
        self._window = window
        self.samples: list[np.ndarray] = []
        self.yaw_deg: float = 0.0
        self.last_seen: float = 0.0

    def update(self, xyz_base: np.ndarray, yaw_deg: float):
        self.samples.append(xyz_base)
        if len(self.samples) > self._window:
            self.samples.pop(0)
        self.yaw_deg = yaw_deg
        self.last_seen = time.monotonic()

    def fresh(self, max_age_s: float) -> bool:
        return (self.last_seen > 0
                and time.monotonic() - self.last_seen < max_age_s)

    @property
    def position(self) -> np.ndarray:
        return np.median(np.stack(self.samples), axis=0)


class MarkerTracker:
    """Tracks every ArUco id seen by aruco_ros marker_publisher, positions in
    base_link via the manual chain (base→link6 TF ∘ calibration ∘ detection).
    Can spawn marker_publisher as a child process if it isn't running."""

    def __init__(self, node: EihBaseNode, marker_size: float = 0.047,
                 window: int = 5, fresh_s: float = 2.0):
        from aruco_msgs.msg import MarkerArray as ArucoMarkerArray
        self._node = node
        self._marker_size = marker_size
        self._window = window
        self.fresh_s = fresh_s
        self._lock = threading.Lock()
        self._markers: dict[int, TrackedMarker] = {}
        self._detector_proc: subprocess.Popen | None = None
        node.create_subscription(ArucoMarkerArray, '/marker_publisher/markers',
                                 self._on_markers, 10)

    def ensure_detector(self, wait_s: float = 3.0):
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if self._node.count_publishers('/marker_publisher/markers') > 0:
                self._node._log('[DETECT] marker_publisher already running.')
                return
            time.sleep(0.3)
        self._node._log('[DETECT] Starting aruco_ros marker_publisher '
                        '(child process)...')
        self._detector_proc = subprocess.Popen(
            ['ros2', 'run', 'aruco_ros', 'marker_publisher', '--ros-args',
             '-p', f'marker_size:={self._marker_size}',
             '-p', 'reference_frame:=camera_color_optical_frame',
             '-p', 'camera_frame:=camera_color_optical_frame',
             '-r', '/camera_info:=/camera/camera/color/camera_info',
             '-r', '/image:=/camera/camera/color/image_raw'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )   # same process group → dies when the GUI stops the script

    def shutdown(self):
        if self._detector_proc is not None and self._detector_proc.poll() is None:
            self._detector_proc.terminate()

    def _on_markers(self, msg):
        l6 = self._node.link6_in_base()
        if l6 is None:
            return
        t_b_l6, R_b_l6 = l6
        R_b_c = R_b_l6 * R_LINK6_CAM
        t_b_c = t_b_l6 + R_b_l6.apply(CAM_T)
        for mk in msg.markers:
            p = mk.pose.pose.position
            q = mk.pose.pose.orientation
            xyz = t_b_c + R_b_c.apply(np.array([p.x, p.y, p.z]))
            R_b_m = R_b_c * Rotation.from_quat([q.x, q.y, q.z, q.w])
            yaw = float(R_b_m.as_euler('zyx', degrees=True)[0])
            with self._lock:
                self._markers.setdefault(
                    mk.id, TrackedMarker(self._window)).update(xyz, yaw)

    def get(self, marker_id: int) -> TrackedMarker | None:
        """Fresh marker or None."""
        with self._lock:
            m = self._markers.get(marker_id)
            return m if (m is not None and m.fresh(self.fresh_s)) else None

    def fresh_ids(self) -> list[int]:
        with self._lock:
            return [i for i, m in self._markers.items() if m.fresh(self.fresh_s)]

    def inject(self, marker_id: int, xyz, yaw_deg: float = 0.0):
        """Testing hook: pretend a detection arrived (used by self-tests)."""
        with self._lock:
            self._markers.setdefault(
                marker_id, TrackedMarker(self._window)).update(
                    np.asarray(xyz, dtype=float), yaw_deg)
