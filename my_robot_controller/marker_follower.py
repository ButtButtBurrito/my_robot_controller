#!/usr/bin/env python3
"""
marker_follower — continuously follow an ArUco marker (AGX Piper + D435).

Show a marker to the camera and the arm follows it for as long as the script
runs: it keeps the camera FOLLOW_DIST_M from the marker, optical axis aimed
at it, biased to sit above / in front of it. Every control cycle it takes a
small clamped step toward the ideal standoff pose:

  - Marker moves → the target moves → the arm follows in short straight-line
    (Cartesian) segments. No joint-space fallback: follower motion must stay
    predictable, so blocked = stop, never a big planner swing.
  - Marker moves AGAIN while a segment is still executing → the segment is
    cancelled mid-flight and replanned at the fresh position (PREEMPT_MOVE_M
    gates this so detection jitter/smear can't thrash it). Combined with
    zero idle sleep while actively following, this is what makes tracking
    feel immediate instead of "finish the old move, then notice".
  - Ideal pose unreachable (IK fails, workspace edge, obstacle) → it tries
    half then quarter steps, then rotation-only ("keep facing it"), then
    holds until the marker moves somewhere new.
  - A Cartesian path blocked partway (e.g. by a planning-scene obstacle) is
    executed up to the blockage: the arm goes as far as it can, stops short,
    and keeps tracking with orientation only.
  - Marker lost → hold pose and wait. If the locked marker stays gone for
    RELOCK_AFTER_S, any marker that appears is adopted instead.

There is NO end condition: the loop runs until the GUI's Stop Script /
released F5 deadman kills the process. Deadman safety is owned entirely by
agx_arm_gui (SIGINT→SIGTERM→SIGKILL on the process group + independent
go-home) — this script has no deadman logic of its own; do not run it
outside the GUI without an equivalent external safety layer.

OBSTACLE TEST (custom models in the planning scene)
  If ~/.ros2/follower_obstacles.yaml exists, its boxes / cylinders / spheres
  / STL meshes are added to the MoveIt planning scene at startup. All IK and
  Cartesian planning here runs with avoid_collisions=True, so the arm stops
  short of them instead of hitting them. See
  follower_obstacles.example.yaml next to this file for the format; copy it
  to ~/.ros2/follower_obstacles.yaml to activate. Obstacles are removed
  again on clean exit (Stop Script sends SIGINT first, so this normally
  runs); after a hard kill run with --clear-obstacles once, or restart
  MoveIt, otherwise the leftover boxes will silently block OTHER scripts'
  planning (pick_and_place!). RViz: obstacles show in any MotionPlanning /
  PlanningScene display, and are mirrored on /follower/viz (MarkerArray)
  together with the live marker, the target pose and a status line.

SAFETY: EXECUTE = False by default — computes and visualises the target
every cycle, calls IK (so reachability is real), but never moves. Watch
/follower/viz in RViz, then set EXECUTE = True.

Offline check (no hardware): python3 marker_follower.py --selftest
"""

import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml
import rclpy
from rclpy.time import Time                      # noqa: F401 (kept for parity)
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray

from my_robot_controller.eih_core import (
    CAM_T, CAMERA_TOOL, EihBaseNode, JOINT_LIMITS, MarkerTracker,
    R_LINK6_CAM, make_pose, pt,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

EXECUTE = True       # ← False = track/plan/visualise ONLY (no motion at all).
                       #   Set True only after a clean EXECUTE=False run.

FOLLOW_MARKER_ID = None    # None = lock onto the first marker seen; or an int id
MARKER_SIZE_M    = 0.047   # must match the printed marker

# Standoff: camera lens to marker along the line of sight. Larger than
# eye_in_hand's 0.15 so the marker stays inside the 69°×42° FOV while both
# the marker and the arm are moving.
FOLLOW_DIST_M = 0.25

MAX_STEP_M       = 0.08    # max translation per control cycle (short segments;
                           # 0.05→0.08 2026-07-12: fewer plan/execute round
                           # trips per travelled metre = snappier tracking)
POS_DEADBAND_M   = 0.02    # already this close to the ideal standoff → don't replan
ANG_DEADBAND_DEG = 8.0     # optical axis this close to the marker → don't replan
LOOP_PERIOD_S    = 0.2     # sleep between cycles while IDLE (holding/no marker);
                           # while actively following the loop replans immediately
PREEMPT_MOVE_M   = 0.04    # marker moved this far mid-segment → cancel + replan
                           # NOW. Floor it above detection smear: samples taken
                           # while the camera moves are off by up to ~1-2 cm
                           # (image 100-300 ms older than the TF used).
RETRY_MOVE_M     = 0.02    # after "unreachable", marker must move this far to retry

MARKER_STALE_S  = 1.0      # detection older than this = marker lost → hold
RELOCK_AFTER_S  = 5.0      # locked marker gone this long → adopt any other marker
MEDIAN_WINDOW   = 3        # small median filter (rejects single-frame jumps)

# Standoff direction = current camera bearing, but forced at least this far
# "up" (unit-vector z-component; 0.26 ≈ 15° above the marker's horizontal
# plane) so the camera prefers to be above / in front rather than level.
MIN_UP_COMPONENT = 0.26

# Camera-position workspace clamp (base_link). The ideal standoff is clamped
# into this region BEFORE reachability is tried, so a marker carried out of
# range makes the arm track along the boundary instead of thrashing on IK.
WORKSPACE_MAX_R      = 0.55    # max distance from base origin
WORKSPACE_MIN_Z      = 0.05    # camera never below this height
WORKSPACE_MAX_Z      = 0.65
WORKSPACE_MIN_R_XY   = 0.12    # keep-out cylinder around the base axis

ROLL_SWEEP_DEG = (0, 90, 180, 270)   # about the optical axis; last good tried first
IK_TIMEOUT_S   = 0.3                 # per-candidate /compute_ik budget

MIN_EXEC_FRACTION = 0.15   # execute a partial Cartesian path down to this fraction

# NOTE (verified against moveit2 humble source 2026-07-12): the Cartesian
# path service time-parameterizes with a HARDCODED scaling of 1.0 (TOTG), so
# these scales never applied to the follow segments — they only pace the
# joint-space fallback branch. Raised 0.10→0.25 so that branch isn't 10×
# slower than the segments around it. Live behaviour confirmed by the user
# ("works"); the GUI deadman remains the safety layer.
MAX_VELOCITY_SCALE = 0.25
MAX_ACCEL_SCALE    = 0.25

OBSTACLES_FILE = Path.home() / '.ros2/follower_obstacles.yaml'
REMOVE_OBSTACLES_ON_EXIT = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure geometry (module-level so --selftest can hit it without a ROS graph)
# ═══════════════════════════════════════════════════════════════════════════════

def lookat_rotation(direction, roll_deg: float = 0.0) -> Rotation:
    """Rotation whose +Z (optical axis) points along `direction` (world),
    then rolled about that axis. For direction = straight down and roll 0
    this reproduces eye_in_hand's Rx(180°) reference orientation."""
    z = np.asarray(direction, dtype=float)
    n = np.linalg.norm(z)
    z = np.array([0.0, 0.0, -1.0]) if n < 1e-9 else z / n
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, z)) > 0.95:           # looking along ±X → use Y reference
        ref = np.array([0.0, 1.0, 0.0])
    x = ref - z * np.dot(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    base = Rotation.from_matrix(np.column_stack([x, y, z]))
    return base * Rotation.from_euler('z', roll_deg, degrees=True)


def standoff_direction(cam_pos, marker_xyz, min_up: float = MIN_UP_COMPONENT
                       ) -> np.ndarray:
    """Unit vector marker → desired camera position: keep the camera's
    current bearing, but at least `min_up` above the horizontal."""
    d = np.asarray(cam_pos, dtype=float) - np.asarray(marker_xyz, dtype=float)
    n = np.linalg.norm(d)
    d = np.array([0.0, 0.0, 1.0]) if n < 1e-6 else d / n
    if d[2] < min_up:
        h = float(np.hypot(d[0], d[1]))
        h_scale = float(np.sqrt(1.0 - min_up ** 2)) / max(h, 1e-9)
        d[:2] *= h_scale
        d[2] = min_up
    return d


def clamp_step(current, desired, max_step: float = MAX_STEP_M) -> np.ndarray:
    """Desired position, clamped to at most max_step from current."""
    current = np.asarray(current, dtype=float)
    delta = np.asarray(desired, dtype=float) - current
    n = np.linalg.norm(delta)
    if n <= max_step:
        return np.asarray(desired, dtype=float)
    return current + delta * (max_step / n)


def clamp_workspace(p) -> np.ndarray:
    """Clamp a camera position into the reachable envelope (see constants)."""
    p = np.asarray(p, dtype=float).copy()
    p[2] = float(np.clip(p[2], WORKSPACE_MIN_Z, WORKSPACE_MAX_Z))
    n = np.linalg.norm(p)
    if n > WORKSPACE_MAX_R:
        p *= WORKSPACE_MAX_R / n
    r_xy = float(np.hypot(p[0], p[1]))
    if r_xy < WORKSPACE_MIN_R_XY:
        if r_xy < 1e-9:
            p[0] = WORKSPACE_MIN_R_XY
        else:
            p[:2] *= WORKSPACE_MIN_R_XY / r_xy
    p[2] = float(np.clip(p[2], WORKSPACE_MIN_Z, WORKSPACE_MAX_Z))
    return p


def angle_between_deg(v1, v2) -> float:
    v1 = np.asarray(v1, dtype=float); v2 = np.asarray(v2, dtype=float)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-12:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


# ═══════════════════════════════════════════════════════════════════════════════
#  STL loading (for custom mesh obstacles) — no external mesh deps
# ═══════════════════════════════════════════════════════════════════════════════

def load_stl(path: Path, scale: float = 1.0
             ) -> tuple[np.ndarray, np.ndarray]:
    """Parse an STL file (binary or ASCII). Returns (vertices Nx3 float,
    faces Mx3 int). `scale` converts units (0.001 for STLs exported in mm)."""
    raw = Path(path).expanduser().read_bytes()
    if raw[:5] == b'solid' and b'facet' in raw[:2000]:
        verts = []
        for line in raw.decode('ascii', errors='replace').splitlines():
            parts = line.split()
            if parts[:1] == ['vertex']:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        v = np.asarray(verts, dtype=float)
    else:
        n_tri = struct.unpack('<I', raw[80:84])[0]
        dt = np.dtype([('normal', '<f4', 3), ('v', '<f4', (3, 3)),
                       ('attr', '<u2')])
        tris = np.frombuffer(raw, dtype=dt, count=n_tri, offset=84)
        v = tris['v'].reshape(-1, 3).astype(float)
    if len(v) == 0 or len(v) % 3 != 0:
        raise ValueError(f'{path}: no triangles parsed')
    v = v * float(scale)
    uniq, inverse = np.unique(v.round(9), axis=0, return_inverse=True)
    return uniq, inverse.reshape(-1, 3)


def obstacle_to_collision_object(entry: dict) -> CollisionObject:
    """YAML obstacle entry → moveit CollisionObject (frame base_link).
    Types: box (dims [x,y,z]), sphere (radius), cylinder (height, radius),
    mesh (file, optional scale). Optional: rpy_deg [r,p,y]."""
    co = CollisionObject()
    co.header.frame_id = 'base_link'
    co.id = str(entry['id'])
    co.operation = CollisionObject.ADD

    pose = Pose()
    px, py, pz = [float(v) for v in entry['position']]
    pose.position.x, pose.position.y, pose.position.z = px, py, pz
    q = Rotation.from_euler(
        'xyz', [float(v) for v in entry.get('rpy_deg', (0, 0, 0))],
        degrees=True).as_quat()
    pose.orientation.x, pose.orientation.y = float(q[0]), float(q[1])
    pose.orientation.z, pose.orientation.w = float(q[2]), float(q[3])

    kind = entry['type']
    if kind == 'mesh':
        verts, faces = load_stl(entry['file'], float(entry.get('scale', 1.0)))
        mesh = Mesh()
        for v in verts:
            mesh.vertices.append(Point(x=float(v[0]), y=float(v[1]), z=float(v[2])))
        for f in faces:
            mesh.triangles.append(MeshTriangle(vertex_indices=[int(i) for i in f]))
        co.meshes = [mesh]
        co.mesh_poses = [pose]
        return co

    prim = SolidPrimitive()
    if kind == 'box':
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(v) for v in entry['dims']]
    elif kind == 'sphere':
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [float(entry['radius'])]
    elif kind == 'cylinder':
        prim.type = SolidPrimitive.CYLINDER
        prim.dimensions = [float(entry['height']), float(entry['radius'])]
    else:
        raise ValueError(f'unknown obstacle type {kind!r}')
    co.primitives = [prim]
    co.primitive_poses = [pose]
    return co


# ═══════════════════════════════════════════════════════════════════════════════
#  Node
# ═══════════════════════════════════════════════════════════════════════════════

class MarkerFollowerNode(EihBaseNode):

    def __init__(self):
        super().__init__('marker_follower',
                         max_velocity_scale=MAX_VELOCITY_SCALE,
                         max_accel_scale=MAX_ACCEL_SCALE)
        self.tracker = MarkerTracker(self, marker_size=MARKER_SIZE_M,
                                     window=MEDIAN_WINDOW, fresh_s=MARKER_STALE_S)
        self._scene_cli = self.create_client(ApplyPlanningScene,
                                             '/apply_planning_scene')

        self._locked_id: int | None = FOLLOW_MARKER_ID
        self._lock_lost_at: float | None = None
        self._last_good_roll: float = 0.0
        self._unreach_marker_pos: np.ndarray | None = None
        self._exec_fail_streak = 0

        self._status = 'starting'
        self._target_viz: tuple | None = None   # (cam_target, marker_xyz)
        self._obstacle_entries: list[dict] = []

        self._viz_pub = self.create_publisher(MarkerArray, '/follower/viz', 10)
        self.create_timer(0.5, self._publish_viz)

        mode = 'EXECUTE' if EXECUTE else 'PLAN-ONLY (no motion; set EXECUTE=True when happy)'
        self._log(f'Mode: {mode}')
        self._log(f'Following: {"first marker seen" if FOLLOW_MARKER_ID is None else f"id {FOLLOW_MARKER_ID}"}'
                  f'  standoff {FOLLOW_DIST_M:.2f} m  step ≤ {MAX_STEP_M*100:.0f} cm/cycle')

    # ── Current camera pose ───────────────────────────────────────────────────

    def current_camera_pose(self) -> tuple[np.ndarray, Rotation] | None:
        """(cam_pos, R_world_cam) via base→link6 TF ∘ calibration (manual
        chain — never the contested camera TF edge)."""
        l6 = self.link6_in_base()
        if l6 is None:
            return None
        t_b_l6, R_b_l6 = l6
        return t_b_l6 + R_b_l6.apply(CAM_T), R_b_l6 * R_LINK6_CAM

    # ── Marker acquisition / lock ─────────────────────────────────────────────

    def acquire_marker(self):
        """Fresh TrackedMarker to follow, or None. Locks onto one id and only
        relocks after RELOCK_AFTER_S without a sighting (unless a fixed
        FOLLOW_MARKER_ID was configured)."""
        if self._locked_id is not None:
            m = self.tracker.get(self._locked_id)
            if m is not None:
                self._lock_lost_at = None
                return m
            if FOLLOW_MARKER_ID is None:
                if self._lock_lost_at is None:
                    self._lock_lost_at = time.monotonic()
                elif time.monotonic() - self._lock_lost_at > RELOCK_AFTER_S:
                    self._log(f'[LOCK] Marker {self._locked_id} gone '
                              f'>{RELOCK_AFTER_S:.0f}s — will follow any marker.')
                    self._locked_id = None
                    self._lock_lost_at = None
            return None
        ids = sorted(self.tracker.fresh_ids())
        if not ids:
            return None
        self._locked_id = ids[0]
        self._log(f'[LOCK] Following marker id {self._locked_id}.')
        return self.tracker.get(self._locked_id)

    # ── Reachability ──────────────────────────────────────────────────────────

    def reachable_lookat(self, cam_pos: np.ndarray, marker_xyz: np.ndarray):
        """First reachable (link6_pos, link6_quat, ik_solution, roll) with the
        camera at cam_pos looking at marker_xyz, sweeping roll (last good
        first). None if every roll fails IK / joint limits."""
        rolls = [self._last_good_roll] + [r for r in ROLL_SWEEP_DEG
                                          if r != self._last_good_roll]
        for roll in rolls:
            R_cam = lookat_rotation(np.asarray(marker_xyz) - np.asarray(cam_pos), roll)
            link6_pos, link6_quat = CAMERA_TOOL.to_link6(cam_pos, R_cam)
            sol = self.call_ik(link6_pos, link6_quat, timeout_s=IK_TIMEOUT_S)
            if sol is None:
                continue
            if any(not (JOINT_LIMITS[j][0] <= sol[j] <= JOINT_LIMITS[j][1])
                   for j in JOINT_LIMITS if j in sol):
                continue
            self._last_good_roll = roll
            return link6_pos, link6_quat, sol, roll
        return None

    # ── Partial Cartesian planning ────────────────────────────────────────────

    def plan_cartesian_partial(self, pos: np.ndarray, quat: np.ndarray):
        """Straight-line Cartesian path current → target. Unlike
        eih_core.plan_cartesian (which rejects fraction < 0.9) any fraction
        ≥ MIN_EXEC_FRACTION is returned: the service truncates the path where
        collision/IK stops it, so executing it = 'go as far as you can, then
        stop'. Returns (trajectory|None, fraction)."""
        if not self._cartesian_cli.wait_for_service(timeout_sec=5.0):
            self._log('[FOLLOW] /compute_cartesian_path not available.', 'ERROR')
            return None, 0.0
        l6 = self.link6_in_base()
        if l6 is None:
            return None, 0.0
        t_b_l6, R_b_l6 = l6

        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_link'
        req.header.stamp    = self.get_clock().now().to_msg()
        if self._current_joint_state is not None:
            req.start_state.joint_state = self._current_joint_state
        req.group_name       = 'arm'
        req.link_name        = 'link6'
        # Target ONLY — the service always interpolates from start_state, and
        # including the current pose as waypoint[0] poisons the fraction: a
        # totally blocked path still reports 0.50 ("achieved" the zero-length
        # first leg), which passed MIN_EXEC_FRACTION and executed a zero-motion
        # trajectory forever (observed live 2026-07-06: arm never moved, every
        # cycle "2 points, followed 50%", controller success in 4 ms).
        req.waypoints        = [make_pose(pos, quat)]
        req.max_step         = 0.01
        req.jump_threshold   = 0.0
        req.avoid_collisions = True

        future = self._cartesian_cli.call_async(req)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=15.0):
            self._log('[FOLLOW] Cartesian path service timed out.', 'ERROR')
            return None, 0.0
        resp = future.result()
        if resp.fraction < MIN_EXEC_FRACTION:
            return None, resp.fraction
        return resp.solution, resp.fraction

    # ── Preemptable execution ─────────────────────────────────────────────────

    def execute_follow(self, traj, marker_at_plan: np.ndarray) -> str:
        """Execute a follow segment, watching the marker while it runs: if it
        moves > PREEMPT_MOVE_M from where the segment was planned, cancel the
        trajectory mid-flight and return 'preempted' so the caller replans
        immediately. Returns 'done' | 'preempted' | 'failed'."""
        result: dict = {}

        def _worker():
            result['ok'] = self.execute_trajectory(traj, 'FOLLOW')

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        preempted = False
        while t.is_alive():
            t.join(timeout=0.05)
            if preempted or self._locked_id is None:
                continue
            m = self.tracker.get(self._locked_id)
            if (m is not None and
                    np.linalg.norm(m.position - marker_at_plan) > PREEMPT_MOVE_M):
                self._log('[FOLLOW] Marker moved mid-segment — cancelling to '
                          'replan at the fresh position.')
                self.cancel_active_execution()
                preempted = True
        if preempted:
            return 'preempted'
        return 'done' if result.get('ok') else 'failed'

    # ── One control cycle ─────────────────────────────────────────────────────

    def step_once(self) -> str:
        """One follow cycle. Returns a coarse status string (used for
        change-only logging)."""
        cam = self.current_camera_pose()
        if cam is None:
            return 'waiting for TF (base_link → link6)'
        cam_pos, R_cam = cam

        m = self.acquire_marker()
        if m is None:
            self._target_viz = None
            return 'no marker — holding, show one to the camera'
        marker_xyz = m.position

        # Ideal standoff pose for this cycle
        desired = clamp_workspace(
            marker_xyz + standoff_direction(cam_pos, marker_xyz) * FOLLOW_DIST_M)
        pos_err = float(np.linalg.norm(desired - cam_pos))
        ang_err = angle_between_deg(R_cam.apply([0.0, 0.0, 1.0]),
                                    marker_xyz - cam_pos)
        self._target_viz = (desired, marker_xyz)

        if pos_err < POS_DEADBAND_M and ang_err < ANG_DEADBAND_DEG:
            self._unreach_marker_pos = None
            return 'on target — tracking'

        # After a full unreachable sweep, wait for the marker to actually move
        # before burning IK calls on the same spot again.
        if (self._unreach_marker_pos is not None
                and np.linalg.norm(marker_xyz - self._unreach_marker_pos) < RETRY_MOVE_M):
            return 'target unreachable — holding until the marker moves'

        # Candidate camera positions: full / half / quarter step, then
        # rotation-only (face it from where we are).
        step_target = clamp_step(cam_pos, desired, MAX_STEP_M)
        candidates: list[tuple[str, np.ndarray]] = []
        if pos_err >= POS_DEADBAND_M:
            for frac, name in ((1.0, 'step'), (0.5, 'half-step'), (0.25, 'quarter-step')):
                candidates.append((name, cam_pos + (step_target - cam_pos) * frac))
        if ang_err >= ANG_DEADBAND_DEG:
            candidates.append(('rotate-in-place', cam_pos))

        for name, p in candidates:
            found = self.reachable_lookat(p, marker_xyz)
            if found is None:
                continue
            link6_pos, link6_quat, _sol, roll = found

            if not EXECUTE:
                return (f'PLAN-ONLY: would {name} toward marker '
                        '(reachable — set EXECUTE=True to move)')

            traj, fraction = self.plan_cartesian_partial(link6_pos, link6_quat)
            if traj is None:
                # Straight line blocked from the current joint config (e.g. a
                # wrist branch flip is needed — IK says reachable but the
                # Cartesian service can't get there continuously). Fall back
                # to a joint-space plan to the IK solution we already have;
                # the step is ≤ MAX_STEP_M so the detour stays small.
                traj = self.plan_joint_goal(_sol, 'FOLLOW-JOINT')
                if traj is None:
                    continue                 # truly blocked → try a smaller step
                fraction = 1.0
                name += ' (joint-space)'
            self._log(f'[FOLLOW] {name} → cam ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})'
                      f'  roll {roll:.0f}°  fraction {fraction:.2f}'
                      f'  (marker at {marker_xyz[0]:+.3f}, {marker_xyz[1]:+.3f}, '
                      f'{marker_xyz[2]:+.3f})')
            outcome = self.execute_follow(traj, marker_xyz)
            if outcome in ('done', 'preempted'):
                self._exec_fail_streak = 0
                self._unreach_marker_pos = None
                return ('following' if outcome == 'done'
                        else 'following (preempted — replanning)')
            self._exec_fail_streak += 1
            if self._exec_fail_streak >= 3:
                self._log('[FOLLOW] 3 consecutive execution failures — check '
                          'the arm/controller state.', 'ERROR')
            return 'execution failed — retrying'

        self._unreach_marker_pos = marker_xyz.copy()
        return ('cannot reach further — holding at limit, still facing marker'
                if ang_err < ANG_DEADBAND_DEG else
                'cannot reach or face marker from here — holding')

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.tracker.ensure_detector()
        self.load_obstacles()
        if EXECUTE:
            self.enable_control_gate()
        self._log('Follow loop running — ends only on Stop Script / deadman release.')
        last_status_log = 0.0
        while rclpy.ok():
            status = self.step_once()
            now = time.monotonic()
            # log on change, plus a 10 s heartbeat while idle
            if status != self._status or now - last_status_log > 10.0:
                self._log(f'[STATE] {status}')
                last_status_log = now
            self._status = status
            # Actively following → replan immediately (the executed segment
            # already paced the cycle); sleep only while idle/holding.
            time.sleep(0.02 if status.startswith('following')
                       else LOOP_PERIOD_S)

    # ── Obstacles (planning scene) ────────────────────────────────────────────

    def _apply_scene(self, collision_objects: list[CollisionObject]) -> bool:
        if not self._scene_cli.wait_for_service(timeout_sec=5.0):
            self._log('[SCENE] /apply_planning_scene not available — obstacles '
                      'NOT applied (is MoveIt running?).', 'WARN')
            return False
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = collision_objects
        future = self._scene_cli.call_async(
            ApplyPlanningScene.Request(scene=scene))
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=5.0):
            self._log('[SCENE] apply_planning_scene timed out.', 'WARN')
            return False
        return bool(future.result().success)

    def load_obstacles(self) -> int:
        """Load OBSTACLES_FILE into the planning scene. Returns count added."""
        if not OBSTACLES_FILE.exists():
            self._log(f'[SCENE] No {OBSTACLES_FILE} — planning scene unchanged. '
                      '(Copy follower_obstacles.example.yaml there to test '
                      'obstacle avoidance.)')
            return 0
        try:
            entries = yaml.safe_load(OBSTACLES_FILE.read_text()).get('obstacles', [])
            objects = [obstacle_to_collision_object(e) for e in entries]
        except Exception as e:
            self._log(f'[SCENE] Failed to parse {OBSTACLES_FILE}: {e}', 'ERROR')
            return 0
        if not objects:
            return 0
        if self._apply_scene(objects):
            self._obstacle_entries = entries
            self._log(f'[SCENE] Added {len(objects)} obstacle(s) to the planning '
                      f'scene: {[o.id for o in objects]}. Planning now avoids '
                      'them; view via RViz PlanningScene display or /follower/viz.')
            if not REMOVE_OBSTACLES_ON_EXIT:
                self._log('[SCENE] They persist after exit — clear with '
                          '--clear-obstacles when done.', 'WARN')
            return len(objects)
        return 0

    def clear_obstacles(self) -> bool:
        """Remove every obstacle id named in OBSTACLES_FILE from the scene."""
        if not OBSTACLES_FILE.exists():
            self._log(f'[SCENE] No {OBSTACLES_FILE} — nothing to clear.')
            return True
        try:
            entries = yaml.safe_load(OBSTACLES_FILE.read_text()).get('obstacles', [])
        except Exception as e:
            self._log(f'[SCENE] Failed to parse {OBSTACLES_FILE}: {e}', 'ERROR')
            return False
        removals = []
        for e in entries:
            co = CollisionObject()
            co.header.frame_id = 'base_link'
            co.id = str(e['id'])
            co.operation = CollisionObject.REMOVE
            removals.append(co)
        ok = self._apply_scene(removals)
        if ok:
            self._log(f'[SCENE] Removed {len(removals)} obstacle(s).')
        return ok

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _publish_viz(self):
        ms: list[Marker] = []

        # Live marker (green) — only while fresh
        if self._locked_id is not None:
            m = self.tracker.get(self._locked_id)
            if m is not None:
                p = m.position
                ms.append(self.viz_sphere('follow', 0, p, (0, 1, 0, 0.9),
                                          diam=0.05, lifetime_s=2))
                ms.append(self.viz_text('follow', 1, p + [0, 0, 0.06],
                                        f'id {self._locked_id}',
                                        (0, 1, 0, 0.9), lifetime_s=2))

        # Target camera pose (orange) + line of sight (magenta)
        if self._target_viz is not None:
            desired, marker_xyz = self._target_viz
            ms.append(self.viz_sphere('follow', 2, desired, (1, 0.5, 0, 0.8),
                                      diam=0.035, lifetime_s=2))
            ms.append(self.viz_arrow_between('follow', 3, desired, marker_xyz,
                                             (1, 0, 1, 0.8), lifetime_s=2))

        # Obstacles (translucent red)
        mid = 10
        for e in self._obstacle_entries:
            try:
                ms.append(self._obstacle_viz(e, mid))
            except Exception:
                pass
            mid += 1

        mode = 'EXECUTE' if EXECUTE else 'plan-only'
        ms.append(self.viz_text('status', 0, [0.0, 0.0, 0.55],
                                f'marker_follower [{mode}] {self._status}'))
        self._viz_pub.publish(MarkerArray(markers=ms))

    def _obstacle_viz(self, e: dict, mid: int) -> Marker:
        rgba = (1.0, 0.15, 0.15, 0.45)
        pos = [float(v) for v in e['position']]
        q = Rotation.from_euler('xyz', [float(v) for v in e.get('rpy_deg', (0, 0, 0))],
                                degrees=True).as_quat()
        if e['type'] == 'box':
            m = self.viz_cube('obstacles', mid, pos, e['dims'], rgba)
        elif e['type'] == 'sphere':
            m = self.viz_sphere('obstacles', mid, pos, rgba,
                                diam=2 * float(e['radius']))
        elif e['type'] == 'cylinder':
            m = self.viz_marker('obstacles', mid, Marker.CYLINDER)
            m.pose.position = pt(pos)
            m.scale.x = m.scale.y = 2 * float(e['radius'])
            m.scale.z = float(e['height'])
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
        else:                                   # mesh → triangle list
            verts, faces = load_stl(e['file'], float(e.get('scale', 1.0)))
            m = self.viz_marker('obstacles', mid, Marker.TRIANGLE_LIST)
            m.pose.position = pt(pos)
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.points = [pt(verts[i]) for f in faces for i in f]
        m.pose.orientation.x, m.pose.orientation.y = float(q[0]), float(q[1])
        m.pose.orientation.z, m.pose.orientation.w = float(q[2]), float(q[3])
        return m

    def shutdown(self):
        self.tracker.shutdown()
        if REMOVE_OBSTACLES_ON_EXIT and self._obstacle_entries:
            self.clear_obstacles()


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test (no hardware, no ROS graph peers needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> int:
    global EXECUTE
    fails = 0

    def check(name, cond, detail=''):
        nonlocal fails
        print(f'  {"PASS" if cond else "FAIL"}  {name}  {detail}')
        fails += 0 if cond else 1

    print('1) geometry')
    for d in ([0, 0, -1], [1, 0, 0], [0.3, -0.4, 0.2], [0, 1, 0]):
        R = lookat_rotation(d, 137.0)
        axis = R.apply([0, 0, 1])
        want = np.asarray(d, float) / np.linalg.norm(d)
        check(f'lookat +Z aims along {d}', np.allclose(axis, want, atol=1e-9))
    down0 = lookat_rotation([0, 0, -1], 0.0)
    check('straight-down roll 0 == Rx(180°) (eye_in_hand reference)',
          np.allclose(down0.as_matrix(),
                      Rotation.from_euler('x', 180, degrees=True).as_matrix(),
                      atol=1e-9))

    d = standoff_direction([0.3, 0.0, 0.05], [0.3, 0.0, 0.05])
    check('standoff zero-offset → straight up', np.allclose(d, [0, 0, 1]))
    d = standoff_direction([0.5, 0.0, 0.10], [0.3, 0.0, 0.10])
    check('standoff enforces min elevation',
          d[2] >= MIN_UP_COMPONENT - 1e-9 and abs(np.linalg.norm(d) - 1) < 1e-9,
          f'd={d}')

    s = clamp_step([0, 0, 0], [1, 0, 0], 0.05)
    check('step clamped to MAX_STEP', abs(np.linalg.norm(s) - 0.05) < 1e-9)
    s = clamp_step([0, 0, 0], [0.01, 0, 0], 0.05)
    check('short step passes through', np.allclose(s, [0.01, 0, 0]))

    p = clamp_workspace([2.0, 0.0, 1.5])
    check('workspace: radius + z clamped',
          np.linalg.norm(p) <= WORKSPACE_MAX_R + 1e-9 and p[2] <= WORKSPACE_MAX_Z)
    p = clamp_workspace([0.0, 0.0, 0.30])
    check('workspace: base keep-out',
          np.hypot(p[0], p[1]) >= WORKSPACE_MIN_R_XY - 1e-9, f'p={p}')

    print('2) STL parsing (generated binary cube face)')
    import tempfile
    tri = [((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0), (0, 1, 0))]
    buf = b'\0' * 80 + struct.pack('<I', len(tri))
    for t in tri:
        buf += struct.pack('<3f', 0, 0, 1)
        for v in t:
            buf += struct.pack('<3f', *v)
        buf += struct.pack('<H', 0)
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        f.write(buf)
        stl_path = Path(f.name)
    verts, faces = load_stl(stl_path, scale=0.001)
    check('binary STL: 4 unique verts, 2 faces',
          len(verts) == 4 and len(faces) == 2, f'{len(verts)}v {len(faces)}f')
    check('mesh scale applied', np.isclose(verts.max(), 0.001))
    co = obstacle_to_collision_object(
        {'id': 'm1', 'type': 'mesh', 'file': str(stl_path),
         'position': [0.3, 0, 0.1], 'scale': 0.001})
    check('mesh CollisionObject built',
          len(co.meshes) == 1 and len(co.meshes[0].triangles) == 2)
    stl_path.unlink()

    co = obstacle_to_collision_object(
        {'id': 'b1', 'type': 'box', 'position': [0.3, 0.1, 0.05],
         'dims': [0.1, 0.1, 0.1], 'rpy_deg': [0, 0, 45]})
    check('box CollisionObject built',
          co.primitives[0].type == SolidPrimitive.BOX
          and abs(co.primitive_poses[0].orientation.z - np.sin(np.radians(22.5))) < 1e-6)
    co = obstacle_to_collision_object(
        {'id': 'c1', 'type': 'cylinder', 'position': [0.3, -0.1, 0.15],
         'height': 0.3, 'radius': 0.04})
    check('cylinder CollisionObject built',
          list(co.primitives[0].dimensions) == [0.3, 0.04])

    print('3) follow cycle (stubbed IK/plan/execute)')
    rclpy.init()
    node = MarkerFollowerNode()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    marker = np.array([0.35, 0.05, 0.05])

    def set_camera(cam_pos, look_dir=None):
        """Stub TF so the camera sits at cam_pos looking at the marker (or
        along look_dir)."""
        R_cam = lookat_rotation((marker - np.asarray(cam_pos, float))
                                if look_dir is None else look_dir)
        R_l6 = R_cam * R_LINK6_CAM.inv()
        t_l6 = np.asarray(cam_pos, float) - R_l6.apply(CAM_T)
        node.link6_in_base = lambda: (t_l6, R_l6)

    def put_marker(mid, xyz):
        """Inject 3× so the median window fully reflects the new position."""
        for _ in range(MEDIAN_WINDOW):
            node.tracker.inject(mid, xyz)

    exec_calls: list[str] = []
    node.execute_trajectory = lambda traj, label: exec_calls.append(label) or True
    node.plan_cartesian_partial = lambda pos, quat: (object(), 1.0)
    # a joint solution inside every JOINT_LIMITS range (note joint3 ≤ 0)
    _sol = {'joint1': 0.1, 'joint2': 0.5, 'joint3': -0.5,
            'joint4': 0.1, 'joint5': 0.1, 'joint6': 0.1}
    ik_ok = lambda pos, quat, timeout_s=0.5: dict(_sol)
    node.call_ik = ik_ok

    # no marker yet
    set_camera([0.15, -0.20, 0.45])
    st = node.step_once()
    check('no marker → holding', 'no marker' in st, st)
    check('no motion without marker', not exec_calls)

    # fresh marker, camera far away, EXECUTE=True → one step executed
    EXECUTE = True
    put_marker(7, marker)
    st = node.step_once()
    check('locked onto first marker', node._locked_id == 7)
    check('far marker → following step executed',
          st == 'following' and exec_calls == ['FOLLOW'], st)

    # second marker appears → lock keeps original id
    node.tracker.inject(9, marker + [0.1, 0, 0])
    put_marker(7, marker)
    node.step_once()
    check('lock survives a second marker', node._locked_id == 7)

    # camera exactly at the ideal standoff, facing the marker → deadband hold
    standoff = marker + np.array([0.0, 0.0, FOLLOW_DIST_M])
    set_camera(standoff)
    put_marker(7, marker)
    exec_calls.clear()
    st = node.step_once()
    check('on target → deadband hold', st == 'on target — tracking', st)
    check('deadband → no motion', not exec_calls)

    # at standoff but facing the wrong way → rotation-in-place only
    set_camera(standoff, look_dir=[1, 0, 0])
    put_marker(7, marker)
    exec_calls.clear()
    st = node.step_once()
    check('wrong facing at standoff → rotate-in-place executed',
          st == 'following' and exec_calls == ['FOLLOW'], st)

    # marker moves while a segment is executing → cancelled mid-flight
    set_camera([0.15, -0.20, 0.45])
    put_marker(7, marker)
    exec_release = threading.Event()

    def slow_exec(traj, label):
        # simulates a long-running segment; returns False when cancelled
        return not exec_release.wait(timeout=3.0)

    node.execute_trajectory = slow_exec
    node.cancel_active_execution = lambda: exec_release.set()
    threading.Timer(0.15, lambda: put_marker(7, marker + [0.10, 0, 0])).start()
    st = node.step_once()
    check('marker moved mid-segment → preempted for replan',
          'preempted' in st and exec_release.is_set(), st)
    node.execute_trajectory = (
        lambda traj, label: exec_calls.append(label) or True)
    node.cancel_active_execution = lambda: None
    put_marker(7, marker)          # restore the scene for the next checks

    # everything unreachable → hold + retry gate until the marker moves
    node.call_ik = lambda pos, quat, timeout_s=0.5: None
    put_marker(7, marker + [0.25, 0.0, 0.0])
    exec_calls.clear()
    st = node.step_once()
    check('all candidates unreachable → holding', 'cannot reach' in st, st)
    st2 = node.step_once()
    check('unreachable → waits for marker to move (no IK hammering)',
          'until the marker moves' in st2, st2)
    check('unreachable → no motion', not exec_calls)

    # EXECUTE=False never executes
    EXECUTE = False
    node.call_ik = ik_ok
    set_camera([0.15, -0.20, 0.45])
    put_marker(7, marker)
    exec_calls.clear()
    st = node.step_once()
    check('plan-only reports intent', st.startswith('PLAN-ONLY'), st)
    check('plan-only → zero motion', not exec_calls)

    print(f'\n{"ALL PASS" if fails == 0 else f"{fails} FAILURE(S)"}')
    # NB: no destroy_node() here — the daemon spin thread still holds the
    # node; destroying it under the spinner aborts the interpreter.
    rclpy.shutdown()
    spin.join(timeout=2.0)
    return 1 if fails else 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if '--selftest' in sys.argv:
        sys.exit(_selftest())

    rclpy.init()
    node = MarkerFollowerNode()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    try:
        if '--clear-obstacles' in sys.argv:
            node.clear_obstacles()
        else:
            node.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.shutdown()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
