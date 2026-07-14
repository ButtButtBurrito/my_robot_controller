#!/usr/bin/env python3
"""
pick_and_place_topdown — pick and place driven ONLY by the top-down phone
camera. The eye-in-hand wrist camera is not used at all: no hover/scan
stage, no calibration chain — the phone sees the whole table the whole
time. Runs against MOCK hardware (GUI MoveIt Sim + RViz) or the real arm.
Parallel experiment to pick_and_place.py (untouched); the two get merged
only after both are validated (phone = coarse global map, EIH = local
refinement — the Stage 2 endgame in THESIS_ROADMAP.md).

PIXEL→BASE MAPPING (no phone intrinsics, no arm needed to calibrate)
  Everything sits on the table plane, so pixels map to base_link XY with a
  single affine transform, fitted from ANCHOR markers at KNOWN positions:
  print markers 100 (A), 101 (B) and 102 (REF) at MARKER_SIZE_M, tape them
  flat in an L shape (NOT in a line), measure each centre from the arm's
  base with a ruler (±5 mm is fine for a first pass) and enter them in
  ANCHORS below. A and B double as the pick/place stations. The fit is
  cross-checked against the apparent marker sizes — a big mismatch means
  the phone is far from overhead or a measurement is off.

  (Affine ignores perspective: with the phone roughly overhead, residual
  error is a few mm — good enough for this test phase. scene_map.py's
  full homography bootstrap replaces this once the wrist camera is in the
  loop.)

RUN
  Phone: IP Webcam app → Start server (same WiFi), like topdown_viewer.py.
  1. GUI: MoveIt2 (Sim checked = mock, unchecked = real) + RViz.
     Real hardware additionally: arm controller running, deadman held.
  2. This file, via GUI Custom Script Runner or:
       python3 pick_and_place_topdown.py <phone ip or url>
     The GUI runner passes no arguments and has no terminal: the URL then
     comes from PHONE_URL below, or ~/.topdown_url (saved by the last
     topdown run/viewer). The phone's IP CHANGES between WiFi sessions —
     a stale saved IP is the classic "script hangs then FAILED at SCAN"
     cause; the pipeline now probes the phone, says exactly what is wrong,
     and keeps retrying instead of dying silently.
  EXECUTE=False (default) scans, maps, plans and draws only — check RViz
  topic /topdown_pp/viz. Set EXECUTE=True only after a clean plan-only run.

  A live preview window (like topdown_viewer.py) opens automatically while
  the phone pipeline runs: every detected marker is drawn with its pixel
  position and — once the anchor map is fitted — its estimated base_link
  coordinates and size ratio (should be ~1.00), plus tilted crosses where
  each ANCHORS entry SHOULD appear. Crosses not on the detected anchors, or
  ratios far from 1 → the mapping is wrong; fix before trusting positions.
  Window keys:  s save snapshot to ~/topdown_snaps/   q close the preview
  (the pipeline keeps running without it).

  --no-phone     skip vision; INJECT_* constants below act as the detections
                 (pure motion test in RViz, no phone needed)
  --no-preview   phone pipeline without the preview window (headless)
  --selftest     no ROS, no phone: validates the mapping math and exits

TUNE: constants marked (sync) must match pick_and_place.py.
"""

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AttachedCollisionObject, CollisionObject, PlanningScene,
)
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray

from my_robot_controller.eih_core import (
    EFFECTOR_LINK, EihBaseNode, GRIPPER_MAX_WIDTH_M, lookat_candidates,
    make_pose, make_tcp_tool,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

EXECUTE = True    # ← False = detect/map/plan/visualise ONLY (no motion).

# Phone stream for GUI-runner launches (no argv there). Bare IP is fine
# ('192.168.3.231' → http://…:8080/video). Empty = fall back to the URL
# remembered in ~/.topdown_url, then to a terminal prompt (terminal runs
# only — the GUI runner has no stdin).
PHONE_URL = ''

# Anchor markers: id → measured centre (x, y) in base_link, metres. ≥3, not
# collinear. 100/101 double as pick/place stations A/B.
ANCHORS = {
    100: (0.36, 0.136),  # station A (pick side)
    101: (0.36, -0.07),   # station B (place target)
    102: (0.46,  0.00),   # REF — only for the mapping fit
}
MARKER_A_ID = 100
MARKER_B_ID = 101

MARKER_C_ID   = 582                          # (sync) on top of the object.
                                             # ← EDIT THIS to pick a different
                                             # object marker: any ARUCO_ORIGINAL
                                             # id printed at MARKER_SIZE_M works;
                                             # nothing else references the id.
MARKER_SIZE_M = 0.047                        # (sync) ALL printed markers
OBJECT_DIMS_M = (0.03, 0.03, 0.03)     # (sync) cube x, y, z
TCP_OFFSET_Z  = 0.175                        # (sync) fingertip point, see
                                             # pick_and_place.py line ~81
TABLE_Z = 0.0          # table surface height in base_link (arm sits on it)

# NOTE 2026-07-04: 0.04 — mock-MoveIt IK probing showed straight-down TCP
# poses are unreachable above z≈0.065 m (joint5 hits its ±1.562 rad limit),
# so 0.10 clearance fails EVERYWHERE on this arm. pick_and_place.py and
# skill_server carry the same fix; keep all three in sync.
PRE_GRASP_CLEAR_M = 0.04    # TCP above object centre before descent
LIFT_CLEAR_M      = 0.04
PLACE_DROP_M      = 0.004   # (sync)
RETREAT_CLEAR_M   = 0.04
GRIP_SQUEEZE_M    = 0.003   # (sync)
GRIPPER_SETTLE_S  = 2.0     # (sync)

MARKER_FRESH_S       = 2.0
MARKER_MEDIAN_WINDOW = 5
WAIT_MARKERS_TIMEOUT_S = 60.0
SIZE_CHECK_TOL = 0.15       # warn if fitted scale vs marker size is off >15%

MAX_VELOCITY_SCALE = 0.10
MAX_ACCEL_SCALE    = 0.10
CONFIRM_DELAY_S    = 4.0

USE_TABLE_COLLISION = True
OBJECT_ID = 'pick_object'
TABLE_ID  = 'table_plane'

# --no-phone injected scene (motion-only testing, e.g. mock hardware at home)
INJECT_C_XY      = (0.30, -0.09)
INJECT_C_YAW_DEG = 20.0

DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
PARAMS = cv2.aruco.DetectorParameters_create()
URL_MEMORY = Path.home() / '.topdown_url'    # shared with topdown_viewer.py
PREVIEW_WINDOW = 'topdown_pp'
SNAP_DIR = Path.home() / 'topdown_snaps'     # shared with topdown_viewer.py


# ═══════════════════════════════════════════════════════════════════════════════
#  Pixel→base mapping (pure math — covered by --selftest)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_pixel_to_base(px_pts: np.ndarray, base_pts: np.ndarray):
    """Affine 2×3 A st. base_xy ≈ A·[u, v, 1]. None if degenerate."""
    A, _ = cv2.estimateAffine2D(
        np.asarray(px_pts, np.float32).reshape(-1, 1, 2),
        np.asarray(base_pts, np.float32).reshape(-1, 1, 2))
    return A


def px_to_base(A: np.ndarray, uv) -> np.ndarray:
    u, v = float(uv[0]), float(uv[1])
    return A @ np.array([u, v, 1.0])


def yaw_from_px_axis(A: np.ndarray, d_px) -> float:
    """Marker x-axis direction in pixels → yaw about base Z, degrees.
    (Grasp rolls repeat every 90°, so the corner-order convention only
    matters modulo 90° — any consistent edge works.)"""
    v = A[:, :2] @ np.asarray(d_px, float)
    return float(np.degrees(np.arctan2(v[1], v[0])))


def metres_per_pixel(A: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(A[:, :2]))))


def normalize_url(s: str) -> str:
    """Same convenience as topdown_viewer.py: bare IP → IP Webcam MJPEG URL."""
    s = s.strip().rstrip('/')
    if not s:
        return s
    if not s.startswith('http'):
        s = 'http://' + s
    if ':' not in s.split('//', 1)[1]:
        s += ':8080'
    if not s.endswith('/video'):
        s += '/video'
    return s


# ═══════════════════════════════════════════════════════════════════════════════
#  Phone detection thread
# ═══════════════════════════════════════════════════════════════════════════════

def _preview_colour(mid: int) -> tuple:
    """BGR, matched to the RViz anchor colours."""
    if mid == MARKER_A_ID:
        return (255, 120, 40)      # blue — station A
    if mid == MARKER_B_ID:
        return (0, 200, 255)       # yellow — station B
    if mid in ANCHORS:
        return (200, 200, 200)     # grey — REF
    if mid == MARKER_C_ID:
        return (80, 220, 80)       # green — object
    return (0, 0, 255)             # red — id the pipeline does not expect


class PhoneDetections(threading.Thread):
    """Grabs the phone stream, detects ArUco every frame, keeps a rolling
    median window per id: {id: (centre_px, xaxis_px, size_px, monotonic)}.
    With preview=True also shows the annotated frames in a window (all cv2
    GUI calls happen on this thread only)."""

    def __init__(self, url: str, log, preview: bool = True):
        super().__init__(daemon=True)
        self._url = url
        self._log = log
        self._lock = threading.Lock()
        self._centres: dict[int, list[np.ndarray]] = {}
        self._latest: dict[int, tuple] = {}
        self.frames = 0
        self.connected = False
        self._stop = threading.Event()
        self._preview = preview
        self._win_open = False
        self.affine = None             # set by the node once the map is fitted
        self.status_text = 'starting'  # mirrored from the node's stage

    def _probe(self) -> str | None:
        """2 s TCP probe of the stream host. None = reachable, else a reason
        string. Prevents cv2.VideoCapture's silent ~30 s hang on a stale IP
        (the phone's IP changes between WiFi sessions!)."""
        import socket
        from urllib.parse import urlparse
        u = urlparse(self._url)
        host, port = u.hostname, u.port or 80
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return None
        except OSError as e:
            return (f'{host}:{port} unreachable ({e}) — is IP Webcam '
                    '"Start server" running, phone screen on, same WiFi, '
                    'and is the IP still current? (it changes; check the '
                    'IP Webcam screen and update PHONE_URL / rerun with '
                    'the new address)')

    def _connect(self) -> cv2.VideoCapture | None:
        """Probe + open, retrying every 3 s until it works or stop is set.
        Never lets the pipeline die on a stale IP — it says WHY it waits."""
        last_log = 0.0
        while not self._stop.is_set():
            reason = self._probe()
            if reason is None:
                cap = cv2.VideoCapture(self._url)
                if cap.isOpened():
                    self.connected = True
                    self._log(f'[PHONE] Streaming from {self._url}')
                    return cap
                cap.release()
                reason = (f'{self._url} reachable but not a video stream — '
                          'wrong port/path? (IP Webcam default is :8080/video)')
            self.connected = False
            if time.monotonic() - last_log > 6.0:
                self._log(f'[PHONE] {reason} Retrying...', 'WARN')
                last_log = time.monotonic()
            self._stop.wait(3.0)
        return None

    def run(self):
        cap = self._connect()
        if cap is None:
            return
        drop_streak = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                drop_streak += 1
                if drop_streak >= 10:          # ~stream gone → full reconnect
                    self._log('[PHONE] Stream lost — reconnecting...', 'WARN')
                    cap.release()
                    cap = self._connect()
                    if cap is None:
                        return
                    drop_streak = 0
                    continue
                self._log('[PHONE] Stream dropped a frame; retrying...', 'WARN')
                time.sleep(0.5)
                continue
            drop_streak = 0
            self.frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, DICT,
                                                      parameters=PARAMS)
            if ids is not None:
                now = time.monotonic()
                with self._lock:
                    for quad, mid in zip(corners, ids.flatten()):
                        q = quad.reshape(4, 2)
                        centre = q.mean(axis=0)
                        xaxis = q[1] - q[0]                  # top edge
                        size_px = float(np.mean(
                            [np.linalg.norm(q[i] - q[i - 1])
                             for i in range(4)]))
                        w = self._centres.setdefault(int(mid), [])
                        w.append(centre)
                        if len(w) > MARKER_MEDIAN_WINDOW:
                            w.pop(0)
                        self._latest[int(mid)] = (
                            np.median(np.stack(w), axis=0), xaxis, size_px, now)
            if self._preview:
                self._show(frame, corners, ids)
        cap.release()
        if self._win_open:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except cv2.error:
                pass

    # ── Live preview (same look as topdown_viewer.py + base-frame overlay) ────

    def _show(self, frame, corners, ids):
        try:
            if not self._win_open:
                cv2.namedWindow(PREVIEW_WINDOW,
                                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(PREVIEW_WINDOW, 960, 540)
                self._win_open = True
            A = self.affine
            n = 0
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for quad, mid in zip(corners, ids.flatten()):
                    n += 1
                    q = quad.reshape(4, 2)
                    centre = q.mean(axis=0)
                    u, v = int(centre[0]), int(centre[1])
                    col = _preview_colour(int(mid))
                    if A is None:
                        label = f'{mid}  px({u},{v})'
                    else:
                        x, y = px_to_base(A, centre)
                        size_px = float(np.mean(
                            [np.linalg.norm(q[i] - q[i - 1])
                             for i in range(4)]))
                        ratio = size_px * metres_per_pixel(A) / MARKER_SIZE_M
                        label = (f'{mid}  base({x:+.3f},{y:+.3f})m'
                                 f'  size x{ratio:.2f}')
                        if int(mid) == MARKER_C_ID:
                            yaw = yaw_from_px_axis(A, q[1] - q[0])
                            label += f'  yaw {yaw:+.0f}'
                    cv2.putText(frame, label, (u + 12, v - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            if A is not None:
                # Where each ANCHORS measurement lands in the image under the
                # fitted map — these crosses must sit ON the printed anchors.
                inv = cv2.invertAffineTransform(A)
                for mid, xy in ANCHORS.items():
                    uv = inv @ np.array([xy[0], xy[1], 1.0])
                    if not (np.all(np.isfinite(uv)) and np.all(np.abs(uv) < 1e6)):
                        continue           # degenerate map — nothing to draw
                    pu, pv = int(uv[0]), int(uv[1])
                    col = _preview_colour(mid)
                    cv2.drawMarker(frame, (pu, pv), col,
                                   cv2.MARKER_TILTED_CROSS, 28, 2)
                    cv2.putText(frame, f'{mid} expected', (pu + 12, pv + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            head = (f'{self.status_text}   {n} marker(s)   '
                    + (f'map {metres_per_pixel(A) * 1000:.2f} mm/px'
                       if A is not None else 'map not fitted yet'))
            cv2.putText(frame, head, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(PREVIEW_WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                SNAP_DIR.mkdir(exist_ok=True)
                p = SNAP_DIR / f'pp_snap_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
                cv2.imwrite(str(p), frame)
                self._log(f'[PREVIEW] saved {p}')
            elif key == ord('q'):
                self._preview = False
                self._win_open = False
                cv2.destroyWindow(PREVIEW_WINDOW)
                self._log('[PREVIEW] closed (pipeline continues).')
        except cv2.error:
            self._preview = False
            self._log('[PREVIEW] Disabled — cv2 GUI unavailable (headless? '
                      'use --no-preview to silence this).', 'WARN')

    def get(self, mid: int):
        """(centre_px, xaxis_px, size_px) if fresh, else None."""
        with self._lock:
            entry = self._latest.get(mid)
        if entry is None or time.monotonic() - entry[3] > MARKER_FRESH_S:
            return None
        return entry[:3]

    def fresh_ids(self) -> list[int]:
        with self._lock:
            items = list(self._latest.items())
        now = time.monotonic()
        return [i for i, e in items if now - e[3] < MARKER_FRESH_S]

    def stop(self):
        self._stop.set()


class InjectedDetections:
    """--no-phone stand-in: identity mapping, constants as detections."""

    connected = True
    frames = -1
    affine = None          # written by the node, unused here (no preview)
    status_text = ''

    def get(self, mid: int):
        if mid in ANCHORS:
            return np.array(ANCHORS[mid]), np.array([1.0, 0.0]), MARKER_SIZE_M
        if mid == MARKER_C_ID:
            yaw = np.radians(INJECT_C_YAW_DEG)
            return (np.array(INJECT_C_XY),
                    np.array([np.cos(yaw), np.sin(yaw)]), MARKER_SIZE_M)
        return None

    def fresh_ids(self):
        return sorted(ANCHORS) + [MARKER_C_ID]

    def stop(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Node
# ═══════════════════════════════════════════════════════════════════════════════

class TopdownPickPlace(EihBaseNode):

    def __init__(self, detections):
        super().__init__('pick_and_place_topdown',
                         max_velocity_scale=MAX_VELOCITY_SCALE,
                         max_accel_scale=MAX_ACCEL_SCALE)
        self.det = detections
        self.tcp = make_tcp_tool(TCP_OFFSET_Z)
        self.A = None                      # pixel→base affine (2×3)

        self._viz_pub = self.create_publisher(MarkerArray, '/topdown_pp/viz', 10)
        self._planned_viz: list[Marker] = []
        self._status_text = 'starting'
        self.create_timer(0.5, self._publish_viz)
        self._scene_cli = self.create_client(ApplyPlanningScene,
                                             '/apply_planning_scene')

        mode = 'EXECUTE' if EXECUTE else 'PLAN-ONLY (no motion; set EXECUTE=True when happy)'
        self._log(f'Mode: {mode}   (top-down phone pipeline — EIH not used)')
        self._log(f'Anchors: { {i: xy for i, xy in ANCHORS.items()} }  '
                  f'object marker C={MARKER_C_ID}')

    # ── Vision → base frame ───────────────────────────────────────────────────

    def marker_in_base(self, mid: int):
        """(xy_base, yaw_deg) of a fresh detection, or None."""
        entry = self.det.get(mid)
        if entry is None or self.A is None:
            return None
        centre, xaxis, _ = entry
        return px_to_base(self.A, centre), yaw_from_px_axis(self.A, xaxis)

    def wait_for_ids(self, ids: list[int], timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        last_note = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            missing = [i for i in ids if self.det.get(i) is None]
            if not missing:
                return True
            if time.monotonic() - last_note > 2.0:
                extra = ('' if self.det.frames
                         else ' (no frames from the phone yet!)')
                self._log(f'[SCAN] Waiting for marker(s) {missing}{extra}...')
                last_note = time.monotonic()
            time.sleep(0.3)
        return False

    def fit_mapping(self) -> bool:
        px, base = [], []
        for mid, xy in ANCHORS.items():
            entry = self.det.get(mid)
            if entry is None:
                self._log(f'[MAP] Anchor {mid} lost during fit.', 'ERROR')
                return False
            px.append(entry[0])
            base.append(xy)
        self.A = fit_pixel_to_base(np.array(px), np.array(base))
        if self.A is None:
            self._log('[MAP] Affine fit failed — anchors collinear or '
                      'detections degenerate. Move anchor 102 off the A–B '
                      'line.', 'ERROR')
            return False
        self.det.affine = self.A       # preview switches to base-frame overlay
        scale = metres_per_pixel(self.A)
        self._log(f'[MAP] Fitted pixel→base affine, scale {scale*1000:.2f} mm/px.')
        ok = True
        for mid in ANCHORS:
            size_px = self.det.get(mid)[2]
            est = size_px * scale
            ratio = est / MARKER_SIZE_M
            lvl = 'INFO' if abs(ratio - 1.0) < SIZE_CHECK_TOL else 'WARN'
            if lvl == 'WARN':
                ok = False
            self._log(f'[MAP] Anchor {mid}: apparent size {est*1000:.1f} mm '
                      f'vs printed {MARKER_SIZE_M*1000:.0f} mm '
                      f'(ratio {ratio:.2f})', lvl)
        if not ok:
            self._log('[MAP] Size check failed — phone not overhead enough, '
                      'ANCHORS measurements off, or wrong print size. '
                      'Positions may be inaccurate; fix before EXECUTE=True.',
                      'WARN')
        return True

    # ── Visualisation (mirrors pick_and_place.py) ─────────────────────────────

    def _publish_viz(self):
        ms: list[Marker] = []
        for i, (mid, xy) in enumerate(sorted(ANCHORS.items())):
            rgba = ((0.2, 0.4, 1.0, 0.9) if mid == MARKER_A_ID else
                    (1.0, 0.8, 0.0, 0.9) if mid == MARKER_B_ID else
                    (0.6, 0.6, 0.6, 0.9))
            p = np.array([xy[0], xy[1], TABLE_Z])
            ms.append(self.viz_sphere('anchors', i * 2, p, rgba, diam=0.03))
            ms.append(self.viz_text('anchors', i * 2 + 1, p + [0, 0, 0.05],
                                    f'{mid}', rgba))
        c = self.marker_in_base(MARKER_C_ID)
        if c is not None:
            (x, y), yaw = c
            centre = np.array([x, y, TABLE_Z + OBJECT_DIMS_M[2] / 2])
            ms.append(self.viz_cube('object', 0, centre, OBJECT_DIMS_M,
                                    (0.1, 0.9, 0.2, 0.8)))
            ms.append(self.viz_text('object', 1, centre + [0, 0, 0.06],
                                    f'C  object (yaw {yaw:+.0f}°)',
                                    (0.1, 0.9, 0.2, 0.9)))
        ms.append(self.viz_text('status', 0, [0.0, 0.0, 0.60],
                                f'topdown_pp: {self._status_text}'))
        ms.extend(self._planned_viz)
        self._viz_pub.publish(MarkerArray(markers=ms))

    def add_planned_pose_viz(self, tool_pos, R_world_tool, label, rgba):
        base = len(self._planned_viz)
        axis_end = np.asarray(tool_pos) + R_world_tool.apply([0, 0, 0.05])
        self._planned_viz.append(self.viz_arrow_between(
            'planned', base, tool_pos, axis_end, rgba))
        self._planned_viz.append(self.viz_cube(
            'planned', base + 1, tool_pos, (0.01, 0.01, 0.01), rgba))
        self._planned_viz.append(self.viz_text(
            'planned', base + 2, np.asarray(tool_pos) + [0, 0, 0.03],
            label, rgba, height=0.02))

    def status(self, text: str):
        self._status_text = text
        self.det.status_text = text
        self._log(f'[STAGE] {text}')

    # ── Planning scene (same shapes as pick_and_place.py) ─────────────────────

    def _apply_scene(self, scene: PlanningScene) -> bool:
        scene.is_diff = True
        if not self._scene_cli.wait_for_service(timeout_sec=3.0):
            self._log('[SCENE] /apply_planning_scene unavailable — continuing '
                      'without collision objects.', 'WARN')
            return False
        future = self._scene_cli.call_async(
            ApplyPlanningScene.Request(scene=scene))
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=5.0):
            self._log('[SCENE] apply_planning_scene timed out.', 'WARN')
            return False
        return future.result().success

    @staticmethod
    def _box_object(oid: str, frame: str, pose: Pose, dims) -> CollisionObject:
        co = CollisionObject()
        co.id = oid
        co.header.frame_id = frame
        prim = SolidPrimitive(type=SolidPrimitive.BOX,
                              dimensions=[float(d) for d in dims])
        co.primitives = [prim]
        co.primitive_poses = [pose]
        co.operation = CollisionObject.ADD
        return co

    def scene_add_table(self, table_z: float, centre_xy: np.ndarray):
        pose = make_pose([centre_xy[0], centre_xy[1], table_z - 0.011],
                         [0, 0, 0, 1])
        scene = PlanningScene()
        scene.world.collision_objects.append(
            self._box_object(TABLE_ID, 'base_link', pose, (0.8, 0.8, 0.02)))
        if self._apply_scene(scene):
            self._log(f'[SCENE] Table plane added at z={table_z:.3f}.')

    def scene_attach_object(self):
        aco = AttachedCollisionObject()
        aco.link_name = EFFECTOR_LINK
        aco.touch_links = [EFFECTOR_LINK, 'gripper_base',
                           'gripper_link1', 'gripper_link2']
        aco.object = self._box_object(
            OBJECT_ID, EFFECTOR_LINK,
            make_pose([0, 0, TCP_OFFSET_Z], [0, 0, 0, 1]), OBJECT_DIMS_M)
        scene = PlanningScene()
        scene.robot_state.attached_collision_objects.append(aco)
        scene.robot_state.is_diff = True
        if self._apply_scene(scene):
            self._log('[SCENE] Object attached to gripper.')

    def scene_detach_object(self, place_xyz_centre: np.ndarray):
        aco = AttachedCollisionObject()
        aco.link_name = EFFECTOR_LINK
        aco.object.id = OBJECT_ID
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.robot_state.attached_collision_objects.append(aco)
        scene.robot_state.is_diff = True
        scene.world.collision_objects.append(
            self._box_object(OBJECT_ID, 'base_link',
                             make_pose(place_xyz_centre, [0, 0, 0, 1]),
                             OBJECT_DIMS_M))
        if self._apply_scene(scene):
            self._log('[SCENE] Object detached and placed in world.')

    # ── Grasp geometry (same as pick_and_place.py) ────────────────────────────

    def grasp_rolls_for_yaw(self, yaw_deg: float) -> tuple:
        base = (-90.0 - yaw_deg) % 90.0
        return tuple((base + k * 90.0) % 360.0 for k in range(4))

    def find_grasp(self, target_xyz: np.ndarray, dist: float,
                   yaw_deg: float, label: str):
        rolls = self.grasp_rolls_for_yaw(yaw_deg)
        self._log(f'[{label}] Grasp rolls for object yaw {yaw_deg:+.0f}°: '
                  f'{[f"{r:.0f}" for r in rolls]}')
        n = 0
        for tilt, az, roll, R_world_tool in lookat_candidates(
                tilts=(0,), rolls=rolls):
            axis = R_world_tool.apply([0.0, 0.0, 1.0])
            tool_pos = np.asarray(target_xyz) - axis * dist
            link6_pos, link6_quat = self.tcp.to_link6(tool_pos, R_world_tool)
            n += 1
            sol = self.call_ik(link6_pos, link6_quat)
            if sol is not None:
                self._log(f'[{label}] Reachable grasp candidate #{n}: '
                          f'roll {roll:.0f}°')
                return (tool_pos, R_world_tool, link6_pos, link6_quat, sol,
                        f'roll {roll:.0f}°')
        self._log(f'[{label}] No straight-down grasp reachable ({n} tried).',
                  'ERROR')
        return None

    # ── Main sequence ─────────────────────────────────────────────────────────

    def run(self) -> bool:
        # ── MAP: anchors → pixel→base affine ──
        self.status('MAP: waiting for anchor markers')
        if not self.wait_for_ids(list(ANCHORS), WAIT_MARKERS_TIMEOUT_S):
            self._log(f'Anchors {sorted(ANCHORS)} never all visible — are '
                      'they printed, flat and inside the phone frame?', 'ERROR')
            return False
        if not self.fit_mapping():
            return False

        # ── LOCATE the object ──
        self.status('LOCATE: finding object marker C')
        if not self.wait_for_ids([MARKER_C_ID], 15.0):
            self._log('Marker C (object) not visible to the phone.', 'ERROR')
            return False
        (cx, cy), yaw = self.marker_in_base(MARKER_C_ID)
        obj_centre = np.array([cx, cy, TABLE_Z + OBJECT_DIMS_M[2] / 2])
        b_xy = np.array(ANCHORS[MARKER_B_ID])
        self._log(f'[LOCATE] object at ({cx:+.3f}, {cy:+.3f}) yaw {yaw:+.0f}°, '
                  f'place target B at ({b_xy[0]:+.3f}, {b_xy[1]:+.3f})')

        if USE_TABLE_COLLISION:
            a_xy = np.array(ANCHORS[MARKER_A_ID])
            self.scene_add_table(TABLE_Z, (a_xy + b_xy) / 2)

        # ── PLAN: all poses validated up front (same shape as pick_and_place) ──
        self.status('PLAN: computing grasp and place poses')
        pre_grasp = self.find_grasp(
            obj_centre, OBJECT_DIMS_M[2] / 2 + PRE_GRASP_CLEAR_M, yaw,
            'PRE_GRASP')
        if pre_grasp is None:
            return False
        pg_pos, pg_R, pg_l6p, pg_l6q, pg_sol, pg_desc = pre_grasp

        g_l6p, g_l6q = self.tcp.to_link6(obj_centre, pg_R)
        g_sol = self.call_ik(g_l6p, g_l6q, timeout_s=1.0)
        if g_sol is None:
            self._log('[GRASP] Grasp depth pose not reachable — lower '
                      'TCP_OFFSET_Z or raise the grasp point.', 'ERROR')
            return False

        place_centre = np.array([b_xy[0], b_xy[1],
                                 TABLE_Z + OBJECT_DIMS_M[2] / 2 + PLACE_DROP_M])
        pre_place = self.find_grasp(
            place_centre, OBJECT_DIMS_M[2] / 2 + PRE_GRASP_CLEAR_M, yaw,
            'PRE_PLACE')
        if pre_place is None:
            return False
        pp_pos, pp_R, pp_l6p, pp_l6q, pp_sol, pp_desc = pre_place
        pl_l6p, pl_l6q = self.tcp.to_link6(place_centre, pp_R)
        pl_sol = self.call_ik(pl_l6p, pl_l6q, timeout_s=1.0)
        if pl_sol is None:
            self._log('[PLACE] Place depth pose not reachable.', 'ERROR')
            return False

        self.add_planned_pose_viz(pg_pos, pg_R, f'pre-grasp ({pg_desc})',
                                  (0.1, 0.9, 0.2, 0.9))
        self.add_planned_pose_viz(obj_centre, pg_R, 'grasp', (1.0, 0.2, 0.2, 0.9))
        self.add_planned_pose_viz(pp_pos, pp_R, f'pre-place ({pp_desc})',
                                  (1.0, 0.8, 0.0, 0.9))
        self.add_planned_pose_viz(place_centre, pp_R, 'place',
                                  (1.0, 0.5, 0.0, 0.9))
        self._log('[PLAN] All four poses IK-validated. Grasp width target: '
                  f'{OBJECT_DIMS_M[0] - GRIP_SQUEEZE_M:.3f} m.')

        if not EXECUTE:
            traj = self.plan_to_link6_pose(pg_l6p, pg_l6q, pg_sol,
                                           'PRE_GRASP(plan-only)')
            self.status('PLAN-ONLY complete — check RViz, then set EXECUTE=True')
            self._log('[DONE] Plan-only run finished '
                      f'({"pre-grasp trajectory OK" if traj else "pre-grasp plan FAILED — investigate before executing"}). '
                      'No motion was commanded.')
            return traj is not None

        # ── PICK ──
        self.enable_control_gate()
        self._log(f'[PREVIEW] Moving in {CONFIRM_DELAY_S:.0f}s — check RViz now.')
        time.sleep(CONFIRM_DELAY_S)

        self.status('PICK: opening gripper, moving to pre-grasp')
        self.set_gripper_width(GRIPPER_MAX_WIDTH_M, GRIPPER_SETTLE_S)
        if not self.move_to(pg_l6p, pg_l6q, pg_sol, 'PRE_GRASP'):
            return False

        self.status('PICK: descending to grasp')
        if not self.move_to(g_l6p, g_l6q, g_sol, 'GRASP_DESCEND',
                            avoid_collisions=False):
            return False

        self.status('PICK: closing gripper')
        self.set_gripper_width(max(OBJECT_DIMS_M[0] - GRIP_SQUEEZE_M, 0.0),
                               GRIPPER_SETTLE_S)
        self.scene_attach_object()

        self.status('PICK: lifting')
        lift_l6p = g_l6p + np.array([0, 0, LIFT_CLEAR_M])
        lift_sol = self.call_ik(lift_l6p, g_l6q, timeout_s=1.0) or g_sol
        if not self.move_to(lift_l6p, g_l6q, lift_sol, 'LIFT',
                            avoid_collisions=False):
            return False

        # ── PLACE ──
        self.status('PLACE: moving above place station')
        if not self.move_to(pp_l6p, pp_l6q, pp_sol, 'PRE_PLACE'):
            return False

        self.status('PLACE: descending')
        if not self.move_to(pl_l6p, pl_l6q, pl_sol, 'PLACE_DESCEND',
                            avoid_collisions=False):
            return False

        self.status('PLACE: releasing')
        self.set_gripper_width(GRIPPER_MAX_WIDTH_M, GRIPPER_SETTLE_S)
        self.scene_detach_object(place_centre)

        self.status('PLACE: retreating')
        ret_l6p = pl_l6p + np.array([0, 0, RETREAT_CLEAR_M])
        ret_sol = self.call_ik(ret_l6p, pl_l6q, timeout_s=1.0) or pl_sol
        self.move_to(ret_l6p, pl_l6q, ret_sol, 'RETREAT', avoid_collisions=False)

        # ── VERIFY (the phone still sees the whole table — the top-down win) ──
        self.status('VERIFY: checking object at place station')
        if isinstance(self.det, InjectedDetections):
            self._log('[VERIFY] Skipped (--no-phone injects a static scene).')
        else:
            time.sleep(2.0)
            c2 = self.marker_in_base(MARKER_C_ID)
            if c2 is None:
                self._log('[VERIFY] Marker C not visible — arm may be '
                          'occluding it; check the phone view.', 'WARN')
            else:
                err = float(np.linalg.norm(np.asarray(c2[0]) - b_xy))
                lvl = 'INFO' if err < 0.03 else 'WARN'
                self._log(f'[VERIFY] Object is {err*100:.1f} cm from station B '
                          f'(horizontal). {"PASS" if err < 0.03 else "off target"}',
                          lvl)
        self.status('DONE')
        return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Selftest (no ROS, no phone): synthetic camera → mapping math round-trip
# ═══════════════════════════════════════════════════════════════════════════════

def selftest() -> bool:
    rng = np.random.default_rng(7)
    theta, scale_px = np.radians(25.0), 900.0     # 900 px per metre
    S = scale_px * np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]])
    S[1, :] *= -1.0                               # v axis down = reflection
    t_px = np.array([412.0, 305.0])

    def project(xy):
        return S @ np.asarray(xy, float) + t_px

    truth_c = np.array([0.30, -0.09])
    truth_yaw = 20.0

    px_anchors = [project(xy) + rng.normal(0, 0.4, 2)
                  for xy in ANCHORS.values()]
    A = fit_pixel_to_base(np.array(px_anchors),
                          np.array(list(ANCHORS.values())))
    assert A is not None, 'affine fit failed'

    c_px = project(truth_c) + rng.normal(0, 0.4, 2)
    d_base = np.array([np.cos(np.radians(truth_yaw)),
                       np.sin(np.radians(truth_yaw))])
    est_xy = px_to_base(A, c_px)
    est_yaw = yaw_from_px_axis(A, S @ d_base)
    est_size = MARKER_SIZE_M * scale_px * metres_per_pixel(A)

    pos_err_mm = float(np.linalg.norm(est_xy - truth_c)) * 1000
    yaw_err = abs((est_yaw - truth_yaw + 180) % 360 - 180)
    size_ratio = est_size / MARKER_SIZE_M
    print(f'[SELFTEST] position error {pos_err_mm:.2f} mm '
          f'(limit 2.0), yaw error {yaw_err:.2f}° (limit 1.0), '
          f'size ratio {size_ratio:.3f} (limit ±0.05)')
    ok = pos_err_mm < 2.0 and yaw_err < 1.0 and abs(size_ratio - 1) < 0.05
    print(f'[SELFTEST] {"PASS" if ok else "FAIL"} — mapping math '
          f'{"recovers" if ok else "DOES NOT recover"} the synthetic scene '
          '(rotated, scaled, reflected camera).')
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_stream_url(args: list[str]) -> tuple[str, str]:
    """(url, source) — CLI arg → PHONE_URL constant → ~/.topdown_url →
    terminal prompt. Never blocks on input() without a terminal (the GUI
    Custom Script Runner has no stdin — input() there dies with EOFError
    before anything runs). ('', reason) if nothing usable."""
    for a in args:
        if not a.startswith('-'):
            return normalize_url(a), 'command line'
    if PHONE_URL.strip():
        return normalize_url(PHONE_URL), 'PHONE_URL constant'
    if URL_MEMORY.exists():
        return (normalize_url(URL_MEMORY.read_text()),
                f'{URL_MEMORY} (LAST USED — stale if the phone rejoined WiFi)')
    if sys.stdin.isatty():
        try:
            return normalize_url(input('Phone IP or stream URL: ')), 'prompt'
        except EOFError:
            pass
    return '', ('no URL: pass one on the command line, set PHONE_URL in the '
                'config header (GUI runner has no argv/stdin), or run '
                'topdown_viewer.py once from a terminal to save one')


def main():
    args = sys.argv[1:]
    if '--selftest' in args:
        sys.exit(0 if selftest() else 1)

    if '--no-phone' in args:
        det = InjectedDetections()
    else:
        url, source = _resolve_stream_url(args)
        if not url:
            print(f'ERROR: {source}', flush=True)
            sys.exit(1)
        print(f'[PHONE] Using {url}  (from {source})', flush=True)
        URL_MEMORY.write_text(url)
        det = PhoneDetections(url, lambda m, lvl='INFO':
                              print(f'[{time.strftime("%H:%M:%S")}] {lvl}: {m}',
                                    flush=True),
                              preview='--no-preview' not in args)
        det.start()

    rclpy.init()
    node = TopdownPickPlace(det)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    ok = False
    try:
        ok = node.run()
    except KeyboardInterrupt:
        pass
    finally:
        det.stop()
    node._log(f'Result: {"SUCCESS" if ok else "FAILED"}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
