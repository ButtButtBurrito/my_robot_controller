#!/usr/bin/env python3
"""
Pick-and-place practice — AGX Piper / RealSense D435 (headless, GUI-run).

Markers (only B and C are required — A was dropped from the pipeline
2026-07-16 per the 2026-07-12 design; it may stay taped as a topdown anchor):
  B (MARKER_B_ID) — place station on the table (goal, never covered at scan)
  C (MARKER_C_ID) — on TOP of the object (dimensions in OBJECT_DIMS_M);
                    the pick target comes entirely from C, wherever it sits

Sequence:
  SCAN      measure station B live, re-measure after a settle delay
            (stability gate), z-sanity gate, straight-down place IK probe —
            all BEFORE any motion; then prompt to place the object
  SCAN_C    wait for C (up to WAIT_OBJECT_TIMEOUT_S — time to put it down)
  PICK      pre-grasp above the object → open gripper → straight descend →
            close on the object → attach collision box → lift
  PLACE     move above B → straight descend until the object sits on the
            table → open → detach → retreat
  VERIFY    check marker C is now detected near B

All three markers, the object box, and every commanded pose are published to
/pick_place/viz (MarkerArray) continuously — add it in RViz.

An rqt_image_view window opens automatically on /pick_place/camera_view: the
live camera image with detected markers drawn, plus an overlay panel showing
the current stage and each marker's id, base-frame coordinates and detection
state (A/B/C + any other ids in view).

SAFETY: EXECUTE = False by default. In that mode the script scans, computes,
IK-validates and *plans* everything it can from the current arm pose,
publishes all visualisation, and stops without moving anything (gripper
included). Read the log + RViz, then set EXECUTE = True for a real run.
Run only from agx_arm_gui's Custom Script Runner — the GUI owns deadman
safety (F5 release kills this process group and homes the arm).

Detection uses aruco_ros `marker_publisher` (all markers, any ID, one
marker_size). If it isn't running, this script starts it as a child process
(dies with the script). The GUI's single-marker tracker can stay on or off —
it is not used here.
"""

import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.time import Time
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation

from aruco_msgs.msg import MarkerArray as ArucoMarkerArray
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AttachedCollisionObject, CollisionObject, PlanningScene,
)
from moveit_msgs.srv import ApplyPlanningScene
from sensor_msgs.msg import Image
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray

from my_robot_controller.eih_core import (
    CAM_T, EFFECTOR_LINK, EihBaseNode, GRIPPER_MAX_WIDTH_M, JOINT_LIMITS,
    R_LINK6_CAM, CAMERA_TOOL, lookat_candidates, make_pose, make_tcp_tool,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

EXECUTE = True        # ← False = scan/plan/visualise ONLY (no motion at all).
                       #   Set True only after a clean EXECUTE=False run.
                       #   (Reset to False 2026-07-12 during the bug-fix pass —
                       #   the 2026-07-06 live status was never recorded, so the
                       #   next run starts from a clean plan-only baseline.)

HOME_AFTER_RUN = True  # After a SUCCESSFUL EXECUTE run, joint-move back to the
                       # pose the arm was in when the script started (the scan
                       # viewpoint — guaranteed reachable, camera back on the
                       # scene, ready for the next run). Failed runs stay put
                       # for inspection. No effect when EXECUTE=False.

# Marker IDs. marker_publisher assumes ONE size (MARKER_SIZE_M) for every
# marker it detects; a marker printed at a DIFFERENT size goes in TRUE_SIZE_M.
MARKER_A_ID   = 100    # pick station (table)
MARKER_B_ID   = 101    # place station (table)
MARKER_C_ID   = 582    # on top of the object
MARKER_SIZE_M = 0.047

# TRUE printed size per id, when it differs from MARKER_SIZE_M. ArUco
# translation scales exactly linearly with the assumed size, so one detector
# serves mixed sizes:  position ×= TRUE_SIZE_M[id] / MARKER_SIZE_M.
# e.g. object marker C reprinted at 21 mm:  TRUE_SIZE_M = {MARKER_C_ID: 0.021}
# (sync: fake_marker_publisher.py TRUE_SIZE_M, so the offline mock simulates
# the same detector mis-scale and tests this correction.)
TRUE_SIZE_M: dict[int, float] = {MARKER_C_ID: 0.021}

# Stations are detected LIVE each run (de-hardcoded 2026-07-16 per the
# 2026-07-12 design): marker A is NOT needed by this pipeline at all — the
# pick target comes entirely from C, and the table height / sanity checks /
# collision centre all come from B, which is never occluded at scan time
# (the object only arrives there at the end). The object may therefore sit
# on station A from the start; nothing has to be uncovered. A=100 can stay
# taped as a topdown anchor; if visible it is logged as a cross-check only.
#   Optional override (debug/repeatability): set STATION_B_XYZ to freeze B,
#   e.g. from a --capture-stations printout. None = live detection.
STATION_B_XYZ = None
   # e.g. (0.288, -0.069, -0.035)
# Station-B measurement validation (before anything else runs):
STATION_RESAMPLE_S     = 2.0    # settle, then re-measure B after this long
STATION_STABLE_TOL_M   = 0.010  # the two reads must agree within this

# Object under marker C. Default: 3×3×3 cm cube. Marker C sits on its top
# face, so object centre = C − (0, 0, height/2).
OBJECT_DIMS_M = (0.03, 0.03, 0.03)   # x, y, z (height)

# Gripper TCP: fingertip grasp point along link6 +Z. Finger joints mount at
# 0.1358 m (URDF); the pads sit a few cm further out. VERIFY IN RVIZ on the
# first EXECUTE=False run (a small axis cube is drawn at the planned TCP).
TCP_OFFSET_Z = 0.175

SCAN_DIST_M       = 0.20    # camera→marker distance when hovering over A
# Measured 2026-07-06: close straight-down hover views mis-measure marker
# positions by up to ~13 cm (reproducible, static), while readings from the
# oblique start/home viewpoint put all table markers at z≈0 (ground truth)
# AND share the error field of the --capture-stations readings (also taken
# from home), so pick/place stay RELATIVELY consistent. Scan C from the
# start pose; leave the hover off until the viewpoint error is understood.
HOVER_BEFORE_SCAN = False
# Clearances ≤0.04: straight-down TCP poses are unreachable above z≈0.065 m
# (joint5 hits ±1.562 rad keeping the tool vertical — IK-probed 2026-07-04,
# 0.10 failed everywhere; 0.04 passed the full mock cycle). Sync skill_server.
PRE_GRASP_CLEAR_M = 0.04    # TCP this far above object centre before descent
LIFT_CLEAR_M      = 0.04    # straight lift after grasping
PLACE_DROP_M      = 0.004   # release the object this far above the table
RETREAT_CLEAR_M   = 0.04    # straight retreat after releasing

GRIP_SQUEEZE_M    = 0.003   # close to object width minus this
GRIPPER_SETTLE_S  = 2.0     # wait after a gripper command

MARKER_FRESH_S       = 2.0  # how recent a detection must be to be used
MARKER_MEDIAN_WINDOW = 5    # median filter over this many detections
WAIT_MARKERS_TIMEOUT_S = 60.0
WAIT_OBJECT_TIMEOUT_S  = 60.0  # time to place the object (marker C) after
                               # station B is verified — the run prompts and
                               # starts as soon as C is detected stably

# Vision sanity gates (lesson from pick_and_place_topdown 2026-07-04: a broken
# pixel→base map put the grasp target 190 m away and the only symptom was
# "No straight-down grasp reachable", which reads as a REACH problem. Catch
# nonsense positions before IK, and refuse to EXECUTE on a failed check.)
STATION_Z_TOL_M = 0.05            # A/B must be ≈ base plane (arm on table)
WORKSPACE_R_MAX_M = 0.55          # any target beyond this is a vision error
WORKSPACE_Z_M     = (-0.05, 0.30) # plausible target z window

MAX_VELOCITY_SCALE = 0.10
MAX_ACCEL_SCALE    = 0.10
CONFIRM_DELAY_S    = 4.0    # pause between preview and first motion

USE_TABLE_COLLISION = True  # thin box at table height so MoveIt avoids it
OBJECT_ID = 'pick_object'
TABLE_ID  = 'table_plane'


# ═══════════════════════════════════════════════════════════════════════════════
#  Marker tracking
# ═══════════════════════════════════════════════════════════════════════════════

class TrackedMarker:
    def __init__(self):
        self.samples: list[np.ndarray] = []   # base-frame positions (rolling)
        self.yaws: list[float] = []           # yaw samples (same window)
        self.yaw_deg: float = 0.0             # about world Z, from last sample
        self.last_seen: float = 0.0           # time.monotonic()

    def update(self, xyz_base: np.ndarray, yaw_deg: float):
        self.samples.append(xyz_base)
        self.yaws.append(yaw_deg)
        if len(self.samples) > MARKER_MEDIAN_WINDOW:
            self.samples.pop(0)
            self.yaws.pop(0)
        self.yaw_deg = yaw_deg
        self.last_seen = time.monotonic()

    @property
    def fresh(self) -> bool:
        return (self.last_seen > 0
                and time.monotonic() - self.last_seen < MARKER_FRESH_S)

    @property
    def position(self) -> np.ndarray:
        return np.median(np.stack(self.samples), axis=0)

    @property
    def grasp_yaw_deg(self) -> float:
        """Yaw for grasp-roll alignment, averaged over the whole window in
        the mod-90° domain (the symmetric fingers repeat every 90°). A single
        ArUco planar-ambiguity flip (±90/180°) in the last frame therefore
        cannot rotate the grasp onto the cube's corners, which the previous
        last-sample-only yaw allowed."""
        a = np.radians(np.asarray(self.yaws) * 4.0)
        return float(np.degrees(np.arctan2(np.mean(np.sin(a)),
                                           np.mean(np.cos(a)))) / 4.0)


class PickAndPlaceNode(EihBaseNode):

    def __init__(self):
        super().__init__('pick_and_place',
                         max_velocity_scale=MAX_VELOCITY_SCALE,
                         max_accel_scale=MAX_ACCEL_SCALE)
        self.tcp = make_tcp_tool(TCP_OFFSET_Z)
        self.markers: dict[int, TrackedMarker] = {}
        self._marker_lock = threading.Lock()
        self._detector_proc: subprocess.Popen | None = None
        self._viewer_proc: subprocess.Popen | None = None
        self._bridge = CvBridge()

        self.create_subscription(ArucoMarkerArray, '/marker_publisher/markers',
                                 self._on_markers, 10)
        # Subscribing to /marker_publisher/result also makes marker_publisher
        # render it (it skips drawing when nobody listens).
        self.create_subscription(Image, '/marker_publisher/result',
                                 self._on_result_image, 1)
        self._annot_pub = self.create_publisher(Image, '/pick_place/camera_view', 1)

        self._viz_pub = self.create_publisher(MarkerArray, '/pick_place/viz', 10)
        self._planned_viz: list[Marker] = []   # extra markers from planning
        self._status_text = 'starting'
        self.create_timer(0.5, self._publish_viz)

        self._scene_cli = self.create_client(ApplyPlanningScene, '/apply_planning_scene')

        mode = 'EXECUTE' if EXECUTE else 'PLAN-ONLY (no motion; set EXECUTE=True when happy)'
        self._log(f'Mode: {mode}')
        self._log(f'Markers: B={MARKER_B_ID} (place), '
                  f'C={MARKER_C_ID} (object {OBJECT_DIMS_M[0]*1000:.0f}×'
                  f'{OBJECT_DIMS_M[1]*1000:.0f}×{OBJECT_DIMS_M[2]*1000:.0f} mm); '
                  f'A={MARKER_A_ID} informational only (not required)')

    # ── Detection ─────────────────────────────────────────────────────────────

    def ensure_detector(self):
        """Start aruco_ros marker_publisher if nothing publishes the topic."""
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.count_publishers('/marker_publisher/markers') > 0:
                self._log('[DETECT] marker_publisher already running — VERIFY '
                          f'its marker_size is {MARKER_SIZE_M} m (a foreign '
                          'detector with another size scales every position '
                          'and this script cannot check its parameters).',
                          'WARN')
                return
            time.sleep(0.3)
        self._log('[DETECT] Starting aruco_ros marker_publisher (child process)...')
        self._detector_proc = subprocess.Popen(
            ['ros2', 'run', 'aruco_ros', 'marker_publisher', '--ros-args',
             '-p', f'marker_size:={MARKER_SIZE_M}',
             '-p', 'reference_frame:=camera_color_optical_frame',
             '-p', 'camera_frame:=camera_color_optical_frame',
             '-r', '/camera_info:=/camera/camera/color/camera_info',
             '-r', '/image:=/camera/camera/color/image_raw'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )   # same process group → dies when the GUI stops this script

    def ensure_viewer(self):
        """Open rqt_image_view on the annotated camera feed (child process)."""
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            self._log('[VIEW] No display available — skipping rqt_image_view. '
                      'View /pick_place/camera_view manually.', 'WARN')
            return
        self._log('[VIEW] Opening rqt_image_view on /pick_place/camera_view...')
        self._viewer_proc = subprocess.Popen(
            ['ros2', 'run', 'rqt_image_view', 'rqt_image_view',
             '/pick_place/camera_view'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )   # same process group → dies when the GUI stops this script

    def _on_result_image(self, msg: Image):
        """Overlay marker status (id, base-frame xyz, freshness) + stage on the
        aruco result image and republish for rqt_image_view."""
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return
        rows = [(f'pick_and_place: {self._status_text}', (255, 255, 255))]
        stations = [('A pick ', MARKER_A_ID, (255, 102, 51)),
                    ('B place', MARKER_B_ID, (0, 204, 255)),
                    ('C object', MARKER_C_ID, (51, 230, 26))]
        now = time.monotonic()
        with self._marker_lock:
            for label, mid, bgr in stations:
                mk = self.markers.get(mid)
                if mk is None or not mk.samples:
                    rows.append((f'{label} id {mid}: NOT DETECTED', (64, 64, 230)))
                    continue
                age = now - mk.last_seen
                state = 'DETECTED' if age < MARKER_FRESH_S else f'stale {age:.0f}s'
                p = mk.position
                txt = (f'{label} id {mid}: {state}  '
                       f'({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m')
                if mid == MARKER_C_ID:
                    txt += f'  yaw {mk.yaw_deg:+.0f}'
                rows.append((txt, bgr if age < MARKER_FRESH_S else (128, 128, 128)))
            extra = sorted(i for i in self.markers
                           if i not in (MARKER_A_ID, MARKER_B_ID, MARKER_C_ID)
                           and now - self.markers[i].last_seen < MARKER_FRESH_S)
        if extra:
            rows.append((f'other ids in view: {extra}', (200, 200, 200)))
        line_h, pad = 22, 8
        panel_h = pad * 2 + line_h * len(rows)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], panel_h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        for i, (txt, bgr) in enumerate(rows):
            cv2.putText(img, txt, (pad, pad + line_h * (i + 1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 1, cv2.LINE_AA)
        out = self._bridge.cv2_to_imgmsg(img, encoding='bgr8')
        out.header = msg.header
        self._annot_pub.publish(out)

    def _on_markers(self, msg: ArucoMarkerArray):
        l6 = self.link6_in_base()
        if l6 is None:
            return
        t_b_l6, R_b_l6 = l6
        R_b_c = R_b_l6 * R_LINK6_CAM
        t_b_c = t_b_l6 + R_b_l6.apply(CAM_T)
        for mk in msg.markers:
            p = mk.pose.pose.position
            q = mk.pose.pose.orientation
            p_cam = np.array([p.x, p.y, p.z])
            if mk.id in TRUE_SIZE_M:
                p_cam = p_cam * (TRUE_SIZE_M[mk.id] / MARKER_SIZE_M)
            xyz_base = t_b_c + R_b_c.apply(p_cam)
            R_b_m = R_b_c * Rotation.from_quat([q.x, q.y, q.z, q.w])
            yaw = float(R_b_m.as_euler('zyx', degrees=True)[0])
            with self._marker_lock:
                self.markers.setdefault(mk.id, TrackedMarker()).update(xyz_base, yaw)

    def get_marker(self, mid: int, min_samples: int = 1) -> TrackedMarker | None:
        with self._marker_lock:
            m = self.markers.get(mid)
            return (m if (m is not None and m.fresh
                          and len(m.samples) >= min_samples) else None)

    def flush_markers(self):
        """Drop all tracked samples. Call after ANY arm motion: samples taken
        while the camera moved are smeared (the detection is transformed with
        link6's pose at callback time, but the image was captured 100-300 ms
        earlier) and would poison the median for the next 2 s."""
        with self._marker_lock:
            self.markers.clear()

    def sane_target(self, xyz, label: str) -> bool:
        """Reject positions no real detection could produce (vision gate)."""
        r = float(np.hypot(xyz[0], xyz[1]))
        if r <= WORKSPACE_R_MAX_M and WORKSPACE_Z_M[0] <= xyz[2] <= WORKSPACE_Z_M[1]:
            return True
        self._log(f'[{label}] Target ({xyz[0]:+.3f}, {xyz[1]:+.3f}, '
                  f'{xyz[2]:+.3f}) is outside any plausible workspace '
                  f'(r={r:.3f} m) — this is a vision/calibration problem, '
                  'NOT a reach problem. Aborting.', 'ERROR')
        return False

    def wait_for_markers(self, ids: list[int], timeout_s: float,
                         min_samples: int = 1) -> bool:
        deadline = time.monotonic() + timeout_s
        last_note = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            missing = [i for i in ids
                       if self.get_marker(i, min_samples) is None]
            if not missing:
                return True
            if time.monotonic() - last_note > 2.0:
                self._log(f'[SCAN] Waiting for marker(s) {missing}...')
                last_note = time.monotonic()
            time.sleep(0.3)
        return False

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _publish_viz(self):
        ms: list[Marker] = []
        stations = [
            (MARKER_A_ID, 'A  pick',  (0.2, 0.4, 1.0, 0.9)),
            (MARKER_B_ID, 'B  place', (1.0, 0.8, 0.0, 0.9)),
        ]
        for i, (mid, label, rgba) in enumerate(stations):
            mk = self.get_marker(mid)
            if mk is None:
                continue
            p = mk.position
            ms.append(self.viz_sphere('stations', i * 2, p, rgba, diam=0.04))
            ms.append(self.viz_text('stations', i * 2 + 1,
                                    p + [0, 0, 0.05], label, rgba))
        mk = self.get_marker(MARKER_C_ID)
        if mk is not None:
            c = mk.position
            centre = c - np.array([0, 0, OBJECT_DIMS_M[2] / 2])
            ms.append(self.viz_cube('object', 0, centre, OBJECT_DIMS_M,
                                    (0.1, 0.9, 0.2, 0.8)))
            ms.append(self.viz_text('object', 1, c + [0, 0, 0.04],
                                    f'C  object (yaw {mk.yaw_deg:+.0f}°)',
                                    (0.1, 0.9, 0.2, 0.9)))
        ms.append(self.viz_text('status', 0, [0.0, 0.0, 0.55],
                                f'pick_and_place: {self._status_text}'))
        ms.extend(self._planned_viz)
        self._viz_pub.publish(MarkerArray(markers=ms))

    def add_planned_pose_viz(self, tool_pos, R_world_tool, label, rgba):
        """Persistent arrow (tool axis) + small cube + label at a planned pose."""
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
        self._log(f'[STAGE] {text}')

    # ── Planning scene ────────────────────────────────────────────────────────

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
        """Attach the object box to link6 at the TCP (called after closing)."""
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

    def scene_clear(self):
        scene = PlanningScene()
        for oid in (OBJECT_ID, TABLE_ID):
            co = CollisionObject()
            co.id = oid
            co.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(co)
        aco = AttachedCollisionObject()
        aco.link_name = EFFECTOR_LINK
        aco.object.id = OBJECT_ID
        aco.object.operation = CollisionObject.REMOVE
        scene.robot_state.attached_collision_objects.append(aco)
        scene.robot_state.is_diff = True
        self._apply_scene(scene)

    # ── Grasp geometry ────────────────────────────────────────────────────────

    def grasp_rolls_for_yaw(self, yaw_deg: float) -> tuple:
        """Rolls (about the vertical tool axis) that align the finger-closing
        axis (tool Y = link6 Y) perpendicular to the cube faces. Fingers are
        symmetric so the pattern repeats every 90°."""
        base = (-90.0 - yaw_deg) % 90.0
        return tuple((base + k * 90.0) % 360.0 for k in range(4))

    def find_grasp(self, target_xyz: np.ndarray, dist: float,
                   yaw_deg: float, label: str):
        """Look-at sweep for the TCP, straight-down only, rolls aligned to the
        object's yaw. Returns same tuple as find_reachable_tool_pose."""
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
                self._log(f'[{label}] Reachable grasp candidate #{n}: roll {roll:.0f}°')
                return tool_pos, R_world_tool, link6_pos, link6_quat, sol, f'roll {roll:.0f}°'
        self._log(f'[{label}] No straight-down grasp reachable ({n} tried).', 'ERROR')
        return None

    # ── Main sequence ─────────────────────────────────────────────────────────

    def run(self) -> bool:
        self.ensure_detector()
        self.ensure_viewer()
        # HOME_AFTER_RUN target — must be captured before any motion.
        start_sol = (self.current_arm_joints()
                     if (EXECUTE and HOME_AFTER_RUN) else None)

        # ── SCAN: station B only (2026-07-12 design — A is not needed: the
        # pick target comes entirely from C; table z / sanity / collision
        # centre come from B, which the object never covers at scan time) ──
        if STATION_B_XYZ is not None:
            b = np.array(STATION_B_XYZ, dtype=float)
            self.status('SCAN: using frozen station B (override set)')
            self._log(f'[SCAN] B (override) at ({b[0]:+.3f}, {b[1]:+.3f}, {b[2]:+.3f})')
        else:
            self.status('SCAN: measuring place station B')
            if not self.wait_for_markers([MARKER_B_ID], WAIT_MARKERS_TIMEOUT_S,
                                         min_samples=MARKER_MEDIAN_WINDOW):
                self._log('Marker B never seen — aborting. Is it printed at '
                          f'{MARKER_SIZE_M} m, uncovered and in the camera '
                          'view from the start pose? Is marker_publisher '
                          'receiving images?', 'ERROR')
                return False
            b1 = self.get_marker(MARKER_B_ID).position
            # Measurement-stability gate: re-measure after a settle delay and
            # require agreement — catches motion smear, a jostled phone/arm,
            # or a marker still being taped down mid-scan.
            self._log(f'[SCAN] B first read ({b1[0]:+.3f}, {b1[1]:+.3f}, '
                      f'{b1[2]:+.3f}); re-measuring in '
                      f'{STATION_RESAMPLE_S:.0f}s to confirm...')
            time.sleep(STATION_RESAMPLE_S)
            self.flush_markers()
            if not self.wait_for_markers([MARKER_B_ID], 10.0,
                                         min_samples=MARKER_MEDIAN_WINDOW):
                self._log('[SCAN] Station B vanished during the stability '
                          're-measure — aborting.', 'ERROR')
                return False
            b = self.get_marker(MARKER_B_ID).position
            drift = float(np.linalg.norm(b - b1))
            if drift > STATION_STABLE_TOL_M:
                self._log(f'[SCAN] Station B unstable: two reads '
                          f'{STATION_RESAMPLE_S:.0f}s apart disagree by '
                          f'{drift*1000:.0f} mm (tol '
                          f'{STATION_STABLE_TOL_M*1000:.0f}) — is the camera '
                          'or table still moving? Aborting.', 'ERROR')
                return False
            self._log(f'[SCAN] B stable at ({b[0]:+.3f}, {b[1]:+.3f}, '
                      f'{b[2]:+.3f}) (drift {drift*1000:.1f} mm)')
        if not self.sane_target(b, 'SCAN'):
            return False
        if abs(b[2]) > STATION_Z_TOL_M:
            msg = (f'[SCAN] Station B detected at z={b[2]:+.3f} m, '
                   'but the arm stands on the table (expect ≈0) — '
                   'hand-eye calibration, marker_size or the TF chain '
                   'is off.')
            if EXECUTE:
                self._log(msg + ' Refusing to EXECUTE.', 'ERROR')
                return False
            self._log(msg, 'WARN')
        # Early feasibility: a straight-down place at B must have IK NOW,
        # before any motion — clearer than failing deep inside PLAN.
        place_probe = self.find_grasp(
            np.array([b[0], b[1], b[2] + OBJECT_DIMS_M[2] / 2 + PLACE_DROP_M]),
            OBJECT_DIMS_M[2] / 2 + PRE_GRASP_CLEAR_M, 0.0, 'B_FEASIBLE')
        if place_probe is None:
            self._log('[SCAN] Station B is measured but NOT reachable for a '
                      'straight-down place — move it closer to the base '
                      '(keep x ≤ 0.26 m; Joint5 envelope).', 'ERROR')
            return False
        # A is optional: log it as a cross-check if it happens to be visible.
        mk_a = self.get_marker(MARKER_A_ID)
        if mk_a is not None:
            self._log(f'[SCAN] A visible at ({mk_a.position[0]:+.3f}, '
                      f'{mk_a.position[1]:+.3f}, {mk_a.position[2]:+.3f}) '
                      '(cross-check only — A is not used).')

        # ── HOVER over B to see the table precisely ──
        if HOVER_BEFORE_SCAN:
            self.status('HOVER_B: aiming camera above place station')
            hover = self.find_reachable_tool_pose(CAMERA_TOOL, b, SCAN_DIST_M,
                                                  label='HOVER_B')
            if hover is None:
                return False
            h_pos, h_R, h_l6p, h_l6q, h_sol, h_desc = hover
            self.add_planned_pose_viz(h_pos, h_R, f'hover B ({h_desc})',
                                      (0.2, 0.4, 1.0, 0.9))

            if EXECUTE:
                if not self.enable_control_gate():
                    self._log('[GATE] control_enable failed — driver not '
                              'ready. Aborting before any motion.', 'ERROR')
                    return False
                self._log(f'[PREVIEW] Moving in {CONFIRM_DELAY_S:.0f}s — check RViz now.')
                time.sleep(CONFIRM_DELAY_S)
                if not self.move_to(h_l6p, h_l6q, h_sol, 'HOVER_B'):
                    return False
                time.sleep(1.0)   # let in-flight (mid-motion) detections drain
                self.flush_markers()

        self.status('SCAN_C: locating the object')
        # Drop anything sampled while the GUI homed/jogged the arm just before
        # this script started — those detections carry motion smear exactly
        # like post-HOVER ones do (same reason flush_markers exists at all).
        self.flush_markers()
        self._log('[SCAN_C] Station B verified. Place the object (marker C) '
                  'on the pick spot now if it is not there yet — waiting up '
                  f'to {WAIT_OBJECT_TIMEOUT_S:.0f}s for C.')
        if not self.wait_for_markers([MARKER_C_ID], WAIT_OBJECT_TIMEOUT_S,
                                     min_samples=MARKER_MEDIAN_WINDOW):
            self._log('Marker C (object) not detected — is the cube in the '
                      'camera view from the start pose?', 'ERROR')
            return False
        mk_c = self.get_marker(MARKER_C_ID)
        c = mk_c.position
        obj_centre = c - np.array([0, 0, OBJECT_DIMS_M[2] / 2])
        table_z = float(b[2])
        self._log(f'[SCAN_C] C at ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}), '
                  f'object centre z={obj_centre[2]:+.3f}, table z={table_z:+.3f}')
        if abs((c[2] - OBJECT_DIMS_M[2]) - b[2]) > 0.02:
            msg = ('[SCAN_C] Object height vs station-B height mismatch >2 cm — '
                   'check OBJECT_DIMS_M and marker detections.')
            if EXECUTE:
                self._log(msg + ' Refusing to EXECUTE.', 'ERROR')
                return False
            self._log(msg, 'WARN')

        if USE_TABLE_COLLISION:
            self.scene_add_table(table_z, (c[:2] + b[:2]) / 2)

        # ── Grasp + place geometry (all computed and validated up front) ──
        self.status('PLAN: computing grasp and place poses')
        if not self.sane_target(obj_centre, 'PRE_GRASP'):
            return False
        grasp_yaw = mk_c.grasp_yaw_deg   # window-averaged, ambiguity-flip-proof
        pre_grasp = self.find_grasp(
            obj_centre, OBJECT_DIMS_M[2] / 2 + PRE_GRASP_CLEAR_M,
            grasp_yaw, 'PRE_GRASP')
        if pre_grasp is None:
            return False
        pg_pos, pg_R, pg_l6p, pg_l6q, pg_sol, pg_desc = pre_grasp

        # grasp: same orientation, TCP at object centre
        g_pos = obj_centre
        g_l6p, g_l6q = self.tcp.to_link6(g_pos, pg_R)
        g_sol = self.call_ik(g_l6p, g_l6q, timeout_s=1.0)
        if g_sol is None:
            self._log('[GRASP] Grasp depth pose not reachable — lower '
                      'TCP_OFFSET_Z or raise the grasp point.', 'ERROR')
            return False

        # place: object centre ends at B + half height (+ drop margin)
        place_centre = np.array([b[0], b[1],
                                 b[2] + OBJECT_DIMS_M[2] / 2 + PLACE_DROP_M])
        if not self.sane_target(place_centre, 'PRE_PLACE'):
            return False
        pre_place = self.find_grasp(
            place_centre, OBJECT_DIMS_M[2] / 2 + PRE_GRASP_CLEAR_M,
            grasp_yaw, 'PRE_PLACE')
        if pre_place is None:
            return False
        pp_pos, pp_R, pp_l6p, pp_l6q, pp_sol, pp_desc = pre_place
        pl_l6p, pl_l6q = self.tcp.to_link6(place_centre, pp_R)
        pl_sol = self.call_ik(pl_l6p, pl_l6q, timeout_s=1.0)
        if pl_sol is None:
            self._log('[PLACE] Place depth pose not reachable.', 'ERROR')
            return False

        # LIFT and RETREAT validated NOW, not mid-cycle: the old code fell
        # back to `ik(...) or g_sol` at execution time, and a failed lift IK
        # then planned to the pose the arm was ALREADY at — a silent no-op
        # "lift" followed by dragging the object at table height.
        lift_l6p = g_l6p + np.array([0, 0, LIFT_CLEAR_M])
        lift_sol = self.call_ik(lift_l6p, g_l6q, timeout_s=1.0)
        if lift_sol is None:
            lift_sol = self.call_ik(lift_l6p, g_l6q, timeout_s=1.0)  # IK is
            # single-attempt flaky (RUNBOOK) — one retry before concluding
        if lift_sol is None:
            self._log('[PLAN] Lift pose has no IK — executing would strand '
                      'the arm holding the object at table height. Aborting '
                      'before any motion.', 'ERROR')
            return False
        ret_l6p = pl_l6p + np.array([0, 0, RETREAT_CLEAR_M])
        ret_sol = (self.call_ik(ret_l6p, pl_l6q, timeout_s=1.0)
                   or self.call_ik(ret_l6p, pl_l6q, timeout_s=1.0))
        if ret_sol is None:
            self._log('[PLAN] Retreat pose has no IK — will skip the retreat '
                      'after releasing (object is safe by then).', 'WARN')

        self.add_planned_pose_viz(pg_pos, pg_R, f'pre-grasp ({pg_desc})',
                                  (0.1, 0.9, 0.2, 0.9))
        self.add_planned_pose_viz(g_pos, pg_R, 'grasp', (1.0, 0.2, 0.2, 0.9))
        self.add_planned_pose_viz(pp_pos, pp_R, f'pre-place ({pp_desc})',
                                  (1.0, 0.8, 0.0, 0.9))
        self.add_planned_pose_viz(place_centre, pp_R, 'place', (1.0, 0.5, 0.0, 0.9))
        self._log('[PLAN] All poses IK-validated (pre-grasp, grasp, lift, '
                  'pre-place, place'
                  f'{", retreat" if ret_sol is not None else " — retreat SKIPPED"}). '
                  f'Grasp width target: {OBJECT_DIMS_M[0] - GRIP_SQUEEZE_M:.3f} m.')

        if not EXECUTE:
            traj = self.plan_to_link6_pose(pg_l6p, pg_l6q, pg_sol, 'PRE_GRASP(plan-only)')
            self.status('PLAN-ONLY complete — check RViz, then set EXECUTE=True')
            self._log('[DONE] Plan-only run finished '
                      f'({"pre-grasp trajectory OK" if traj else "pre-grasp plan FAILED — investigate before executing"}). '
                      'No motion was commanded.')
            return traj is not None

        # ── PICK ──
        # idempotent; needed when HOVER was skipped
        if not self.enable_control_gate():
            self._log('[GATE] control_enable failed — driver not ready. '
                      'Aborting before any motion.', 'ERROR')
            return False
        self._log(f'[PREVIEW] Moving in {CONFIRM_DELAY_S:.0f}s — check RViz now.')
        time.sleep(CONFIRM_DELAY_S)
        self.status('PICK: opening gripper, moving to pre-grasp')
        self.set_gripper_width(GRIPPER_MAX_WIDTH_M, GRIPPER_SETTLE_S)
        if not self.move_to(pg_l6p, pg_l6q, pg_sol, 'PRE_GRASP'):
            return False

        self.status('PICK: descending to grasp')
        # short straight descent into the grasp — collision checking off, the
        # volume was validated by the pre-grasp plan and the IK check above
        if not self.move_to(g_l6p, g_l6q, g_sol, 'GRASP_DESCEND',
                            avoid_collisions=False):
            return False

        self.status('PICK: closing gripper')
        self.set_gripper_width(max(OBJECT_DIMS_M[0] - GRIP_SQUEEZE_M, 0.0),
                               GRIPPER_SETTLE_S)
        self.scene_attach_object()

        self.status('PICK: lifting')
        if not self.move_to(lift_l6p, g_l6q, lift_sol, 'LIFT',
                            avoid_collisions=False):
            return False

        # ── Grasp verification (vision — the gripper is open-loop, no width
        # readback exists). After the lift, marker C must NOT still be at the
        # pick spot: if it is, the fingers closed on air / pushed the cube
        # aside, and continuing would run the whole PLACE cycle empty-handed
        # and still report SUCCESS (enumerated bug #16, 2026-07-12). C hidden
        # by the gripper is the common good case — inconclusive, continue.
        self.status('PICK: verifying grasp')
        time.sleep(1.0)              # drain in-flight (mid-lift) detections
        self.flush_markers()
        self.wait_for_markers([MARKER_C_ID], 3.0, min_samples=2)
        mk_chk = self.get_marker(MARKER_C_ID, min_samples=2)
        if mk_chk is not None and np.linalg.norm(mk_chk.position - c) < 0.02:
            self._log('[GRASP] Marker C is still at the pick position — the '
                      'grasp MISSED (cube pushed aside or fingers closed on '
                      'air). Opening gripper and aborting.', 'ERROR')
            self.set_gripper_width(GRIPPER_MAX_WIDTH_M, GRIPPER_SETTLE_S)
            self.scene_detach_object(obj_centre)   # scene box back on the table
            return False
        if mk_chk is not None:
            self._log(f'[GRASP] Marker C moved with the gripper '
                      f'(now z={mk_chk.position[2]:+.3f}) — grasp confirmed.')
        else:
            self._log('[GRASP] Marker C not visible after lift (usually the '
                      'gripper occludes it) — cannot confirm the grasp; '
                      'continuing. VERIFY at the end is the backstop.', 'WARN')

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

        if ret_sol is not None:
            self.status('PLACE: retreating')
            self.move_to(ret_l6p, pl_l6q, ret_sol, 'RETREAT',
                         avoid_collisions=False)
        else:
            self._log('[RETREAT] Skipped (no IK at plan time).', 'WARN')

        # ── VERIFY ──
        self.status('VERIFY: checking object at place station')
        time.sleep(2.0)   # drain in-flight detections from the retreat motion
        self.flush_markers()
        self.wait_for_markers([MARKER_C_ID], 5.0,
                              min_samples=MARKER_MEDIAN_WINDOW)
        mk_c2 = self.get_marker(MARKER_C_ID)
        if mk_c2 is None:
            self._log('[VERIFY] Marker C not visible from here — move the '
                      'camera over B to confirm placement visually.', 'WARN')
        else:
            err = np.linalg.norm(mk_c2.position[:2] - b[:2])
            lvl = 'INFO' if err < 0.03 else 'WARN'
            self._log(f'[VERIFY] Object is {err*100:.1f} cm from station B '
                      f'(horizontal). {"PASS" if err < 0.03 else "off target"}',
                      lvl)
        # ── HOME ──
        if EXECUTE and HOME_AFTER_RUN:
            if start_sol is None:
                self._log('[HOME] No joint feedback was available at start — '
                          'skipping the return move.', 'WARN')
            else:
                self.status('HOME: returning to start pose')
                traj = self.plan_joint_goal(start_sol, 'HOME')
                if traj is None or not self.execute_trajectory(traj, 'HOME'):
                    self._log('[HOME] Return-to-start failed — arm stays put '
                              '(the pick-and-place itself succeeded).', 'WARN')

        self.status('DONE')
        return True

    def current_arm_joints(self, timeout_s: float = 2.0) -> dict | None:
        """Snapshot of the six arm joints from /feedback/joint_states, in the
        same dict form plan_joint_goal() takes. None if no feedback arrives
        (e.g. the mock stack, which has no encoder feedback topic)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            js = self._current_joint_state
            if js is not None:
                sol = {n: p for n, p in zip(js.name, js.position)
                       if n in JOINT_LIMITS}
                if len(sol) == len(JOINT_LIMITS):
                    return sol
            time.sleep(0.05)
        return None

    def capture_stations(self) -> bool:
        """Diagnostic, no motion: print what the live SCAN would measure.
        Stations are detected live per run since 2026-07-16 (B only — A is
        not used by the pipeline); paste the B line into STATION_B_XYZ only
        if you want to FREEZE it for repeatability testing."""
        self.ensure_detector()
        self.status('CAPTURE: waiting for markers A and B (table must be clear)')
        if not self.wait_for_markers([MARKER_A_ID, MARKER_B_ID],
                                     WAIT_MARKERS_TIMEOUT_S):
            self._log('Markers A/B never seen — is the table clear and are '
                      'both markers in view?', 'ERROR')
            return False
        a = self.get_marker(MARKER_A_ID).position
        b = self.get_marker(MARKER_B_ID).position
        self._log('[CAPTURE] Live station readings (A is informational only):')
        self._log(f'A = ({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f})')
        self._log(f'STATION_B_XYZ = ({b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f})')
        return True

    def shutdown(self):
        for proc in (self._detector_proc, self._viewer_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()


def main():
    rclpy.init()
    node = PickAndPlaceNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    ok = False
    try:
        if '--capture-stations' in sys.argv:
            ok = node.capture_stations()
        else:
            ok = node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
    node._log(f'Result: {"SUCCESS" if ok else "FAILED"}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
