#!/usr/bin/env python3
"""
Transform audit — is the camera→base chain CORRECT? (READ-ONLY, never moves)

Answers the professor's question ("are your transforms right, not upside
down?") with live evidence instead of trust:

  1. STARTUP GATE   decomposes the hand-eye calibration and refuses axes
                    that could not be a wrist-mounted camera (the actual
                    "upside down" check).
  2. AXIS CHECK     you slide a marker along base X/Y with a tape measure
                    and watch the live base-frame readout follow (signs +
                    magnitude). Manual, but definitive.
  3. VIEWPOINT AUDIT the same STATIC marker is measured from every arm pose
                    you park the camera at (jog via GUI/RViz — this script
                    NEVER moves the arm). A correct chain reports the same
                    base-frame position from everywhere; the error PATTERN
                    otherwise names the culprit:
                      error along the view ray, ∝ distance → printed marker
                        size vs marker_size, or camera intrinsics
                      constant offset in the CAMERA frame → CAM_T translation
                      error ⊥ view ray, growing with distance → R_LINK6_CAM
                        rotation (recalibrate)
                    This is the instrument for the 2026-07-06 "hover reads
                    differ ~13 cm from home-pose reads" mystery.
  4. TF RACE SNIFFER lists every TF parent claiming camera_color_optical_frame
                    (RViz can look wrong from the known race while the manual
                    chain here is right — show the professor this line).

Workflow (stack up per RUNBOOK §0, then GUI → Custom Script Runner):
  - keep ONE marker fixed on the table (any audited id);
  - park the arm at the home/start pose, keep it STILL until the overlay
    says the viewpoint is captured (~1 s);
  - jog to the next viewpoint (vary height, tilt, side — include the close
    straight-down hover that misbehaved) and hold still again; 5+ viewpoints
    make the diagnosis meaningful;
  - stop the script (or just read the log): report + CSV are REWRITTEN to
    ~/ros2_ws/audit_results/ after every captured viewpoint, so a hard kill
    loses nothing.

Topics:  in  /marker_publisher/markers, /marker_publisher/result,
             /camera/camera/color/camera_info, /tf, /tf_static
         out /transform_audit/camera_view (annotated image),
             /transform_audit/viz (RViz MarkerArray)

Never-do: no motion, no gripper, no planning-scene writes, no service calls
that change state. Samples are ONLY taken while link6 has been still for
STATIC_WINDOW_S (aruco images lag 100–300 ms; stillness outlasts the lag).

Offline check anytime:  python3 transform_audit.py --selftest
"""

import csv
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

from aruco_msgs.msg import MarkerArray as ArucoMarkerArray
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from my_robot_controller.eih_core import (
    CALIB_SOURCE, CAM_T, EihBaseNode, R_LINK6_CAM,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Size handed to marker_publisher (one value for every marker it sees).
MARKER_SIZE_M = 0.047

# Per-id TRUE printed size, when it differs from MARKER_SIZE_M. ArUco
# translation scales exactly linearly with the assumed size, so one detector
# serves mixed sizes:  position ×= TRUE_SIZE_M[id] / MARKER_SIZE_M.
# e.g. a future 0.021 m object marker:  TRUE_SIZE_M = {582: 0.021}
TRUE_SIZE_M: dict[int, float] = {582: 0.021}   # cube marker C, 21 mm reprint

# Tape-measured base-frame positions (m) to audit against, when you have
# them. e.g. GROUND_TRUTH_XYZ = {100: (0.412, 0.136, 0.000)}
GROUND_TRUTH_XYZ: dict[int, tuple] = {}

AUDIT_IDS = None            # None = audit every id seen; or e.g. (100, 101)

STATIC_WINDOW_S     = 0.7   # arm must be still this long before sampling
STATIC_TRANS_TOL_M  = 0.002
STATIC_ROT_TOL_DEG  = 0.4

VIEWPOINT_NEW_DIST_M  = 0.03   # camera moved this far   → new viewpoint
VIEWPOINT_NEW_ANG_DEG = 5.0    # view axis turned this far → new viewpoint
MIN_SAMPLES_PER_VP    = 8      # viewpoint counts once it has this many

SPREAD_PASS_M = 0.010       # cross-viewpoint spread ≤ this = chain CONSISTENT

RESULTS_DIR = Path.home() / 'ros2_ws' / 'audit_results'

CAMERA_FRAME = 'camera_color_optical_frame'


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure core (no ROS) — everything here is covered by --selftest
# ═══════════════════════════════════════════════════════════════════════════════

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def calib_gross_check(R_l6_cam: Rotation, t_l6_cam: np.ndarray) -> tuple[bool, str]:
    """The literal 'not upside down' gate: for a wrist camera looking along
    the tool, the optical +Z (view axis) must be broadly ALONG link6 +Z and
    the lens within arm's-hand distance of link6. Catches inverted
    transforms, optical-vs-camera_link mixups and axis-order errors — all of
    which throw the view axis ≥ 90° off or the offset to silly values."""
    view = R_l6_cam.apply([0.0, 0.0, 1.0])
    ang = float(np.degrees(np.arccos(np.clip(view[2], -1.0, 1.0))))
    dist = float(np.linalg.norm(t_l6_cam))
    msg = (f'view axis {ang:.1f}° off link6 +Z, lens {dist*100:.1f} cm '
           f'from link6 origin')
    if ang > 60.0:
        return False, msg + ' — FAILS gross check: calibration looks ' \
            'inverted or in the wrong frame convention (camera_link vs ' \
            'optical?). Do not trust any position until recalibrated.'
    if not 0.02 <= dist <= 0.20:
        return False, msg + ' — FAILS gross check: implausible mount offset.'
    return True, msg + ' — plausible wrist mount (gross convention check PASS).'


class StaticGate:
    """True only when the pose history spans the window and never left the
    translation/rotation tolerance of the newest pose."""

    def __init__(self, window_s: float = STATIC_WINDOW_S,
                 trans_tol_m: float = STATIC_TRANS_TOL_M,
                 rot_tol_deg: float = STATIC_ROT_TOL_DEG):
        self.window_s = window_s
        self.trans_tol = trans_tol_m
        self.rot_tol = rot_tol_deg
        self._hist: list[tuple[float, np.ndarray, Rotation]] = []
        self.is_static = False

    def push(self, t: float, pos: np.ndarray, R: Rotation) -> bool:
        self._hist.append((t, np.asarray(pos, float), R))
        while self._hist and t - self._hist[0][0] > self.window_s * 1.5:
            self._hist.pop(0)
        span = t - self._hist[0][0]
        if len(self._hist) < 3 or span < self.window_s:
            self.is_static = False
            return False
        ok = True
        for ht, hp, hR in self._hist:
            if t - ht > self.window_s:
                continue
            if np.linalg.norm(hp - pos) > self.trans_tol:
                ok = False
                break
            if np.degrees((hR.inv() * R).magnitude()) > self.rot_tol:
                ok = False
                break
        self.is_static = ok
        return ok


class Viewpoint:
    """One parked camera pose, accumulating samples per marker id."""

    def __init__(self, idx: int, cam_pos: np.ndarray, cam_R: Rotation):
        self.idx = idx
        self.cam_pos = np.asarray(cam_pos, float)
        self.cam_R = cam_R
        self.view_axis = cam_R.apply([0.0, 0.0, 1.0])
        # per marker id: base-frame xyz samples + camera-frame distances
        self.xyz: dict[int, list[np.ndarray]] = {}
        self.dist: dict[int, list[float]] = {}

    def add(self, mid: int, xyz_base: np.ndarray, dist_m: float):
        self.xyz.setdefault(mid, []).append(np.asarray(xyz_base, float))
        self.dist.setdefault(mid, []).append(float(dist_m))

    def n(self, mid: int) -> int:
        return len(self.xyz.get(mid, ()))

    def median(self, mid: int) -> np.ndarray:
        return np.median(np.stack(self.xyz[mid]), axis=0)

    def median_dist(self, mid: int) -> float:
        return float(np.median(self.dist[mid]))

    @property
    def tilt_deg(self) -> float:
        """View-axis angle from straight down (0° = the hover pose)."""
        return float(np.degrees(np.arccos(np.clip(-self.view_axis[2], -1, 1))))


class ViewpointClusters:
    def __init__(self, new_dist_m: float = VIEWPOINT_NEW_DIST_M,
                 new_ang_deg: float = VIEWPOINT_NEW_ANG_DEG):
        self.new_dist = new_dist_m
        self.new_ang = new_ang_deg
        self.vps: list[Viewpoint] = []

    def assign(self, cam_pos: np.ndarray, cam_R: Rotation) -> Viewpoint:
        axis = cam_R.apply([0.0, 0.0, 1.0])
        for vp in self.vps:
            if (np.linalg.norm(vp.cam_pos - cam_pos) <= self.new_dist
                    and np.degrees(np.arccos(np.clip(
                        float(np.dot(vp.view_axis, axis)), -1, 1)))
                    <= self.new_ang):
                return vp
        vp = Viewpoint(len(self.vps) + 1, cam_pos, cam_R)
        self.vps.append(vp)
        return vp


def _corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def analyse_marker(vps: list[Viewpoint], mid: int,
                   ground_truth=None, min_n: int = MIN_SAMPLES_PER_VP) -> dict | None:
    """Cross-viewpoint consistency + error-signature hints for one marker.
    Returns dict(rows, spread_m, axis_range_m, ref, hints, verdict) or None
    if fewer than 2 viewpoints saw it."""
    use = [vp for vp in vps if vp.n(mid) >= min_n]
    if len(use) < 2:
        return None
    meds = np.stack([vp.median(mid) for vp in use])
    ref = (np.asarray(ground_truth, float) if ground_truth is not None
           else np.mean(meds, axis=0))

    rows, along_frac, perp_norm, dists, e_cam = [], [], [], [], []
    for vp, m in zip(use, meds):
        e = m - ref
        ray = _unit(m - vp.cam_pos)
        d = vp.median_dist(mid)
        along = float(np.dot(e, ray))
        perp = e - along * ray
        along_frac.append(along / d if d > 1e-6 else 0.0)
        perp_norm.append(float(np.linalg.norm(perp)))
        dists.append(d)
        e_cam.append(vp.cam_R.inv().apply(e))
        rows.append(dict(vp=vp.idx, n=vp.n(mid), dist=d, tilt=vp.tilt_deg,
                         xyz=m, err=float(np.linalg.norm(e))))

    spread = 0.0
    for i in range(len(meds)):
        for j in range(i + 1, len(meds)):
            spread = max(spread, float(np.linalg.norm(meds[i] - meds[j])))
    axis_range = meds.max(axis=0) - meds.min(axis=0)

    hints = []
    if spread <= SPREAD_PASS_M and (ground_truth is None or
                                    max(r['err'] for r in rows) <= 0.015):
        verdict = f'CONSISTENT (spread {spread*100:.1f} cm ≤ {SPREAD_PASS_M*100:.0f} cm)'
    else:
        verdict = f'INCONSISTENT (spread {spread*100:.1f} cm)'
        # Scale-like error (wrong printed size / intrinsics) stretches every
        # measurement along its own view ray:  m_i = T + (1+s)(T − c_i) + c_i
        # = u − s·c_i with u = (1+s)T — LINEAR in (u, s), so fit it directly.
        A = np.zeros((3 * len(use), 4))
        bvec = np.zeros(3 * len(use))
        for k, (vp, m) in enumerate(zip(use, meds)):
            A[3 * k:3 * k + 3, :3] = np.eye(3)
            A[3 * k:3 * k + 3, 3] = -vp.cam_pos
            bvec[3 * k:3 * k + 3] = m
        sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        s = float(sol[3])
        resid = (A @ sol - bvec).reshape(-1, 3)
        rms_model = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
        rms_base = float(np.sqrt(np.mean(np.sum((meds - meds.mean(0))**2, axis=1))))
        if abs(s) > 0.015 and rms_model < 0.5 * rms_base:
            true_sz = MARKER_SIZE_M / (1.0 + s)
            hints.append(
                f'measurements stretched {s*100:+.1f}% along every view ray '
                f'→ scale-like: printed size vs marker_size or intrinsics. '
                f'If size: true size ≈ {true_sz*1000:.1f} mm (configured '
                f'{MARKER_SIZE_M*1000:.1f} mm) — measure the print.')
        ec = np.stack(e_cam)
        ec_mean = ec.mean(axis=0)
        ec_norm = float(np.linalg.norm(ec_mean))
        ec_scatter = float(np.mean(np.linalg.norm(ec - ec_mean, axis=1)))
        if ec_norm > 0.010 and ec_scatter < 0.5 * ec_norm:
            hints.append(
                f'≈constant offset in the CAMERA frame '
                f'({ec_mean[0]*100:+.1f}, {ec_mean[1]*100:+.1f}, '
                f'{ec_mean[2]*100:+.1f}) cm → CAM_T translation error — '
                'recalibrate hand-eye.')
        mean_perp = float(np.mean(perp_norm))
        mean_along_abs = float(np.mean(np.abs(np.array(along_frac)) * np.array(dists)))
        if (mean_perp > 0.010 and mean_perp > 1.5 * mean_along_abs
                and _corr(dists, perp_norm) > 0.6):
            ang = float(np.degrees(np.mean(np.array(perp_norm) / np.array(dists))))
            hints.append(
                f'error ⊥ view ray growing with distance (≈{ang:.1f}° '
                'equivalent) → R_LINK6_CAM rotation error — recalibrate '
                'hand-eye with better-spread samples.')
        if not hints:
            hints.append('no single clean signature — mixed/other error '
                         '(lighting? marker flatness? try more viewpoints, '
                         'then recalibrate).')
    return dict(rows=rows, spread_m=spread, axis_range_m=axis_range,
                ref=ref, hints=[f'HINT (heuristic): {h}' for h in hints],
                verdict=verdict, ground_truth=ground_truth)


def render_report(vps: list[Viewpoint], tf_parents: dict[str, int],
                  min_n: int = MIN_SAMPLES_PER_VP) -> str:
    ids = sorted({mid for vp in vps for mid in vp.xyz})
    out = ['════ transform_audit report ════',
           f'calibration: {CALIB_SOURCE}',
           f'viewpoints captured (≥{min_n} samples): '
           f'{sum(1 for vp in vps if any(vp.n(i) >= min_n for i in ids))}'
           f' / {len(vps)} parked poses']
    if tf_parents:
        out.append(f'TF parents claiming {CAMERA_FRAME}: '
                   + ', '.join(f'{p} ({n} msgs)' for p, n in sorted(tf_parents.items())))
        if len(tf_parents) > 1:
            out.append('  ⚠ >1 parent = the known TF race. RViz may render this '
                       'frame wrong; the audit math above never uses that edge.')
    for mid in ids:
        res = analyse_marker(vps, mid, GROUND_TRUTH_XYZ.get(mid), min_n)
        out.append(f'── marker {mid} ──')
        if res is None:
            out.append('   seen from <2 settled viewpoints — park the arm at '
                       'more poses with this marker in view.')
            continue
        gt = res['ground_truth']
        out.append('   vp    n   dist   tilt        x        y        z    '
                   + ('err_vs_gt' if gt is not None else 'err_vs_mean'))
        for r in res['rows']:
            p = r['xyz']
            out.append(f'   {r["vp"]:2d} {r["n"]:4d}  {r["dist"]*100:4.0f}cm '
                       f'{r["tilt"]:5.0f}°  {p[0]:+.3f}  {p[1]:+.3f}  '
                       f'{p[2]:+.3f}    {r["err"]*100:5.1f} cm')
        if gt is not None:
            g = np.asarray(gt, float)
            out.append(f'   ground truth          {g[0]:+.3f}  {g[1]:+.3f}  {g[2]:+.3f}')
        ar = res['axis_range_m']
        out.append(f'   VERDICT: {res["verdict"]}   per-axis range '
                   f'({ar[0]*100:.1f}, {ar[1]*100:.1f}, {ar[2]*100:.1f}) cm')
        out.extend(f'   {h}' for h in res['hints'])
    out.append('════ end report ════')
    return '\n'.join(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  Node
# ═══════════════════════════════════════════════════════════════════════════════

class TransformAuditNode(EihBaseNode):

    def __init__(self):
        super().__init__('transform_audit')
        self._lock = threading.Lock()
        self.clusters = ViewpointClusters()
        self.gate = StaticGate()
        self._tf_parents: dict[str, int] = {}
        self._cam_info: CameraInfo | None = None
        self._bridge = CvBridge()
        self._detector_proc: subprocess.Popen | None = None
        self._viewer_proc: subprocess.Popen | None = None
        self._last_live: dict[int, tuple[np.ndarray, float]] = {}  # id → (xyz, t)
        self._reported_vps: set[int] = set()
        self._dropped_moving = 0

        ts = time.strftime('%Y%m%d_%H%M%S')
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_path = RESULTS_DIR / f'transform_audit_{ts}.csv'
        self.report_path = RESULTS_DIR / f'transform_audit_{ts}.txt'
        self._csv_rows: list[list] = []

        self.create_subscription(ArucoMarkerArray, '/marker_publisher/markers',
                                 self._on_markers, 10)
        self.create_subscription(Image, '/marker_publisher/result',
                                 self._on_result_image, 1)
        self._annot_pub = self.create_publisher(Image,
                                                '/transform_audit/camera_view', 1)
        self._viz_pub = self.create_publisher(MarkerArray,
                                              '/transform_audit/viz', 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self._on_cam_info, 1)
        latched = QoSProfile(depth=100,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(TFMessage, '/tf_static',
                                 self._on_tf_static, latched)
        self.create_subscription(TFMessage, '/tf', self._on_tf, 50)
        self.create_timer(0.05, self._poll_static)   # 20 Hz stillness gate
        self.create_timer(0.5, self._publish_viz)

        ok, msg = calib_gross_check(R_LINK6_CAM, CAM_T)
        self._log(f'[GATE] {msg}', 'INFO' if ok else 'ERROR')
        self.gross_ok = ok

    # ── passive listeners ─────────────────────────────────────────────────────

    def _on_cam_info(self, msg: CameraInfo):
        if self._cam_info is None:
            self._log(f'[CAMINFO] {msg.width}×{msg.height}, fx={msg.k[0]:.0f} '
                      f'— a {MARKER_SIZE_M*1000:.0f} mm marker is '
                      f'~{msg.k[0]*MARKER_SIZE_M/0.35:.0f} px across at 35 cm.')
        self._cam_info = msg

    def _tf_note(self, msg: TFMessage):
        for tr in msg.transforms:
            if tr.child_frame_id.lstrip('/') == CAMERA_FRAME:
                p = tr.header.frame_id.lstrip('/')
                with self._lock:
                    known = p in self._tf_parents
                    self._tf_parents[p] = self._tf_parents.get(p, 0) + 1
                if not known:
                    self._log(f'[TF] parent of {CAMERA_FRAME}: {p}'
                              + ('  (ours — expected)' if p == 'link6' else
                                 '  ⚠ competing parent (known race; RViz may '
                                 'render the camera frame wrong — audit math '
                                 'is immune)'),
                              'INFO' if p == 'link6' else 'WARN')

    def _on_tf_static(self, msg: TFMessage):
        self._tf_note(msg)

    def _on_tf(self, msg: TFMessage):
        self._tf_note(msg)

    def _poll_static(self):
        l6 = self.link6_in_base()
        if l6 is None:
            return
        self.gate.push(time.monotonic(), l6[0], l6[1])

    # ── sampling ──────────────────────────────────────────────────────────────

    def _on_markers(self, msg: ArucoMarkerArray):
        l6 = self.link6_in_base()
        if l6 is None:
            return
        t_b_l6, R_b_l6 = l6
        R_b_c = R_b_l6 * R_LINK6_CAM
        t_b_c = t_b_l6 + R_b_l6.apply(CAM_T)
        now = time.monotonic()
        static = self.gate.is_static
        with self._lock:
            vp = self.clusters.assign(t_b_c, R_b_c) if static else None
            for mk in msg.markers:
                if AUDIT_IDS is not None and mk.id not in AUDIT_IDS:
                    continue
                p = mk.pose.pose.position
                p_cam = np.array([p.x, p.y, p.z])
                if mk.id in TRUE_SIZE_M:
                    p_cam = p_cam * (TRUE_SIZE_M[mk.id] / MARKER_SIZE_M)
                xyz = t_b_c + R_b_c.apply(p_cam)
                self._last_live[mk.id] = (xyz, now)
                if not static:
                    self._dropped_moving += 1
                    continue
                vp.add(mk.id, xyz, float(np.linalg.norm(p_cam)))
                self._csv_rows.append(
                    [f'{now:.3f}', vp.idx, mk.id,
                     *(f'{v:.4f}' for v in xyz),
                     *(f'{v:.4f}' for v in t_b_c),
                     f'{np.linalg.norm(p_cam):.4f}', f'{vp.tilt_deg:.1f}'])
                if (vp.n(mk.id) == MIN_SAMPLES_PER_VP
                        and (vp.idx, mk.id) not in self._reported_vps):
                    self._reported_vps.add((vp.idx, mk.id))
                    m = vp.median(mk.id)
                    self._log(f'[VP {vp.idx}] marker {mk.id} captured: '
                              f'({m[0]:+.3f}, {m[1]:+.3f}, {m[2]:+.3f}) m '
                              f'from {vp.median_dist(mk.id)*100:.0f} cm, '
                              f'tilt {vp.tilt_deg:.0f}° — jog to the next pose '
                              'when ready.')
                    self._write_results()

    # ── outputs ───────────────────────────────────────────────────────────────

    def _write_results(self):
        """Called with self._lock held (and once more at shutdown)."""
        report = render_report(self.clusters.vps, dict(self._tf_parents))
        self.report_path.write_text(report + '\n')
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t_mono', 'viewpoint', 'marker_id', 'x', 'y', 'z',
                        'cam_x', 'cam_y', 'cam_z', 'cam_dist_m', 'tilt_deg'])
            w.writerows(self._csv_rows)

    def _on_result_image(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return
        now = time.monotonic()
        with self._lock:
            n_vp = len(self.clusters.vps)
            captured = len({v for v, _ in self._reported_vps})
            state = ('STATIC — sampling' if self.gate.is_static
                     else 'MOVING — hold still to sample')
            rows = [(f'transform_audit  vp {captured} captured / {n_vp} seen'
                     f'   [{state}]',
                     (255, 255, 255) if self.gate.is_static else (0, 200, 255))]
            for mid in sorted(self._last_live):
                xyz, t = self._last_live[mid]
                if now - t > 2.0:
                    continue
                spread_txt = ''
                res = analyse_marker(self.clusters.vps, mid,
                                     GROUND_TRUTH_XYZ.get(mid))
                if res is not None:
                    spread_txt = f'   spread {res["spread_m"]*100:.1f} cm'
                rows.append((f'id {mid}: ({xyz[0]:+.3f}, {xyz[1]:+.3f}, '
                             f'{xyz[2]:+.3f}) m{spread_txt}', (51, 230, 26)))
            if len(self._tf_parents) > 1:
                rows.append((f'TF race: {len(self._tf_parents)} parents claim '
                             f'{CAMERA_FRAME} (RViz suspect; audit immune)',
                             (0, 200, 255)))
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

    def _publish_viz(self):
        ms: list[Marker] = []
        palette = [(0.2, 0.4, 1.0, 0.9), (1.0, 0.8, 0.0, 0.9),
                   (0.1, 0.9, 0.2, 0.9), (1.0, 0.2, 0.2, 0.9),
                   (0.8, 0.2, 1.0, 0.9), (0.2, 0.9, 0.9, 0.9)]
        i = 0
        with self._lock:
            for vp in self.clusters.vps:
                rgba = palette[(vp.idx - 1) % len(palette)]
                for mid in vp.xyz:
                    if vp.n(mid) < MIN_SAMPLES_PER_VP:
                        continue
                    p = vp.median(mid)
                    ms.append(self.viz_sphere('audit', i, p, rgba, diam=0.015))
                    ms.append(self.viz_text('audit', i + 1, p + [0, 0, 0.02],
                                            f'{mid}@vp{vp.idx}', rgba,
                                            height=0.012))
                    i += 2
        for mid, gt in GROUND_TRUTH_XYZ.items():
            ms.append(self.viz_cube('audit_gt', mid, np.asarray(gt, float),
                                    (0.02, 0.02, 0.002), (1, 1, 1, 0.8)))
        self._viz_pub.publish(MarkerArray(markers=ms))

    # ── child processes (same pattern as pick_and_place) ──────────────────────

    def ensure_detector(self):
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.count_publishers('/marker_publisher/markers') > 0:
                self._log('[DETECT] marker_publisher already running — '
                          'VERIFY its marker_size matches '
                          f'{MARKER_SIZE_M} m (this script cannot check a '
                          'foreign detector\'s parameters).', 'WARN')
                return
            time.sleep(0.3)
        self._log('[DETECT] Starting aruco_ros marker_publisher (child)...')
        self._detector_proc = subprocess.Popen(
            ['ros2', 'run', 'aruco_ros', 'marker_publisher', '--ros-args',
             '-p', f'marker_size:={MARKER_SIZE_M}',
             '-p', f'reference_frame:={CAMERA_FRAME}',
             '-p', f'camera_frame:={CAMERA_FRAME}',
             '-r', '/camera_info:=/camera/camera/color/camera_info',
             '-r', '/image:=/camera/camera/color/image_raw'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def ensure_viewer(self):
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            self._log('[VIEW] No display — view /transform_audit/camera_view '
                      'manually.', 'WARN')
            return
        self._viewer_proc = subprocess.Popen(
            ['ros2', 'run', 'rqt_image_view', 'rqt_image_view',
             '/transform_audit/camera_view'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ── main ──────────────────────────────────────────────────────────────────

    def run(self):
        self.ensure_detector()
        self.ensure_viewer()
        self._log('[AUDIT] Read-only: this script NEVER moves the arm. '
                  'Jog via the GUI/RViz between viewpoints.')
        self._log('[AUDIT] AXIS CHECK: put a marker at a tape-measured spot, '
                  'compare the overlay readout; slide it +10 cm along base X '
                  'then Y and watch sign + magnitude follow.')
        self._log(f'[AUDIT] Hold still ~{STATIC_WINDOW_S + 0.5:.1f} s per pose; '
                  f'a viewpoint is captured at {MIN_SAMPLES_PER_VP} samples. '
                  'Aim for 5+ viewpoints incl. the close hover. Results '
                  f'rewritten to {self.report_path} after each.')
        while rclpy.ok():
            time.sleep(0.5)

    def shutdown(self):
        with self._lock:
            report = render_report(self.clusters.vps, dict(self._tf_parents))
            self._write_results()
        self._log('\n' + report)
        self._log(f'[AUDIT] Raw samples: {self.csv_path}')
        self._log(f'[AUDIT] Report:      {self.report_path} '
                  f'({self._dropped_moving} detections dropped while moving)')
        for proc in (self._detector_proc, self._viewer_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test (offline, no ROS graph, no camera)
# ═══════════════════════════════════════════════════════════════════════════════

def _lookat_R(cam_pos: np.ndarray, target: np.ndarray) -> Rotation:
    z = _unit(np.asarray(target, float) - np.asarray(cam_pos, float))
    up = np.array([1.0, 0.0, 0.0]) if abs(z[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    x = _unit(np.cross(up, z))
    y = np.cross(z, x)
    return Rotation.from_matrix(np.column_stack([x, y, z]))


def _synth_vps(true_xyz, cam_poses, err_fn, n=MIN_SAMPLES_PER_VP, mid=100):
    """Viewpoints observing true_xyz through an error model err_fn(cam_pos,
    cam_R, p_cam_true) → measured p_cam."""
    true_xyz = np.asarray(true_xyz, float)
    vps = []
    for i, cam_pos in enumerate(cam_poses):
        cam_pos = np.asarray(cam_pos, float)
        R = _lookat_R(cam_pos, true_xyz)
        vp = Viewpoint(i + 1, cam_pos, R)
        p_cam_true = R.inv().apply(true_xyz - cam_pos)
        for _ in range(n):
            p_cam = err_fn(cam_pos, R, p_cam_true)
            vp.add(mid, cam_pos + R.apply(p_cam), float(np.linalg.norm(p_cam)))
        vps.append(vp)
    return vps


def selftest() -> int:
    checks: list[tuple[str, bool]] = []
    target = np.array([0.40, 0.10, 0.00])
    cams = [np.array([0.15, -0.05, 0.35]), np.array([0.40, 0.10, 0.20]),
            np.array([0.55, 0.30, 0.30]), np.array([0.25, 0.25, 0.15]),
            np.array([0.40, -0.10, 0.40])]

    # 1–2 gross calibration gate
    ok, _ = calib_gross_check(R_LINK6_CAM, CAM_T)
    checks.append(('gross check passes real calibration', ok))
    bad = Rotation.from_euler('x', 180, degrees=True) * R_LINK6_CAM
    ok, m = calib_gross_check(bad, CAM_T)
    checks.append(('gross check catches upside-down camera', not ok and 'FAILS' in m))

    # 3–4 static gate
    g = StaticGate(window_s=0.5)
    res = [g.push(t, np.zeros(3), Rotation.identity())
           for t in np.arange(0, 1.0, 0.05)]
    checks.append(('static gate: still arm → static', res[-1] is True))
    g2 = StaticGate(window_s=0.5)
    res2 = [g2.push(t, np.array([0.02 * t, 0, 0]), Rotation.identity())
            for t in np.arange(0, 1.0, 0.05)]
    checks.append(('static gate: creeping arm → not static', res2[-1] is False))

    # 5–6 viewpoint clustering
    cl = ViewpointClusters()
    R0 = _lookat_R(cams[0], target)
    vp_a = cl.assign(cams[0], R0)
    vp_b = cl.assign(cams[0] + [0.005, 0, 0], R0)      # 5 mm → same vp
    vp_c = cl.assign(cams[0] + [0.10, 0, 0], R0)       # 10 cm → new vp
    checks.append(('clustering: nearby pose reuses viewpoint', vp_a is vp_b))
    checks.append(('clustering: distant pose makes new viewpoint', vp_c is not vp_a))

    # 7 size-override math: a 21 mm print detected as 47 mm reads 47/21×
    #   too far; the TRUE_SIZE_M correction must recover the true position
    p_true = np.array([0.01, -0.02, 0.30])
    p_meas = p_true * (0.047 / 0.021)
    checks.append(('size override recovers true position',
                   np.allclose(p_meas * (0.021 / 0.047), p_true)))

    # 8 clean chain → CONSISTENT, no hints
    vps = _synth_vps(target, cams, lambda c, R, pc: pc)
    res = analyse_marker(vps, 100)
    checks.append(('clean data → CONSISTENT, no hints',
                   res is not None and res['verdict'].startswith('CONSISTENT')
                   and not res['hints']))

    # 9 scale error → scale hint (5% size error)
    vps = _synth_vps(target, cams, lambda c, R, pc: pc * 1.05)
    res = analyse_marker(vps, 100)
    checks.append(('scale error → scale hint',
                   res is not None and any('scale-like' in h for h in res['hints'])))

    # 10 constant camera-frame offset → CAM_T hint
    dt = np.array([0.02, 0.015, 0.0])
    vps = _synth_vps(target, cams, lambda c, R, pc: pc + dt)
    res = analyse_marker(vps, 100)
    checks.append(('camera-frame offset → CAM_T hint',
                   res is not None and any('CAM_T' in h for h in res['hints'])))

    # 11 rotation error → rotation hint (4° about camera x)
    R_err = Rotation.from_euler('x', 4, degrees=True)
    vps = _synth_vps(target, cams, lambda c, R, pc: R_err.apply(pc))
    res = analyse_marker(vps, 100)
    checks.append(('rotation error → rotation hint',
                   res is not None and any('rotation' in h.lower() for h in res['hints'])))

    # 12 ground truth wiring: err measured against GT, not the mean
    vps = _synth_vps(target, cams, lambda c, R, pc: pc)
    res = analyse_marker(vps, 100, ground_truth=target + [0.05, 0, 0])
    checks.append(('ground truth shifts the error reference',
                   res is not None and all(abs(r['err'] - 0.05) < 0.005
                                           for r in res['rows'])))

    # 13 <2 viewpoints → None
    vps = _synth_vps(target, cams[:1], lambda c, R, pc: pc)
    checks.append(('single viewpoint → no verdict', analyse_marker(vps, 100) is None))

    # 14 report renders
    vps = _synth_vps(target, cams, lambda c, R, pc: pc * 1.05)
    rep = render_report(vps, {'link6': 10, 'camera_color_frame': 3})
    checks.append(('report renders with race warning',
                   'marker 100' in rep and '>1 parent' in rep))

    n_ok = sum(ok for _, ok in checks)
    for name, ok in checks:
        print(f'  {"PASS" if ok else "FAIL"}  {name}')
    print(f'--selftest: {n_ok}/{len(checks)} PASS')
    return 0 if n_ok == len(checks) else 1


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if '--selftest' in sys.argv:
        sys.exit(selftest())
    rclpy.init()
    node = TransformAuditNode()

    def _sigterm(*_):   # GUI "Stop Script" sends SIGTERM — still write results
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
