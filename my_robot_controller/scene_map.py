#!/usr/bin/env python3
"""
scene_map — Stage 2 of the thesis stack (see ~/sessions/THESIS_ROADMAP.md).

Turns the top-down phone camera into a global marker map in base_link
coordinates, published for the skill server to consume as a fallback when
the wrist camera can't see a target.

KEY DESIGN — homography, not PnP: every marker lies on the table plane, so
pixel→base is a single 3×3 homography. No phone intrinsics, no checkerboard.
Each detected marker contributes its 4 corners as correspondences, so the
three thesis markers alone (12 points) over-determine the fit.

CALIBRATION (one command, arm + phone both watching the table):
  ros2 topic pub --once /scene/command std_msgs/msg/String "{data: calibrate}"
  For every marker visible to BOTH cameras at that moment, the wrist
  camera's base-frame pose (via MarkerTracker) + marker size give 4 base-
  frame corners; the phone frame gives the 4 pixel corners. findHomography
  (RANSAC) fits the map, residuals are logged LOUDLY, and the result is
  saved to ~/.ros2/topdown_map.yaml (auto-loaded on startup). Re-calibrate
  whenever the phone moves.

OUTPUT
  /scene/markers (std_msgs/String, JSON, ~2 Hz):
    {"stamp": <epoch>, "table_z": <m>, "markers":
        {"582": {"x":.., "y":.., "z":.., "yaw_deg":.., "px":[u,v]}, ...}}
  /scene/viz (MarkerArray): flat discs + ids at mapped positions in RViz.

RUN
  python3 scene_map.py http://<PHONE_IP>:8080/video     (needs colcon build
  after edits — this file imports from the installed package)
  python3 scene_map.py --selftest        # no phone, no arm: synthetic camera

This node is a read-only observer: it never commands motion. The skill
server decides what to do with the map.
"""

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

MAP_FILE = Path.home() / '.ros2' / 'topdown_map.yaml'
STREAM_URL = 'http://192.168.1.42:8080/video'
MARKER_SIZE_M = 0.047          # same size for all printed markers
PUBLISH_HZ = 2.0
DETECT_HZ = 5.0                # top-down markers don't move fast

# cv2.aruco corner order (TL,TR,BR,BL of the pattern) in the marker's own
# frame, x right / y up. The wrist side must generate corners in the SAME
# order for correspondences to line up; RANSAC + the residual report will
# expose it loudly if this convention is ever wrong.
_CORNER_OFFSETS = np.array([[-0.5, +0.5], [+0.5, +0.5],
                            [+0.5, -0.5], [-0.5, -0.5]])


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure mapper (no ROS — fully covered by the self-test)
# ═══════════════════════════════════════════════════════════════════════════════

class TableMap:
    """Planar pixel→base_link map: fit from correspondences, apply, persist."""

    def __init__(self):
        self.H = None            # 3×3 homography, pixel → base xy
        self.table_z = 0.0
        self.rms_m = None        # fit residual (metres, on the base plane)

    # ── fitting ────────────────────────────────────────────────────────────

    @staticmethod
    def base_corners(centre_xy, yaw_deg, size_m=MARKER_SIZE_M) -> np.ndarray:
        """4 marker corners (TL,TR,BR,BL) in base xy from centre + yaw."""
        a = np.radians(yaw_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return np.asarray(centre_xy)[None, :] + (_CORNER_OFFSETS * size_m) @ R.T

    def fit(self, px_pts: np.ndarray, base_pts: np.ndarray,
            table_z: float) -> float:
        """Fit pixel→base homography from N≥4 correspondences.
        Returns RMS residual in metres (also stored)."""
        px = np.asarray(px_pts, dtype=np.float64).reshape(-1, 2)
        ba = np.asarray(base_pts, dtype=np.float64).reshape(-1, 2)
        if len(px) < 4:
            raise ValueError(f'need ≥4 correspondences, got {len(px)}')
        H, mask = cv2.findHomography(px, ba, cv2.RANSAC, 0.01)
        if H is None:
            raise ValueError('findHomography failed')
        self.H = H
        self.table_z = float(table_z)
        mapped = self.px_to_base_many(px)
        self.rms_m = float(np.sqrt(np.mean(np.sum((mapped - ba) ** 2, axis=1))))
        return self.rms_m

    # ── applying ───────────────────────────────────────────────────────────

    def px_to_base_many(self, px: np.ndarray) -> np.ndarray:
        pts = np.asarray(px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def px_to_base(self, u: float, v: float) -> np.ndarray:
        x, y = self.px_to_base_many(np.array([[u, v]]))[0]
        return np.array([x, y, self.table_z])

    def yaw_from_px_corners(self, corners_px: np.ndarray) -> float:
        """Marker yaw in base frame: map TL and TR corners, the TL→TR edge
        is the marker's +x axis."""
        tl, tr = self.px_to_base_many(corners_px[:2])
        d = tr - tl
        return float(np.degrees(np.arctan2(d[1], d[0])))

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, path: Path = MAP_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({
            'homography': self.H.tolist(),
            'table_z': self.table_z,
            'rms_m': self.rms_m,
            'saved': time.strftime('%Y-%m-%d %H:%M:%S'),
        }))

    def load(self, path: Path = MAP_FILE) -> bool:
        try:
            d = yaml.safe_load(path.read_text())
            self.H = np.array(d['homography'], dtype=np.float64)
            self.table_z = float(d['table_z'])
            self.rms_m = d.get('rms_m')
            return True
        except Exception:
            return False

    @property
    def ready(self) -> bool:
        return self.H is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  ROS node
# ═══════════════════════════════════════════════════════════════════════════════

def _make_node(stream_url: str):
    """Node factory (import-inside so --selftest runs without ROS installed
    in the environment doing the testing)."""
    import rclpy
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray

    from my_robot_controller.eih_core import EihBaseNode, MarkerTracker
    from my_robot_controller.topdown_viewer import detect, open_stream

    class SceneMapNode(EihBaseNode):
        def __init__(self):
            super().__init__('scene_map')
            self.map = TableMap()
            if self.map.load():
                self._log(f'[MAP] Loaded {MAP_FILE} '
                          f'(rms {self.map.rms_m if self.map.rms_m is None else round(self.map.rms_m*1000,1)} mm)')
            else:
                self._log('[MAP] No calibration yet — publish "calibrate" on '
                          '/scene/command once both cameras see the markers.', 'WARN')

            # wrist-side tracker: provides base-frame truth for calibration
            self.tracker = MarkerTracker(self)

            self._latest_px: dict[int, np.ndarray] = {}   # id → 4×2 px corners
            self._px_lock = threading.Lock()
            self._px_stamp = 0.0

            self.create_subscription(String, '/scene/command', self._on_cmd, 10)
            self._pub = self.create_publisher(String, '/scene/markers', 10)
            self._viz = self.create_publisher(MarkerArray, '/scene/viz', 10)
            self.create_timer(1.0 / PUBLISH_HZ, self._publish)

            self._cap_thread = threading.Thread(
                target=self._capture_loop, args=(stream_url,), daemon=True)
            self._cap_thread.start()

        # ── phone capture ────────────────────────────────────────────────

        def _capture_loop(self, url):
            cap = None
            while True:
                if cap is None:
                    cap = open_stream(url)
                    if cap is None:
                        time.sleep(2.0)
                        continue
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    cap = None
                    continue
                found, corners, ids = detect(frame)
                with self._px_lock:
                    self._latest_px = {}
                    if ids is not None:
                        for mid, c in zip(ids.flatten(), corners):
                            self._latest_px[int(mid)] = c[0].copy()
                    self._px_stamp = time.monotonic()
                time.sleep(1.0 / DETECT_HZ)

        # ── calibration ──────────────────────────────────────────────────

        def _on_cmd(self, msg):
            if msg.data.strip().lower() != 'calibrate':
                return
            with self._px_lock:
                px_now = dict(self._latest_px)
            px_pts, base_pts, zs, used = [], [], [], []
            for mid, corners in px_now.items():
                mk = self.tracker.get(mid)
                if mk is None:
                    continue
                p = mk.position
                px_pts.append(corners)
                base_pts.append(TableMap.base_corners(p[:2], mk.yaw_deg))
                zs.append(p[2])
                used.append(mid)
            if len(used) < 1 or sum(len(c) for c in px_pts) < 4:
                self._log(f'[MAP] Calibration failed: only {used} visible to '
                          'BOTH cameras — need at least 1 marker (4 corners); '
                          'more markers = better fit.', 'ERROR')
                return
            rms = self.map.fit(np.vstack(px_pts), np.vstack(base_pts),
                               float(np.mean(zs)))
            self.map.save()
            lvl = 'INFO' if rms < 0.01 else 'WARN'
            self._log(f'[MAP] Calibrated from markers {used}: '
                      f'rms {rms*1000:.1f} mm '
                      f'{"(GOOD)" if rms < 0.01 else "(HIGH — corner-order or wrist-pose problem?)"} '
                      f'→ saved {MAP_FILE}', lvl)

        # ── output ───────────────────────────────────────────────────────

        def _publish(self):
            from std_msgs.msg import String
            from visualization_msgs.msg import MarkerArray
            if not self.map.ready:
                return
            with self._px_lock:
                px_now = dict(self._latest_px)
                age = time.monotonic() - self._px_stamp
            if age > 3.0:
                return                      # phone stream stale — publish nothing
            out = {}
            ms = []
            i = 0
            for mid, corners in sorted(px_now.items()):
                centre_px = corners.mean(axis=0)
                pos = self.map.px_to_base(*centre_px)
                yaw = self.map.yaw_from_px_corners(corners)
                out[str(mid)] = {'x': round(float(pos[0]), 4),
                                 'y': round(float(pos[1]), 4),
                                 'z': round(float(pos[2]), 4),
                                 'yaw_deg': round(yaw, 1),
                                 'px': [round(float(centre_px[0]), 1),
                                        round(float(centre_px[1]), 1)]}
                ms.append(self.viz_cube('scene', i, pos,
                                        (MARKER_SIZE_M, MARKER_SIZE_M, 0.002),
                                        (1.0, 0.3, 0.8, 0.7)))
                ms.append(self.viz_text('scene', i + 1,
                                        pos + np.array([0, 0, 0.03]),
                                        f'td:{mid}', (1.0, 0.3, 0.8, 0.9),
                                        height=0.02))
                i += 2
            self._pub.publish(String(data=json.dumps(
                {'stamp': time.time(), 'table_z': self.map.table_z,
                 'markers': out})))
            self._viz.publish(MarkerArray(markers=ms))

    return SceneMapNode()


def main():
    if '--selftest' in sys.argv:
        sys.exit(_selftest())
    import rclpy
    rclpy.init()
    url = next((a for a in sys.argv[1:] if not a.startswith('-')), STREAM_URL)
    node = _make_node(url)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test: synthetic top-down camera, no phone / arm / ROS graph
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> int:
    fails = 0

    def check(name, cond, detail=''):
        nonlocal fails
        print(f'  {"PASS" if cond else "FAIL"}  {name}  {detail}')
        fails += 0 if cond else 1

    # Ground truth: a plausible oblique phone view of the table plane.
    # base (x,y) → pixel via a projective map (rotation+tilt+scale+offset).
    H_true = np.array([[900.0, -120.0, 300.0],
                       [80.0,  -950.0, 620.0],
                       [0.05,   -0.10, 1.0]])

    def base_to_px(xy):
        p = H_true @ np.array([xy[0], xy[1], 1.0])
        return p[:2] / p[2]

    table_z = 0.021
    truth = {100: ((0.30, -0.12), 0.0),
             101: ((0.30, +0.12), 15.0),
             582: ((0.42,  0.00), -40.0)}

    print('1) calibration fit from the 3 thesis markers (12 corners)')
    px_pts, base_pts = [], []
    for mid, (centre, yaw) in truth.items():
        bc = TableMap.base_corners(centre, yaw)
        base_pts.append(bc)
        px_pts.append(np.array([base_to_px(c) for c in bc]))
    m = TableMap()
    rms = m.fit(np.vstack(px_pts), np.vstack(base_pts), table_z)
    check('fit rms < 1 mm', rms < 0.001, f'rms={rms*1000:.3f} mm')

    print('2) held-out point accuracy')
    worst = 0.0
    rng = np.random.default_rng(42)
    for _ in range(50):
        xy = rng.uniform([0.15, -0.30], [0.50, 0.30])
        est = m.px_to_base(*base_to_px(xy))
        worst = max(worst, float(np.hypot(*(est[:2] - xy))))
    check('50 random table points within 2 mm', worst < 0.002,
          f'worst={worst*1000:.2f} mm')
    check('table z propagated', abs(m.px_to_base(400, 300)[2] - table_z) < 1e-9)

    print('3) yaw recovery through the perspective map')
    worst_yaw = 0.0
    for mid, (centre, yaw) in truth.items():
        bc = TableMap.base_corners(centre, yaw)
        px_corners = np.array([base_to_px(c) for c in bc])
        est = m.yaw_from_px_corners(px_corners)
        err = abs((est - yaw + 180) % 360 - 180)
        worst_yaw = max(worst_yaw, err)
    check('yaw within 0.5° for all markers', worst_yaw < 0.5,
          f'worst={worst_yaw:.3f}°')

    print('4) persistence round-trip')
    tmp = Path('/tmp/claude-1000') if Path('/tmp/claude-1000').exists() else Path('/tmp')
    f = tmp / 'topdown_map_test.yaml'
    m.save(f)
    m2 = TableMap()
    check('load succeeds', m2.load(f))
    est1 = m.px_to_base(512, 384)
    est2 = m2.px_to_base(512, 384)
    check('loaded map identical', np.allclose(est1, est2, atol=1e-12))
    f.unlink()

    print('5) degenerate input rejected')
    try:
        TableMap().fit(np.zeros((3, 2)), np.zeros((3, 2)), 0.0)
        check('<4 points raises', False)
    except ValueError:
        check('<4 points raises', True)

    print(f'\n{"ALL PASS" if fails == 0 else f"{fails} FAILURES"}')
    return 1 if fails else 0


if __name__ == '__main__':
    main()
