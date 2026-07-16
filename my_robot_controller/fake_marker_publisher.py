#!/usr/bin/env python3
"""
fake_marker_publisher — offline stand-in for aruco_ros marker_publisher.

Publishes synthetic ArUco detections on /marker_publisher/markers so that
pick_and_place.py can run end-to-end against MoveIt MOCK hardware in RViz
with no camera and no arm. TESTING ONLY — nothing in the Monday hardware
stack imports or depends on this file, and pick_and_place.py is unmodified.

HOW IT WORKS
  A virtual world is defined below in base_link coordinates: stations A/B
  flat on the table, object cube with marker C on its top face. Every tick
  the node reads the (mock) base_link→link6 TF, composes the SAME
  calibration chain pick_and_place uses (eih_core CAM_T / R_LINK6_CAM), and
  publishes each marker's pose in the camera optical frame — the exact
  inverse of pick_and_place._on_markers. So the script's base-frame
  estimates reproduce this virtual world, plus the simulated noise (which
  exercises the median filter).

  The gripper is simulated too: the node listens on control/joint_states
  for 'gripper' width commands (the same dispatch the real driver uses).
  Closing near the object → the cube and marker C ride along with the TCP;
  opening → the cube drops back to the table under the TCP. The full
  pick → place → VERIFY cycle therefore behaves like reality.

RUN (order matters — start this BEFORE pick_and_place, so its
     ensure_detector() sees a publisher and does not spawn aruco_ros):
  1. GUI: launch MoveIt2 with Sim checked (mock hardware) + RViz.
  2. GUI Custom Script Runner (or terminal): this file.
  3. GUI Custom Script Runner: pick_and_place.py — EXECUTE=False first;
     EXECUTE=True is also safe in this setup (the "arm" is mock).
  RViz: add a MarkerArray display on /fake_markers/viz for the ground
  truth (green) and compare with /pick_place/viz (the script's estimates).

TUNE: if the IK sweep reports poses unreachable, nudge STATION_*_XY /
OBJECT_XY closer to the base and re-run. Constants marked (sync) must
match pick_and_place.py.
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener

from aruco_msgs.msg import Marker as ArucoMarker
from aruco_msgs.msg import MarkerArray as ArucoMarkerArray
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray

from my_robot_controller.eih_core import CAM_T, R_LINK6_CAM, EFFECTOR_LINK


# ═══════════════════════════════════════════════════════════════════════════════
#  VIRTUAL WORLD (base_link coordinates)
# ═══════════════════════════════════════════════════════════════════════════════

MARKER_A_ID   = 100      # (sync) pick station
MARKER_B_ID   = 101      # (sync) place station
MARKER_C_ID   = 582      # (sync) on top of the object
OBJECT_DIMS_M = (0.03, 0.03, 0.03)   # (sync) cube x, y, z (measured 2026-07-16)
TCP_OFFSET_Z  = 0.175    # (sync) used only to simulate the grasp attach point
MARKER_SIZE_M = 0.047    # (sync) the ONE size the detector assumes
# (sync) per-id TRUE printed size. A real detector assuming MARKER_SIZE_M for
# a marker actually printed at TRUE_SIZE_M[id] reports its position scaled by
# MARKER_SIZE_M / TRUE_SIZE_M[id]; this publisher simulates that mis-scale so
# pick_and_place's TRUE_SIZE_M correction is exercised by the mock test.
TRUE_SIZE_M: dict[int, float] = {MARKER_C_ID: 0.021}

TABLE_Z      = 0.0                 # markers lie flat at this height
STATION_A_XY = (0.26, -0.02)       # directly under the object — realistic:
                                   # the cube sits ON station A and covers it
                                   # (pick_and_place no longer needs A at all)
STATION_B_XY = (0.24, +0.10)       # place target — must be inside the
                                   # straight-down envelope at pre-place
                                   # height (probe 2026-07-16: x ≤ 0.26)
OBJECT_XY    = (0.26, -0.02)       # on station A; r≈0.26 — inside the
                                   # straight-down IK envelope (r≈0.31+ has
                                   # no vertical IK)
OBJECT_YAW_DEG = 20.0              # non-zero to exercise roll alignment

NOISE_POS_STD_M   = 0.0015         # per-axis gaussian on detections
NOISE_YAW_STD_DEG = 0.8
PUBLISH_HZ        = 10.0

# Camera visibility. ALWAYS_VISIBLE=True publishes every marker regardless
# (so the SCAN stage cannot dead-lock from an unlucky start pose) but still
# LOGS what a real D435 would/wouldn't see, so FOV issues stay visible.
ALWAYS_VISIBLE = True
FOV_H_DEG, FOV_V_DEG = 69.4, 42.5  # RealSense D435 colour FOV
FOV_MIN_Z, FOV_MAX_Z = 0.07, 1.5   # usable detection distance band

GRASP_ATTACH_RADIUS_M = 0.035      # TCP within this of object centre → grab
GRIP_CLOSE_W = OBJECT_DIMS_M[0] + 0.002   # width below this counts as closed
GRIP_OPEN_W  = OBJECT_DIMS_M[0] + 0.010   # width above this counts as open


class FakeMarkerPublisher(Node):

    def __init__(self):
        super().__init__('fake_marker_publisher')
        self._tf  = Buffer(node=self)
        self._tfl = TransformListener(self._tf, self, spin_thread=False)

        self._pub     = self.create_publisher(ArucoMarkerArray,
                                              '/marker_publisher/markers', 10)
        self._viz_pub = self.create_publisher(MarkerArray, '/fake_markers/viz', 10)
        self.create_subscription(JointState, 'control/joint_states',
                                 self._on_control_joint_state, 10)
        # Mock stand-in for the arm driver's /control_enable gate (the real
        # SetBool service only exists when agx_arm_ctrl is up, so without this
        # every EXECUTE run aborts at the pre-motion gate in the mock stack).
        self._gate_srv = self.create_service(SetBool, '/control_enable',
                                             self._on_control_enable)

        self._lock = threading.Lock()
        # object state (marker C sits on the top face)
        self._obj_centre = np.array([OBJECT_XY[0], OBJECT_XY[1],
                                     TABLE_Z + OBJECT_DIMS_M[2] / 2])
        self._obj_yaw_deg = OBJECT_YAW_DEG
        self._attached = False
        self._attach_yaw_offset = 0.0
        self._gripper_width = None
        self._visible_last: set[int] = set()
        self._warned_no_tf = False

        self.create_timer(1.0 / PUBLISH_HZ, self._tick)
        self._log(f'Virtual world: A={STATION_A_XY} B={STATION_B_XY} '
                  f'object={OBJECT_XY} yaw={OBJECT_YAW_DEG:.0f}° table_z={TABLE_Z}')
        self._log(f'Publishing /marker_publisher/markers at {PUBLISH_HZ:.0f} Hz '
                  f'(noise σ={NOISE_POS_STD_M*1000:.1f} mm, '
                  f'always_visible={ALWAYS_VISIBLE}). Ground truth: /fake_markers/viz')

    def _log(self, msg: str, level: str = 'INFO'):
        ts = time.strftime('%H:%M:%S')
        print(f'[{ts}] {level}: {msg}', flush=True)

    def _on_control_enable(self, request, response):
        self._log(f'[GATE] /control_enable(data={request.data}) → mock OK')
        response.success = True
        response.message = 'mock gate (fake_marker_publisher)'
        return response

    # ── Arm / camera pose ─────────────────────────────────────────────────────

    def _link6_in_base(self):
        try:
            tf = self._tf.lookup_transform('base_link', EFFECTOR_LINK,
                                           Time(), Duration(seconds=0.1))
        except Exception:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return (np.array([t.x, t.y, t.z]),
                Rotation.from_quat([q.x, q.y, q.z, q.w]))

    # ── Simulated gripper (control/joint_states 'gripper' dispatch) ───────────

    def _on_control_joint_state(self, msg: JointState):
        if 'gripper' not in msg.name:
            return
        width = float(msg.position[msg.name.index('gripper')])
        self._gripper_width = width
        l6 = self._link6_in_base()
        with self._lock:
            if not self._attached and width <= GRIP_CLOSE_W:
                if l6 is None:
                    return
                t_b_l6, R_b_l6 = l6
                tcp = t_b_l6 + R_b_l6.apply([0.0, 0.0, TCP_OFFSET_Z])
                dist = float(np.linalg.norm(tcp - self._obj_centre))
                if dist <= GRASP_ATTACH_RADIUS_M:
                    self._attached = True
                    link6_yaw = float(R_b_l6.as_euler('zyx', degrees=True)[0])
                    self._attach_yaw_offset = self._obj_yaw_deg - link6_yaw
                    self._log(f'[GRIP] width {width:.3f} m, TCP {dist*100:.1f} cm '
                              'from object → GRABBED (object now follows TCP)')
                else:
                    self._log(f'[GRIP] width {width:.3f} m but TCP is '
                              f'{dist*100:.1f} cm from object → grabbed air', 'WARN')
            elif self._attached and width >= GRIP_OPEN_W:
                self._attached = False
                self._obj_centre[2] = TABLE_Z + OBJECT_DIMS_M[2] / 2
                self._log(f'[GRIP] width {width:.3f} m → RELEASED (object dropped '
                          f'to table at ({self._obj_centre[0]:+.3f}, '
                          f'{self._obj_centre[1]:+.3f}))')

    def _update_attached_object(self, t_b_l6, R_b_l6):
        tcp = t_b_l6 + R_b_l6.apply([0.0, 0.0, TCP_OFFSET_Z])
        self._obj_centre = tcp
        link6_yaw = float(R_b_l6.as_euler('zyx', degrees=True)[0])
        self._obj_yaw_deg = link6_yaw + self._attach_yaw_offset

    # ── Publishing ────────────────────────────────────────────────────────────

    def _world_markers(self):
        """(id, xyz_base, yaw_deg) of every virtual marker, current state."""
        c_pos = self._obj_centre + np.array([0, 0, OBJECT_DIMS_M[2] / 2])
        return [
            (MARKER_A_ID, np.array([*STATION_A_XY, TABLE_Z]), 0.0),
            (MARKER_B_ID, np.array([*STATION_B_XY, TABLE_Z]), 0.0),
            (MARKER_C_ID, c_pos, self._obj_yaw_deg),
        ]

    def _in_fov(self, p_cam: np.ndarray) -> bool:
        if not (FOV_MIN_Z < p_cam[2] < FOV_MAX_Z):
            return False
        return (abs(np.degrees(np.arctan2(p_cam[0], p_cam[2]))) < FOV_H_DEG / 2
                and abs(np.degrees(np.arctan2(p_cam[1], p_cam[2]))) < FOV_V_DEG / 2)

    def _tick(self):
        l6 = self._link6_in_base()
        if l6 is None:
            if not self._warned_no_tf:
                self._log('base_link→link6 TF not available yet — is MoveIt '
                          '(mock) running? Will keep retrying quietly.', 'WARN')
                self._warned_no_tf = True
            return
        self._warned_no_tf = False
        t_b_l6, R_b_l6 = l6
        R_b_c = R_b_l6 * R_LINK6_CAM
        t_b_c = t_b_l6 + R_b_l6.apply(CAM_T)

        with self._lock:
            if self._attached:
                self._update_attached_object(t_b_l6, R_b_l6)
            world = self._world_markers()

        msg = ArucoMarkerArray()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.header.stamp = self.get_clock().now().to_msg()

        visible_now = set()
        for mid, xyz_b, yaw_deg in world:
            p_cam = R_b_c.inv().apply(xyz_b - t_b_c)
            if self._in_fov(p_cam):
                visible_now.add(mid)
            elif not ALWAYS_VISIBLE:
                continue
            R_b_m = Rotation.from_euler(
                'z', yaw_deg + np.random.normal(0.0, NOISE_YAW_STD_DEG),
                degrees=True)
            q_cm = (R_b_c.inv() * R_b_m).as_quat()
            if mid in TRUE_SIZE_M:
                p_cam = p_cam * (MARKER_SIZE_M / TRUE_SIZE_M[mid])
            p_noisy = p_cam + np.random.normal(0.0, NOISE_POS_STD_M, 3)

            mk = ArucoMarker()
            mk.header = msg.header
            mk.id = int(mid)
            mk.confidence = 1.0
            mk.pose.pose.position.x = float(p_noisy[0])
            mk.pose.pose.position.y = float(p_noisy[1])
            mk.pose.pose.position.z = float(p_noisy[2])
            mk.pose.pose.orientation.x = float(q_cm[0])
            mk.pose.pose.orientation.y = float(q_cm[1])
            mk.pose.pose.orientation.z = float(q_cm[2])
            mk.pose.pose.orientation.w = float(q_cm[3])
            msg.markers.append(mk)

        if visible_now != self._visible_last:
            gained = sorted(visible_now - self._visible_last)
            lost = sorted(self._visible_last - visible_now)
            note = ' (published anyway)' if ALWAYS_VISIBLE and lost else ''
            self._log(f'[FOV] real D435 would see {sorted(visible_now)} '
                      f'(+{gained} -{lost}){note}')
            self._visible_last = visible_now

        self._pub.publish(msg)
        self._publish_viz(world)

    # ── Ground-truth RViz overlay ─────────────────────────────────────────────

    def _publish_viz(self, world):
        ms = []
        stamp = self.get_clock().now().to_msg()

        def base_marker(mid, mtype):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp = stamp
            m.ns = 'truth'; m.id = mid
            m.type = mtype; m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            return m

        for i, (mid, xyz, yaw) in enumerate(world):
            sq = base_marker(i * 2, Marker.CUBE)
            sq.pose.position.x, sq.pose.position.y = float(xyz[0]), float(xyz[1])
            sq.pose.position.z = float(xyz[2]) + 0.001
            q = Rotation.from_euler('z', yaw, degrees=True).as_quat()
            (sq.pose.orientation.x, sq.pose.orientation.y,
             sq.pose.orientation.z, sq.pose.orientation.w) = [float(v) for v in q]
            sq.scale.x = sq.scale.y = 0.047; sq.scale.z = 0.001
            sq.color.r, sq.color.g, sq.color.b, sq.color.a = 0.0, 0.8, 0.1, 0.9
            ms.append(sq)
            tx = base_marker(i * 2 + 1, Marker.TEXT_VIEW_FACING)
            tx.pose.position.x, tx.pose.position.y = float(xyz[0]), float(xyz[1])
            tx.pose.position.z = float(xyz[2]) + 0.03
            tx.scale.z = 0.02
            tx.color.r = tx.color.g = tx.color.b = 1.0; tx.color.a = 0.9
            tx.text = f'truth {mid}'
            ms.append(tx)

        with self._lock:
            obj = self._obj_centre.copy()
            attached = self._attached
        cube = base_marker(100, Marker.CUBE)
        cube.pose.position.x, cube.pose.position.y, cube.pose.position.z = \
            [float(v) for v in obj]
        cube.scale.x, cube.scale.y, cube.scale.z = OBJECT_DIMS_M
        cube.color.r, cube.color.g, cube.color.b, cube.color.a = \
            (1.0, 0.3, 0.0, 0.9) if attached else (0.0, 0.6, 0.9, 0.9)
        ms.append(cube)

        self._viz_pub.publish(MarkerArray(markers=ms))


def main():
    rclpy.init()
    node = FakeMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
