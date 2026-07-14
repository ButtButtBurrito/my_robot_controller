#!/usr/bin/env python3
"""
Eye-in-hand calibration verification and practice script.

What it does:
  1. Publishes the calibrated link6 → camera_color_optical_frame TF
  2. Prints the ArUco marker position in base_link frame every second
  3. Press Enter: plan above the marker, open the arm control gate,
     then immediately execute the trajectory.

Prerequisites:
  - Arm Controller + MoveIt2 + RealSense + ArUco detector
"""

import threading
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time, Duration
from tf2_ros import StaticTransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, MoveItErrorCodes,
)
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import SetBool
import tf2_geometry_msgs  # noqa: F401

# ── Calibration from piper_eih.calib ─────────────────────────────────────────
CAMERA_TF = dict(
    x  = -0.05593476016675718,
    y  = -0.031925813829647605,
    z  =  0.044513090224353104,
    qx = -0.10433398988297468,
    qy =  0.08134981927603824,
    qz = -0.5141169607716367,
    qw =  0.8474552354583642,
)

_qx, _qy, _qz, _qw = CAMERA_TF['qx'], CAMERA_TF['qy'], CAMERA_TF['qz'], CAMERA_TF['qw']
# Camera optical-Z direction expressed in link6 frame (derived from calibration rotation)
CAM_Z_IN_LINK6 = np.array([
    2*(_qx*_qz + _qy*_qw),
    2*(_qy*_qz - _qx*_qw),
    1 - 2*(_qx*_qx + _qy*_qy),
])
# Camera origin offset from link6 origin in link6 frame (calibration translation)
CAM_ORIGIN_IN_LINK6 = np.array([CAMERA_TF['x'], CAMERA_TF['y'], CAMERA_TF['z']])

MOVE_GROUP_NAME     = 'arm'
EFFECTOR_LINK       = 'tcp_link'
APPROACH_HEIGHT     = 0.30
MINIMUM_SAFE_Z      = 0.25
MARKER_CACHE_EXPIRY = 5.0


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _qmul(q1, q2):
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return np.array([
        w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2])

def _qrot(q, v):
    x,y,z,w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
    return R @ np.array(v)

def _lookat_quat(cur_q, cam_z_link6, look_dir):
    cur_look = _qrot(cur_q, cam_z_link6); cur_look /= np.linalg.norm(cur_look)
    target   = np.array(look_dir, dtype=float); target /= np.linalg.norm(target)
    cross = np.cross(cur_look, target); dot = float(np.dot(cur_look, target))
    n = np.linalg.norm(cross)
    if n < 1e-6:
        delta = np.array([0.,0.,0.,1.]) if dot > 0 else np.array([1.,0.,0.,0.])
    else:
        ax = cross/n; angle = np.arctan2(n, dot)
        s = np.sin(angle/2); c = np.cos(angle/2)
        delta = np.array([ax[0]*s, ax[1]*s, ax[2]*s, c])
    r = _qmul(delta, cur_q); return r / np.linalg.norm(r)


def _call_service(node, cli, data=True, timeout=3.0):
    """Call a SetBool service safely from a thread while the node is spinning."""
    if not cli.wait_for_service(timeout_sec=timeout):
        node.get_logger().warn(f'Service not available: {cli.srv_name}')
        return False
    future = cli.call_async(SetBool.Request(data=data))
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout=timeout):
        node.get_logger().warn(f'Service call timed out: {cli.srv_name}')
        return False
    result = future.result()
    node.get_logger().info(f'{cli.srv_name}: {result.message}')
    return result.success


class EyeInHandPractice(Node):

    def __init__(self):
        super().__init__('eye_in_hand_practice')

        self._cam_broadcaster = StaticTransformBroadcaster(self)
        self._publish_camera_tf()

        # Buffer needs node= for ROS clock access; spin_thread=False avoids
        # conflicting with our manual rclpy.spin thread in main()
        self._tf_buffer   = Buffer(node=self)
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

        self._plan_client    = ActionClient(self, MoveGroup,          '/move_action')
        self._execute_client = ActionClient(self, ExecuteTrajectory,  '/execute_trajectory')
        self._enable_cli     = self.create_client(SetBool, '/enable_agx_arm')
        self._gate_cli       = self.create_client(SetBool, '/control_enable')

        self.create_subscription(PoseStamped, '/aruco_single/pose', lambda _: None, 10)

        self._last_marker: tuple | None = None
        self._pending: dict = {}
        self._busy: bool = False   # prevents overlapping move sequences

        self.create_timer(1.0, self._report_marker)
        self.get_logger().info(
            f'CAM_Z_IN_LINK6     = [{CAM_Z_IN_LINK6[0]:+.3f}, {CAM_Z_IN_LINK6[1]:+.3f}, {CAM_Z_IN_LINK6[2]:+.3f}]')
        self.get_logger().info(
            f'CAM_ORIGIN_IN_LINK6= [{CAM_ORIGIN_IN_LINK6[0]:+.3f}, {CAM_ORIGIN_IN_LINK6[1]:+.3f}, {CAM_ORIGIN_IN_LINK6[2]:+.3f}]')
        self.get_logger().info('Ready.')

    # ── Camera TF ──────────────────────────────────────────────────────────────

    def _publish_camera_tf(self):
        t = TransformStamped()
        t.header.stamp            = self.get_clock().now().to_msg()
        t.header.frame_id         = 'link6'
        t.child_frame_id          = 'camera_color_optical_frame'
        t.transform.translation.x = CAMERA_TF['x']
        t.transform.translation.y = CAMERA_TF['y']
        t.transform.translation.z = CAMERA_TF['z']
        t.transform.rotation.x    = CAMERA_TF['qx']
        t.transform.rotation.y    = CAMERA_TF['qy']
        t.transform.rotation.z    = CAMERA_TF['qz']
        t.transform.rotation.w    = CAMERA_TF['qw']
        self._cam_broadcaster.sendTransform(t)

    # ── Marker ─────────────────────────────────────────────────────────────────

    def _report_marker(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                'base_link', 'marker_frame', Time(), Duration(seconds=0.5))
            t = tf.transform.translation
            r = tf.transform.rotation
            self._last_marker = (t.x, t.y, t.z, time.monotonic())
            self.get_logger().info(
                f'Marker → x={t.x:+.3f} y={t.y:+.3f} z={t.z:+.3f}  '
                f'qx={r.x:+.3f} qy={r.y:+.3f} qz={r.z:+.3f} qw={r.w:+.3f}')
        except Exception:
            age = f'  [cached {time.monotonic()-self._last_marker[3]:.1f}s ago]' \
                  if self._last_marker else ''
            self.get_logger().warn(f'Marker not visible{age}')

    # ── Move ───────────────────────────────────────────────────────────────────

    def move_to_marker(self):
        if self._busy:
            self.get_logger().warn('Already moving — ignoring Enter.')
            return
        self._busy = True
        # Resolve marker position (live or cached)
        mx = my = mz = None
        try:
            tf = self._tf_buffer.lookup_transform(
                'base_link', 'marker_frame', Time(), Duration(seconds=0.5))
            mx, my, mz = tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z
            self._last_marker = (mx, my, mz, time.monotonic())
        except Exception:
            if self._last_marker:
                age = time.monotonic() - self._last_marker[3]
                if age <= MARKER_CACHE_EXPIRY:
                    mx, my, mz = self._last_marker[:3]
                    self.get_logger().warn(f'Using cached marker ({age:.1f}s old)')
                else:
                    self.get_logger().error('Cached marker too old. Point camera at marker first.')
                    self._busy = False; return
            else:
                self.get_logger().error('No marker seen yet.')
                self._busy = False; return

        try:
            tcp_tf  = self._tf_buffer.lookup_transform('base_link', 'tcp_link', Time(), Duration(seconds=1.0))
            cur_rot = tcp_tf.transform.rotation
        except Exception as e:
            self.get_logger().error(f'Cannot get tcp_link: {e}')
            self._busy = False; return

        # Try decreasing approach heights until one is reachable
        heights = [h for h in [APPROACH_HEIGHT, 0.25, 0.20, 0.15] if mz + h >= MINIMUM_SAFE_Z]
        if not heights:
            self.get_logger().error('Marker z too low — all approach heights below safe floor.')
            self._busy = False; return
        tx, ty = mx, my
        tz = mz + heights[0]   # start with the tallest; fallback set in _plan_result_cb

        cur_q    = np.array([cur_rot.x, cur_rot.y, cur_rot.z, cur_rot.w])
        look_dir = np.array([mx-tx, my-ty, mz-tz]); look_dir /= np.linalg.norm(look_dir)
        goal_q   = _lookat_quat(cur_q, CAM_Z_IN_LINK6, look_dir)

        self.get_logger().info(f'Step 1: position-only to ({tx:.3f},{ty:.3f},{tz:.3f})  '
                               f'[approach={tz-mz:.2f}m]...')
        self._pending = dict(tx=tx, ty=ty, tz=tz, goal_q=goal_q,
                             mx=mx, my=my, mz=mz,
                             heights=heights, height_idx=0,
                             with_orientation=False)   # always position-only first
        self._send_plan(with_orientation=False)

    def _send_plan(self, with_orientation: bool):
        p = self._pending
        if not self._plan_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('MoveIt not available.')
            return

        prim = SolidPrimitive(); prim.type = SolidPrimitive.SPHERE; prim.dimensions = [0.03]
        center = Pose(); center.position.x = p['tx']; center.position.y = p['ty']
        center.position.z = p['tz']; center.orientation.w = 1.0
        region = BoundingVolume(); region.primitives.append(prim); region.primitive_poses.append(center)
        pos = PositionConstraint(); pos.header.frame_id = 'base_link'
        pos.link_name = EFFECTOR_LINK; pos.constraint_region = region; pos.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos)

        if with_orientation:
            gq = p['goal_q']
            oc = OrientationConstraint()
            oc.header.frame_id = 'base_link'; oc.link_name = EFFECTOR_LINK
            oc.orientation.x = float(gq[0]); oc.orientation.y = float(gq[1])
            oc.orientation.z = float(gq[2]); oc.orientation.w = float(gq[3])
            oc.absolute_x_axis_tolerance = 0.5
            oc.absolute_y_axis_tolerance = 0.5
            oc.absolute_z_axis_tolerance = 3.14
            oc.weight = 0.8
            constraints.orientation_constraints.append(oc)

        goal = MoveGroup.Goal()
        goal.request.group_name                      = MOVE_GROUP_NAME
        goal.request.allowed_planning_time           = 10.0
        goal.request.num_planning_attempts           = 20
        goal.request.max_velocity_scaling_factor     = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = True   # plan only — we execute separately
        goal.planning_options.replan    = False

        self._pending['with_orientation'] = with_orientation
        future = self._plan_client.send_goal_async(goal)
        future.add_done_callback(self._plan_response_cb)

    def _plan_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Plan goal rejected.')
            return
        self.get_logger().info('Planning...')
        handle.get_result_async().add_done_callback(self._plan_result_cb)

    def _plan_result_cb(self, future):
        result           = future.result().result
        code             = result.error_code.val
        with_orientation = self._pending['with_orientation']

        if code != MoveItErrorCodes.SUCCESS:
            p = self._pending
            if with_orientation:
                # Orientation correction failed — don't retry as position move, just give up
                self.get_logger().error(
                    f'Orientation correction planning failed (code {code}). '
                    f'Try pressing Enter again from a more centred position.')
                self._busy = False
            else:
                # Position move failed — try a lower approach height
                next_idx = p.get('height_idx', 0) + 1
                heights  = p.get('heights', [])
                if next_idx < len(heights):
                    new_tz = p['mz'] + heights[next_idx]
                    p['tz'] = new_tz
                    p['height_idx'] = next_idx
                    p['with_orientation'] = False
                    self.get_logger().warn(
                        f'Unreachable at approach={heights[next_idx-1]:.2f}m. '
                        f'Trying {heights[next_idx]:.2f}m (z={new_tz:.3f})...')
                    self._send_plan(with_orientation=False)
                else:
                    self.get_logger().error(
                        f'Planning failed at all approach heights {heights}. '
                        f'Move marker closer to the front of the arm (reduce y offset).')
                    self._busy = False
            return

        trajectory = result.planned_trajectory
        pts = len(trajectory.joint_trajectory.points)
        label = 'with orientation' if with_orientation else 'position-only'
        self.get_logger().info(f'Plan OK ({pts} pts, {label}). Opening gate and executing...')

        # Run gate + execute in a separate thread — calling services from inside
        # a ROS2 spin callback would deadlock (spin thread blocks waiting for the
        # service response that the spin thread itself needs to deliver)
        phase = 'position' if not with_orientation else 'orientation'
        threading.Thread(
            target=self._gate_then_execute, args=(trajectory, phase), daemon=True
        ).start()

    def _gate_then_execute(self, trajectory, phase='position'):
        """Open control gate then execute — must run in its own thread, not a callback."""
        self._pending['phase'] = phase
        _call_service(self, self._enable_cli)
        _call_service(self, self._gate_cli)
        self._execute(trajectory)

    def _execute(self, trajectory):
        if not self._execute_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('/execute_trajectory not available.')
            return
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        future = self._execute_client.send_goal_async(goal)
        future.add_done_callback(self._exec_response_cb)

    def _exec_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Execute goal rejected.')
            return
        self.get_logger().info('Executing...')
        handle.get_result_async().add_done_callback(self._exec_result_cb)

    def _exec_result_cb(self, future):
        code  = future.result().result.error_code.val
        phase = self._pending.get('phase', 'position')

        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'Execution failed (code {code}) in phase={phase}.')
            self._busy = False
            return

        if phase == 'position':
            self._report_position_reached()
            self.get_logger().info('Step 2: rotating camera to face marker...')
            self._send_orientation_correction()
        else:
            # Orientation correction done — report actual camera direction
            self._report_camera_direction()
            self.get_logger().info('Done — press Enter to move again.')
            self._busy = False

    def _report_position_reached(self):
        """After position move: print where tcp_link landed and how far off-target."""
        try:
            tf = self._tf_buffer.lookup_transform('base_link', 'tcp_link', Time(), Duration(seconds=1.0))
            p  = self._pending
            tx, ty, tz = tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z
            err = np.linalg.norm([tx - p['tx'], ty - p['ty'], tz - p['tz']])
            q   = tf.transform.rotation
            q_arr = np.array([q.x, q.y, q.z, q.w])
            cam_z  = _qrot(q_arr, CAM_Z_IN_LINK6)
            to_m   = np.array([p['mx']-tx, p['my']-ty, p['mz']-tz])
            to_m  /= np.linalg.norm(to_m)
            angle  = np.degrees(np.arccos(np.clip(np.dot(cam_z, to_m), -1, 1)))
            self.get_logger().info(
                f'Step 1 done. tcp_link at ({tx:+.3f},{ty:+.3f},{tz:+.3f})  '
                f'target=({p["tx"]:+.3f},{p["ty"]:+.3f},{p["tz"]:+.3f})  err={err*100:.1f}cm')
            self.get_logger().info(
                f'  cam_z=({cam_z[0]:+.3f},{cam_z[1]:+.3f},{cam_z[2]:+.3f})  '
                f'to_marker=({to_m[0]:+.3f},{to_m[1]:+.3f},{to_m[2]:+.3f})  '
                f'angle={angle:.1f}deg')
        except Exception as e:
            self.get_logger().warn(f'Cannot read tcp_link after position move: {e}')

    def _report_camera_direction(self):
        """After orientation move, print where the camera actually points vs marker."""
        try:
            tf = self._tf_buffer.lookup_transform('base_link', 'tcp_link', Time(), Duration(seconds=1.0))
            q  = tf.transform.rotation
            q_arr = np.array([q.x, q.y, q.z, q.w])
            cam_z = _qrot(q_arr, CAM_Z_IN_LINK6)
            p     = self._pending
            to_marker = np.array([p['mx'] - tf.transform.translation.x,
                                   p['my'] - tf.transform.translation.y,
                                   p['mz'] - tf.transform.translation.z])
            to_marker /= np.linalg.norm(to_marker)
            dot = float(np.dot(cam_z, to_marker))
            angle_deg = np.degrees(np.arccos(np.clip(dot, -1, 1)))
            self.get_logger().info(
                f'Camera Z in base_link: ({cam_z[0]:+.3f},{cam_z[1]:+.3f},{cam_z[2]:+.3f})')
            self.get_logger().info(
                f'Direction to marker:   ({to_marker[0]:+.3f},{to_marker[1]:+.3f},{to_marker[2]:+.3f})')
            self.get_logger().info(
                f'Angle between camera look and marker: {angle_deg:.1f} deg '
                f'(0=perfect, >20=likely out of frame)')
        except Exception as e:
            self.get_logger().warn(f'Cannot compute camera direction: {e}')

    def _send_orientation_correction(self):
        """Second move: keep position fixed, rotate camera to face marker."""
        p = self._pending
        if not self._plan_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('MoveIt not available for orientation correction.')
            return

        # Read the current tcp_link position (where we just arrived)
        try:
            tcp_tf  = self._tf_buffer.lookup_transform('base_link', 'tcp_link', Time(), Duration(seconds=1.0))
            cur_rot = tcp_tf.transform.rotation
            cur_pos = tcp_tf.transform.translation
        except Exception as e:
            self.get_logger().error(f'Cannot get tcp_link for correction: {e}')
            return

        # Recompute look-at from the CAMERA origin (not tcp_link) at our arrived pose.
        # Camera is offset from tcp_link by CAM_ORIGIN_IN_LINK6, rotated by cur_q.
        cur_q      = np.array([cur_rot.x, cur_rot.y, cur_rot.z, cur_rot.w])
        cam_origin = np.array([cur_pos.x, cur_pos.y, cur_pos.z]) + _qrot(cur_q, CAM_ORIGIN_IN_LINK6)
        look_dir   = np.array([p['mx'] - cam_origin[0], p['my'] - cam_origin[1], p['mz'] - cam_origin[2]])
        look_dir  /= np.linalg.norm(look_dir)
        goal_q     = _lookat_quat(cur_q, CAM_Z_IN_LINK6, look_dir)

        self.get_logger().info(
            f'Camera origin=({cam_origin[0]:+.3f},{cam_origin[1]:+.3f},{cam_origin[2]:+.3f})  '
            f'look=({look_dir[0]:+.2f},{look_dir[1]:+.2f},{look_dir[2]:+.2f})')

        # Orientation-only — no position constraint so the planner has freedom to find a solution.
        # Removing position hold lets the arm drift slightly but drastically improves plan success.
        oc = OrientationConstraint()
        oc.header.frame_id = 'base_link'; oc.link_name = EFFECTOR_LINK
        oc.orientation.x = float(goal_q[0]); oc.orientation.y = float(goal_q[1])
        oc.orientation.z = float(goal_q[2]); oc.orientation.w = float(goal_q[3])
        oc.absolute_x_axis_tolerance = 0.25  # ±14° on camera X
        oc.absolute_y_axis_tolerance = 0.25  # ±14° on camera Y
        oc.absolute_z_axis_tolerance = 0.5   # ±29° — NOT free; camera Z is 15° off tcp_link Z
                                             #  so free roll sweeps camera through a 30° cone
        oc.weight = 1.0

        constraints = Constraints()
        constraints.orientation_constraints.append(oc)

        goal = MoveGroup.Goal()
        goal.request.group_name                      = MOVE_GROUP_NAME
        goal.request.allowed_planning_time           = 15.0
        goal.request.num_planning_attempts           = 30
        goal.request.max_velocity_scaling_factor     = 0.05   # slow — small corrective rotation
        goal.request.max_acceleration_scaling_factor = 0.05
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = True
        goal.planning_options.replan    = False

        self._pending['with_orientation'] = True   # so _plan_result_cb tags this as 'orientation' phase
        future = self._plan_client.send_goal_async(goal)
        future.add_done_callback(self._plan_response_cb)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = EyeInHandPractice()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print('\n=== Eye-in-Hand Practice ===')
    print('Marker position printed every second.')
    print('Press Enter → plan path, open arm gate, execute.')
    print('Ctrl+C to quit.\n')

    try:
        while rclpy.ok():
            try:
                input()
            except EOFError:
                break
            node.move_to_marker()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
