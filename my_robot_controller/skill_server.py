#!/usr/bin/env python3
"""
skill_server — Stage 1 of the thesis stack (see ~/sessions/THESIS_ROADMAP.md).

The ONLY component that moves the arm. Consumes JSON action lists
(Interface 1), validates them atomically, executes them sequentially through
the skills API (Interface 2), and reports every step. Intent sources
(keyboard now, gestures/speech in Stage 3, LLM planner in Stage 4) are just
publishers of Interface 1 — they never touch motion.

Interface 1 — /skills/command (std_msgs/String, JSON):
    {"id": "demo-1", "actions": [
        {"action": "look_at", "target": "station_A"},
        {"action": "pick",    "target": "obj_C"},
        {"action": "place",   "target": "station_B"}
    ]}
  A bare list is also accepted. {"actions":[{"action":"stop"}]} (or the word
  "stop" alone as the payload) preempts: clears the queue and cancels the
  in-flight trajectory.

Status — /skills/status (std_msgs/String, JSON), one event per transition:
    {"id": ..., "index": ..., "action": ..., "event":
        received|rejected|started|done|failed|aborted, "detail": ...}

Skills (Interface 2):
    scan                       report all fresh markers (no motion)
    look_at   target           aim the camera at a target from LOOK_DIST_M
    pick      target           pre-grasp → descend → close → attach → lift
    place     target           pre-place → descend → open → detach → retreat
    lift      [dz]             straight vertical lift (default LIFT_CLEAR_M)
    home                       driver's /move_home (same call the GUI uses)
    open_gripper / close_gripper [width]
    wait      seconds
    say       text             log only (feedback channel placeholder)
    stop                       preempt everything (handled out-of-band)

SAFETY
  - EXECUTE=False by default: everything validates, resolves and plans;
    nothing moves (gripper included). Motion skills report what they WOULD do.
  - All speed caps and the /control_enable gate live here.
  - Run under agx_arm_gui's Custom Script Runner: the GUI deadman kills this
    process group and homes the arm — that remains the emergency stop.

Test without hardware:
    python3 skill_server.py --selftest        # stubbed motion, full logic
Test with the stack up (arm can stay disabled for plan-only):
    ros2 topic pub --once /skills/command std_msgs/msg/String \
      "{data: '{\"actions\": [{\"action\": \"scan\"}]}'}"
"""

import json
import queue
import sys
import threading
import time

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from std_srvs.srv import Empty as EmptySrv
from visualization_msgs.msg import Marker, MarkerArray

from my_robot_controller.eih_core import (
    CAMERA_TOOL, EihBaseNode, GRIPPER_MAX_WIDTH_M, MarkerTracker,
    make_tcp_tool,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

EXECUTE = False      # ← False = validate/resolve/plan only, NO motion at all.

# Named targets — the only things actions may reference. Stations are table
# waypoints; objects are graspable and carry their dimensions (m).
TARGETS = {
    'station_A': {'marker_id': 100, 'type': 'station'},
    'station_B': {'marker_id': 101, 'type': 'station'},
    'obj_C':     {'marker_id': 582, 'type': 'object',
                  'dims': (0.0254, 0.0254, 0.0254)},   # 1-inch cube
}

# Reachable table volume (base_link, metres) — resolved target positions
# outside this are refused. Derived from observed workspace; tune as needed.
WORKSPACE = {'x': (0.10, 0.50), 'y': (-0.35, 0.35), 'z': (-0.05, 0.40)}

MARKER_SIZE_M      = 0.047
LOOK_DIST_M        = 0.20     # camera→target for look_at
# Clearances ≤0.04: straight-down TCP poses are unreachable above z≈0.065 m
# (joint5 limit — IK-probed 2026-07-04; 0.04 mock-verified). Sync pick_and_place.
PRE_GRASP_CLEAR_M  = 0.04
LIFT_CLEAR_M       = 0.04
PLACE_DROP_M       = 0.004
RETREAT_CLEAR_M    = 0.04
GRIP_SQUEEZE_M     = 0.003
GRIPPER_SETTLE_S   = 2.0
TCP_OFFSET_Z       = 0.175    # verify in RViz before EXECUTE=True (see roadmap)
RESOLVE_WAIT_S     = 5.0      # max wait for a fresh detection of a target
SCENE_FRESH_S      = 3.0      # max age of top-down scene data for fallback
MAX_VELOCITY_SCALE = 0.10
MAX_ACCEL_SCALE    = 0.10

# action name → required params, optional params
ACTION_SPECS = {
    'scan':          ((), ()),
    'look_at':       (('target',), ()),
    'pick':          (('target',), ()),
    'place':         (('target',), ()),
    'lift':          ((), ('dz',)),
    'home':          ((), ()),
    'open_gripper':  ((), ('width',)),
    'close_gripper': ((), ('width',)),
    'wait':          (('seconds',), ()),
    'say':           (('text',), ()),
    'stop':          ((), ()),
}


class SkillError(Exception):
    """Raised inside a skill to fail the current action with a message."""


class SkillServer(EihBaseNode):

    def __init__(self):
        super().__init__('skill_server',
                         max_velocity_scale=MAX_VELOCITY_SCALE,
                         max_accel_scale=MAX_ACCEL_SCALE)
        self.tcp = make_tcp_tool(TCP_OFFSET_Z)
        self.tracker = MarkerTracker(self, marker_size=MARKER_SIZE_M)
        self.holding: str | None = None      # target name currently grasped

        self._queue: queue.Queue = queue.Queue()
        self._stop_flag = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)

        self._home_cli = self.create_client(EmptySrv, '/move_home')

        # Optional global map from the top-down camera (scene_map node,
        # Stage 2). Purely a fallback: wrist detections always win.
        self._scene: tuple[dict, float] | None = None   # (markers, recv time)
        self.create_subscription(String, '/scene/markers', self._on_scene, 10)

        self.create_subscription(String, '/skills/command', self._on_command, 10)
        self._status_pub = self.create_publisher(String, '/skills/status', 10)
        self._status_history: list[dict] = []   # inspected by self-tests

        self._viz_pub = self.create_publisher(MarkerArray, '/skills/viz', 10)
        self.create_timer(1.0, self._publish_viz)

        mode = 'EXECUTE' if EXECUTE else 'PLAN-ONLY (no motion; set EXECUTE=True)'
        self._log(f'skill_server ready. Mode: {mode}')
        self._log(f'Targets: {", ".join(TARGETS)}')
        self._worker.start()

    # ── Interface 1: command intake ───────────────────────────────────────────

    def _on_command(self, msg: String):
        payload = msg.data.strip()
        try:
            if payload.lower() == 'stop':
                data = {'actions': [{'action': 'stop'}]}
            else:
                data = json.loads(payload)
            if isinstance(data, list):
                data = {'actions': data}
            seq_id = str(data.get('id', f'seq-{int(time.time())}'))
            actions = data['actions']
            assert isinstance(actions, list) and actions
        except Exception as e:
            self._emit('?', -1, '?', 'rejected', f'malformed command: {e}')
            return

        if any(a.get('action') == 'stop' for a in actions if isinstance(a, dict)):
            self._do_stop(seq_id)
            return

        errors = self.validate(actions)
        if errors:
            self._emit(seq_id, -1, '?', 'rejected', '; '.join(errors))
            return
        self._emit(seq_id, -1, '?', 'received', f'{len(actions)} action(s) queued')
        self._queue.put((seq_id, actions))

    def _do_stop(self, seq_id: str):
        self._stop_flag.set()
        drained = 0
        try:
            while True:
                self._queue.get_nowait()
                drained += 1
        except queue.Empty:
            pass
        self.cancel_active_execution()
        self._emit(seq_id, -1, 'stop', 'done',
                   f'queue cleared ({drained} pending), execution cancel requested')

    # ── Validation (atomic: whole list accepted or whole list rejected) ──────

    def validate(self, actions: list) -> list[str]:
        errors = []
        will_hold = self.holding
        for i, a in enumerate(actions):
            if not isinstance(a, dict) or 'action' not in a:
                errors.append(f'#{i}: not an action object')
                continue
            name = a['action']
            spec = ACTION_SPECS.get(name)
            if spec is None:
                errors.append(f'#{i}: unknown action "{name}"')
                continue
            required, optional = spec
            for p in required:
                if p not in a:
                    errors.append(f'#{i} {name}: missing param "{p}"')
            for p in a:
                if p != 'action' and p not in required + optional:
                    errors.append(f'#{i} {name}: unexpected param "{p}"')
            tgt = a.get('target')
            if 'target' in (required + optional) and tgt is not None:
                if tgt not in TARGETS:
                    errors.append(f'#{i} {name}: unknown target "{tgt}" '
                                  f'(known: {", ".join(TARGETS)})')
                    continue
                if name == 'pick':
                    if TARGETS[tgt]['type'] != 'object':
                        errors.append(f'#{i} pick: "{tgt}" is not an object')
                    if will_hold is not None:
                        errors.append(f'#{i} pick: already holding "{will_hold}"')
                    will_hold = tgt
                if name == 'place':
                    if TARGETS[tgt]['type'] != 'station':
                        errors.append(f'#{i} place: "{tgt}" is not a station')
                    if will_hold is None:
                        errors.append(f'#{i} place: nothing is held')
                    will_hold = None
            if name == 'wait':
                try:
                    s = float(a.get('seconds', -1))
                    assert 0 < s <= 30
                except Exception:
                    errors.append(f'#{i} wait: seconds must be 0–30')
        return errors

    # ── Target resolution ─────────────────────────────────────────────────────

    def _on_scene(self, msg: String):
        try:
            data = json.loads(msg.data)
            markers = {int(k): v for k, v in data['markers'].items()}
            self._scene = (markers, time.monotonic())
        except Exception:
            pass

    def _check_bounds(self, name: str, p: np.ndarray):
        for axis, (lo, hi) in zip('xyz', WORKSPACE.values()):
            v = p['xyz'.index(axis)]
            if not lo <= v <= hi:
                raise SkillError(f'target "{name}" at {axis}={v:+.3f} outside '
                                 f'workspace [{lo}, {hi}] — refusing')

    def resolve_target(self, name: str) -> tuple[np.ndarray, float]:
        """Fresh base-frame position (+yaw) of a target, waiting up to
        RESOLVE_WAIT_S for the wrist camera, then falling back to the
        top-down scene map. Enforces workspace bounds. Raises SkillError."""
        marker_id = TARGETS[name]['marker_id']
        deadline = time.monotonic() + RESOLVE_WAIT_S
        while time.monotonic() < deadline:
            mk = self.tracker.get(marker_id)
            if mk is not None:
                p = mk.position
                self._check_bounds(name, p)
                return p, mk.yaw_deg
            time.sleep(0.2)

        if self._scene is not None:
            markers, stamp = self._scene
            if time.monotonic() - stamp < SCENE_FRESH_S and marker_id in markers:
                e = markers[marker_id]
                p = np.array([e['x'], e['y'], e['z']])
                self._check_bounds(name, p)
                self._log(f'[RESOLVE] "{name}" not seen by wrist camera — '
                          f'using TOP-DOWN estimate ({p[0]:+.3f}, {p[1]:+.3f}, '
                          f'{p[2]:+.3f}); expect cm-level accuracy, look_at '
                          'before pick to refine.', 'WARN')
                return p, float(e.get('yaw_deg', 0.0))

        raise SkillError(f'target "{name}" (marker {marker_id}) not detected '
                         f'within {RESOLVE_WAIT_S:.0f}s (wrist), and no fresh '
                         'top-down scene data')

    # ── Motion gate ───────────────────────────────────────────────────────────

    def _move(self, l6p, l6q, sol, label, **kw) -> None:
        """move_to that honours EXECUTE and the stop flag. Raises SkillError."""
        if self._stop_flag.is_set():
            raise SkillError('stopped')
        if not EXECUTE:
            self._log(f'[{label}] PLAN-ONLY: would move link6 to '
                      f'({l6p[0]:+.3f}, {l6p[1]:+.3f}, {l6p[2]:+.3f})')
            return
        if not self.move_to(l6p, l6q, sol, label, **kw):
            raise SkillError(f'{label}: motion failed')

    def _gripper(self, width: float):
        if not EXECUTE:
            self._log(f'[GRIPPER] PLAN-ONLY: would set width to {width:.3f} m')
            return
        self.set_gripper_width(width, GRIPPER_SETTLE_S)

    # ── Skills (Interface 2) ──────────────────────────────────────────────────

    def skill_scan(self, a: dict) -> str:
        ids = self.tracker.fresh_ids()
        named = {n: t['marker_id'] for n, t in TARGETS.items()
                 if t['marker_id'] in ids}
        return f'fresh markers: {ids or "none"}; known targets visible: {named or "none"}'

    def skill_look_at(self, a: dict) -> str:
        pos, _ = self.resolve_target(a['target'])
        found = self.find_reachable_tool_pose(CAMERA_TOOL, pos, LOOK_DIST_M,
                                              label='LOOK_AT')
        if found is None:
            raise SkillError('no reachable camera pose')
        _, _, l6p, l6q, sol, desc = found
        self._move(l6p, l6q, sol, 'LOOK_AT')
        return f'aimed at {a["target"]} ({desc})'

    def skill_pick(self, a: dict) -> str:
        name = a['target']
        dims = TARGETS[name]['dims']
        pos, yaw = self.resolve_target(name)
        obj_centre = pos - np.array([0, 0, dims[2] / 2])

        # straight-down grasp, wrist roll aligned to the object's yaw
        rolls = tuple(((-90.0 - yaw) % 90.0 + k * 90.0) % 360.0 for k in range(4))
        found = None
        from my_robot_controller.eih_core import lookat_candidates
        for tilt, az, roll, R in lookat_candidates(tilts=(0,), rolls=rolls):
            axis = R.apply([0.0, 0.0, 1.0])
            tool_pos = obj_centre - axis * (dims[2] / 2 + PRE_GRASP_CLEAR_M)
            l6p, l6q = self.tcp.to_link6(tool_pos, R)
            sol = self.call_ik(l6p, l6q)
            if sol is not None:
                found = (R, l6p, l6q, sol, roll)
                break
        if found is None:
            raise SkillError('no reachable grasp orientation')
        R, pg_l6p, pg_l6q, pg_sol, roll = found

        g_l6p, g_l6q = self.tcp.to_link6(obj_centre, R)
        g_sol = self.call_ik(g_l6p, g_l6q, timeout_s=1.0)
        if g_sol is None:
            raise SkillError('grasp depth unreachable — check TCP_OFFSET_Z')

        self._gripper(GRIPPER_MAX_WIDTH_M)
        self._move(pg_l6p, pg_l6q, pg_sol, 'PRE_GRASP')
        self._move(g_l6p, g_l6q, g_sol, 'GRASP_DESCEND', avoid_collisions=False)
        self._gripper(max(dims[0] - GRIP_SQUEEZE_M, 0.0))
        lift_p = g_l6p + np.array([0, 0, LIFT_CLEAR_M])
        lift_sol = self.call_ik(lift_p, g_l6q, timeout_s=1.0) or g_sol
        self._move(lift_p, g_l6q, lift_sol, 'LIFT', avoid_collisions=False)
        self.holding = name
        return f'picked {name} (grasp roll {roll:.0f}°)'

    def skill_place(self, a: dict) -> str:
        if self.holding is None:
            raise SkillError('nothing is held')
        held_dims = TARGETS[self.holding]['dims']
        pos, yaw = self.resolve_target(a['target'])
        place_centre = np.array([pos[0], pos[1],
                                 pos[2] + held_dims[2] / 2 + PLACE_DROP_M])

        rolls = tuple(((-90.0 - yaw) % 90.0 + k * 90.0) % 360.0 for k in range(4))
        from my_robot_controller.eih_core import lookat_candidates
        found = None
        for tilt, az, roll, R in lookat_candidates(tilts=(0,), rolls=rolls):
            axis = R.apply([0.0, 0.0, 1.0])
            tool_pos = place_centre - axis * (held_dims[2] / 2 + PRE_GRASP_CLEAR_M)
            l6p, l6q = self.tcp.to_link6(tool_pos, R)
            sol = self.call_ik(l6p, l6q)
            if sol is not None:
                found = (R, l6p, l6q, sol)
                break
        if found is None:
            raise SkillError('no reachable place approach')
        R, pp_l6p, pp_l6q, pp_sol = found
        pl_l6p, pl_l6q = self.tcp.to_link6(place_centre, R)
        pl_sol = self.call_ik(pl_l6p, pl_l6q, timeout_s=1.0)
        if pl_sol is None:
            raise SkillError('place depth unreachable')

        self._move(pp_l6p, pp_l6q, pp_sol, 'PRE_PLACE')
        self._move(pl_l6p, pl_l6q, pl_sol, 'PLACE_DESCEND', avoid_collisions=False)
        self._gripper(GRIPPER_MAX_WIDTH_M)
        ret_p = pl_l6p + np.array([0, 0, RETREAT_CLEAR_M])
        ret_sol = self.call_ik(ret_p, pl_l6q, timeout_s=1.0) or pl_sol
        self._move(ret_p, pl_l6q, ret_sol, 'RETREAT', avoid_collisions=False)
        placed = self.holding
        self.holding = None
        return f'placed {placed} at {a["target"]}'

    def skill_lift(self, a: dict) -> str:
        dz = float(a.get('dz', LIFT_CLEAR_M))
        l6 = self.link6_in_base()
        if l6 is None:
            raise SkillError('current pose unknown (TF)')
        t, R = l6
        target = t + np.array([0, 0, dz])
        sol = self.call_ik(target, R.as_quat(), timeout_s=1.0)
        if sol is None:
            raise SkillError(f'lift +{dz:.2f} m unreachable')
        self._move(target, R.as_quat(), sol, 'LIFT', avoid_collisions=False)
        return f'lifted {dz:.2f} m'

    def skill_home(self, a: dict) -> str:
        if not EXECUTE:
            self._log('[HOME] PLAN-ONLY: would call /move_home')
            return 'would home (plan-only)'
        if not self._home_cli.wait_for_service(timeout_sec=3.0):
            raise SkillError('/move_home unavailable')
        future = self._home_cli.call_async(EmptySrv.Request())
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=15.0):
            raise SkillError('/move_home timed out')
        return 'homed'

    def skill_open_gripper(self, a: dict) -> str:
        w = float(a.get('width', GRIPPER_MAX_WIDTH_M))
        self._gripper(w)
        return f'gripper → {w:.3f} m'

    def skill_close_gripper(self, a: dict) -> str:
        w = float(a.get('width', 0.0))
        self._gripper(w)
        return f'gripper → {w:.3f} m'

    def skill_wait(self, a: dict) -> str:
        s = float(a['seconds'])
        end = time.monotonic() + s
        while time.monotonic() < end:
            if self._stop_flag.is_set():
                raise SkillError('stopped')
            time.sleep(0.1)
        return f'waited {s:.1f} s'

    def skill_say(self, a: dict) -> str:
        self._log(f'[SAY] {a["text"]}')
        return a['text']

    # ── Executor ──────────────────────────────────────────────────────────────

    def _worker_loop(self):
        while rclpy.ok():
            try:
                seq_id, actions = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._stop_flag.clear()
            if EXECUTE:
                self.enable_control_gate()
            for i, a in enumerate(actions):
                name = a['action']
                if self._stop_flag.is_set():
                    self._emit(seq_id, i, name, 'aborted', 'stopped')
                    break
                self._emit(seq_id, i, name, 'started', json.dumps(a))
                try:
                    detail = getattr(self, f'skill_{name}')(a)
                    self._emit(seq_id, i, name, 'done', detail)
                except SkillError as e:
                    self._emit(seq_id, i, name, 'failed', str(e))
                    break
                except Exception as e:                     # noqa: BLE001
                    self._emit(seq_id, i, name, 'failed', f'internal: {e!r}')
                    break

    def _emit(self, seq_id, index, action, event, detail=''):
        rec = {'id': seq_id, 'index': index, 'action': action,
               'event': event, 'detail': detail}
        self._status_history.append(rec)
        self._status_pub.publish(String(data=json.dumps(rec)))
        lvl = 'ERROR' if event in ('rejected', 'failed') else (
            'WARN' if event == 'aborted' else 'INFO')
        self._log(f'[{seq_id}#{index} {action}] {event}: {detail}', lvl)

    # ── Viz ───────────────────────────────────────────────────────────────────

    def _publish_viz(self):
        ms: list[Marker] = []
        i = 0
        for name, t in TARGETS.items():
            mk = self.tracker.get(t['marker_id'])
            if mk is None:
                continue
            p = mk.position
            col = ((0.1, 0.9, 0.2, 0.8) if t['type'] == 'object'
                   else (0.2, 0.4, 1.0, 0.9))
            if t['type'] == 'object':
                centre = p - np.array([0, 0, t['dims'][2] / 2])
                ms.append(self.viz_cube('targets', i, centre, t['dims'], col))
            else:
                ms.append(self.viz_sphere('targets', i, p, col, diam=0.04))
            ms.append(self.viz_text('targets', i + 1, p + [0, 0, 0.05], name, col))
            i += 2
        held = f' | holding: {self.holding}' if self.holding else ''
        mode = 'EXECUTE' if EXECUTE else 'plan-only'
        ms.append(self.viz_text('status', 0, [0.0, 0.0, 0.60],
                                f'skill_server [{mode}]{held}'))
        self._viz_pub.publish(MarkerArray(markers=ms))


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test (no hardware, no ROS graph peers needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> int:
    rclpy.init()
    node = SkillServer()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    calls: list[str] = []
    node.move_to = lambda p, q, s, label, **kw: calls.append(label) or True
    node.set_gripper_width = lambda w, settle_s=0: calls.append(f'grip:{w:.3f}')
    node.call_ik = lambda p, q, timeout_s=0.5: {f'joint{i}': 0.1 for i in range(1, 7)}
    node.find_reachable_tool_pose = (
        lambda tool, xyz, dist, label='', tilts=None:
        (xyz, Rotation.from_euler('x', 180, degrees=True),
         np.zeros(3), np.array([0, 0, 0, 1.0]),
         {f'joint{i}': 0.1 for i in range(1, 7)}, 'stub'))
    node.enable_control_gate = lambda: True
    node.tracker.inject(100, [0.30, -0.10, 0.02])
    node.tracker.inject(101, [0.30, +0.10, 0.02])
    node.tracker.inject(582, [0.30, -0.10, 0.02 + 0.0254], yaw_deg=20.0)

    fails = 0
    def check(name, cond, detail=''):
        nonlocal fails
        print(f'  {"PASS" if cond else "FAIL"}  {name}  {detail}')
        fails += 0 if cond else 1

    def send(payload):
        node._on_command(String(data=payload))

    def events(seq_id):
        return [r for r in node._status_history if r['id'] == seq_id]

    def wait_done(seq_id, n_actions, timeout=10.0):
        """Terminal when rejected, or a failed/aborted event arrived, or all
        n_actions have a 'done' event."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            ev = events(seq_id)
            if any(r['event'] in ('rejected', 'failed', 'aborted') for r in ev):
                time.sleep(0.2)          # let trailing events land
                return events(seq_id)
            if sum(r['event'] == 'done' for r in ev) >= n_actions:
                return ev
            time.sleep(0.1)
        return events(seq_id)

    print('1) validation')
    send('not json at all {{{')
    check('malformed json rejected',
          node._status_history[-1]['event'] == 'rejected')
    send(json.dumps({'id': 'v1', 'actions': [{'action': 'flyaway'}]}))
    check('unknown action rejected', events('v1')[-1]['event'] == 'rejected',
          events('v1')[-1]['detail'])
    send(json.dumps({'id': 'v2', 'actions': [{'action': 'pick'}]}))
    check('missing param rejected', events('v2')[-1]['event'] == 'rejected')
    send(json.dumps({'id': 'v3', 'actions': [{'action': 'pick', 'target': 'station_A'}]}))
    check('pick a station rejected', events('v3')[-1]['event'] == 'rejected')
    send(json.dumps({'id': 'v4', 'actions': [{'action': 'place', 'target': 'station_B'}]}))
    check('place while empty-handed rejected', events('v4')[-1]['event'] == 'rejected')
    send(json.dumps({'id': 'v5', 'actions': [
        {'action': 'pick', 'target': 'obj_C'},
        {'action': 'pick', 'target': 'obj_C'}]}))
    check('double pick rejected', events('v5')[-1]['event'] == 'rejected')

    print('2) plan-only execution of a full sequence')
    calls.clear()
    send(json.dumps({'id': 'run1', 'actions': [
        {'action': 'scan'},
        {'action': 'look_at', 'target': 'station_A'},
        {'action': 'pick', 'target': 'obj_C'},
        {'action': 'place', 'target': 'station_B'},
        {'action': 'say', 'text': 'all done'}]}))
    ev = wait_done('run1', 5)
    done_ev = [r for r in ev if r['event'] == 'done']
    check('all 5 actions completed', len(done_ev) == 5,
          f'{[ (r["action"], r["event"]) for r in ev ]}')
    check('scan saw all targets', 'obj_C' in (done_ev[0]['detail'] if done_ev else ''))
    check('holding cleared after place', node.holding is None)
    check('EXECUTE=False → zero real motion calls', not calls, str(calls))
    order = [r['action'] for r in ev if r['event'] == 'started']
    check('actions ran in order',
          order == ['scan', 'look_at', 'pick', 'place', 'say'], str(order))

    print('3) failure propagation')
    node.tracker._markers.clear()
    node.tracker.inject(100, [0.30, -0.10, 0.02])
    old_wait = sys.modules[__name__].RESOLVE_WAIT_S
    sys.modules[__name__].RESOLVE_WAIT_S = 0.5
    send(json.dumps({'id': 'run2', 'actions': [
        {'action': 'pick', 'target': 'obj_C'},
        {'action': 'say', 'text': 'never reached'}]}))
    ev = wait_done('run2', 2)
    sys.modules[__name__].RESOLVE_WAIT_S = old_wait
    check('undetected target fails the action',
          any(r['event'] == 'failed' and r['action'] == 'pick' for r in ev))
    check('sequence halts after failure',
          not any(r['action'] == 'say' and r['event'] == 'started' for r in ev))

    print('4) workspace bounds')
    node.tracker.inject(582, [1.50, 0.0, 0.05])   # far outside reach
    sys.modules[__name__].RESOLVE_WAIT_S = 0.5
    send(json.dumps({'id': 'run3', 'actions': [{'action': 'pick', 'target': 'obj_C'}]}))
    ev = wait_done('run3', 1)
    sys.modules[__name__].RESOLVE_WAIT_S = old_wait
    fail_ev = [r for r in ev if r['event'] == 'failed']
    check('out-of-workspace target refused',
          fail_ev and 'workspace' in fail_ev[0]['detail'],
          fail_ev[0]['detail'] if fail_ev else 'no failure event')

    print('5) top-down scene fallback')
    node.tracker._markers.clear()                      # wrist camera blind
    node._on_scene(String(data=json.dumps({
        'stamp': time.time(), 'table_z': 0.02, 'markers': {
            '100': {'x': 0.30, 'y': -0.10, 'z': 0.02, 'yaw_deg': 5.0,
                    'px': [400, 300]}}})))
    sys.modules[__name__].RESOLVE_WAIT_S = 0.5
    send(json.dumps({'id': 'run5', 'actions': [
        {'action': 'look_at', 'target': 'station_A'}]}))
    ev = wait_done('run5', 1)
    sys.modules[__name__].RESOLVE_WAIT_S = old_wait
    check('wrist-blind target resolved via top-down map',
          any(r['event'] == 'done' and r['action'] == 'look_at' for r in ev),
          str([(r['action'], r['event'], r['detail']) for r in ev
               if r['event'] in ('done', 'failed')]))
    node._on_scene(String(data=json.dumps({
        'stamp': time.time(), 'table_z': 0.02, 'markers': {
            '100': {'x': 1.5, 'y': 0.0, 'z': 0.02, 'yaw_deg': 0.0,
                    'px': [10, 10]}}})))
    sys.modules[__name__].RESOLVE_WAIT_S = 0.5
    send(json.dumps({'id': 'run5b', 'actions': [
        {'action': 'look_at', 'target': 'station_A'}]}))
    ev = wait_done('run5b', 1)
    sys.modules[__name__].RESOLVE_WAIT_S = old_wait
    check('out-of-bounds top-down estimate still refused',
          any(r['event'] == 'failed' and 'workspace' in r['detail'] for r in ev))

    print('6) stop preempts')
    node.tracker.inject(582, [0.30, -0.10, 0.045], yaw_deg=0.0)
    send(json.dumps({'id': 'run4', 'actions': [
        {'action': 'wait', 'seconds': 8},
        {'action': 'say', 'text': 'should be aborted'}]}))
    time.sleep(0.8)
    send('stop')
    ev = wait_done('run4', 2, timeout=5.0)
    check('wait aborted by stop',
          any(r['event'] == 'failed' and r['detail'] == 'stopped' for r in ev)
          or any(r['event'] == 'aborted' for r in ev),
          str([(r['action'], r['event']) for r in ev]))
    check('queued say never ran',
          not any(r['action'] == 'say' and r['event'] == 'done' for r in ev))

    print(f'\n{"ALL PASS" if fails == 0 else f"{fails} FAILURES"}')
    rclpy.shutdown()
    return 1 if fails else 0


def main():
    if '--selftest' in sys.argv:
        sys.exit(_selftest())
    rclpy.init()
    node = SkillServer()
    node.tracker.ensure_detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tracker.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
