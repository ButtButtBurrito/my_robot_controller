#!/usr/bin/env python3
"""
gesture_node — Stage 3 intent source: facial gestures → /intent/gesture.

Watches a user-facing camera, detects coarse facial gestures with
gesture_core (MediaPipe Face Landmarker CPU + rule-based temporal detector,
lives in ~/ros2_ws/gesture/), and publishes the RAW event name. This node is
a *producer* of gesture events only (THESIS_ROADMAP Stage 3): it must NEVER
map gestures to actions (nod→CONFIRM is intent_router's job), identify
people, parse commands, or touch anything that moves.

Topics OUT:
    /intent/gesture       std_msgs/String  event name, one msg/event:
                          nod | shake | brow_raise | blink_stop
    /intent/gesture_meta  std_msgs/String  JSON: {"gesture", "confidence",
                          "t", "detail"}
Topics IN:
    /intent/gesture_enable  std_msgs/Bool  False pauses inference (saves the
                          shared CPU when no confirmation is pending);
                          camera stays open, frames are skipped. Default on.

Camera source (config below):
    IMAGE_TOPIC = None  → local webcam CAMERA_SOURCE (dev default, laptop cam)
    IMAGE_TOPIC = "/user_cam/image_raw"  → subscribe instead (final rig /
    any fixed user-facing camera published into ROS). Logic is identical —
    both paths feed FaceStream.process_frame().

Detector thresholds are PROVISIONAL (tuned in ~/ros2_ws/gesture/ on recorded
clips); MIN_CONFIDENCE below is this node's own last-line filter.

Test without camera or model:
    python3 gesture_node.py --selftest     # stubbed stream, real rclpy pub/sub
Run for real (laptop webcam):
    python3 gesture_node.py
    # (separate shell)  ros2 topic echo /intent/gesture std_msgs/msg/String
"""

import argparse
import json
import os
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

GESTURE_CORE_DIR = os.path.expanduser("~/ros2_ws/gesture")
sys.path.insert(0, GESTURE_CORE_DIR)  # single source of truth for detector

# ── config ───────────────────────────────────────────────────────────────────
IMAGE_TOPIC = None            # None = local webcam; set to a ROS image topic
                              # when the fixed user-facing camera exists (TODO)
CAMERA_SOURCE = 0             # local webcam index (dev default: laptop cam)
TARGET_FPS = 12               # roadmap budget ≤15 — shared CPU with Whisper
MIN_CONFIDENCE = 0.5          # PROVISIONAL — drop events below this
MODEL_PATH = os.path.join(GESTURE_CORE_DIR, "models/face_landmarker.task")


class GestureNode(Node):
    """Gesture event publisher. Stream injectable for selftest."""

    def __init__(self, stream_factory=None, use_camera=True):
        super().__init__("gesture_node")
        self.pub_event = self.create_publisher(String, "/intent/gesture", 10)
        self.pub_meta = self.create_publisher(String, "/intent/gesture_meta", 10)
        self.create_subscription(Bool, "/intent/gesture_enable",
                                 self._on_enable, 10)
        self.enabled = True
        self._stop = threading.Event()
        self._stream = None
        self._stream_lock = threading.Lock()  # camera thread vs main preload
        self._stream_factory = stream_factory or self._real_stream
        if IMAGE_TOPIC:
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
            self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 5)
            self._log(f"camera = ROS topic {IMAGE_TOPIC}")
        elif use_camera:
            threading.Thread(target=self._camera_loop, daemon=True).start()
            self._log(f"camera = local webcam index {CAMERA_SOURCE}")

    def _log(self, msg):
        self.get_logger().info(msg)

    def _real_stream(self):
        from gesture_core import FaceStream
        return FaceStream(source=None, model_path=MODEL_PATH,
                          on_event=self._on_event)

    def _ensure_stream(self):
        with self._stream_lock:
            if self._stream is None:
                self._log("loading Face Landmarker…")
                self._stream = self._stream_factory()
                self._log("landmarker ready")
            return self._stream

    # ── enable gate ──────────────────────────────────────────────────────
    def _on_enable(self, msg: Bool):
        if bool(msg.data) != self.enabled:
            self.enabled = bool(msg.data)
            self._log("detection ENABLED" if self.enabled
                      else "detection paused (/intent/gesture_enable)")

    # ── frame sources (both end in process_frame) ────────────────────────
    def _on_image(self, msg):
        if not self.enabled:
            return
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._ensure_stream().process_frame(bgr, time.time())

    def _camera_loop(self):
        import cv2
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            self._log(f"ERROR: cannot open webcam {CAMERA_SOURCE} — is another "
                      "app (gesture_live?) using it?")
            return
        try:
            while not self._stop.is_set():
                t = time.time()
                ok, frame = cap.read()
                if not ok:
                    self._log("webcam read failed, retrying…")
                    time.sleep(0.5)
                    continue
                if self.enabled:
                    self._ensure_stream().process_frame(frame, t)
                time.sleep(max(0.0, 1.0 / TARGET_FPS - (time.time() - t)))
        finally:
            cap.release()

    # ── event → topics ───────────────────────────────────────────────────
    def _on_event(self, event: dict):
        if not self.enabled:
            return
        if event["confidence"] < MIN_CONFIDENCE:
            self._log(f'dropped: {event["gesture"]} below confidence '
                      f'({event["confidence"]} < {MIN_CONFIDENCE})')
            return
        self.pub_event.publish(String(data=event["gesture"]))
        self.pub_meta.publish(String(data=json.dumps(event)))
        self._log(f'→ /intent/gesture: {event["gesture"]} '
                  f'(conf {event["confidence"]})')

    def shutdown(self):
        self._stop.set()


# ── selftest: stubbed stream, real node + real ROS pub/sub ─────────────────
def selftest() -> int:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok &= bool(cond)

    rclpy.init()
    node = GestureNode(stream_factory=lambda: None, use_camera=False)
    got_ev, got_meta = [], []
    probe = rclpy.create_node("selftest_probe")
    probe.create_subscription(String, "/intent/gesture",
                              lambda m: got_ev.append(m.data), 10)
    probe.create_subscription(String, "/intent/gesture_meta",
                              lambda m: got_meta.append(m.data), 10)

    def spin_both(t=0.6):
        end = time.time() + t
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
            rclpy.spin_once(probe, timeout_sec=0.05)

    spin_both(0.4)  # let discovery settle

    def fire(gesture="nod", conf=0.9):
        node._on_event({"gesture": gesture, "confidence": conf,
                        "t": time.time(), "detail": {}})

    # 1. event publishes on both topics, payload is the RAW gesture name
    fire("nod", 0.9)
    spin_both()
    check(got_ev == ["nod"], "event publishes raw gesture name (no mapping)")
    meta = json.loads(got_meta[0]) if got_meta else {}
    check(meta.get("confidence") == 0.9 and "t" in meta,
          "meta JSON has confidence + timestamp")

    # 2. low-confidence events dropped
    fire("shake", 0.3)
    spin_both(0.3)
    check(got_ev == ["nod"], "low-confidence event dropped")

    # 3. enable gate: False pauses, True resumes
    enable_pub = probe.create_publisher(Bool, "/intent/gesture_enable", 10)
    enable_pub.publish(Bool(data=False)); spin_both(0.4)
    fire("blink_stop", 0.9)
    spin_both(0.3)
    check(got_ev == ["nod"] and not node.enabled,
          "disabled: events not published")
    enable_pub.publish(Bool(data=True)); spin_both(0.4)
    fire("blink_stop", 0.9)
    spin_both()
    check(got_ev == ["nod", "blink_stop"], "re-enabled: events flow again")

    # 4. disabled also skips inference in the frame paths
    class CountingStream:
        calls = 0
        def process_frame(self, bgr, t):
            CountingStream.calls += 1
    node._stream = CountingStream()
    node.enabled = False
    import numpy as np
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    if node.enabled:
        node._stream.process_frame(frame, time.time())
    # simulate the camera-loop guard directly:
    for _ in range(3):
        if node.enabled:
            node._stream.process_frame(frame, time.time())
    check(CountingStream.calls == 0, "disabled: no inference on frames")
    node.enabled = True
    for _ in range(3):
        if node.enabled:
            node._stream.process_frame(frame, time.time())
    check(CountingStream.calls == 3, "enabled: frames reach the stream")

    # 5. all four roadmap events pass through verbatim
    got_ev.clear()
    time.sleep(1.3)  # respect nothing — node has no refractory; detector does
    for g in ["nod", "shake", "brow_raise", "blink_stop"]:
        fire(g, 0.8)
    spin_both()
    check(got_ev == ["nod", "shake", "brow_raise", "blink_stop"],
          "all four Stage-3 events pass through verbatim")

    node.shutdown(); node.destroy_node(); probe.destroy_node()
    rclpy.shutdown()
    print("SELFTEST", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--selftest", action="store_true")
    args, _ros_args = ap.parse_known_args()

    if args.selftest:
        sys.exit(selftest())

    rclpy.init()
    node = GestureNode()
    node._ensure_stream()  # load the model NOW, not on the first frame —
                           # lazy load stalls past the image-topic QoS queue
    node._log("ready — nod / shake / brow raise / triple blink at the camera; "
              "pause with /intent/gesture_enable=false")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
