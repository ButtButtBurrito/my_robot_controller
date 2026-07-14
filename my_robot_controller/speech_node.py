#!/usr/bin/env python3
"""
speech_node — Stage 3 intent source: local Whisper ASR → /intent/speech.

Captures microphone audio gated by push-to-talk, transcribes the utterance
with faster-whisper (CPU, int8), and publishes the VERBATIM transcript.
This node is a *producer* of intent text only (THESIS_ROADMAP Stage 3):
it must NEVER parse commands, decide actions, or touch anything that moves.
Downstream (intent_router / LLM planner) consumes /intent/speech.

Topics OUT:
    /intent/speech       std_msgs/String   raw transcript, one msg/utterance
    /intent/speech_meta  std_msgs/String   JSON: {"text", "avg_logprob",
                          "duration_s", "latency_s", "rtf", "language"}
Topics IN:
    /intent/ptt          std_msgs/Bool     True = start listening,
                          False = stop + transcribe (remote PTT, e.g. a GUI)

Push-to-talk (either source, whichever comes first):
  - terminal: ENTER toggles listening on/off (line-based, works under the
    GUI Custom Script Runner's terminal too)
  - topic:    /intent/ptt Bool (for a future GUI button / foot switch)

Config constants below (MODEL_SIZE/LANGUAGE/INITIAL_PROMPT) are PROVISIONAL
until the Taglish benchmark (~/ros2_ws/speech/, Whisper Lab step 3-4) picks
the final condition — update them when the decision table exists.

Test without mic or model:
    python3 speech_node.py --selftest      # stubbed capture + transcriber
Test the real model + ROS path without a mic:
    python3 speech_node.py --wav ~/ros2_ws/speech/corpus/clip_001.wav
    # (separate shell)  ros2 topic echo /intent/speech
Run for real (mic PTT):
    python3 speech_node.py                 # ENTER to talk, ENTER to stop
"""

import argparse
import json
import statistics
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

# ── config (PROVISIONAL until the Whisper Lab benchmark decides) ─────────────
MODEL_SIZE = "small"          # tiny | base | small — idle RTF 0.09/0.16/0.46
LANGUAGE = "tl"               # None = auto-detect (benchmark condition)
INITIAL_PROMPT = (            # decoder bias toward the command grammar
    "Kunin mo yung red block. Ilagay mo sa station B. "
    "Balik ka sa home position. Stop. Buksan mo yung gripper."
)
SAMPLE_RATE = 16000
CPU_THREADS = 4
BEAM_SIZE = 5
MIN_UTTERANCE_S = 0.4         # shorter captures are dropped, not transcribed
SILENT_PEAK = 300             # |int16| below this ⇒ mic muted/dead, warn


class MicCapture:
    """Push-to-talk microphone capture (16 kHz mono int16)."""

    def __init__(self, log):
        self._log = log
        self._stream = None
        self._chunks = []

    def start(self):
        import sounddevice as sd
        self._chunks = []
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            callback=lambda data, *_: self._chunks.append(bytes(data)))
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        raw = b"".join(self._chunks)
        return np.frombuffer(raw, dtype=np.int16)


class WhisperTranscriber:
    """Lazy-loaded faster-whisper wrapper. transcribe() -> meta dict."""

    def __init__(self, log):
        self._log = log
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._log(f"loading Whisper '{MODEL_SIZE}' (int8, {CPU_THREADS} threads)…")
            from faster_whisper import WhisperModel
            self._model = WhisperModel(MODEL_SIZE, device="cpu",
                                       compute_type="int8",
                                       cpu_threads=CPU_THREADS)
            self._log("model ready")

    def transcribe(self, audio_int16: np.ndarray) -> dict:
        self._ensure_model()
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio_int16.astype(np.float32) / 32768.0,
            language=LANGUAGE, beam_size=BEAM_SIZE,
            initial_prompt=INITIAL_PROMPT, temperature=0.0)
        segments = list(segments)  # generator — consuming it IS the inference
        latency = time.perf_counter() - t0
        return {
            "text": " ".join(s.text.strip() for s in segments),
            "avg_logprob": (statistics.mean(s.avg_logprob for s in segments)
                            if segments else 0.0),
            "duration_s": round(info.duration, 2),
            "latency_s": round(latency, 2),
            "rtf": round(latency / max(info.duration, 0.1), 2),
            "language": info.language,
        }


class SpeechNode(Node):
    """PTT-gated ASR publisher. capture/transcriber injectable for selftest."""

    def __init__(self, capture=None, transcriber=None):
        super().__init__("speech_node")
        self.pub_text = self.create_publisher(String, "/intent/speech", 10)
        self.pub_meta = self.create_publisher(String, "/intent/speech_meta", 10)
        self.create_subscription(Bool, "/intent/ptt", self._on_ptt_msg, 10)
        self.capture = capture or MicCapture(self._log)
        self.transcriber = transcriber or WhisperTranscriber(self._log)
        self._listening = False
        self._lock = threading.Lock()  # ENTER and /intent/ptt may race

    def _log(self, msg):
        self.get_logger().info(msg)

    # ── PTT gate (both sources funnel here) ──────────────────────────────
    def _on_ptt_msg(self, msg: Bool):
        self.set_listening(bool(msg.data), source="topic")

    def set_listening(self, on: bool, source: str = "key"):
        with self._lock:
            if on == self._listening:
                return
            self._listening = on
            if on:
                try:
                    self.capture.start()
                    self._log(f"● LISTENING ({source})")
                except Exception as e:
                    self._listening = False
                    self._log(f"mic error: {e}")
            else:
                audio = self.capture.stop()
                self._log("■ stopped, transcribing…")
                self._process(audio)

    # ── utterance → topics ───────────────────────────────────────────────
    def _process(self, audio: np.ndarray):
        dur = len(audio) / SAMPLE_RATE
        if dur < MIN_UTTERANCE_S:
            self._log(f"dropped: too short ({dur:.2f}s)")
            return
        if len(audio) and int(np.abs(audio).max()) < SILENT_PEAK:
            self._log("dropped: SILENT capture — mic muted? "
                      "(RUNBOOK 'Known hazards' has the fix)")
            return
        try:
            meta = self.transcriber.transcribe(audio)
        except Exception as e:
            self._log(f"transcription error: {e}")
            return
        text = meta["text"].strip()
        if not text:
            self._log("dropped: empty transcript")
            return
        self.pub_text.publish(String(data=text))
        self.pub_meta.publish(String(data=json.dumps(meta)))
        self._log(f'→ /intent/speech: "{text}"  '
                  f'({meta["duration_s"]}s → {meta["latency_s"]}s, '
                  f'logprob {meta["avg_logprob"]:.2f})')


def _stdin_ptt_loop(node: SpeechNode):
    """ENTER toggles listening. Daemon thread; EOF (no TTY) exits quietly."""
    while True:
        try:
            input()
        except EOFError:
            return
        node.set_listening(not node._listening, source="key")


# ── selftest: stubbed mic + model, real node + real ROS publishing ─────────
def selftest() -> int:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok &= bool(cond)

    class FakeCapture:
        def __init__(self, audio):
            self.audio = audio

        def start(self):
            pass

        def stop(self):
            return self.audio

    class FakeTranscriber:
        def __init__(self, text="kunin mo yung red block"):
            self.text = text
            self.calls = 0

        def transcribe(self, audio):
            self.calls += 1
            return {"text": self.text, "avg_logprob": -0.3,
                    "duration_s": len(audio) / SAMPLE_RATE,
                    "latency_s": 0.5, "rtf": 0.2, "language": "tl"}

    rclpy.init()
    speech = (np.sin(np.linspace(0, 4000, SAMPLE_RATE * 2)) * 8000).astype(np.int16)
    node = SpeechNode(capture=FakeCapture(speech),
                      transcriber=FakeTranscriber())
    got_text, got_meta = [], []
    probe = rclpy.create_node("selftest_probe")
    probe.create_subscription(String, "/intent/speech",
                              lambda m: got_text.append(m.data), 10)
    probe.create_subscription(String, "/intent/speech_meta",
                              lambda m: got_meta.append(m.data), 10)

    def spin_both(t=1.0):
        end = time.time() + t
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
            rclpy.spin_once(probe, timeout_sec=0.05)

    # 1. full PTT cycle publishes both topics
    node.set_listening(True)
    node.set_listening(False)
    spin_both()
    check(got_text == ["kunin mo yung red block"], "PTT cycle publishes transcript")
    meta = json.loads(got_meta[0]) if got_meta else {}
    check(meta.get("language") == "tl" and "latency_s" in meta,
          "meta JSON has language + latency")
    check(node.transcriber.calls == 1, "transcriber called exactly once")

    # 2. redundant PTT transitions are no-ops
    node.set_listening(False)
    check(node.transcriber.calls == 1, "duplicate stop is a no-op")

    # 3. too-short capture dropped
    node.capture = FakeCapture(speech[: int(0.2 * SAMPLE_RATE)])
    node.set_listening(True); node.set_listening(False)
    spin_both(0.3)
    check(node.transcriber.calls == 1 and len(got_text) == 1,
          "short capture dropped before transcription")

    # 4. silent capture dropped
    node.capture = FakeCapture(np.zeros(SAMPLE_RATE, dtype=np.int16))
    node.set_listening(True); node.set_listening(False)
    spin_both(0.3)
    check(node.transcriber.calls == 1, "silent capture dropped (mic-mute guard)")

    # 5. empty transcript not published
    node.capture = FakeCapture(speech)
    node.transcriber = FakeTranscriber(text="  ")
    node.set_listening(True); node.set_listening(False)
    spin_both(0.3)
    check(len(got_text) == 1, "empty transcript not published")

    # 6. /intent/ptt topic drives the same gate
    node.capture = FakeCapture(speech)
    node.transcriber = FakeTranscriber(text="ilagay mo sa station b")
    pub = probe.create_publisher(Bool, "/intent/ptt", 10)
    pub.publish(Bool(data=True)); spin_both(0.4)
    pub.publish(Bool(data=False)); spin_both(0.6)
    check(got_text[-1:] == ["ilagay mo sa station b"],
          "topic PTT publishes transcript")

    node.destroy_node(); probe.destroy_node()
    rclpy.shutdown()
    print("SELFTEST", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--wav", metavar="FILE",
                    help="transcribe FILE once, publish, exit (no mic needed)")
    args, _ros_args = ap.parse_known_args()

    if args.selftest:
        sys.exit(selftest())

    rclpy.init()
    node = SpeechNode()
    if args.wav:
        import wave
        with wave.open(args.wav, "rb") as w:
            assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1, \
                f"--wav expects {SAMPLE_RATE} Hz mono (Whisper Lab clips are)"
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        node._log(f"transcribing {args.wav} ({len(audio)/SAMPLE_RATE:.1f}s)…")
        node._process(audio)
        # spin briefly so subscribers with a late join still get the latched-ish burst
        end = time.time() + 1.0
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
    else:
        threading.Thread(target=_stdin_ptt_loop, args=(node,), daemon=True).start()
        node._log("ready — ENTER to talk, ENTER again to stop (or /intent/ptt)")
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
