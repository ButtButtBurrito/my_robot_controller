#!/usr/bin/env python3
"""
offline_eih_test — one-click offline test for the GUI's Custom Script Runner.

The runner holds a single script slot (starting a second script stops the
first), but the offline EIH test needs TWO processes in a fixed order:
fake_marker_publisher BEFORE pick_and_place (so ensure_detector() sees a
publisher and does not spawn aruco_ros). This wrapper runs both:

  1. starts fake_marker_publisher.py (child), waits for its topic,
  2. runs pick_and_place.py to completion (EXECUTE flag read from that
     file as usual — plan-only by default),
  3. on any exit — including the GUI's Stop Script, which signals the whole
     process group — the publisher is shut down too.

Use with MoveIt MOCK hardware (GUI: Sim checked) + RViz. TESTING ONLY.
"""

import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLISHER = HERE / 'fake_marker_publisher.py'
PICK_AND_PLACE = HERE / 'pick_and_place.py'


def log(msg: str):
    print(f'[{time.strftime("%H:%M:%S")}] WRAPPER: {msg}', flush=True)


def main() -> int:
    pub = subprocess.Popen([sys.executable, '-u', str(PUBLISHER)])
    log(f'fake_marker_publisher started (pid {pub.pid}); giving it 3 s...')
    time.sleep(3.0)
    if pub.poll() is not None:
        log(f'fake_marker_publisher exited early (code {pub.returncode}) — '
            'aborting. Is the workspace sourced / MoveIt mock running?')
        return 1

    code = 1
    try:
        log('starting pick_and_place.py (its own EXECUTE flag applies)...')
        code = subprocess.call([sys.executable, '-u', str(PICK_AND_PLACE)])
        log(f'pick_and_place finished with exit code {code}.')
        if code < 0:
            log('(negative code = killed by a signal during rclpy teardown — '
                'known quirk, harmless; trust the "Result: SUCCESS/FAILED" '
                'line above.)')
    except KeyboardInterrupt:
        log('interrupted — shutting down.')
    finally:
        if pub.poll() is None:
            pub.send_signal(signal.SIGINT)
            try:
                pub.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pub.kill()
        log('fake_marker_publisher stopped.')
    return code


if __name__ == '__main__':
    sys.exit(main())
