#!/usr/bin/env python3
"""
topdown_viewer — Stage 2 groundwork (see ~/sessions/THESIS_ROADMAP.md).

Receives the phone's camera over WiFi and "sees" it: detects all ArUco
markers in every frame and reports their 2D pixel positions — the raw
material for the top-down marker map. Standalone: needs only python3 +
OpenCV, no ROS, no build (single file). The Stage 2 `scene_map` node will
later reuse this capture+detect loop and add intrinsics + extrinsics to turn
pixels into base_link coordinates.

PHONE SETUP (one-time)
  Android: install "IP Webcam" (free, no account). Open it → scroll down →
    "Start server". It shows e.g.  http://192.168.1.42:8080
    The MJPEG video stream is that URL + /video
  iPhone: any RTSP/MJPEG camera app works; give this script the URL it shows.
  Tips: same WiFi as the laptop, phone plugged in, mount pointing straight
  down at the table. In IP Webcam, 1280x720 @ ~10 fps is plenty for markers
  (Video preferences → resolution / fps limit) and easy on this laptop.

RUN
  python3 topdown_viewer.py            # asks for the phone's IP/URL
                                       # (remembers the last one as default)
  python3 topdown_viewer.py 192.168.1.6              # bare IP is fine
  python3 topdown_viewer.py http://192.168.1.6:8080/video
  python3 topdown_viewer.py --selftest # no phone needed: synthetic frames

The video window is resizable — drag its corner (aspect ratio is kept).
Window keys:  q quit   s save annotated snapshot to ~/topdown_snaps/
Console: one summary line per second — marker id → pixel (u,v) + normalised
(0..1) coordinates. Uses DICT_ARUCO_ORIGINAL — the same dictionary family
aruco_ros uses, so the SAME printed markers (582, 100, 101, ...) work for
both the wrist camera and the phone.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Fallback stream URL — the interactive prompt and ~/.topdown_url (last
# successfully used) both take precedence over this.
STREAM_URL = 'http://192.168.3.231/video'

DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
PARAMS = cv2.aruco.DetectorParameters_create()

SNAP_DIR = Path.home() / 'topdown_snaps'
URL_MEMORY = Path.home() / '.topdown_url'
WINDOW = 'topdown'


def normalize_url(s: str) -> str:
    """Accept whatever the user types — '192.168.1.6', '192.168.1.6:8080',
    'http://192.168.1.6:8080' or a full stream URL — and return a usable
    stream URL (IP Webcam's MJPEG endpoint is <base>/video)."""
    s = s.strip().rstrip('/')
    if not s:
        return s
    if '://' not in s:
        s = 'http://' + s
    scheme, rest = s.split('://', 1)
    host, _, path = rest.partition('/')
    if ':' not in host:
        host += ':8080'
    if scheme == 'http' and path in ('', 'video'):
        path = 'video'
        return f'{scheme}://{host}/{path}'
    return f'{scheme}://{host}/{path}' if path else f'{scheme}://{host}'


def ask_url() -> str:
    """Prompt for the phone's address, defaulting to the last-used URL."""
    default = STREAM_URL
    try:
        default = URL_MEMORY.read_text().strip() or default
    except OSError:
        pass
    print('Enter the address shown by the IP Webcam app on the phone')
    print('(a bare IP like 192.168.1.6 is fine).')
    typed = input(f'Phone address [{default}]: ').strip()
    return normalize_url(typed) if typed else default


def remember_url(url: str):
    try:
        URL_MEMORY.write_text(url + '\n')
    except OSError:
        pass


def detect(frame_bgr):
    """Return {marker_id: (u_px, v_px)} centres + raw corners for drawing."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, DICT, parameters=PARAMS)
    found = {}
    if ids is not None:
        for mid, c in zip(ids.flatten(), corners):
            centre = c[0].mean(axis=0)          # 4 corners → centre
            found[int(mid)] = (float(centre[0]), float(centre[1]))
    return found, corners, ids


def annotate(frame, corners, ids, found):
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    for mid, (u, v) in found.items():
        cv2.drawMarker(frame, (int(u), int(v)), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f'{mid} ({u:.0f},{v:.0f})',
                    (int(u) + 12, int(v) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def open_stream(url: str):
    print(f'[STREAM] Connecting to {url} ...', flush=True)
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'[STREAM] Connected: {w}x{h}', flush=True)
        return cap
    cap.release()
    return None


def run(url: str) -> int:
    SNAP_DIR.mkdir(exist_ok=True)
    # WINDOW_NORMAL = user-resizable (drag the corner); KEEPRATIO avoids
    # squashing the image while resizing.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, 960, 540)
    cap = None
    last_report = 0.0
    frames = 0
    fps_t0 = time.monotonic()
    fps = 0.0
    while True:
        if cap is None:
            cap = open_stream(url)
            if cap is None:
                print('[STREAM] Connect failed — retrying in 2 s '
                      '(is IP Webcam started? same WiFi? URL ends in /video?)',
                      flush=True)
                time.sleep(2.0)
                continue
            remember_url(url)          # connected once → default for next run
        ok, frame = cap.read()
        if not ok:
            print('[STREAM] Frame read failed — reconnecting...', flush=True)
            cap.release()
            cap = None
            continue

        frames += 1
        now = time.monotonic()
        if now - fps_t0 >= 2.0:
            fps = frames / (now - fps_t0)
            frames = 0
            fps_t0 = now

        found, corners, ids = detect(frame)
        h, w = frame.shape[:2]

        if now - last_report >= 1.0:
            if found:
                parts = [f'{mid}: px({u:.0f},{v:.0f}) norm({u/w:.3f},{v/h:.3f})'
                         for mid, (u, v) in sorted(found.items())]
                print(f'[{time.strftime("%H:%M:%S")}] {fps:.1f} fps | '
                      + '  '.join(parts), flush=True)
            else:
                print(f'[{time.strftime("%H:%M:%S")}] {fps:.1f} fps | '
                      'no markers in view', flush=True)
            last_report = now

        annotate(frame, corners, ids, found)
        cv2.putText(frame, f'{fps:.1f} fps  {len(found)} marker(s)',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            p = SNAP_DIR / f'snap_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
            cv2.imwrite(str(p), frame)
            print(f'[SNAP] saved {p}', flush=True)
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test: synthetic frames, no phone required
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> int:
    fails = 0

    def check(name, cond, detail=''):
        nonlocal fails
        print(f'  {"PASS" if cond else "FAIL"}  {name}  {detail}')
        fails += 0 if cond else 1

    # Build a fake top-down photo: three markers (the thesis trio) pasted at
    # known pixel positions on a light "table", with margins, plus one rotated.
    canvas = np.full((720, 1280, 3), 190, np.uint8)
    placements = {100: (200, 180, 0), 101: (900, 200, 0), 582: (550, 500, 30)}
    side = 120
    for mid, (x, y, angle) in placements.items():
        tile = cv2.aruco.drawMarker(DICT, mid, side)
        pad = 30   # white quiet zone, required by the detector
        padded = np.full((side + 2 * pad,) * 2, 255, np.uint8)
        padded[pad:pad + side, pad:pad + side] = tile
        if angle:
            M = cv2.getRotationMatrix2D(((side + 2 * pad) / 2,) * 2, angle, 1.0)
            padded = cv2.warpAffine(padded, M, padded.shape[::-1],
                                    borderValue=255)
        bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
        hh, ww = bgr.shape[:2]
        canvas[y:y + hh, x:x + ww] = bgr

    found, corners, ids = detect(canvas)
    check('all 3 thesis markers detected', set(found) == {100, 101, 582},
          f'found {sorted(found)}')
    for mid, (x, y, _) in placements.items():
        if mid not in found:
            continue
        expected = (x + side / 2 + 30, y + side / 2 + 30)   # centre incl. pad
        err = np.hypot(found[mid][0] - expected[0], found[mid][1] - expected[1])
        check(f'marker {mid} centre within 3 px', err < 3.0, f'err={err:.1f} px')

    empty_found, *_ = detect(np.full((480, 640, 3), 190, np.uint8))
    check('empty frame → no detections', empty_found == {})

    cases = {
        '192.168.1.6':                   'http://192.168.1.6:8080/video',
        '192.168.1.6:8080':              'http://192.168.1.6:8080/video',
        'http://192.168.1.6:8080':       'http://192.168.1.6:8080/video',
        'http://192.168.1.6:8080/video': 'http://192.168.1.6:8080/video',
        ' 10.0.0.5:9000/ ':              'http://10.0.0.5:9000/video',
        'rtsp://10.0.0.5:554/stream':    'rtsp://10.0.0.5:554/stream',
    }
    bad = {k: (normalize_url(k), v) for k, v in cases.items()
           if normalize_url(k) != v}
    check('normalize_url handles all input forms', not bad, str(bad))

    out = annotate(canvas.copy(), corners, ids, found)
    check('annotate returns a frame', out.shape == canvas.shape)

    p = Path('/tmp/claude-1000') if Path('/tmp/claude-1000').exists() else Path('/tmp')
    snap = p / 'topdown_selftest.jpg'
    cv2.imwrite(str(snap), out)
    print(f'  (annotated self-test frame saved to {snap})')

    print(f'\n{"ALL PASS" if fails == 0 else f"{fails} FAILURES"}')
    return 1 if fails else 0


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(_selftest())
    arg = next((a for a in sys.argv[1:] if not a.startswith('-')), None)
    url = normalize_url(arg) if arg else ask_url()
    sys.exit(run(url))
