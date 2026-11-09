"""Offline checks for the Live-mode face tracker in run_vcam.py (synthetic frames, no webcam).

Run: venv/bin/python tests/vcam_tracker_check.py
"""
import os
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
sys.argv = ["run_vcam.py"]

import cv2
import numpy as np

import run_vcam as rv
import modules.globals as g
from modules import ui
from modules.face_analyser import get_face_analyser

g.execution_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
g.many_faces = False
g.mouth_mask = False

IMAGES = os.path.join(PROJECT, "venv/insightface/data/images")
face_img = cv2.imread(os.path.join(IMAGES, "Tom_Hanks_54745.png"))
group = cv2.imread(os.path.join(IMAGES, "t1.jpg"))
H, W = 720, 1280
rng = np.random.default_rng(0)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def canvas(x, y, size=260, angle=0.0, gain=1.0, noise=6.0, base=None):
    frame = np.full((H, W, 3), 120, np.uint8) if base is None else base.copy()
    face = cv2.resize(face_img, (size, size))
    if angle:
        m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        face = cv2.warpAffine(face, m, (size, size), borderMode=cv2.BORDER_REPLICATE)
    frame[y:y + size, x:x + size] = face
    out = frame.astype(np.float32) * gain + rng.normal(0, noise, frame.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def reset():
    rv._track.update(filter=rv._OneEuro(), last=0.0, box=None, seen=0.0, face=None,
                     pending=rv._NOT_RUN, hook=0.0, calls=0)


get_face_analyser()  # Warm up models.
fa = get_face_analyser()

# 1. Hook is wired in place of upstream's detector.
check("upstream detector replaced", ui.detect_one_face_fast is rv._counted_detect)

# 2. Points come straight from the full-frame detector (no bias added).
reset()
f = canvas(500, 220, size=180)
b, k = fa.det_model.detect(f, max_num=0, metric="default")
face = rv._detect_best(f)
check("tracked points match the detector exactly", np.allclose(face.kps, k[0]), "")

# 3. Second, bigger-left face appears: upstream takes leftmost, tracker stays on you.
reset()
for i in range(5):
    rv._detect_best(canvas(700, 250))
mine = rv._track["box"].copy()
two = canvas(700, 250)
two[200:200 + 300, 60:60 + 300] = cv2.resize(face_img, (300, 300))
upstream_pick = rv._run_detector(two)
leftmost = min(upstream_pick, key=lambda f: f[0][0])
face = rv._detect_best(two)
check("stays on the same person with someone else in view",
      rv._iou(face.bbox, mine) > 0.5 and rv._iou(leftmost[0], mine) < 0.1,
      f"iou(tracked)={rv._iou(face.bbox, mine):.2f} iou(leftmost)={rv._iou(leftmost[0], mine):.2f}")
reset()
cold = rv._detect_best(two)
check("fresh start prefers the bigger/central face", cold.bbox[0] < 400,
      f"picked x={cold.bbox[0]:.0f} (300px face at left vs 260px)")

# 4. Dim room.
reset()
rv._detect_best(canvas(600, 230, gain=1.0))
dim = canvas(600, 230, size=220, gain=0.025, noise=3.0)
plain = rv._run_detector(dim)
got = rv._detect_best(dim)
check("very dark frame rescued", got is not None and len(plain) == 0, f"plain detector found {len(plain)}, tracker {'found' if got is not None else 'missed'}")

# 5. Tilted head: the detector handles it alone, so no rotated retry is needed.
reset()
got = rv._detect_best(canvas(600, 230, angle=60))
check("60 degree tilt found without retries", got is not None)

# 6. Upstream-like loop: detect every 2nd frame, swap every frame; misses bridged, one detection per frame.
reset()
calls = {"n": 0}
real = rv._run_detector


def counting(image):
    calls["n"] += 1
    return real(image)


rv._run_detector = counting
cached = None
swapped, flashes, det_per_frame = 0, 0, []
blank = np.full((H, W, 3), 120, np.uint8)
plan = ["face"] * 10 + ["blank"] * 3 + ["face"] * 6 + ["blank"] * 25
for i, kind in enumerate(plan):
    frame = canvas(600 + i * 3, 230) if kind == "face" else blank
    before = calls["n"]
    if i % 2 == 0:
        cached = rv._counted_detect(frame)
    target = rv._steady(cached, frame) if cached is not None else None
    det_per_frame.append(calls["n"] - before)
    if target is not None:
        swapped += 1
    elif kind == "face" or i < 13:
        flashes += 1
    time.sleep(1 / 30)
rv._run_detector = real
check("no real-face flash through a 3-frame miss", flashes == 0, f"flashes={flashes}, swapped={swapped}/{len(plan)}")
check("swap stops once the face is really gone", target is None)
first_face = [d for d, k in zip(det_per_frame, plan) if k == "face"]
check("exactly one detection per frame while tracking", all(d == 1 for d in first_face), f"{first_face}")

# 7. Timing: tracked detection vs upstream's detector on the same frames.
reset()
frames = [canvas(600 + i, 230) for i in range(30)]
rv._detect_best(frames[0])
t = time.perf_counter()
for fr in frames:
    rv._detect_best(fr)
ms_track = (time.perf_counter() - t) / 30 * 1000
from modules.face_analyser import detect_one_face_fast
t = time.perf_counter()
for fr in frames:
    detect_one_face_fast(fr)
ms_up = (time.perf_counter() - t) / 30 * 1000
check("tracking adds no real cost over upstream detection", ms_track < ms_up * 1.15 + 1,
      f"tracked {ms_track:.1f} ms vs upstream {ms_up:.1f} ms")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
