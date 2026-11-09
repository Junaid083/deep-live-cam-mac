#!/usr/bin/env python3
"""Deep-Live-Cam with its live preview also streamed to OBS Virtual Camera.

Local addition (not part of upstream): wraps the stock app without editing it,
so `git pull` keeps working. Pick "OBS Virtual Camera" in Meet/Zoom.
"""

import os
import sys

import cv2
import numpy as np

# `deeplive photo.jpg` pre-selects the face to wear (upstream's -s would skip the UI).
_photo = None
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    _photo = os.path.abspath(os.path.expanduser(sys.argv.pop(1)))
    if not os.path.isfile(_photo):
        sys.exit(f"Photo not found: {_photo}")
# --head off (default): plain Deep-Live-Cam face swap, the most natural look.
# --head hair: experimental, the photo's hair and hairline on top of the swap.
# --head full: experimental, also the photo's cheeks/jaw, and your own hair painted out.
_head_mode = "off"
if "--head" in sys.argv:
    index = sys.argv.index("--head")
    _head_mode = sys.argv[index + 1] if index + 1 < len(sys.argv) else ""
    del sys.argv[index:index + 2]
    if _head_mode not in ("full", "hair", "off"):
        sys.exit("--head must be full, hair or off")
if "--execution-provider" not in sys.argv:
    sys.argv += ["--execution-provider", "coreml"]

import run  # noqa: F401  Sets up PATH and prints the banner, like run.py.
import modules.globals
from modules import ui

_camera = None
_failed = False


def _send(frame: np.ndarray) -> None:
    global _camera, _failed
    if _failed:
        return
    height, width = frame.shape[:2]
    if _camera is None or (_camera.width, _camera.height) != (width, height):
        try:
            import pyvirtualcam

            if _camera is not None:
                _camera.close()
            _camera = pyvirtualcam.Camera(width, height, 30, fmt=pyvirtualcam.PixelFormat.BGR,
                                          backend="obs")
            print(f"[vcam] Streaming to {_camera.device} ({width}x{height}).", flush=True)
        except Exception as error:  # Keep the preview working without OBS.
            _failed = True
            print(f"[vcam] OBS Virtual Camera not available, preview only: {error}", flush=True)
            return
    # Meeting viewers should see the true orientation, not the mirrored preview.
    if modules.globals.live_mirror:
        frame = cv2.flip(frame, 1)
    _camera.send(np.ascontiguousarray(frame))


def _tick(self) -> None:
    if self._stop_event.is_set():
        self.close()
        return
    try:
        bgr_frame = self._processed_queue.get_nowait()
    except ui.queue.Empty:
        return
    _send(bgr_frame)
    bgr_frame = ui.fit_image_to_size(bgr_frame, self.width(), self.height())
    self._image_label.setPixmap(ui._bgr_to_qpixmap(bgr_frame))


ui.WebcamPreviewWindow._tick = _tick

import vcam_random  # noqa: E402  Women/Men/beard filter for the random-face button.

vcam_random.install(ui)

# Live health line every 2 s: was a face found, and did the swap run?
import time
from modules.processors.frame import face_swapper

_stats = {"frames": 0, "faces": 0, "swaps": 0, "held": 0, "rescued": 0, "no_source": 0,
          "since": time.time()}
_swap = face_swapper.swap_face


_source = {"key": None, "face": None}
_head = {"layer": None, "failed": False, "ms": 0.0, "count": 0}


def _head_layer():
    if _head["layer"] is None and _head_mode != "off":
        import vcam_head

        _head["layer"] = vcam_head.HeadTransplant(modules.globals.execution_providers, _head_mode)
        if not _head["layer"].enabled:
            _head["failed"] = True
    return None if _head["failed"] else _head["layer"]


def _find_face(image):
    """The detector misses faces that fill the whole photo (common for close-up
    portraits and the random-face site): retry with a plain border around it."""
    from modules.face_analyser import get_many_faces

    def largest(picture):  # Upstream takes the leftmost face; the main subject is the biggest.
        faces = get_many_faces(picture) or []
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), default=None)

    face = largest(image)
    for pad in (0.25, 0.6):
        if face is not None:
            break
        border = int(max(image.shape[:2]) * pad)
        padded = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT,
                                    value=[int(v) for v in np.median(image.reshape(-1, 3), axis=0)])
        face = largest(padded)
        if face is not None:
            image = padded
    return image, face


def _current_source_face():
    """The selected face, reloaded whenever the photo file changes.

    Upstream only reloads when the path changes, but the random-face button
    always reuses one temp path, so a new random face was never picked up; and
    a read during the download left the swap off for good. Retrying on the
    file's modification time fixes both.
    """
    path = modules.globals.source_path
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path))
    except (TypeError, OSError):
        return None
    now = time.time()
    # A failed read is retried every 2 s (the file may still be downloading), not every frame.
    if key != _source["key"] or (_source["face"] is None and now - _source.get("tried", 0) > 2):
        from modules import imread_unicode

        image = imread_unicode(path)
        image, face = _find_face(image) if image is not None else (None, None)
        if face is None and key != _source["key"]:
            print(f"[vcam] No face found in {os.path.basename(path)}: pick a clearer photo.", flush=True)
        _source.update(key=key, face=face, tried=now)
        if face is not None:
            print(f"[vcam] Now wearing: {os.path.basename(path)}", flush=True)
            layer = _head_layer()
            if layer is not None:
                try:
                    from modules.face_analyser import ensure_landmarks

                    ensure_landmarks(image, [face])
                    for note in layer.set_source(image, face.landmark_2d_106):
                        print(f"[head] Note: {note}.", flush=True)
                    print(f"[head] Hair ready ({_head_mode} mode).", flush=True)
                except Exception as error:
                    layer.source = None
                    print(f"[head] Could not prepare this photo, using the plain swap: {error}", flush=True)
    return _source["face"]


class _OneEuro:
    """Removes jitter from still faces without adding lag to moving ones (Casiez 2012)."""

    def __init__(self, min_cutoff=1.5, beta=0.05):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.value = self.speed = self.time = None

    def __call__(self, value, now):
        if self.value is None or now <= self.time or value.shape != self.value.shape:
            self.value, self.speed, self.time = value.copy(), np.zeros_like(value), now
            return value
        dt, self.time = now - self.time, now
        alpha = lambda cutoff: 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))
        self.speed += alpha(1.0) * ((value - self.value) / dt - self.speed)
        cutoff = self.min_cutoff + self.beta * np.abs(self.speed)
        self.value = self.value + alpha(cutoff) * (value - self.value)
        return self.value


# --- Face tracking (single-face Live mode) ------------------------------------
# Upstream detects every other frame, takes the leftmost face, and stops swapping
# on the first missed frame, so your real face flashes through. This tracker
# detects once per frame, follows the same person, retries very dark frames, and
# bridges short misses with the last smoothed position.
HOLD_SECONDS = 0.3   # Keep swapping through a miss this short (a blink of a turn, a hand).
LOST_SECONDS = 1.0   # After this long without a face, stop preferring the old spot.
_NOT_RUN = object()
_track = {"filter": _OneEuro(), "last": 0.0, "box": None, "seen": 0.0, "face": None,
          "pending": _NOT_RUN, "hook": 0.0, "calls": 0}
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def _iou(a, b) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def _run_detector(image):
    """(box, kps, score) for every face in the image."""
    from modules.face_analyser import get_face_analyser

    bboxes, kpss = get_face_analyser().det_model.detect(image, max_num=0, metric="default")
    if bboxes.shape[0] == 0 or kpss is None:
        return []
    return [(box[:4].astype(np.float32), kps.astype(np.float32), float(box[4]))
            for box, kps in zip(bboxes, kpss)]


def _brightened(frame):
    """Detector-only contrast boost for dim rooms; the video itself is untouched."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _pick(found, shape, last_box):
    """The same person as last frame; otherwise the biggest, most central face."""
    if not found:
        return None
    if last_box is not None:
        best = max(found, key=lambda f: _iou(f[0], last_box))
        if _iou(best[0], last_box) > 0.2:
            return best
    height, width = shape[:2]

    def prominence(f):
        box = f[0]
        off_centre = np.hypot((box[0] + box[2]) / (2 * width) - 0.5, (box[1] + box[3]) / (2 * height) - 0.5)
        return (box[2] - box[0]) * (box[3] - box[1]) * (1.0 - min(0.8, off_centre))

    return max(found, key=prominence)


def _detect_best(frame):
    """One tracked detection for this frame."""
    from insightface.app.common import Face

    now = time.perf_counter()
    _track["calls"] += 1
    recent = _track["box"] if now - _track["seen"] < LOST_SECONDS else None
    found = _run_detector(frame)
    dark = not found and float(frame[::8, ::8].mean()) < 90
    # Retry only while a face was just here (or every 3rd frame), so a dark empty
    # room does not cost two detections on every frame.
    if dark and (recent is not None or _track["calls"] % 3 == 0):
        found = _run_detector(_brightened(frame))
        _stats["rescued"] += bool(found)
    _stats["frames"] += 1
    pick = _pick(found, frame.shape, recent)
    if pick is None:
        return None
    _stats["faces"] += 1
    _track["box"], _track["seen"] = pick[0], now
    return Face(bbox=pick[0], kps=pick[1], det_score=pick[2])


def _held_face(now):
    """The last smoothed face, if it was seen recently enough to bridge a miss."""
    if _track["face"] is not None and now - _track["seen"] < HOLD_SECONDS:
        return _track["face"]
    return None


def _counted_detect(frame):
    """Replaces upstream's live detection. A short miss returns the held face, so
    upstream keeps swapping instead of flashing your real face."""
    now = time.perf_counter()
    face = _detect_best(frame)
    _track["hook"], _track["pending"] = now, (face,)  # _steady reuses it for this frame.
    _report()
    return face if face is not None else _held_face(now)


def _steady(target_face, frame):
    """Exactly one detection per frame (upstream's, or ours on the frames it skips),
    smoothed eye/nose/mouth points so the face does not shimmer, and the held face
    through short misses. None means: do not swap this frame.
    Only for single-face Live mode; Many faces and file processing keep upstream behaviour."""
    if modules.globals.many_faces or target_face is None:
        return target_face
    now = time.perf_counter()
    if now - _track["hook"] > 1.0:  # Not in Live mode (e.g. processing a file).
        return target_face
    from insightface.app.common import Face

    pending, _track["pending"] = _track["pending"], _NOT_RUN
    fresh = pending[0] if pending is not _NOT_RUN else _detect_best(frame)
    if fresh is None:
        held = _held_face(now)
        _stats["held"] += held is not None
        return held
    if now - _track["last"] > 0.5:  # Face was lost for a moment: start clean.
        _track["filter"] = _OneEuro()
    _track["last"] = now
    raw = np.asarray(fresh.kps, np.float32)
    face = Face(dict(fresh))
    face.kps = _track["filter"](raw, now)
    face.bbox = np.asarray(fresh.bbox, np.float32) + np.tile(face.kps.mean(0) - raw.mean(0), 2)
    if modules.globals.mouth_mask:  # The mouth mask needs the detailed landmarks.
        from modules.face_analyser import ensure_landmarks

        ensure_landmarks(frame, [face])
    _track["face"] = face
    return face


def _counted_swap(source_face, target_face, frame):
    source_face = _current_source_face() or source_face
    if source_face is None:
        _stats["no_source"] += 1
        return frame
    target_face = _steady(target_face, frame)
    if target_face is None:
        return frame
    _stats["swaps"] += 1
    result = _swap(source_face, target_face, frame)
    layer = _head_layer()
    if layer is None or layer.source is None or modules.globals.many_faces:
        return result
    started = time.perf_counter()
    try:
        from modules.face_analyser import ensure_landmarks

        ensure_landmarks(frame, [target_face])
        points = getattr(target_face, "landmark_2d_106", None)
        result = layer.apply(result, frame, points, started)
    except Exception as error:  # Never let the extra layer break the live video.
        _head["failed"] = True
        print(f"[head] Turned off after an error, plain swap continues: {error}", flush=True)
    _head["ms"] += (time.perf_counter() - started) * 1000
    _head["count"] += 1
    return result


def _report():
    now = time.time()
    if now - _stats["since"] < 2:
        return
    frames = max(_stats["frames"], 1)
    message = f"[vcam] face found in {_stats['faces']}/{frames} checks, {_stats['swaps']} frames swapped"
    if _stats["rescued"]:
        message += f", {_stats['rescued']} dark frames recovered"
    if _stats["held"]:
        message += f", {_stats['held']} short misses bridged"
    if _head["count"]:
        message += f", head layer {_head['ms'] / _head['count']:.0f} ms"
        _head.update(ms=0.0, count=0)
    if _stats["no_source"]:
        message += f", {_stats['no_source']} skipped: no face found in the selected face photo"
    elif not _stats["faces"]:
        message += " - no face seen: face the camera, add light, or set Det size to 320"
    print(message, flush=True)
    _stats.update(frames=0, faces=0, swaps=0, held=0, rescued=0, no_source=0, since=now)


ui.detect_one_face_fast = _counted_detect
face_swapper.swap_face = _counted_swap

if __name__ == "__main__":
    from modules import core

    if _photo:
        _parse_args = core.parse_args

        def parse_args() -> None:
            _parse_args()
            modules.globals.source_path = _photo
            print(f"[vcam] Face: {os.path.basename(_photo)} - click Live to start.", flush=True)

        core.parse_args = parse_args
    core.run()
