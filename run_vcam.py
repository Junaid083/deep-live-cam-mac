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

# Live health line every 2 s: was a face found, and did the swap run?
import time
from modules.processors.frame import face_swapper

_stats = {"frames": 0, "faces": 0, "swaps": 0, "no_source": 0, "since": time.time()}
_detect, _swap = ui.detect_one_face_fast, face_swapper.swap_face


def _counted_detect(frame):
    face = _detect(frame)
    _stats["frames"] += 1
    _stats["faces"] += face is not None
    _report()
    return face


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
    from modules.face_analyser import get_one_face

    face = get_one_face(image)
    for pad in (0.25, 0.6):
        if face is not None:
            break
        border = int(max(image.shape[:2]) * pad)
        padded = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT,
                                    value=[int(v) for v in np.median(image.reshape(-1, 3), axis=0)])
        face = get_one_face(padded)
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


_track = {"filter": _OneEuro(), "last": 0.0}


def _steady(target_face, frame):
    """Precise placement: fresh detection on every frame (upstream reuses one for
    two frames), and smoothed eye/nose/mouth points so the face does not shimmer.
    Only for the single-face mode; Many faces keeps upstream behaviour."""
    if modules.globals.many_faces or target_face is None:
        return target_face
    from insightface.app.common import Face

    fresh = _detect(frame) or target_face
    now = time.perf_counter()
    if now - _track["last"] > 0.5:  # Face was lost for a moment: start clean.
        _track["filter"] = _OneEuro()
    _track["last"] = now
    face = Face(dict(fresh))
    face.kps = _track["filter"](np.asarray(fresh.kps, np.float32), now)
    if modules.globals.mouth_mask:  # The mouth mask needs the detailed landmarks.
        from modules.face_analyser import ensure_landmarks

        ensure_landmarks(frame, [face])
    return face


def _counted_swap(source_face, target_face, frame):
    source_face = _current_source_face() or source_face
    if source_face is None:
        _stats["no_source"] += 1
        return frame
    _stats["swaps"] += 1
    target_face = _steady(target_face, frame)
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
    if _head["count"]:
        message += f", head layer {_head['ms'] / _head['count']:.0f} ms"
        _head.update(ms=0.0, count=0)
    if _stats["no_source"]:
        message += f", {_stats['no_source']} skipped: no face found in the selected face photo"
    elif not _stats["faces"]:
        message += " - no face seen: face the camera, add light, or set Det size to 320"
    print(message, flush=True)
    _stats.update(frames=0, faces=0, swaps=0, no_source=0, since=now)


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
