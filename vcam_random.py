"""Filters for the random-face button (local addition, not upstream).

thispersondoesnotexist.com has no filters: every request is a random person.
This keeps fetching in the background until a face matches the chosen filter:

- Women / Men: insightface's gender model (genderage.onnx, already downloaded
  with buffalo_l). Right on 27 of 28 adults in a 30-face sample; it mistook a
  young boy for a girl, so children are its weak spot.
- Facial hair / clean-shaven: no model; two measurements of the chin and jaw
  against the cheeks. Texture catches stubble and patchy beards; loss of skin
  colour catches dense dark beards, which look smooth. On 37 labelled men the
  facial-hair rule found all 8 beards, and the clean-shaven rule picked none.
  It cannot tell heavy stubble from a full beard (in live searches most hits
  were heavy stubble), hence "facial hair", not "beard". Light stubble counts
  as clean-shaven; borderline faces are skipped by both filters.

The age output of that model was unusable on these photos (25-year-olds read as
45-55), so there is deliberately no age filter.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

URL = "https://thispersondoesnotexist.com/random-person.jpeg"
TEMP_PATH = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
MAX_TRIES = 60
PARALLEL = 4  # Downloads in flight at once.


def _bearded(jaw):
    """jaw = (texture ratio, skin-colour ratio) of chin/jaw against the cheeks."""
    if jaw is None:
        return False
    texture, colour = jaw
    return texture >= 2.55 or colour <= 0.70 or (texture >= 2.0 and colour <= 0.80)


def _shaven(jaw):
    if jaw is None:
        return False
    texture, colour = jaw
    return texture < 1.8 and colour > 0.75


FILTERS = {
    "Anyone": lambda gender, jaw: True,
    "Women": lambda gender, jaw: gender == "F",
    "Men": lambda gender, jaw: gender == "M",
    "Men with facial hair": lambda gender, jaw: gender == "M" and _bearded(jaw),
    "Men clean-shaven": lambda gender, jaw: gender == "M" and _shaven(jaw),
}

_gender_model = None
_gender_lock = threading.Lock()


def _gender_age():
    global _gender_model
    with _gender_lock:
        if _gender_model is None:
            import insightface
            from modules.face_analyser import get_face_analyser

            folder = os.path.dirname(get_face_analyser().det_model.model_file)
            _gender_model = insightface.model_zoo.get_model(os.path.join(folder, "genderage.onnx"),
                                                            providers=["CPUExecutionProvider"])
            _gender_model.prepare(ctx_id=-1)
    return _gender_model


def _main_face(image):
    """The biggest face, padding the photo when the face fills it (the detector misses those)."""
    from insightface.app.common import Face
    from modules.face_analyser import get_face_analyser

    detector = get_face_analyser().det_model
    for pad in (0.0, 0.25):
        border = int(max(image.shape[:2]) * pad)
        picture = image if not border else cv2.copyMakeBorder(
            image, border, border, border, border, cv2.BORDER_CONSTANT,
            value=[int(v) for v in np.median(image.reshape(-1, 3), axis=0)])
        boxes, points = detector.detect(picture, max_num=0, metric="default")
        if len(boxes):
            i = int(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).argmax())
            return picture, Face(bbox=boxes[i, :4], kps=points[i], det_score=boxes[i, 4])
    return image, None


def _jaw_measure(image, landmarks) -> tuple[float, float] | None:
    """(texture, skin colour) of the chin and jaw below the lips, each relative to
    the cheeks beside the nose. Beard hair is more textured and less skin-coloured."""
    size = float(np.linalg.norm(landmarks[0] - landmarks[43:52].mean(0)))
    if size < 40:
        return None
    lips = landmarks[52:72]
    lip_bottom = lips[:, 1].max()
    jaw = landmarks[[0, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24]]
    jaw = jaw[jaw[:, 1] > lip_bottom - 0.02 * size]
    if len(jaw) < 3:
        return None
    chin = np.zeros(image.shape[:2], np.uint8)
    outline = np.vstack([jaw, [[lips[:, 0].min(), lip_bottom], [lips[:, 0].max(), lip_bottom]]])
    cv2.fillPoly(chin, [np.round(cv2.convexHull(outline.astype(np.float32))).astype(np.int32)], 1)
    cheeks = np.zeros(image.shape[:2], np.uint8)
    top, bottom = int(landmarks[33:43][:, 1].max() + 0.08 * size), int(landmarks[86][1])
    centre = landmarks[86][0]
    for side in (-1, 1):
        x0, x1 = sorted([centre + side * 0.25 * size, centre + side * 0.46 * size])
        cheeks[max(top, 0):max(bottom, 0), max(int(x0), 0):max(int(x1), 0)] = 1
    if chin.sum() < 50 or cheeks.sum() < 50:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    colour = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    return (float(texture[chin > 0].mean() / max(texture[cheeks > 0].mean(), 1e-3)),
            float(colour[chin > 0].mean() / max(colour[cheeks > 0].mean(), 1e-3)))


def describe(image):
    """(gender 'F'/'M', jaw measure or None), or None when no face is found."""
    from modules.face_analyser import get_face_analyser

    image, face = _main_face(image)
    if face is None:
        return None
    _gender_age().get(image, face)
    gender = "M" if int(face.gender) == 1 else "F"
    jaw = None
    if gender == "M":
        get_face_analyser().models["landmark_2d_106"].get(image, face)
        jaw = _jaw_measure(image, face.landmark_2d_106)
    return gender, jaw


def _download() -> bytes | None:
    """One random face as JPEG bytes; None if the site sent something else."""
    response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    response.raise_for_status()
    if not response.headers.get("content-type", "").startswith("image/"):
        return None
    return response.content


def find_face(choice: str, stop: threading.Event, progress=lambda tries: None) -> tuple[str | None, str]:
    """Fetch until a face matches `choice`. Returns (saved path or None, message).

    Downloads run PARALLEL at a time while faces are checked one by one, so a
    rare filter (about 1 face in 9 has facial hair) takes seconds, not a minute.
    """
    wanted = FILTERS[choice]
    pool = ThreadPoolExecutor(PARALLEL)
    queued = deque(pool.submit(_download) for _ in range(PARALLEL))
    tries, errors = 0, 0
    try:
        while queued and tries < MAX_TRIES:
            if stop.is_set():
                return None, "Random face search cancelled."
            future = queued.popleft()
            if tries + len(queued) + 1 < MAX_TRIES:
                queued.append(pool.submit(_download))
            try:
                content = future.result()
                errors = 0
            except requests.RequestException as error:
                errors += 1
                if errors >= 3:  # One flaky request is fine; a dead connection is not.
                    return None, f"Could not fetch a random face: {error}"
                continue
            tries += 1
            progress(tries)
            image = None if content is None else cv2.imdecode(np.frombuffer(content, np.uint8),
                                                               cv2.IMREAD_COLOR)
            if image is None:
                continue
            found = describe(image) if choice != "Anyone" else ("?", None)
            if found is None or not wanted(*found):
                continue
            # Stage then rename, so the face in use is never half-written (Live reloads on change).
            staging = TEMP_PATH + ".part"
            with open(staging, "wb") as file:
                file.write(content)
            os.replace(staging, TEMP_PATH)
            return TEMP_PATH, f"Random face: {choice.lower()} (found in {tries} tries)."
        return None, f"No {choice.lower()} found in {tries} faces: click again to keep looking."
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def install(ui) -> None:
    """Add the filter menu next to the random-face button and run searches off the UI thread."""
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QComboBox

    import modules.globals

    class _Bridge(QObject):
        done = Signal(object, str)
        tried = Signal(int)

    original_build = ui.MainWindow._build_image_row

    def find_layout(layout, widget):
        if layout.indexOf(widget) >= 0:
            return layout
        for i in range(layout.count()):
            child = layout.itemAt(i).layout()
            found = child and find_layout(child, widget)
            if found:
                return found
        return None

    def warm_up():
        try:  # Load the face and gender models now, not on the first click (~10 s once).
            describe(np.zeros((256, 256, 3), np.uint8))
            _gender_age()
        except Exception as error:
            print(f"[random] Could not preload the face models: {error}", flush=True)

    def on_filter(text):
        modules.globals.random_face_filter = text
        if text != "Anyone" and _gender_model is None:
            threading.Thread(target=warm_up, daemon=True).start()

    def build_image_row(self):
        row = original_build(self)
        self._random_filter = QComboBox()
        self._random_filter.addItems(list(FILTERS))
        self._random_filter.setToolTip("Which kind of random face the 🔄 button looks for")
        self._random_filter.setCurrentText(getattr(modules.globals, "random_face_filter", "Anyone"))
        self._random_filter.currentTextChanged.connect(on_filter)
        holder = find_layout(row, self.btn_random_face)
        if holder is not None:
            holder.insertWidget(holder.indexOf(self.btn_random_face), self._random_filter)
        self._random_bridge = _Bridge()
        self._random_bridge.done.connect(lambda path, message: on_done(self, path, message))
        self._random_bridge.tried.connect(
            lambda tries: self.btn_random_face.setText(f"{tries}"))
        self._random_stop = None
        return row

    def on_random_face(self):
        if self._random_stop is not None:  # A second click while searching cancels.
            self._random_stop.set()
            return
        if ui._PREVIEW is not None:
            ui._PREVIEW.hide()
        choice = self._random_filter.currentText()
        stop = self._random_stop = threading.Event()
        self.btn_random_face.setText("…")  # Becomes the try count as faces arrive.
        self.btn_random_face.setToolTip("Searching... click to cancel")
        bridge = self._random_bridge

        def work():
            try:
                path, message = find_face(choice, stop, bridge.tried.emit)
            except Exception as error:  # Never leave the button stuck.
                path, message = None, f"Random face search failed: {error}"
            bridge.done.emit(path, message)

        threading.Thread(target=work, daemon=True).start()

    def on_done(self, path, message):
        self._random_stop = None
        self.btn_random_face.setText("🔄")
        self.btn_random_face.setToolTip("Get a random face (filter on the left)")
        print(f"[random] {message}", flush=True)
        ui.update_status(message)
        if path is None:
            return
        modules.globals.source_path = path
        self.source_label.setPixmap(ui.render_image_preview(path, (200, 200)))
        self.source_label.setText("")

    ui.MainWindow._build_image_row = build_image_row
    ui.MainWindow._on_random_face = on_random_face
