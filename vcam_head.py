"""Head transplant for Deep-Live-Cam's live mode (local addition, not upstream).

Deep-Live-Cam swaps only the inner face. This layer adds the rest of the head
from the selected photo - hair, forehead, cheeks, jawline and ears - and tries
to hide your own hair where the new head does not cover it.

How it works
- The photo is parsed once (BiSeNet face parsing: skin, hair, ears, neck...).
- A mesh of the photo's 106 face landmarks plus rings of extra points is warped
  onto the live face: the face outline follows your jaw while you talk, and the
  outer rings (hair) follow your head rigidly so the hair does not wobble.
- The inner face (eyes, nose, mouth) stays Deep-Live-Cam's swap, so your
  expressions come through; the photo's skin is colour-matched to it.
"""

from __future__ import annotations

import math
import os
import urllib.request

import cv2
import numpy as np

PARSER_URL = ("https://github.com/facefusion/facefusion-assets/releases/download/"
              "models-3.0.0/bisenet_resnet_34.onnx")
PARSER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                           "face_parser_bisenet34.onnx")

# BiSeNet (CelebAMask-HQ) classes.
SKIN, L_BROW, R_BROW, L_EYE, R_EYE, GLASSES, L_EAR, R_EAR, EARRING = 1, 2, 3, 4, 5, 6, 7, 8, 9
NOSE, MOUTH, U_LIP, L_LIP, NECK, NECKLACE, CLOTH, HAIR, HAT = 10, 11, 12, 13, 14, 15, 16, 17, 18

# insightface 2d106 landmarks: the face outline in drawing order, and the inner face.
OUTLINE = [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0,
           24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17]
BROWS = list(range(43, 52)) + list(range(97, 106))
INNER = BROWS + list(range(33, 43)) + list(range(87, 97)) + list(range(72, 87)) + list(range(52, 72))
# What Deep-Live-Cam's swap keeps: eyes, nose and mouth carry your expressions.
CORE = list(range(33, 43)) + list(range(87, 97)) + list(range(72, 87)) + list(range(52, 72))
STABLE = BROWS + list(range(33, 43)) + list(range(87, 97)) + [72, 73, 74, 86] + [1, 9, 17, 25]
CHIN, NOSE_TIP, LEFT_TEMPLE, RIGHT_TEMPLE = 0, 86, 1, 17

# Extra rings around the face (multiples of the face size), rigid with the head.
RINGS = (1.35, 1.9, 2.7, 3.8)
RING_POINTS = 28


def _download_parser() -> str | None:
    if os.path.isfile(PARSER_PATH):
        return PARSER_PATH
    try:
        print("[head] Downloading the face parsing model (90 MB, once)...", flush=True)
        partial = PARSER_PATH + ".part"
        urllib.request.urlretrieve(PARSER_URL, partial)
        os.replace(partial, PARSER_PATH)
        return PARSER_PATH
    except OSError as error:
        print(f"[head] Could not download the face parsing model, head layer off: {error}", flush=True)
        return None


class FaceParser:
    def __init__(self, providers):
        import onnxruntime

        path = _download_parser()
        self.session = onnxruntime.InferenceSession(path, providers=providers) if path else None

    def __call__(self, image: np.ndarray, centre, side: float) -> np.ndarray:
        """Class map for a square crop of `side` px around `centre`, mapped back to the image."""
        height, width = image.shape[:2]
        scale = 512.0 / side
        to_crop = np.float32([[scale, 0, 256 - centre[0] * scale], [0, scale, 256 - centre[1] * scale]])
        crop = cv2.warpAffine(image, to_crop, (512, 512), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        blob = (crop[:, :, ::-1].astype(np.float32) / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        logits = self.session.run(None, {self.session.get_inputs()[0].name:
                                         blob.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        labels = logits.argmax(0).astype(np.uint8)
        # Pixels outside the crop are unknown (255), not background.
        valid = np.full((512, 512), 1, np.uint8)
        back = cv2.invertAffineTransform(to_crop)
        labels = cv2.warpAffine(labels, back, (width, height), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inside = cv2.warpAffine(valid, back, (width, height), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        labels[inside == 0] = 255
        return labels


def face_size(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[CHIN] - points[BROWS].mean(axis=0)))


def head_turn(points: np.ndarray) -> float:
    """0 facing the camera, towards 1 when the head turns sideways."""
    left = np.linalg.norm(points[NOSE_TIP] - points[LEFT_TEMPLE])
    right = np.linalg.norm(points[NOSE_TIP] - points[RIGHT_TEMPLE])
    return float(abs(left - right) / max(left + right, 1e-6))


def similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.estimateAffinePartial2D(source.astype(np.float32), target.astype(np.float32),
                                            method=cv2.LMEDS)
    if matrix is None:
        matrix = np.float32([[1, 0, 0], [0, 1, 0]])
    return matrix


def mesh_points(points: np.ndarray, rigid: np.ndarray | None = None) -> np.ndarray:
    """106 landmarks + a widened outline (follows the jaw) + forehead + rigid outer rings.

    `rigid` maps photo-space ring positions into this face; None means this IS the photo.
    """
    centre = points[STABLE].mean(axis=0)
    size = face_size(points)
    outline = points[OUTLINE]
    widened = centre + (outline - centre) * 1.15
    up = points[BROWS].mean(axis=0) - points[CHIN]
    up /= max(np.linalg.norm(up), 1e-6)
    brows = points[BROWS]
    forehead = np.concatenate([brows + up * size * k for k in (0.35, 0.7)])
    extra = [widened, forehead]
    if rigid is None:
        angles = np.linspace(0, 2 * np.pi, RING_POINTS, endpoint=False)
        circle = np.c_[np.cos(angles), np.sin(angles) * 1.1]
        extra.append(np.concatenate([centre + circle * size * k for k in RINGS]))
    else:
        extra.append(rigid)
    return np.concatenate([points, *extra]).astype(np.float32)


def triangulate(points: np.ndarray) -> np.ndarray:
    left, top = np.floor(points.min(axis=0)) - 2
    right, bottom = np.ceil(points.max(axis=0)) + 2
    subdiv = cv2.Subdiv2D((int(left), int(top), int(right - left) + 1, int(bottom - top) + 1))
    lookup = {}
    for index, (x, y) in enumerate(points):
        key = (round(float(x), 2), round(float(y), 2))
        if key not in lookup:
            lookup[key] = index
            subdiv.insert((float(x), float(y)))
    triangles = []
    for t in subdiv.getTriangleList():
        corners = [lookup.get((round(float(t[i]), 2), round(float(t[i + 1]), 2))) for i in (0, 2, 4)]
        if None not in corners:
            triangles.append(corners)
    return np.array(triangles, np.int32)


def warp_mesh(image: np.ndarray, source_points: np.ndarray, target_points: np.ndarray,
              triangles: np.ndarray, size: tuple[int, int], box) -> np.ndarray:
    """Piecewise-affine warp of `image` (photo space) into the target box of the frame."""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    # The warp is piecewise affine, so its coordinate maps are smooth: build them
    # at half resolution and upscale, which is ~4x cheaper with no visible loss.
    step = 2
    local = (target_points - (left, top)) / step
    small = ((height + step - 1) // step, (width + step - 1) // step)
    owner = np.full(small, -1, np.int32)
    for index, triangle in enumerate(triangles):
        cv2.fillConvexPoly(owner, np.round(local[triangle]).astype(np.int32), index)
    corners = np.concatenate([local[triangles], np.ones((len(triangles), 3, 1), np.float32)], axis=2)
    try:
        affine = np.linalg.inv(corners) @ source_points[triangles]
    except np.linalg.LinAlgError:  # A collapsed triangle (extreme head turn).
        affine = np.linalg.pinv(corners) @ source_points[triangles]
    map_x = np.full(small, -1e4, np.float32)
    map_y = np.full(small, -1e4, np.float32)
    ys, xs = np.nonzero(owner >= 0)
    if len(xs):
        matrix = affine[owner[ys, xs]]
        map_x[ys, xs] = xs * matrix[:, 0, 0] + ys * matrix[:, 1, 0] + matrix[:, 2, 0]
        map_y[ys, xs] = xs * matrix[:, 0, 1] + ys * matrix[:, 1, 1] + matrix[:, 2, 1]
    map_x = cv2.resize(map_x, (width, height), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _feather(mask: np.ndarray, size: float, amount: float) -> np.ndarray:
    blur = max(3, int(size * amount)) | 1
    return cv2.GaussianBlur(mask, (blur, blur), 0)


def _polygon(shape, points) -> np.ndarray:
    mask = np.zeros(shape, np.float32)
    cv2.fillPoly(mask, [np.round(cv2.convexHull(points.astype(np.float32))).astype(np.int32)], 1.0,
                 cv2.LINE_AA)
    return mask


class HeadSource:
    """Everything prepared once from the selected photo."""

    def __init__(self, image: np.ndarray, points: np.ndarray, parser: FaceParser, mode: str):
        self.notes: list[str] = []
        scale = min(1.0, 1400.0 / max(image.shape[:2]))
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            points = points * scale
        size = face_size(points)
        cut = image.shape[:2]
        pad = int(size * 0.6)
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        points = points + pad
        self.image = image
        self.points = mesh_points(points.astype(np.float32))
        self.triangles = triangulate(self.points)
        self.size = face_size(points)
        centre = points[STABLE].mean(axis=0)
        self.ring_offsets = self.points[106 + len(OUTLINE) + 2 * len(BROWS):]
        self.stable = points[STABLE].astype(np.float32)
        shape = image.shape[:2]

        # Tight crop for accurate skin/ear labels, wide crop for long hair.
        original = image[pad:pad + cut[0], pad:pad + cut[1]]
        tight = parser(original, centre - pad - (0, self.size * 0.25), self.size * 3.0)
        wide = parser(original, centre - pad + (0, self.size * 0.6), self.size * 5.5)
        tight = np.where(tight == 255, wide, tight)
        tight[tight == 255] = 0
        labels = cv2.copyMakeBorder(tight, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        wide = cv2.copyMakeBorder(np.where(wide == 255, 0, wide).astype(np.uint8), pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=0)

        hair = np.isin(labels, (HAIR, HAT)).astype(np.uint8)
        wide_hair = np.isin(wide, (HAIR, HAT)).astype(np.uint8)
        # Keep only hair connected to this person's head (not someone else in the photo).
        count, components = cv2.connectedComponents(np.maximum(hair, wide_hair))
        head_box = _polygon(shape, np.concatenate([points[OUTLINE], self.points[106 + len(OUTLINE):
                                                   106 + len(OUTLINE) + 2 * len(BROWS)]]))
        near = cv2.dilate(head_box, np.ones((3, 3), np.uint8), iterations=max(1, int(self.size * 0.6)))
        keep = np.zeros(count, bool)
        for label in np.unique(components[(near > 0) & (components > 0)]):
            keep[label] = True
        hair = keep[components].astype(np.float32)
        if hair.sum() < (self.size ** 2) * 0.05:
            self.notes.append("no hair found in this photo (bald, covered or unclear): adding face shape only")
            hair[:] = 0
        else:
            inside = hair[pad:pad + cut[0], pad:pad + cut[1]]
            if inside[:2].any() or inside[:, :2].any() or inside[:, -2:].any():
                self.notes.append("the photo cuts off part of the hair: a photo showing the whole head looks best")

        # Ears, glasses and neck never fit someone else's head: they doubled up
        # with the wearer's own. Skin and brows only.
        skin = np.isin(labels, (SKIN, L_BROW, R_BROW)).astype(np.float32)
        if mode == "hair":
            # Forehead and hairline only, fading out at eye level, so the photo
            # never paints over your cheeks or jaw.
            eye_line = float(points[CORE[:20]][:, 1].min())
            rows = np.arange(shape[0], dtype=np.float32)[:, None]
            skin *= np.clip((eye_line - rows) / (self.size * 0.15), 0.0, 1.0)
        # The photo's own forehead and brows replace yours (no seam above the swap);
        # only the eyes-nose-mouth core is left to Deep-Live-Cam.
        inner = _polygon(shape, points[CORE])
        inner = cv2.dilate(inner, np.ones((3, 3), np.uint8), iterations=max(1, int(self.size * 0.03)))
        inner = _feather(inner, self.size, 0.14)
        outer_skin = np.clip(skin * (1.0 - inner), 0, 1)
        # Pull the hair edge in a little so the photo's background does not halo around it.
        hair = cv2.erode(hair, np.ones((3, 3), np.uint8), iterations=max(1, int(self.size * 0.012)))
        hair = _feather(hair, self.size, 0.03)
        # Fade out towards the photo's edges instead of a hard cut where the photo ends.
        ys, xs = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        border = np.minimum.reduce([xs, ys, shape[1] - 1 - xs, shape[0] - 1 - ys])
        border = np.clip((border - pad) / max(self.size * 0.25, 1.0), 0.0, 1.0)
        self.skin_alpha = _feather(outer_skin, self.size, 0.04) * border
        self.hair_alpha = hair * border
        self.alpha = np.maximum(self.skin_alpha, self.hair_alpha)
        self.skin_pixels = (outer_skin > 0.8)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        sample = lab[self.skin_pixels] if self.skin_pixels.sum() > 50 else lab[skin > 0.5]
        self.skin_mean = sample.mean(0) if len(sample) else np.float32([150, 135, 140])
        self.skin_std = sample.std(0) + 1e-3 if len(sample) else np.float32([20, 6, 6])
        # Photo + both alphas travel through one warp.
        self.layers = np.dstack([lab, self.alpha * 255, self.skin_alpha * 255, self.hair_alpha * 255])
        if head_turn(points) > 0.12:
            self.notes.append("the face in this photo is turned: a front-facing photo fits much better")


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.03):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.value = self.speed = self.time = None

    def __call__(self, value, now):
        if self.value is None or now <= self.time or value.shape != self.value.shape:
            self.value, self.speed, self.time = value.copy(), np.zeros_like(value), now
            return value
        dt = now - self.time
        self.time = now
        alpha = lambda cutoff: 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))
        speed = (value - self.value) / dt
        self.speed += alpha(1.0) * (speed - self.speed)
        cutoff = self.min_cutoff + self.beta * np.abs(self.speed)
        self.value = self.value + alpha(cutoff) * (value - self.value)
        return self.value


class HeadTransplant:
    def __init__(self, providers, mode: str = "full"):
        self.mode = mode
        self.parser = FaceParser(providers) if mode != "off" else None
        self.source: HeadSource | None = None
        self.smooth = OneEuro()
        self.colour = None
        self.own_hair: np.ndarray | None = None
        self.frame_count = 0

    @property
    def enabled(self) -> bool:
        return self.parser is not None and self.parser.session is not None

    def set_source(self, image: np.ndarray, points: np.ndarray) -> list[str]:
        if not self.enabled:
            return []
        self.source = HeadSource(image, points, self.parser, self.mode)
        self.colour = None
        self.own_hair = None
        return self.source.notes

    def apply(self, swapped: np.ndarray, original: np.ndarray, points: np.ndarray, now: float) -> np.ndarray:
        source = self.source
        if source is None or points is None:
            self.smooth = OneEuro()
            return swapped
        points = self.smooth(points.astype(np.float32), now)
        self.frame_count += 1
        turn = head_turn(points)
        fade = float(np.clip((0.55 - turn) / 0.2, 0.0, 1.0))  # Very side-on: show the plain swap.
        if fade <= 0:
            return swapped
        rigid = similarity(source.stable, points[STABLE])
        rings = source.ring_offsets @ rigid[:, :2].T + rigid[:, 2]
        target = mesh_points(points, rings)
        height, width = swapped.shape[:2]
        left, top = np.maximum(np.floor(target.min(axis=0)).astype(int), 0)
        right = min(width, int(np.ceil(target[:, 0].max())) + 1)
        bottom = min(height, int(np.ceil(target[:, 1].max())) + 1)
        if right - left < 8 or bottom - top < 8:
            return swapped
        box = (left, top, right, bottom)
        warped = warp_mesh(source.layers, source.points, target, source.triangles,
                           (height, width), box)
        photo_lab = warped[:, :, :3]
        alpha = warped[:, :, 3] / 255.0 * fade
        skin_alpha = warped[:, :, 4] / 255.0
        hair_alpha = warped[:, :, 5] / 255.0

        output = swapped.copy()
        region = output[top:bottom, left:right].astype(np.float32)
        photo = self._match_colour(photo_lab, region, points - (left, top), skin_alpha, hair_alpha)

        if self.mode == "full":
            region = self._hide_own_hair(region, original, points, box, alpha)
        output[top:bottom, left:right] = np.clip(photo * alpha[:, :, None] + region * (1 - alpha[:, :, None]),
                                                 0, 255).astype(np.uint8)
        return output

    def _match_colour(self, lab, live, local_points, skin_alpha, hair_alpha):
        """Photo skin takes the swapped face's tone; hair only follows the room's brightness."""
        if self.colour is None or self.frame_count % 3 == 0:
            # Sample the swapped inner face at quarter size: plenty for a mean and spread.
            inner = _polygon(live.shape[:2], local_points[INNER])[::4, ::4] > 0.5
            if inner.sum() >= 20:
                small = np.clip(live[::4, ::4], 0, 255).astype(np.uint8)
                live_lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)[inner]
                stats = np.stack([live_lab.mean(0), live_lab.std(0) + 1e-3])
                self.colour = stats if self.colour is None else self.colour + 0.3 * (stats - self.colour)
        if self.colour is None:
            return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        mean, std = self.colour
        source = self.source
        skin = (lab - source.skin_mean) * np.clip(std / source.skin_std, 0.6, 1.6) + mean
        brightness = lab.copy()
        brightness[:, :, 0] *= float(np.clip(mean[0] / max(source.skin_mean[0], 1.0), 0.7, 1.3))
        weight = (skin_alpha / np.maximum(skin_alpha + hair_alpha, 1e-3))[:, :, None]
        mixed = skin * weight + brightness * (1 - weight)
        return cv2.cvtColor(np.clip(mixed, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    def _hide_own_hair(self, region, original, points, box, alpha):
        """Your hair that the new head does not cover is painted over from its surroundings."""
        left, top, right, bottom = box
        if self.own_hair is None or self.own_hair.shape != original.shape[:2] or self.frame_count % 6 == 0:
            size = face_size(points)
            labels = self.parser(original, points[STABLE].mean(axis=0) - (0, size * 0.3), size * 3.2)
            # Generous margin: wisps at the edge of your hair are what show through.
            own = cv2.dilate(np.isin(labels, (HAIR, HAT)).astype(np.uint8),
                             np.ones((3, 3), np.uint8), iterations=max(2, int(size * 0.06)))
            # Only hair on and right around your head; never repaint other things
            # (a scarf, a person behind you) that the parser happens to call hair.
            zone = _polygon(own.shape, mesh_points(points, np.zeros((0, 2), np.float32)))
            zone = cv2.dilate(zone, np.ones((3, 3), np.uint8), iterations=max(1, int(size * 0.35)))
            self.own_hair = own * (zone > 0)
        own = self.own_hair[top:bottom, left:right].astype(np.float32)
        # Anything not fully covered by the new head counts, including under its soft edge.
        exposed = (own > 0) & (alpha < 0.15)
        if exposed.sum() < 30:
            return region
        # Inpaint at quarter size: fast, and smooth enough to read as background.
        small = cv2.resize(np.clip(region, 0, 255).astype(np.uint8), None, fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_AREA)
        hole = cv2.resize(exposed.astype(np.uint8), (small.shape[1], small.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        hole = cv2.dilate(hole, np.ones((3, 3), np.uint8))
        filled = cv2.inpaint(small, hole, 5, cv2.INPAINT_TELEA)
        filled = cv2.resize(filled, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_LINEAR)
        weight = cv2.GaussianBlur(exposed.astype(np.float32), (7, 7), 0)[:, :, None] * (1.0 - alpha[:, :, None])
        return filled.astype(np.float32) * weight + region * (1 - weight)
