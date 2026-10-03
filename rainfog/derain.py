"""Single-image rain removal.

Rain is the bright detail that is thinner than the scene: streaks and droplets.
Those pixels are replaced from a morphological opening, which is the local
background beside the streak. Strong edges of objects are kept.
"""

import cv2
import numpy as np

from rainfog.filters import odd, resize_max_side

SCORE_MAX_SIDE = 960


def estimate_rain(bgr, params):
    """Return a full-size rain mask and a rain score in [0, 1].

    The score is measured at a fixed working size so a dry, detailed scene
    does not look rainy just because the file is many megapixels.
    """
    work, scale = resize_max_side(bgr, SCORE_MAX_SIDE)
    mask_work, _opened = _rain_layer(work, params)
    coverage = _cell_coverage(mask_work, 8)
    fraction = float(mask_work.mean())
    # Widespread fine bright residuals score as rain. A few highlights do not.
    score = coverage * min(1.0, fraction / 0.045)
    if fraction > 0.2:
        score *= 0.25
    if scale != 1.0:
        mask = cv2.resize(
            mask_work, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    else:
        mask = mask_work
    return mask, float(score), 0.0, float(coverage)


def suppress_streaks(bgr):
    """Erase thin bright streaks once the scene behind them is known.

    The frame is turned so the streaks stand upright, and each streak pixel
    is replaced from the wall just beside it. Object edges are left alone.
    """
    angle = _rain_angle(bgr)
    return _suppress_aligned(bgr, angle)


def _rain_angle(bgr):
    """Degrees that make the streaks vertical. Zero means they already are."""
    small, _ = resize_max_side(bgr, 480)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    center = (width / 2.0, height / 2.0)
    best_angle = 0
    best_score = -1.0
    for angle in range(-20, 21, 2):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        blurred = cv2.GaussianBlur(rotated, (0, 0), sigmaX=2.0, sigmaY=0.3)
        score = float(np.clip(rotated - blurred, 0.0, None).mean())
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def _suppress_aligned(bgr, angle):
    height, width = bgr.shape[:2]
    center = (width / 2.0, height / 2.0)
    forward = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(bgr, forward, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY).astype(np.float32)
    reach = max(2, int(round(4.0 * width / 1000.0)))
    left = np.roll(gray, reach, axis=1)
    right = np.roll(gray, -reach, axis=1)
    up = np.roll(gray, reach * 3, axis=0)
    down = np.roll(gray, -reach * 3, axis=0)
    beside = np.minimum(left, right)
    ridge = (gray > beside + 1.0) & (np.abs(gray - up) < 12.0) & (np.abs(gray - down) < 12.0)
    scene = cv2.GaussianBlur(gray, (0, 0), max(1.2, 2.2 * width / 1000.0))
    magnitude = cv2.magnitude(
        cv2.Sobel(scene, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(scene, cv2.CV_32F, 0, 1, ksize=3),
    )
    edge = (magnitude >= np.percentile(magnitude, 97)).astype(np.uint8)
    edge = cv2.dilate(edge, np.ones((5, 5), np.uint8)).astype(bool)
    ridge[edge] = False
    ridge[:, :reach] = False
    ridge[:, -reach:] = False
    # The test hits the middle of a streak. Spread it across the streak's width.
    thickness = max(3, int(round(11.0 * width / 7008.0)))
    along = max(5, int(round(21.0 * width / 7008.0)))
    span = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, along))
    ridge = cv2.dilate(ridge.astype(np.uint8), span).astype(bool)
    ridge[edge] = False
    fill = cv2.GaussianBlur(rotated, (0, 0), sigmaX=max(1.0, reach * 1.4), sigmaY=0.4)
    weight = cv2.GaussianBlur(ridge.astype(np.float32), (0, 0), max(0.6, 0.7 * width / 1000.0))[..., None]
    cleaned = rotated.astype(np.float32) * (1.0 - weight) + fill.astype(np.float32) * weight
    cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)
    backward = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(cleaned, backward, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def remove_rain(bgr, params):
    """Replace thin bright rain pixels with the local background."""
    mask, opened = _rain_layer(bgr, params)
    if not mask.any() or float(mask.mean()) > 0.2:
        return bgr.copy()
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8)
    soft = np.clip(soft, 0.0, 1.0)[..., None]
    original = bgr.astype(np.float32)
    filled = opened.astype(np.float32)
    blended = original * (1.0 - soft) + filled * soft
    return np.clip(blended, 0, 255).astype(np.uint8)


def _rain_layer(bgr, params):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel_size = odd(_scaled_kernel(bgr.shape[1], params))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(bgr, cv2.MORPH_OPEN, kernel)
    opened_gray = cv2.cvtColor(opened, cv2.COLOR_BGR2GRAY)
    delta = gray.astype(np.int16) - opened_gray.astype(np.int16)
    edge = _strong_edges(gray)
    mask = ((delta >= float(params["streak_threshold"])) & (edge == 0)).astype(np.uint8)
    return mask, opened


def _scaled_kernel(width, params):
    scaled = float(params["streak_length"]) * float(width) / float(params["ref_width"])
    return int(np.clip(round(scaled), 3, 31))


def _strong_edges(gray):
    low = cv2.GaussianBlur(gray, (0, 0), 4.0)
    gx = cv2.Sobel(low, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(low, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    edge = (magnitude >= np.percentile(magnitude, 84)).astype(np.uint8)
    return cv2.dilate(edge, np.ones((3, 3), np.uint8))


def _cell_coverage(mask, cells):
    height, width = mask.shape
    occupied = 0
    total = cells * cells
    for row in range(cells):
        y0 = row * height // cells
        y1 = (row + 1) * height // cells
        for col in range(cells):
            x0 = col * width // cells
            x1 = (col + 1) * width // cells
            if mask[y0:y1, x0:x1].any():
                occupied += 1
    return occupied / float(total)
