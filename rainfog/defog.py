"""Dark-channel fog removal.

The veil is estimated from the dark channel prior and lifted with a guided
transmission map. Strength is capped so flat regions are not turned into texture.
"""

import cv2
import numpy as np

from rainfog.filters import guided_filter, odd, scale_to_width


def dark_channel(bgr, patch):
    minimum = np.min(bgr.astype(np.float32), axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(minimum, kernel)


def atmospheric_light(bgr, dark):
    flat_dark = dark.reshape(-1)
    count = max(10, int(0.001 * flat_dark.size))
    index = np.argpartition(flat_dark, -count)[-count:]
    pixels = bgr.reshape(-1, 3)[index].astype(np.float32)
    brightness = pixels.max(axis=1)
    keep = pixels[brightness >= np.percentile(brightness, 50)]
    light = keep.mean(axis=0) if keep.size else pixels.mean(axis=0)
    return np.clip(light, 1.0, 255.0)


def fog_metrics(bgr, params):
    """Contrast, saturation, veil, and a fog score in [0, 1]."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    contrast_std = float(gray.std())
    saturation = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    patch = odd(max(9, scale_to_width(params["patch"], bgr.shape[1], params["ref_width"])))
    # Cap the patch so a multi-megapixel frame still finishes quickly.
    patch = int(min(patch, 61))
    dark = dark_channel(bgr, patch)
    veil = float(dark.mean())
    contrast_term = float(np.clip((42.0 - contrast_std) / 32.0, 0.0, 1.0))
    sat_term = float(np.clip((90.0 - saturation) / 90.0, 0.0, 1.0))
    veil_term = float(np.clip((veil - 40.0) / 90.0, 0.0, 1.0))
    fog_score = 0.5 * contrast_term + 0.2 * sat_term + 0.3 * veil_term
    return {
        "contrast_std": contrast_std,
        "saturation": saturation,
        "veil": veil,
        "fog_score": float(fog_score),
        "whiteout": contrast_std <= float(params["whiteout_std_max"]),
    }


def far_veil(bgr):
    """True when the top of the frame is a bright, flat distant fog."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    top = gray[: max(8, gray.shape[0] // 3)]
    return float(top.std()) < 12.0 and float(top.mean()) > 165.0


def remove_fog(bgr, params):
    """Lift fog. Dense veils use one global boost; lighter fog uses the dark channel."""
    contrast = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).std())
    if contrast < 16.0:
        return _dense_fog(bgr)
    if far_veil(bgr):
        return _long_range(bgr)
    return _dark_channel(bgr, params, contrast)


def _long_range(bgr):
    """Pull vehicles out of a distant fog bank, not only the near traffic.

    The airlight is taken from the gray distance, with headlights left out.
    Transmission is allowed to fall low there so the far lane is recovered,
    then a shadow lift keeps the near road from going black.
    """
    width = bgr.shape[1]
    light = _sky_light(bgr)
    normalized = bgr.astype(np.float32) / light.reshape(1, 1, 3)
    patch = int(np.clip(round(15 * width / 630.0), 9, 31))
    if patch % 2 == 0:
        patch += 1
    transmission = 1.0 - 0.96 * dark_channel(normalized, patch)
    transmission = np.clip(transmission, 0.05, 1.0).astype(np.float32)
    guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    radius = int(np.clip(round(36 * width / 630.0), 20, 80))
    transmission = guided_filter(guide, transmission, radius, 1e-3)
    transmission = np.clip(transmission, 0.05, 1.0)
    recovered = (bgr.astype(np.float32) - light) / transmission[..., None] + light
    recovered = np.clip(recovered, 0, 255)
    recovered = 255.0 * np.power(recovered / 255.0, 0.84)
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    return _soften_color(recovered, min(20.0, 14.0 * width / 630.0))


def _sky_light(bgr):
    """Mean color of the distant fog, excluding small bright lamps."""
    height = bgr.shape[0]
    region = bgr[: max(8, height // 3)].astype(np.float32)
    gray = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, (30, 97))
    mask = (gray >= low) & (gray <= high)
    if float(mask.mean()) < 0.05:
        light = region.reshape(-1, 3).mean(axis=0)
    else:
        light = region[mask].mean(axis=0)
    return np.clip(light, 1.0, 255.0)


def _dense_fog(bgr):
    """Remove a thick veil from the smooth scene, and leave flat surfaces flat.

    Local contrast stretches turn fog noise into blotches. A guided filter
    keeps object edges, and one global boost brings those edges forward.
    Color is taken from the slow veil so a cone stays orange, then the color
    channels are softened so fog specks do not become colored noise.
    """
    width = bgr.shape[1]
    radius = max(2, int(round(4 * width / 1280.0)))
    sigma = 45.0 * (width / 1280.0)
    smoothed = _guided_color(bgr, radius, 0.002)
    veil = cv2.GaussianBlur(smoothed, (0, 0), sigma)
    structure = smoothed - veil
    amount = 70.0 / (np.percentile(np.abs(structure), 92) + 1e-3)
    fog_color = np.median(veil.reshape(-1, 3), axis=0).reshape(1, 1, 3)
    restored = 122.0 + structure * amount + (veil - fog_color) * 1.35
    restored = np.clip(restored, 0, 255).astype(np.uint8)
    return _soften_color(restored, 6.0 * width / 1280.0)


def _guided_color(bgr, radius, eps):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    source = bgr.astype(np.float32)
    channels = [
        guided_filter(gray, source[:, :, channel] / 255.0, radius, eps) * 255.0
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1)


def _soften_color(bgr, sigma):
    """Blur only the color channels. Brightness, including edges, stays put."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    luma, red, blue = cv2.split(ycrcb)
    red = cv2.GaussianBlur(red, (0, 0), sigma)
    blue = cv2.GaussianBlur(blue, (0, 0), sigma)
    return cv2.cvtColor(cv2.merge([luma, red, blue]), cv2.COLOR_YCrCb2BGR)


def _dark_channel(bgr, params, contrast):
    """Recover scene radiance. Heavier fog uses a stronger transmission."""
    patch = odd(max(9, scale_to_width(params["patch"], bgr.shape[1], params["ref_width"])))
    patch = int(min(patch, 61))
    severity = float(np.clip((40.0 - contrast) / 24.0, 0.0, 1.0))
    omega = min(0.97, float(params["omega"]) + 0.12 * severity)
    t0 = max(0.10, float(params["t0"]) - 0.10 * severity)
    dark = dark_channel(bgr, patch)
    light = atmospheric_light(bgr, dark)
    normalized = bgr.astype(np.float32) / light.reshape(1, 1, 3)
    transmission = 1.0 - omega * dark_channel(normalized, patch)
    transmission = np.clip(transmission, t0, 1.0)
    guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    radius = max(15, patch)
    transmission = guided_filter(guide, transmission.astype(np.float32), radius, 1e-3)
    transmission = np.clip(transmission, t0, 1.0)
    recovered = (bgr.astype(np.float32) - light) / transmission[..., None] + light
    return np.clip(recovered, 0, 255).astype(np.uint8)


def mild_contrast(bgr, gain=1.08):
    """Small contrast lift used when fog has wiped the frame."""
    image = bgr.astype(np.float32)
    mean = image.mean(axis=(0, 1), keepdims=True)
    adjusted = (image - mean) * gain + mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)
