"""Shared image filters."""

import cv2
import numpy as np


def box_mean(image, radius):
    size = int(radius) * 2 + 1
    return cv2.boxFilter(
        image, cv2.CV_32F, (size, size), normalize=True, borderType=cv2.BORDER_REFLECT
    )


def guided_filter(guide, src, radius, eps):
    """Grayscale-guided filter. guide and src are float arrays in any range."""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    mean_g = box_mean(guide, radius)
    mean_s = box_mean(src, radius)
    corr_gs = box_mean(guide * src, radius)
    corr_gg = box_mean(guide * guide, radius)
    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    mean_a = box_mean(a, radius)
    mean_b = box_mean(b, radius)
    return mean_a * guide + mean_b


def scale_to_width(value, width, ref_width):
    scaled = int(round(float(value) * float(width) / float(ref_width)))
    return max(1, scaled)


def odd(value):
    value = int(max(1, value))
    if value % 2 == 0:
        value += 1
    return value


def resize_max_side(image, max_side):
    """Return image, scale (resized / original). scale is 1 when no resize."""
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return image, 1.0
    scale = max_side / float(long_side)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale
