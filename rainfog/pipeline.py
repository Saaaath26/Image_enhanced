"""Decide rain versus fog and apply the matching enhancement."""

import shutil
from pathlib import Path

import cv2
import numpy as np

from rainfog.condition import analyze
from rainfog.defog import far_veil, fog_metrics, mild_contrast, remove_fog
from rainfog.derain import remove_rain, suppress_streaks
from rainfog.related import align_nearby_stills, load_and_align, temporal_clean
from rainfog.review import work_through


def read_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def enhance_image(bgr, params, related_path=None, skip_path=None):
    """Enhance one BGR image. related_path may be a folder or a video."""
    info = analyze(bgr, params)
    methods = []
    aligned = []
    if info["decision"] != "clear" and related_path:
        aligned, report = load_and_align(bgr, related_path, params, skip_path=skip_path)
        info["alignment"] = report
    elif info["decision"] != "clear" and skip_path:
        aligned, report = align_nearby_stills(bgr, skip_path, params)
        info["alignment"] = report
    else:
        info["alignment"] = None

    output = bgr
    decision = info["decision"]
    enough = aligned and len(aligned) >= int(params["min_aligned_frames"])

    if decision in ("rain", "rain+fog"):
        if enough:
            output = temporal_clean([bgr] + aligned)
            methods.append("temporal-clean")
        else:
            output = remove_rain(bgr, params)
            methods.append("streak-removal")
        output = suppress_streaks(output)
        methods.append("streak-suppress")

    if decision in ("fog", "rain+fog"):
        if info["whiteout"]:
            output = mild_contrast(output)
            methods.append("whiteout-mild")
            info["recoverable"] = False
        else:
            used_median = "temporal-clean" in methods
            clearer = None if used_median else _clearer_frame(bgr, aligned, info, params)
            if clearer is not None:
                output = clearer
                methods.append("clearer-frame")
            else:
                output = remove_fog(output, params)
                if info["contrast_std"] < 16.0:
                    methods.append("dense-veil")
                elif far_veil(bgr):
                    methods.append("long-range")
                else:
                    methods.append("dark-channel")

    if decision == "clear":
        methods.append("unchanged")

    info["methods"] = methods
    info["aligned_frames"] = len(aligned)
    output, notes = work_through(output, bgr, info, params)
    info["review"] = notes
    return output, info


def write_comparison(original_path, original_bgr, enhanced_bgr, compare_dir):
    """Store the input, the enhanced photo, and a side-by-side view."""
    compare_dir = Path(compare_dir)
    compare_dir.mkdir(parents=True, exist_ok=True)
    source = Path(original_path)
    stem = source.stem
    suffix = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
    original_copy = compare_dir / f"{stem}_original{suffix}"
    enhanced_path = compare_dir / f"{stem}_enhanced{suffix}"
    versus_path = compare_dir / f"{stem}_vs.jpg"
    shutil.copyfile(source, original_copy)
    save_image(enhanced_path, enhanced_bgr)
    save_image(versus_path, side_by_side(original_bgr, enhanced_bgr))
    return original_copy, enhanced_path, versus_path


def side_by_side(original, enhanced, max_height=900):
    """Original on the left, enhanced on the right, with a label on each."""
    left = _fit_height(original, max_height)
    right = _fit_height(enhanced, max_height)
    if left.shape[0] != right.shape[0]:
        right = cv2.resize(right, (max(1, int(right.shape[1] * left.shape[0] / right.shape[0])), left.shape[0]))
    header = 42
    gap = 8
    height, width_left = left.shape[:2]
    width_right = right.shape[1]
    canvas = np.full((height + header, width_left + gap + width_right, 3), 24, np.uint8)
    canvas[header:, :width_left] = left
    canvas[header:, width_left + gap :] = right
    _label(canvas, "Original", 16, 28)
    _label(canvas, "Enhanced", width_left + gap + 16, 28)
    return canvas


def _fit_height(image, max_height):
    height, width = image.shape[:2]
    if height <= max_height:
        return image
    scale = max_height / float(height)
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max_height),
        interpolation=cv2.INTER_AREA,
    )


def _label(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif suffix == ".png":
        ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Could not write image: {path}")
    return path


def _clearer_frame(query, aligned, info, params):
    if not aligned:
        return None
    best = None
    best_fog = float(info["fog_score"])
    fog_limit = best_fog * float(params["clearer_fog_ratio"])
    contrast_need = float(info["contrast_std"]) * float(params["clearer_contrast_ratio"])
    for frame in aligned:
        metrics = fog_metrics(frame, params)
        if metrics["whiteout"]:
            continue
        if metrics["fog_score"] <= fog_limit and metrics["contrast_std"] >= contrast_need:
            if best is None or metrics["fog_score"] < best_fog:
                best = frame
                best_fog = metrics["fog_score"]
    return best
