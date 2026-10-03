"""Load a folder or video of the same scene and align it to the query."""

from pathlib import Path

import cv2
import numpy as np

from rainfog.filters import resize_max_side

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
ALIGN_MAX_SIDE = 640


def temporal_median(frames):
    stack = np.stack([frame.astype(np.float32) for frame in frames], axis=0)
    return np.clip(np.median(stack, axis=0), 0, 255).astype(np.uint8)


def temporal_clean(frames):
    """Keep the still scene and drop bright rain.

    In a heavy downpour the median is still a rain pixel. The dark end of
    the stack is the wall behind the streak, so that sample is kept.
    """
    stack = np.stack(frames, axis=0)
    count = stack.shape[0]
    if count < 4:
        return temporal_median(frames)
    rank = max(0, int(round(0.08 * (count - 1))))
    return np.partition(stack, rank, axis=0)[rank]


def align_nearby_stills(query, image_path, params):
    """Use other photos in the same folder when they show this still scene.

    A DSLR burst keeps the dog and the wall fixed while the rain moves.
    Only neighbors with a small brightness gap are loaded.
    """
    image_path = Path(image_path)
    folder = image_path.parent
    files = [
        item
        for item in sorted(folder.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]
    report = {
        "source": str(folder),
        "loaded": 0,
        "aligned": 0,
        "rejected": 0,
        "errors": [],
    }
    try:
        index = next(i for i, item in enumerate(files) if item.resolve() == image_path.resolve())
    except StopIteration:
        return [], report
    window = files[max(0, index - 16) : index + 17]
    query_thumb = cv2.resize(
        cv2.cvtColor(query, cv2.COLOR_BGR2GRAY),
        (160, max(1, int(round(160 * query.shape[0] / query.shape[1])))),
    )
    scored = []
    for item in window:
        if item.resolve() == image_path.resolve():
            continue
        thumb = cv2.imread(str(item), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if thumb is None:
            continue
        thumb = cv2.resize(thumb, (query_thumb.shape[1], query_thumb.shape[0]))
        mad = float(np.abs(thumb.astype(np.int16) - query_thumb.astype(np.int16)).mean())
        if mad <= 12.0:
            scored.append((mad, item))
    scored.sort(key=lambda pair: pair[0])
    limit = min(11, int(params["max_related_frames"]))
    accepted = []
    for _, item in scored[:limit]:
        frame = cv2.imread(str(item), cv2.IMREAD_COLOR)
        report["loaded"] += 1
        if frame is None:
            report["rejected"] += 1
            continue
        warped, _, reason = _align_frame(query, frame, params)
        if warped is None:
            report["rejected"] += 1
            if reason and len(report["errors"]) < 3:
                report["errors"].append(reason)
            continue
        accepted.append(warped)
    report["aligned"] = len(accepted)
    return accepted, report


def load_and_align(query, related_path, params, skip_path=None):
    """Return frames warped onto the query, plus a short alignment report.

    Frames whose scene does not match are dropped. With fewer than
    min_aligned_frames survivors the caller should use single-image removal.
    """
    related_path = Path(related_path)
    raw_frames = _read_related(related_path, params, skip_path)
    accepted = []
    rejected = 0
    errors = []
    for frame in raw_frames:
        warped, mad, reason = _align_frame(query, frame, params)
        if warped is None:
            rejected += 1
            if reason and len(errors) < 3:
                errors.append(reason)
            continue
        accepted.append(warped)
    report = {
        "source": str(related_path),
        "loaded": len(raw_frames),
        "aligned": len(accepted),
        "rejected": rejected,
        "errors": errors,
    }
    return accepted, report


def _read_related(path, params, skip_path):
    limit = int(params["max_related_frames"])
    if path.is_dir():
        files = [
            item
            for item in sorted(path.iterdir())
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ]
        if skip_path is not None:
            skip = Path(skip_path).resolve()
            files = [item for item in files if item.resolve() != skip]
        files = _even_sample(files, limit)
        frames = []
        for item in files:
            image = cv2.imread(str(item), cv2.IMREAD_COLOR)
            if image is not None:
                frames.append(image)
        return frames
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return _sample_video(path, limit)
    raise FileNotFoundError(
        f"Related path must be an image folder or a video: {path}"
    )


def _even_sample(items, limit):
    if len(items) <= limit:
        return items
    index = np.linspace(0, len(items) - 1, limit)
    chosen = []
    seen = set()
    for value in index:
        slot = int(round(value))
        if slot not in seen:
            chosen.append(items[slot])
            seen.add(slot)
    return chosen


def _sample_video(path, limit):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        if count > 1:
            for index in _even_sample(list(range(count)), limit):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append(frame)
        else:
            while len(frames) < limit:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
    finally:
        capture.release()
    return frames


def _align_frame(query, frame, params):
    mad_limit = float(params["align_mad_max"])
    edge_limit = float(params["edge_miss_max"])
    query_small, _ = resize_max_side(query, ALIGN_MAX_SIDE)
    resized = cv2.resize(frame, (query_small.shape[1], query_small.shape[0]))
    mad = _mad(query_small, resized)
    miss = _edge_miss(query_small, resized)
    # Heavy rain changes the edges but not the scene. A very small brightness
    # gap means the camera and the objects stayed put.
    edges_ok = miss <= edge_limit or mad <= 12.0
    if mad <= mad_limit and edges_ok and _same_aspect(query, frame):
        if frame.shape[:2] == query.shape[:2]:
            return frame, mad, ""
        fitted = cv2.resize(frame, (query.shape[1], query.shape[0]))
        return fitted, mad, ""

    homography = _estimate_homography(query_small, resized)
    if homography is None:
        return None, mad, "no-match"
    full = _scale_homography(homography, query.shape, frame.shape, query_small.shape)
    warped = cv2.warpPerspective(frame, full, (query.shape[1], query.shape[0]))
    ones = np.full(frame.shape[:2], 255, np.uint8)
    coverage = cv2.warpPerspective(ones, full, (query.shape[1], query.shape[0]))
    if float(coverage.mean()) < 0.75 * 255:
        return None, mad, "border"
    warped_small = cv2.resize(warped, (query_small.shape[1], query_small.shape[0]))
    small_coverage = cv2.resize(coverage, (query_small.shape[1], query_small.shape[0]))
    aligned_mad = _mad(query_small, warped_small, coverage=small_coverage)
    aligned_miss = _edge_miss(query_small, warped_small)
    if aligned_mad > mad_limit or aligned_miss > edge_limit:
        return None, aligned_mad, "mad"
    return warped, aligned_mad, ""


def _edge_miss(query, frame):
    """Share of query edges that have no nearby edge in the other frame.

    Rain changes brightness. A moved camera or subject moves the edges.
    """
    query_edges = cv2.Canny(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY), 60, 140)
    frame_edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 140)
    near = cv2.dilate(frame_edges, np.ones((5, 5), np.uint8))
    source = query_edges > 0
    if int(source.sum()) < 50:
        return 1.0
    return float(np.logical_and(source, near == 0).sum() / source.sum())


def _same_aspect(query, frame):
    qh, qw = query.shape[:2]
    fh, fw = frame.shape[:2]
    return abs((qw / qh) - (fw / fh)) < 0.02


def _mad(query, frame, coverage=None):
    diff = np.abs(query.astype(np.float32) - frame.astype(np.float32))
    if coverage is None:
        return float(diff.mean())
    valid = coverage > 127
    if valid.mean() < 0.5:
        return 255.0
    return float(diff[valid].mean())


def _estimate_homography(query_small, frame_small):
    homography = _ecc_homography(query_small, frame_small)
    if homography is not None:
        return homography
    return _orb_homography(query_small, frame_small)


def _ecc_homography(query_small, frame_small):
    query_gray = cv2.cvtColor(query_small, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.GaussianBlur(query_gray, (5, 5), 0)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    warp = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
    try:
        cc, warp = cv2.findTransformECC(
            query_gray.astype(np.float32) / 255.0,
            frame_gray.astype(np.float32) / 255.0,
            warp,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            None,
            5,
        )
    except cv2.error:
        return None
    if cc < 0.55:
        return None
    return warp.astype(np.float64)


def _orb_homography(query_small, frame_small):
    orb = cv2.ORB_create(1500)
    query_gray = cv2.cvtColor(query_small, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    kq, dq = orb.detectAndCompute(query_gray, None)
    kf, df = orb.detectAndCompute(frame_gray, None)
    if dq is None or df is None or len(kq) < 12 or len(kf) < 12:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(dq, df), key=lambda item: item.distance)[:250]
    if len(matches) < 12:
        return None
    src = np.float32([kf[item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
    dst = np.float32([kq[item.queryIdx].pt for item in matches]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if homography is None or inliers is None or int(inliers.sum()) < 10:
        return None
    return homography


def _scale_homography(homography_small, query_shape, frame_shape, small_shape):
    """Map a homography from the shared small canvas back to full resolution."""
    qh, qw = query_shape[:2]
    fh, fw = frame_shape[:2]
    sh, sw = small_shape[:2]
    scale_q = np.array(
        [[sw / float(qw), 0, 0], [0, sh / float(qh), 0], [0, 0, 1]], dtype=np.float64
    )
    scale_f = np.array(
        [[sw / float(fw), 0, 0], [0, sh / float(fh), 0], [0, 0, 1]], dtype=np.float64
    )
    # H_small = S_q * H_full * inv(S_f), and frame_small was stretched onto query_small.
    return np.linalg.inv(scale_q) @ homography_small @ scale_f
