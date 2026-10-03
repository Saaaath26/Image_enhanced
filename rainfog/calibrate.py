"""Tune streak length and threshold on still DSC bursts.

Bursts are consecutive frames of one scene. Their temporal median is a
stand-in clean image: moving rain changes, the background does not.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from rainfog.derain import estimate_rain, remove_rain
from rainfog.params import DEFAULTS
from rainfog.related import temporal_median

LENGTHS = (3, 5, 7, 9)
THRESHOLDS = (8.0, 12.0, 18.0, 26.0)
WORK_WIDTH = 960
BURST_MAD = 12.0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tune rain streak parameters on DSC bursts")
    parser.add_argument("--data", default="DRDO_DATA_29-9;1-10", help="Dataset directory")
    parser.add_argument("--out", default="params.json", help="Where to write parameters")
    parser.add_argument("--max-bursts", type=int, default=6)
    args = parser.parse_args()

    data = Path(args.data)
    files = sorted(data.glob("DSC*.JPG"), key=lambda path: int(path.stem[3:]))
    if not files:
        raise SystemExit(f"No DSC stills in {data}")
    groups = _bursts(files)
    usable = [group for group in groups if len(group) >= 4]
    if not usable:
        raise SystemExit("No still bursts with at least 4 frames")
    chosen = _spread(usable, args.max_bursts)
    print(f"bursts: {len(usable)} usable, tuning on {len(chosen)}")

    loaded = []
    for group in chosen:
        frames = [_load_width(path, WORK_WIDTH) for path in group[:8]]
        frames = [frame for frame in frames if frame is not None]
        if len(frames) < 4:
            continue
        contrast = float(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).std())
        if contrast < 18:
            continue
        loaded.append(frames)
        print(f"  {group[0].name} .. {group[min(7, len(group)-1)].name} ({len(frames)} frames)")
    if not loaded:
        raise SystemExit("Bursts were too foggy or unreadable to tune rain")

    best = None
    for length in LENGTHS:
        for threshold in THRESHOLDS:
            trial = dict(DEFAULTS)
            trial["ref_width"] = WORK_WIDTH
            trial["streak_length"] = length
            trial["streak_threshold"] = threshold
            errors = []
            for frames in loaded:
                clean = temporal_median(frames)
                for frame in frames:
                    removed = remove_rain(frame, trial)
                    errors.append(
                        float(np.abs(removed.astype(np.float32) - clean.astype(np.float32)).mean())
                    )
            score = float(np.mean(errors))
            print(f"length={length} threshold={threshold:.0f} mae={score:.3f}")
            if best is None or score < best[0]:
                best = (score, length, threshold)

    _, length, threshold = best
    params = dict(DEFAULTS)
    params["ref_width"] = WORK_WIDTH
    params["streak_length"] = int(length)
    params["streak_threshold"] = float(threshold)
    params["rain_score_min"] = _rain_threshold(loaded, params)
    out = Path(args.out)
    out.write_text(json.dumps(params, indent=2) + "\n")
    print(
        f"wrote {out} streak_length={params['streak_length']} "
        f"streak_threshold={params['streak_threshold']} "
        f"rain_score_min={params['rain_score_min']:.5f} mae={best[0]:.3f}"
    )


def _rain_threshold(loaded, params):
    frame_scores = []
    clean_scores = []
    for frames in loaded:
        clean_scores.append(estimate_rain(temporal_median(frames), params)[1])
        for frame in frames[::2]:
            frame_scores.append(estimate_rain(frame, params)[1])
    frame_mid = float(np.median(frame_scores)) if frame_scores else 0.6
    clean_mid = float(np.median(clean_scores)) if clean_scores else 0.2
    # Between a rainy frame and its median. Clipped so a dry lab scene stays dry
    # and an obvious rain frame still triggers.
    cutoff = float(np.clip((frame_mid + clean_mid) / 2.0, 0.55, 0.85))
    if frame_mid <= clean_mid:
        cutoff = 0.62
    print(
        f"rain scores frames={frame_mid:.5f} medians={clean_mid:.5f} cutoff={cutoff:.5f}"
    )
    return cutoff


def _bursts(files):
    groups = []
    current = [files[0]]
    previous = _thumb(files[0])
    for path in files[1:]:
        thumb = _thumb(path)
        mad = float(np.abs(previous.astype(np.float32) - thumb.astype(np.float32)).mean())
        if mad <= BURST_MAD:
            current.append(path)
        else:
            groups.append(current)
            current = [path]
        previous = thumb
    groups.append(current)
    return groups


def _thumb(path):
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if image is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return np.zeros((90, 160, 3), np.uint8)
    return cv2.resize(image, (160, 90))


def _load_width(path, width):
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if image is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, current = image.shape[:2]
    target_h = max(1, int(round(height * (width / float(current)))))
    return cv2.resize(image, (width, target_h), interpolation=cv2.INTER_AREA)


def _spread(groups, limit):
    if len(groups) <= limit:
        return groups
    index = np.linspace(0, len(groups) - 1, limit)
    chosen = []
    seen = set()
    for value in index:
        slot = int(round(value))
        if slot not in seen:
            chosen.append(groups[slot])
            seen.add(slot)
    return chosen


if __name__ == "__main__":
    main()
