"""Default parameters and loader for params.json."""

import json
from pathlib import Path

DEFAULTS = {
    "ref_width": 960,
    "streak_length": 5,
    "streak_threshold": 12.0,
    "rain_score_min": 0.50,
    "fog_score_min": 0.42,
    "whiteout_std_max": 8.0,
    "omega": 0.8,
    "t0": 0.25,
    "patch": 15,
    "align_mad_max": 22.0,
    "edge_miss_max": 0.22,
    "min_aligned_frames": 3,
    "max_related_frames": 12,
    "clearer_fog_ratio": 0.75,
    "clearer_contrast_ratio": 1.25,
}

ROOT = Path(__file__).resolve().parent.parent


def load_params(path=None):
    """Merge params.json over the built-in defaults."""
    params = dict(DEFAULTS)
    candidate = Path(path) if path else ROOT / "params.json"
    if candidate.is_file():
        stored = json.loads(candidate.read_text())
        params.update(stored)
    params["_path"] = str(candidate)
    return params
