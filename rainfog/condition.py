"""Unsupervised rain and fog scores. No labels and no object detector."""

from rainfog.defog import fog_metrics
from rainfog.derain import estimate_rain


def analyze(bgr, params):
    """Score the frame and decide clear, rain, fog, or rain+fog."""
    mask, rain_score, dominance, coverage = estimate_rain(bgr, params)
    fog = fog_metrics(bgr, params)
    rainy = rain_score >= float(params["rain_score_min"])
    foggy = fog["fog_score"] >= float(params["fog_score_min"])
    if fog["whiteout"]:
        decision = "fog"
        rainy = False
    elif rainy and foggy:
        decision = "rain+fog"
    elif rainy:
        decision = "rain"
    elif foggy:
        decision = "fog"
    else:
        decision = "clear"
    return {
        "decision": decision,
        "rain_score": rain_score,
        "fog_score": fog["fog_score"],
        "contrast_std": fog["contrast_std"],
        "saturation": fog["saturation"],
        "veil": fog["veil"],
        "whiteout": fog["whiteout"],
        "recoverable": not fog["whiteout"],
        "rain_dominance": dominance,
        "rain_coverage": coverage,
        "rain_pixels": float(mask.mean()),
    }
