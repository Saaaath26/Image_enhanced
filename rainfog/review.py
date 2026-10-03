"""Look at a result the way a person does, then decide the next pass.

The check is the distance against the foreground, leftover rain streaks, and
whether another pass would only invent detail. A blank whiteout stops.
"""

import cv2

from rainfog.condition import analyze
from rainfog.defog import _long_range, remove_fog
from rainfog.derain import suppress_streaks


def inspect(bgr, params):
    """Score the far band, the near band, leftover rain, and leftover fog."""
    reading = analyze(bgr, params)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    far = gray[: max(8, height // 3)]
    near = gray[2 * height // 3 :]
    return {
        "rain_score": reading["rain_score"],
        "fog_score": reading["fog_score"],
        "contrast_std": reading["contrast_std"],
        "whiteout": reading["whiteout"],
        "far_std": float(far.std()),
        "far_mean": float(far.mean()),
        "near_std": float(near.std()),
        "near_mean": float(near.mean()),
    }


def work_through(image, original, info, params, limit=2):
    """Re-read the photo and apply another pass while it still helps."""
    opening = inspect(original, params)
    action, reason = _next_action(opening, info)
    notes = ["input " + _line(opening, reason)]
    if action == "stop":
        return image, notes
    current = image
    for _ in range(limit):
        view = inspect(current, params)
        action, reason = _next_action(view, info)
        notes.append(_line(view, reason))
        if action == "stop":
            break
        updated = _apply(current, action, params)
        after = inspect(updated, params)
        if _rejected(view, after, action, updated):
            notes.append("stop: the next pass did not clear more of the scene")
            break
        current = updated
        info["methods"].append(action)
    if not notes:
        notes.append("stop: nothing further to try")
    return current, notes


def _next_action(view, info):
    """Pick one job from the original condition and what is still left."""
    original = info.get("decision", "clear")
    if info.get("whiteout") or view["whiteout"] or view["contrast_std"] <= 8.0:
        return "stop", "the frame is blank, so nothing further is recoverable"
    if original in ("fog", "rain+fog") and _far_still_hidden(view):
        return "review-far", "the distance is still a bright veil, so lift the far band"
    if original in ("rain", "rain+fog") and view["rain_score"] >= 0.50:
        return "review-streaks", "streaks are still spread across the frame"
    if original in ("fog", "rain+fog") and view["fog_score"] >= 0.42 and view["contrast_std"] < 28.0:
        return "review-fog", "a veil is still over the scene"
    if original == "clear":
        return "stop", "the photo is already clear"
    if view["rain_score"] >= 0.50 and original not in ("rain", "rain+fog"):
        return "stop", "the distance is open. Bright detail here is the scene, not rain"
    return "stop", "the distance has contrast and the leftover streaks are gone"


def _far_still_hidden(view):
    """A far band that is bright and much flatter than the foreground."""
    return (
        view["far_std"] < 16.0
        and view["far_mean"] > 155.0
        and view["far_std"] < view["near_std"] * 0.9
    )


def _apply(image, action, params):
    if action == "review-far":
        return _long_range(image)
    if action == "review-streaks":
        return suppress_streaks(image)
    if action == "review-fog":
        return remove_fog(image, params)
    return image


def _rejected(before, after, action, image):
    """Drop a pass that crushes the photo or does not move the target."""
    gray_mean = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    if gray_mean < 40.0:
        return True
    if action == "review-far":
        return after["far_std"] < before["far_std"] + 2.0
    if action == "review-streaks":
        return after["rain_score"] > before["rain_score"] - 0.05
    if action == "review-fog":
        return after["fog_score"] > before["fog_score"] - 0.03
    return False


def _line(view, reason):
    return (
        f"far {view['far_std']:.1f} / near {view['near_std']:.1f}, "
        f"rain {view['rain_score']:.2f}, fog {view['fog_score']:.2f} -> {reason}"
    )
