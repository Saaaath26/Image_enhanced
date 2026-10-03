"""Enhance one photo: remove rain streaks and fog, keep the scene."""

import argparse
import shutil
import sys
from pathlib import Path

from rainfog.params import load_params
from rainfog.pipeline import enhance_image, read_bgr, save_image, write_comparison

COMPARE_DIR = Path(__file__).resolve().parent / "compare"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Remove rain and fog from one image. "
        "Pass a folder or video of the same scene to clear moving rain."
    )
    parser.add_argument("--image", required=True, help="Path to the photo")
    parser.add_argument(
        "--related",
        default=None,
        help="Optional folder of stills, or a video, of the same scene",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional extra path for the enhanced image. "
        "By default it is saved in the compare folder.",
    )
    parser.add_argument(
        "--compare",
        default=str(COMPARE_DIR),
        help="Folder for the original, the enhanced image, and a side-by-side view",
    )
    parser.add_argument("--params", default=None, help="Parameter JSON (default: params.json)")
    args = parser.parse_args(argv)

    params = load_params(args.params)
    image_path = Path(args.image)
    image = read_bgr(image_path)
    enhanced, info = enhance_image(
        image,
        params,
        related_path=args.related,
        skip_path=image_path,
    )
    original_copy, enhanced_copy, versus_path = write_comparison(
        image_path, image, enhanced, args.compare
    )
    if args.out:
        out_path = Path(args.out)
        if info["methods"] == ["unchanged"] and image_path.suffix.lower() == out_path.suffix.lower():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_path, out_path)
        else:
            save_image(out_path, enhanced)
    else:
        out_path = enhanced_copy
    _report(info, out_path, original_copy, enhanced_copy, versus_path)
    return 0


def _report(info, out_path, original_copy, enhanced_copy, versus_path):
    print(f"condition: {info['decision']}")
    print(f"rain_score: {info['rain_score']:.4f}")
    print(f"fog_score: {info['fog_score']:.4f}")
    print(f"contrast_std: {info['contrast_std']:.2f}")
    print(f"methods: {', '.join(info['methods'])}")
    for note in info.get("review") or []:
        print(f"review: {note}")
    if info["decision"] in ("fog", "rain+fog"):
        print(f"recoverable: {str(info['recoverable']).lower()}")
    alignment = info.get("alignment")
    if alignment:
        print(
            f"related: aligned {alignment['aligned']} of {alignment['loaded']} "
            f"(rejected {alignment['rejected']})"
        )
    print(f"wrote: {out_path}")
    print(f"compare original: {original_copy}")
    print(f"compare enhanced: {enhanced_copy}")
    print(f"compare side by side: {versus_path}")


if __name__ == "__main__":
    sys.exit(main())
