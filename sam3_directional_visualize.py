"""
sam3_directional_visualize.py

Visualize SAM3 forward/backward directional outputs in two forms:

1. directional_overlay/
   RGB frame + bounding boxes for forward and backward separately.

2. directional_png/
   Union of all binary object masks for each frame/direction.

Expected SAM3 layout:

    <sam-dir>/
    ├── directional_tracks/
    │   ├── class_0_forward.json
    │   └── class_0_backward.json
    └── masks/
        ├── 000000_c0_forward_o0.png
        ├── 000000_c0_backward_o0.png
        └── ...

IMPORTANT ABOUT BOX FORMAT
--------------------------
New/fixed CAROECT-D sam3_export_tracks.py stores canonical normalized:

    [cx, cy, width, height]

Use:
    --box-format cxcywh

OLD directional JSON produced before the SAM3 box-convention fix contains
SAM3 raw normalized:

    [x_min, y_min, width, height]

even though the keys are named cx/cy. To inspect those OLD artifacts use:

    --box-format xywh

Example — current/fixed pipeline:

    python sam3_directional_visualize.py \
        --rgb-dir data/rgb/test \
        --sam-dir data/sam3/test \
        --class-id 0

Example — old pre-fix JSON:

    python sam3_directional_visualize.py \
        --rgb-dir data/rgb/test \
        --sam-dir data/sam3/test \
        --class-id 0 \
        --box-format xywh
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


def sorted_rgb_frames(rgb_dir: Path) -> list[Path]:
    """Match SAM3 temporal ordering: numeric stem if possible."""
    paths = list(rgb_dir.glob("*.tif")) + list(rgb_dir.glob("*.tiff"))

    if not paths:
        raise FileNotFoundError(
            f"No .tif/.tiff RGB frames found in {rgb_dir}"
        )

    try:
        return sorted(paths, key=lambda p: int(p.stem))
    except ValueError:
        print(
            "[warning] RGB filenames are not all numeric; "
            "falling back to lexicographic sorting."
        )
        return sorted(paths)


def box_to_pixels(obs: dict, width: int, height: int, box_format: str):
    """
    Convert normalized box stored in directional JSON to pixel xyxy.

    JSON field names are cx/cy/w/h for historical reasons.

    cxcywh:
        values really mean center-x, center-y, width, height.

    xywh:
        OLD artifact mode: values really mean x_min, y_min, width, height.
    """
    a = float(obs["cx"])
    b = float(obs["cy"])
    bw = float(obs["w"])
    bh = float(obs["h"])

    if box_format == "cxcywh":
        x0 = (a - bw / 2.0) * width
        y0 = (b - bh / 2.0) * height
        x1 = (a + bw / 2.0) * width
        y1 = (b + bh / 2.0) * height

    elif box_format == "xywh":
        x0 = a * width
        y0 = b * height
        x1 = (a + bw) * width
        y1 = (b + bh) * height

    else:
        raise ValueError(f"Unknown box format: {box_format}")

    x0 = int(round(np.clip(x0, 0, width - 1)))
    y0 = int(round(np.clip(y0, 0, height - 1)))
    x1 = int(round(np.clip(x1, 0, width - 1)))
    y1 = int(round(np.clip(y1, 0, height - 1)))

    return x0, y0, x1, y1


def render_directional_overlays(
    rgb_dir: Path,
    sam_dir: Path,
    class_id: int,
    box_format: str,
):
    """
    Create:
        directional_overlay/forward/*.png
        directional_overlay/backward/*.png
    """
    out_root = sam_dir / "directional_overlay"

    sources = {
        "forward":
            sam_dir / "directional_tracks" /
            f"class_{class_id}_forward.json",

        "backward":
            sam_dir / "directional_tracks" /
            f"class_{class_id}_backward.json",
    }

    frame_paths = sorted_rgb_frames(rgb_dir)

    print("\n=== RGB FRAME MAPPING ===")
    print(f"RGB frames: {len(frame_paths)}")

    for i, path in enumerate(frame_paths[:10]):
        print(f"  frame_idx {i:>4} -> {path.name}")

    for direction, json_path in sources.items():

        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing directional track file: {json_path}"
            )

        tracks = json.loads(json_path.read_text())

        out_dir = out_root / direction
        out_dir.mkdir(parents=True, exist_ok=True)

        by_frame = {}

        for track_id, track in tracks.items():
            raw_obj_id = track.get("raw_obj_id", track_id)

            for obs in track.get("frames", []):
                frame_idx = int(obs["frame_idx"])

                by_frame.setdefault(frame_idx, []).append(
                    (track_id, raw_obj_id, obs)
                )

        n_written = 0

        for frame_idx, rows in sorted(by_frame.items()):

            if frame_idx >= len(frame_paths):
                print(
                    f"[skip] {direction} frame {frame_idx}: "
                    f"no corresponding RGB frame"
                )
                continue

            img_path = frame_paths[frame_idx]
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

            if img is None:
                print(f"[skip] could not read {img_path}")
                continue

            height, width = img.shape[:2]

            for track_id, raw_obj_id, obs in rows:

                x0, y0, x1, y1 = box_to_pixels(
                    obs,
                    width,
                    height,
                    box_format,
                )

                cv2.rectangle(
                    img,
                    (x0, y0),
                    (x1, y1),
                    (0, 255, 0),
                    2,
                )

                score = float(obs.get("score", 0.0))

                label = (
                    f"{direction} "
                    f"o{raw_obj_id} "
                    f"{score:.2f}"
                )

                cv2.putText(
                    img,
                    label,
                    (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            out_path = out_dir / f"{frame_idx:06d}.png"

            if not cv2.imwrite(str(out_path), img):
                raise RuntimeError(
                    f"cv2.imwrite failed: {out_path}"
                )

            n_written += 1

        print(
            f"[overlay] {direction}: "
            f"{n_written} PNG(s) -> {out_dir}"
        )


def render_directional_mask_unions(
    sam_dir: Path,
    class_id: int,
):
    """
    Create:
        directional_png/forward/*.png
        directional_png/backward/*.png

    Each output PNG is the union of every object mask in that
    direction/frame.
    """
    mask_dir = sam_dir / "masks"
    out_root = sam_dir / "directional_png"

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Mask directory not found: {mask_dir}"
        )

    for direction in ("forward", "backward"):

        out_dir = out_root / direction
        out_dir.mkdir(parents=True, exist_ok=True)

        pattern = (
            f"*_c{class_id}_{direction}_o*.png"
        )

        mask_files = sorted(mask_dir.glob(pattern))

        regex = re.compile(
            rf"(\d+)_c{class_id}_{direction}_o\d+\.png$"
        )

        by_frame = {}

        for path in mask_files:
            match = regex.match(path.name)

            if not match:
                continue

            frame_idx = int(match.group(1))

            by_frame.setdefault(
                frame_idx, []
            ).append(path)

        n_written = 0

        for frame_idx, paths in sorted(by_frame.items()):

            merged = None

            for path in paths:

                mask = cv2.imread(
                    str(path),
                    cv2.IMREAD_GRAYSCALE,
                )

                if mask is None:
                    print(f"[skip] could not read mask: {path}")
                    continue

                if merged is None:
                    merged = np.zeros_like(
                        mask,
                        dtype=np.uint8,
                    )

                merged[mask > 0] = 255

            if merged is None:
                continue

            out_path = out_dir / f"{frame_idx:06d}.png"

            if not cv2.imwrite(str(out_path), merged):
                raise RuntimeError(
                    f"cv2.imwrite failed: {out_path}"
                )

            n_written += 1

        print(
            f"[mask-union] {direction}: "
            f"{n_written} PNG(s) -> {out_dir}"
        )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Render SAM3 forward/backward RGB box overlays "
            "and union-mask PNGs."
        )
    )

    ap.add_argument(
        "--rgb-dir",
        default="data/rgb/test",
        help="RGB TIFF folder used as SAM3 input",
    )

    ap.add_argument(
        "--sam-dir",
        default="data/sam3/test",
        help="SAM3 output root",
    )

    ap.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class ID to visualize",
    )

    ap.add_argument(
        "--box-format",
        choices=("cxcywh", "xywh"),
        default="cxcywh",
        help=(
            "cxcywh = NEW/fixed canonical tracks; "
            "xywh = OLD pre-fix SAM3 raw directional JSON"
        ),
    )

    ap.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete existing directional_overlay/ and "
            "directional_png/ before rendering"
        ),
    )

    args = ap.parse_args()

    rgb_dir = Path(args.rgb_dir)
    sam_dir = Path(args.sam_dir)

    if args.clean:
        for name in (
            "directional_overlay",
            "directional_png",
        ):
            path = sam_dir / name

            if path.exists():
                shutil.rmtree(path)

                print(f"[clean] removed {path}")

    print(
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        " SAM3 directional visualization\n"
        f" RGB       : {rgb_dir}\n"
        f" SAM3      : {sam_dir}\n"
        f" class_id  : {args.class_id}\n"
        f" box format: {args.box_format}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    render_directional_overlays(
        rgb_dir=rgb_dir,
        sam_dir=sam_dir,
        class_id=args.class_id,
        box_format=args.box_format,
    )

    render_directional_mask_unions(
        sam_dir=sam_dir,
        class_id=args.class_id,
    )

    print("\n✓ Done")
    print(
        f"  RGB box overlays : "
        f"{sam_dir / 'directional_overlay'}"
    )
    print(
        f"  Union masks      : "
        f"{sam_dir / 'directional_png'}"
    )


if __name__ == "__main__":
    main()
