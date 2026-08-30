#!/usr/bin/env python3
"""Build causal event-count/time-surface images and exact frame-k labels.

Count channels use one fixed training-derived clip shared by ON and OFF.
Native 1280x720 output is the default. Optional alternate sizes use
aspect-ratio-preserving letterbox transforms for both pixels and labels.
"""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from label_transfer import event_index_bounds


REPRESENTATION_VERSION = "caroectd-count-time-v2"


def load_events_h5(path):
    with h5py.File(path, "r") as handle:
        events = {key: handle[key][:] for key in ("x", "y", "t", "p")}
    order = np.argsort(events["t"], kind="stable")
    return {key: value[order] for key, value in events.items()}


def count_maps(events, lo, hi, width, height):
    x = events["x"][lo:hi].astype(np.int64)
    y = events["y"][lo:hi].astype(np.int64)
    polarity = events["p"][lo:hi] > 0
    valid = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    x, y, polarity = x[valid], y[valid], polarity[valid]
    on = np.zeros((height, width), dtype=np.uint32)
    off = np.zeros((height, width), dtype=np.uint32)
    if len(x):
        np.add.at(on, (y[polarity], x[polarity]), 1)
        np.add.at(off, (y[~polarity], x[~polarity]), 1)
    return on, off, x, y


def fit_count_clip(events, windows, width, height, percentile=99.5):
    """Fit one ON/OFF clip from training windows only."""
    histogram = np.zeros(2, dtype=np.int64)
    for window in windows:
        lo, hi = event_index_bounds(events["t"], window["t_start_us"], window["t_end_us"])
        on, off, _, _ = count_maps(events, lo, hi, width, height)
        values = np.concatenate((on[on > 0], off[off > 0]))
        if not len(values):
            continue
        counts = np.bincount(values.astype(np.int64))
        if len(counts) > len(histogram):
            histogram = np.pad(histogram, (0, len(counts) - len(histogram)))
        histogram[:len(counts)] += counts
    total = int(histogram[1:].sum())
    if not total:
        return 1.0
    target = percentile / 100.0 * total
    cumulative = np.cumsum(histogram)
    return float(max(1, int(np.searchsorted(cumulative, target, side="left"))))


def encode_count_u8(count, count_clip):
    """Encode count amplitude with a fixed scale; never normalize per window."""
    if count_clip <= 0:
        raise ValueError("count_clip must be positive")
    return np.rint(255.0 * np.clip(count, 0, count_clip) / count_clip).astype(np.uint8)


def render_window(events, lo, hi, start_us, end_us, width, height, count_clip):
    on, off, x, y = count_maps(events, lo, hi, width, height)
    time_surface = np.zeros((height, width), dtype=np.float32)
    if hi > lo and len(x):
        raw_x = events["x"][lo:hi].astype(np.int64)
        raw_y = events["y"][lo:hi].astype(np.int64)
        raw_t = events["t"][lo:hi].astype(np.float64)
        valid = (0 <= raw_x) & (raw_x < width) & (0 <= raw_y) & (raw_y < height)
        normalized_t = np.clip(
            (raw_t[valid] - start_us) / max(end_us - start_us, 1.0), 0.0, 1.0
        )
        np.maximum.at(time_surface, (raw_y[valid], raw_x[valid]), normalized_t)
    return np.stack(
        (encode_count_u8(on, count_clip),
         encode_count_u8(off, count_clip),
         np.rint(time_surface * 255.0).astype(np.uint8)),
        axis=2,
    )


def letterbox_image(image, target_width, target_height):
    """Resize without aspect warp and return transform metadata."""
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    canvas = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
    canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    return canvas, {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "source_width": width,
        "source_height": height,
        "target_width": target_width,
        "target_height": target_height,
    }


def transform_box_letterbox(box, transform):
    """Transform normalized center-format coordinates into letterboxed output."""
    source_width = transform["source_width"]
    source_height = transform["source_height"]
    target_width = transform["target_width"]
    target_height = transform["target_height"]
    scale = transform["scale"]
    cx = (float(box["cx"]) * source_width * scale + transform["pad_x"]) / target_width
    cy = (float(box["cy"]) * source_height * scale + transform["pad_y"]) / target_height
    width = float(box["w"]) * source_width * scale / target_width
    height = float(box["h"]) * source_height * scale / target_height
    return {**box, "cx": cx, "cy": cy, "w": width, "h": height}


def write_data_yaml(root, class_names):
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"names:\n{names}\n"
    )


def update_coco(root, split, site_id, class_names, samples):
    """Write/merge a COCO detection file while preserving CAROECT-D provenance."""
    annotation_dir = root / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    path = annotation_dir / f"instances_{split}.json"
    if path.exists():
        coco = json.loads(path.read_text())
    else:
        coco = {
            "info": {
                "description": "CAROECT-D causal frame-observation detector labels",
                "representation_version": REPRESENTATION_VERSION,
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

    # Rebuilding one site replaces only that site's records and leaves other
    # sites in the same split intact.
    removed_ids = {
        int(image["id"]) for image in coco.get("images", [])
        if image.get("site_id") == site_id
    }
    coco["images"] = [
        image for image in coco.get("images", []) if int(image["id"]) not in removed_ids
    ]
    coco["annotations"] = [
        annotation for annotation in coco.get("annotations", [])
        if int(annotation["image_id"]) not in removed_ids
    ]
    coco["categories"] = [
        {"id": index, "name": name, "supercategory": "traffic_participant"}
        for index, name in enumerate(class_names)
    ]

    next_image_id = max((int(row["id"]) for row in coco["images"]), default=0) + 1
    next_annotation_id = max(
        (int(row["id"]) for row in coco["annotations"]), default=0) + 1
    for sample in samples:
        image_id = next_image_id
        next_image_id += 1
        coco["images"].append({
            "id": image_id,
            "file_name": f"{split}/images/{sample['stem']}.png",
            "width": int(sample["width"]),
            "height": int(sample["height"]),
            "site_id": site_id,
            "frame_idx": int(sample["frame_idx"]),
            "t_k_us": float(sample["t_k_us"]),
            "window_us": float(sample["window_us"]),
            "event_index_start": int(sample["event_index_start"]),
            "event_index_end": int(sample["event_index_end"]),
        })
        for box in sample["boxes"]:
            width_px = float(box["w"]) * sample["width"]
            height_px = float(box["h"]) * sample["height"]
            x_px = float(box["cx"]) * sample["width"] - width_px / 2.0
            y_px = float(box["cy"]) * sample["height"] - height_px / 2.0
            coco["annotations"].append({
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": int(box["class_id"]),
                "bbox": [x_px, y_px, width_px, height_px],
                "area": width_px * height_px,
                "iscrowd": 0,
                "track_id": box.get("track_id"),
            })
            next_annotation_id += 1

    path.write_text(json.dumps(coco, indent=2))
    return path


def resolve_representation(args, events, payload, root):
    path = Path(args.representation) if args.representation else root / "representation.json"
    if path.exists():
        representation = json.loads(path.read_text())
        if representation.get("representation_version") != REPRESENTATION_VERSION:
            raise ValueError(f"Incompatible representation manifest: {path}")
        if float(representation["window_us"]) != float(payload["window_us"]):
            raise ValueError("Representation window_us does not match windows.json")
        fit_source = representation.get("fit_source")
        if not isinstance(fit_source, dict) or not fit_source.get("events") or not fit_source.get("windows"):
            raise ValueError(
                f"Representation manifest {path} predates fit-source provenance. "
                "Delete/regenerate it from an explicit training source before reuse."
            )
        if representation.get("frozen_after_fit") is not True:
            raise ValueError(f"Representation manifest is not marked frozen: {path}")
        return representation, path
    if args.split != "train":
        raise RuntimeError(
            "Validation/test builds must reuse a training representation.json via "
            "--representation or an existing output-root manifest."
        )
    clip = args.count_clip
    if clip is None:
        clip = fit_count_clip(
            events, payload["windows"], int(payload["width"]), int(payload["height"]),
            args.count_clip_percentile,
        )
    representation = {
        "representation_version": REPRESENTATION_VERSION,
        "channels": ["on_count", "off_count", "last_timestamp"],
        "count_clip": float(clip),
        "count_clip_percentile": float(args.count_clip_percentile),
        "fitting_split": "train",
        "shared_on_off_scale": True,
        "window_us": float(payload["window_us"]),
        "interval": "[t_k-window_us,t_k)",
        "fit_method": "explicit_count_clip" if args.count_clip is not None else "positive_count_percentile",
        "fit_source": {
            "events": str(Path(args.events).resolve()),
            "windows": str(Path(args.windows).resolve()),
            "site_id": args.site_id,
            "split": args.split,
        },
        "frozen_after_fit": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(representation, indent=2))
    return representation, path


def process(args):
    payload = json.loads(Path(args.windows).read_text())
    if payload.get("label_semantics") != "causal_exact_frame_observation":
        raise RuntimeError("Refusing non-causal/interpolated labels in the normal dataset builder")
    events = load_events_h5(args.events)
    width, height = int(payload["width"]), int(payload["height"])
    root = Path(args.output)
    representation, representation_path = resolve_representation(args, events, payload, root)
    count_clip = float(representation["count_clip"])

    image_dir = root / args.split / "images"
    label_dir = root / args.split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    class_names = args.class_names
    write_data_yaml(root, class_names)

    sample_rows = []
    coco_samples = []
    for index, window in enumerate(payload["windows"]):
        lo, hi = event_index_bounds(events["t"], window["t_start_us"], window["t_end_us"])
        image = render_window(
            events, lo, hi, window["t_start_us"], window["t_end_us"],
            width, height, count_clip,
        )
        boxes = list(window["boxes"])
        transform = None
        if args.img_size is not None:
            image, transform = letterbox_image(image, args.img_size[0], args.img_size[1])
            boxes = [transform_box_letterbox(box, transform) for box in boxes]

        stem = f"{args.site_id}_{int(window.get('frame_idx', index)):06d}"
        cv2.imwrite(str(image_dir / f"{stem}.png"), image)
        with open(label_dir / f"{stem}.txt", "w") as handle:
            for box in boxes:
                handle.write(
                    f"{int(box['class_id'])} {float(box['cx']):.8f} "
                    f"{float(box['cy']):.8f} {float(box['w']):.8f} {float(box['h']):.8f}\n"
                )
        sample_rows.append({
            "sample": stem,
            "frame_idx": int(window["frame_idx"]),
            "t_k_us": float(window["t_end_us"]),
            "window_us": float(payload["window_us"]),
            "event_index_start": lo,
            "event_index_end": hi,
            "n_events": hi - lo,
            "n_labels": len(boxes),
            "image_shape": list(image.shape),
            "letterbox": transform,
        })
        coco_samples.append({
            "stem": stem,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "frame_idx": int(window["frame_idx"]),
            "t_k_us": float(window["t_end_us"]),
            "window_us": float(payload["window_us"]),
            "event_index_start": lo,
            "event_index_end": hi,
            "boxes": boxes,
        })
        if args.preview and len(sample_rows) >= args.preview:
            break

    metadata_path = root / args.split / f"{args.site_id}_samples.json"
    metadata_path.write_text(json.dumps(sample_rows, indent=2))
    coco_path = None
    if args.export_coco:
        coco_path = update_coco(
            root, args.split, args.site_id, class_names, coco_samples)
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": 1,
        "representation": str(representation_path.resolve()),
        "sources": [],
    }
    manifest["sources"] = [
        row for row in manifest["sources"]
        if not (row.get("site_id") == args.site_id and row.get("split") == args.split)
    ]
    manifest["sources"].append({
        "site_id": args.site_id,
        "split": args.split,
        "events": str(Path(args.events).resolve()),
        "windows": str(Path(args.windows).resolve()),
        "window_us": float(payload["window_us"]),
        "sample_metadata": str(metadata_path.resolve()),
        "coco_annotations": str(coco_path.resolve()) if coco_path else None,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"Wrote {len(sample_rows)} samples to {root} using count_clip={count_clip:g}; "
        f"representation={representation_path}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--representation")
    parser.add_argument("--count-clip", type=float)
    parser.add_argument("--count-clip-percentile", type=float, default=99.5)
    parser.add_argument("--preview", type=int)
    parser.add_argument("--export-coco", action="store_true",
                        help="Also write COCO detection JSON with CAROECT-D timing provenance")
    parser.add_argument("--class-names", nargs="+", default=[
        "car", "truck", "bus", "motorcycle", "person", "bicycle",
        "micromobility", "other",
    ])
    process(parser.parse_args())


if __name__ == "__main__":
    main()
