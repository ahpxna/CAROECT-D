#!/usr/bin/env python3
"""Export auditable, bidirectionally merged SAM3 trajectories.

Why this file exists
--------------------
Per-frame YOLO text contains only class_id and normalized cx/cy/w/h values.
It discards SAM object identity, masks, confidence, and propagation provenance.
CAROECT-D needs those fields so an exact observation at frame k can be copied
to the causal detector sample ending at the same timestamp.

Bidirectional method
--------------------
Each class prompt is processed twice in independent SAM3 sessions:

* forward: prompt frame 0, then explicit forward propagation;
* backward: prompt the last frame, then explicit backward propagation.

Both pre-merge artifacts are saved under directional_tracks/. Same-class
trajectories are associated using temporal overlap and mean box IoU. At each
frame the merge chooses continuity first and confidence second. Only when
those candidates are comparable does it prefer the trajectory with stronger
image evidence of recession: decreasing box area and motion toward the
configured horizon. Propagation direction itself never receives a preference.
The selected source and reason remain in every observation for audit.

After directional merging, the existing cross-prompt NMS remains in place so
the same physical object detected by multiple class prompts is not duplicated.

Timestamp and geometry contract
-------------------------------
tracks.json includes n_frames and the complete frame_times_us vector derived
from camera.fps_original, not fps_export. Boxes use SAM3's normalized
center-format convention [cx, cy, width, height]. Overlay images are a required
visual sanity check because an API-version convention mismatch would propagate
directly into every event label. Gaps are preserved; no synthetic observation
is inserted by this exporter.

Usage:
  python sam3_export_tracks.py frames_dir/ --prompt car --output-dir out/ \
      --config config.yaml --class-id 0 --also-yolo
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from sam3_video_to_labels import (
    propagate_in_video, load_sorted_frame_paths, draw_overlay, write_yolo_label,
)
from sam3.model_builder import build_sam3_video_predictor


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _mask_to_uint8(mask, height: int, width: int):
    m = _as_numpy(mask)
    if m.ndim == 3:
        m = np.squeeze(m)
    if m.shape != (height, width):
        m = cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8) * 255


def _xywh_to_xyxy(box: dict):
    cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _box_iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = _xywh_to_xyxy(a)
    bx0, by0, bx1, by1 = _xywh_to_xyxy(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_duplicate_observations(tracks: dict, iou_threshold: float) -> tuple[dict, int]:
    """Frame-wise high-IoU NMS across all SAM3 prompt sessions.

    SAM3 is run once per class prompt in this wrapper, so the same physical
    object can appear in multiple prompt sessions. We keep the highest-score
    observation per frame and drop lower-score duplicates; track IDs remain
    stable for the remaining observations.
    """
    by_frame: dict[int, list[tuple[str, int, dict]]] = {}
    for key, entry in tracks.items():
        for obs_idx, obs in enumerate(entry["frames"]):
            by_frame.setdefault(int(obs["frame_idx"]), []).append((key, obs_idx, obs))

    drop: set[tuple[str, int]] = set()
    for rows in by_frame.values():
        rows = sorted(rows, key=lambda r: float(r[2].get("score", 0.0)), reverse=True)
        kept: list[tuple[str, int, dict]] = []
        for key, obs_idx, obs in rows:
            duplicate = any(_box_iou(obs, kept_obs) >= iou_threshold for _, _, kept_obs in kept)
            if duplicate:
                drop.add((key, obs_idx))
            else:
                kept.append((key, obs_idx, obs))

    if not drop:
        return tracks, 0

    merged = {}
    for key, entry in tracks.items():
        frames = [obs for obs_idx, obs in enumerate(entry["frames"]) if (key, obs_idx) not in drop]
        if frames:
            kept_entry = dict(entry)
            kept_entry["frames"] = frames
            merged[key] = kept_entry
    return merged, len(drop)


def _trajectory_receding_score(entry: dict, horizon_y: float,
                               area_weight: float = 0.65,
                               horizon_weight: float = 0.35) -> float:
    """Score recession from image trajectory, independent of propagation direction.

    A receding road user usually shrinks and moves toward the configured
    horizon/vanishing region. The score is used only after continuity and
    confidence are otherwise comparable.
    """
    frames = sorted(entry.get("frames", []), key=lambda row: row["frame_idx"])
    if len(frames) < 2:
        return 0.0
    first, last = frames[0], frames[-1]
    first_area = max(float(first["w"]) * float(first["h"]), 1e-12)
    last_area = max(float(last["w"]) * float(last["h"]), 1e-12)
    area_score = np.clip((first_area - last_area) / first_area, -1.0, 1.0)
    first_distance = abs((float(first["cy"]) + 0.5 * float(first["h"])) - horizon_y)
    last_distance = abs((float(last["cy"]) + 0.5 * float(last["h"])) - horizon_y)
    horizon_score = np.clip(
        (first_distance - last_distance) / max(first_distance, 1e-12), -1.0, 1.0)
    return float(area_weight * area_score + horizon_weight * horizon_score)


def _mean_temporal_iou(a: dict, b: dict) -> tuple[int, float]:
    by_frame_a = {int(row["frame_idx"]): row for row in a.get("frames", [])}
    by_frame_b = {int(row["frame_idx"]): row for row in b.get("frames", [])}
    overlap = sorted(set(by_frame_a) & set(by_frame_b))
    if not overlap:
        return 0, 0.0
    return len(overlap), float(np.mean([
        _box_iou(by_frame_a[frame], by_frame_b[frame]) for frame in overlap
    ]))


def associate_directional_tracks(forward: dict, backward: dict,
                                 min_mean_iou: float = 0.1):
    """Greedily associate same-class trajectories by overlap and mean IoU."""
    candidates = []
    for forward_key, forward_entry in forward.items():
        for backward_key, backward_entry in backward.items():
            if forward_entry["class_id"] != backward_entry["class_id"]:
                continue
            n_overlap, mean_iou = _mean_temporal_iou(forward_entry, backward_entry)
            if n_overlap and mean_iou >= min_mean_iou:
                candidates.append(
                    (-n_overlap, -mean_iou, forward_key, backward_key))
    used_forward, used_backward, pairs = set(), set(), []
    for _negative_overlap, _negative_iou, forward_key, backward_key in sorted(candidates):
        if forward_key in used_forward or backward_key in used_backward:
            continue
        used_forward.add(forward_key)
        used_backward.add(backward_key)
        pairs.append((forward_key, backward_key))
    return pairs, sorted(set(forward) - used_forward), sorted(set(backward) - used_backward)


def _merge_observation_pair(forward_obs, backward_obs, previous_obs,
                            forward_receding, backward_receding,
                            continuity_tolerance, score_tolerance):
    """Choose one candidate: continuity first, score second, recession last."""
    if forward_obs is None:
        chosen, reason = backward_obs, "only_backward"
    elif backward_obs is None:
        chosen, reason = forward_obs, "only_forward"
    else:
        forward_continuity = _box_iou(previous_obs, forward_obs) if previous_obs else 0.0
        backward_continuity = _box_iou(previous_obs, backward_obs) if previous_obs else 0.0
        if previous_obs and abs(forward_continuity - backward_continuity) > continuity_tolerance:
            chosen = forward_obs if forward_continuity > backward_continuity else backward_obs
            reason = "continuity"
        else:
            forward_score = float(forward_obs.get("score", 0.0))
            backward_score = float(backward_obs.get("score", 0.0))
            if abs(forward_score - backward_score) > score_tolerance:
                chosen = forward_obs if forward_score > backward_score else backward_obs
                reason = "confidence"
            elif abs(forward_receding - backward_receding) > 1e-12:
                chosen = forward_obs if forward_receding > backward_receding else backward_obs
                reason = "receding_tiebreak"
            else:
                chosen, reason = forward_obs, "deterministic_forward_tie"
    selected = dict(chosen)
    selected["chosen_source"] = chosen["source"]
    selected["merge_reason"] = reason
    if forward_obs is not None and backward_obs is not None:
        selected["candidate_scores"] = {
            "forward": float(forward_obs.get("score", 0.0)),
            "backward": float(backward_obs.get("score", 0.0)),
        }
    return selected


def merge_directional_tracks(forward: dict, backward: dict, merge_cfg: dict):
    """Associate and deterministically merge forward/backward SAM3 artifacts."""
    horizon_y = float(merge_cfg.get("horizon_y", 0.45))
    continuity_tolerance = float(merge_cfg.get("continuity_iou_tolerance", 0.02))
    score_tolerance = float(merge_cfg.get("score_tolerance", 0.02))
    area_weight = float(merge_cfg.get("area_trend_weight", 0.65))
    horizon_weight = float(merge_cfg.get("horizon_weight", 0.35))
    pairs, forward_only, backward_only = associate_directional_tracks(forward, backward)
    merged = {}

    for pair_index, (forward_key, backward_key) in enumerate(pairs):
        forward_entry, backward_entry = forward[forward_key], backward[backward_key]
        forward_receding = _trajectory_receding_score(
            forward_entry, horizon_y, area_weight, horizon_weight)
        backward_receding = _trajectory_receding_score(
            backward_entry, horizon_y, area_weight, horizon_weight)
        forward_by_frame = {
            int(row["frame_idx"]): row for row in forward_entry["frames"]}
        backward_by_frame = {
            int(row["frame_idx"]): row for row in backward_entry["frames"]}
        frames, previous = [], None
        for frame_idx in sorted(set(forward_by_frame) | set(backward_by_frame)):
            selected = _merge_observation_pair(
                forward_by_frame.get(frame_idx), backward_by_frame.get(frame_idx),
                previous, forward_receding, backward_receding,
                continuity_tolerance, score_tolerance)
            frames.append(selected)
            previous = selected
        merged[f"{forward_entry['class_id']}_merged_{pair_index}"] = {
            "class_id": forward_entry["class_id"],
            "class_name": forward_entry.get("class_name"),
            "directional_track_ids": {
                "forward": forward_key, "backward": backward_key},
            "receding_scores": {
                "forward": forward_receding, "backward": backward_receding},
            "frames": frames,
        }

    for source, keys, collection in (
        ("forward", forward_only, forward),
        ("backward", backward_only, backward),
    ):
        for key in keys:
            entry = dict(collection[key])
            entry["directional_track_ids"] = {source: key}
            entry["frames"] = [
                {**row, "chosen_source": source, "merge_reason": f"{source}_only_track"}
                for row in entry["frames"]
            ]
            merged[f"{key}_{source}_only"] = entry
    return merged


def _collect_direction(predictor, video_folder: str, prompt: str, class_id: int,
                       prompt_frame: int, direction: str, fps: float,
                       width: int, height: int, score_threshold: float,
                       masks_dir: Path):
    """Run one direction in its own session and retain its pre-merge artifact."""
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=video_folder))
    session_id = response["session_id"]
    response = predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id,
                     frame_index=prompt_frame, text=prompt))
    prompt_output = response["outputs"]
    outputs = propagate_in_video(predictor, session_id, direction)
    outputs[prompt_frame] = outputs.get(prompt_frame, prompt_output)
    tracks = {}
    for frame_idx in sorted(outputs):
        output = outputs[frame_idx]
        masks = output.get("out_binary_masks")
        for index, obj_id in enumerate(output["out_obj_ids"]):
            score = float(output["out_probs"][index])
            if score < score_threshold:
                continue
            raw_obj_id = int(obj_id)
            key = f"{class_id}_{direction}_{raw_obj_id}"
            box = [float(value) for value in output["out_boxes_xywh"][index]]
            mask_path = None
            if masks is not None and index < len(masks):
                mask_name = (
                    f"{frame_idx:06d}_c{class_id}_{direction}_o{raw_obj_id}.png")
                cv2.imwrite(
                    str(masks_dir / mask_name),
                    _mask_to_uint8(masks[index], height, width))
                mask_path = f"masks/{mask_name}"
            entry = tracks.setdefault(key, {
                "class_id": class_id,
                "class_name": prompt,
                "raw_obj_id": raw_obj_id,
                "propagation_direction": direction,
                "frames": [],
            })
            entry["frames"].append({
                "frame_idx": int(frame_idx),
                "t_us": int(round(frame_idx * 1e6 / fps)),
                "cx": box[0], "cy": box[1], "w": box[2], "h": box[3],
                "score": score, "mask_path": mask_path, "source": direction,
            })
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    return tracks


def _run_one_prompt(predictor, video_folder: str, frame_paths: list[str], prompt: str,
                    class_id: int, fps: float, width: int, height: int,
                    score_threshold: float, out_dir: Path, also_yolo: bool,
                    merge_cfg: dict):
    labels_dir, overlay_dir, masks_dir = out_dir / "labels", out_dir / "overlay", out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPrompt '{prompt}' (class_id={class_id})")
    forward = _collect_direction(
        predictor, video_folder, prompt, class_id, 0, "forward", fps,
        width, height, score_threshold, masks_dir)
    backward = _collect_direction(
        predictor, video_folder, prompt, class_id, len(frame_paths) - 1,
        "backward", fps, width, height, score_threshold, masks_dir)
    artifact_dir = out_dir / "directional_tracks"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / f"class_{class_id}_forward.json").write_text(
        json.dumps(forward, indent=2))
    (artifact_dir / f"class_{class_id}_backward.json").write_text(
        json.dumps(backward, indent=2))
    tracks = merge_directional_tracks(forward, backward, merge_cfg)
    print(
        f"  Directional artifacts: {len(forward)} forward, "
        f"{len(backward)} backward -> {len(tracks)} merged track(s)")

    yolo_by_frame: dict[int, list] = {}
    for entry in tracks.values():
        for observation in entry["frames"]:
            yolo_by_frame.setdefault(int(observation["frame_idx"]), []).append(
                (class_id, observation))
    if also_yolo:
        for frame_idx, rows in sorted(yolo_by_frame.items()):
            image = cv2.imread(frame_paths[frame_idx])
            if image is None:
                continue
            boxes = np.asarray([
                [row["cx"], row["cy"], row["w"], row["h"]] for _, row in rows])
            scores = np.asarray([row.get("score", 0.0) for _, row in rows])
            obj_ids = np.arange(len(rows))
            masks = []
            any_mask = False
            for _, row in rows:
                mask = cv2.imread(
                    str(out_dir / row["mask_path"]), cv2.IMREAD_GRAYSCALE
                ) if row.get("mask_path") else None
                masks.append(mask > 0 if mask is not None else np.zeros((height, width), bool))
                any_mask = any_mask or mask is not None
            blended, _ = draw_overlay(
                image, boxes, masks if any_mask else None, obj_ids, scores,
                prompt, score_threshold)
            cv2.imwrite(
                str(overlay_dir / f"{Path(frame_paths[frame_idx]).stem}_{class_id}.png"),
                blended)
    return tracks, yolo_by_frame


def main():
    ap = argparse.ArgumentParser(
        description="Bidirectional SAM3 video propagation -> auditable tracks.json")
    ap.add_argument("video_folder", help="Folder containing 0.tiff, 1.tiff, ...")
    ap.add_argument("--prompt", default=None,
                    help="One text prompt; --all-classes reads the configured class list")
    ap.add_argument("--all-classes", action="store_true",
                    help="Run every label_transfer.class_names prompt")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument("--also-yolo", action="store_true",
                    help="Also write YOLO labels and visual overlay images")
    ap.add_argument("--merge-iou", type=float, default=None,
                    help="Cross-prompt NMS IoU threshold")
    ap.add_argument("--no-merge", action="store_true",
                    help="Disable cross-prompt NMS")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fps = float(cfg["camera"]["fps_original"])  # Capture FPS, never export-timeline FPS.
    W, H = int(cfg["camera"]["width"]), int(cfg["camera"]["height"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir, overlay_dir = out_dir / "labels", out_dir / "overlay"
    if args.also_yolo:
        labels_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = load_sorted_frame_paths(args.video_folder)
    if not frame_paths:
        raise FileNotFoundError(f"No .tif/.tiff frames in {args.video_folder}")
    print(f"{len(frame_paths)} frame(s) @ fps_original={fps}  "
          f"(duration={len(frame_paths)/fps:.2f}s)")

    if args.all_classes:
        prompts = list(cfg.get("label_transfer", {}).get("class_names", []))
        if not prompts:
            raise ValueError("--all-classes requires label_transfer.class_names in config.yaml")
        prompt_class_pairs = [(p, i) for i, p in enumerate(prompts)]
    else:
        prompt = args.prompt or "car"
        prompt_class_pairs = [(prompt, args.class_id)]

    print("Loading SAM3 video predictor (first run downloads checkpoint)...")
    predictor = build_sam3_video_predictor()

    tracks: dict = {}
    yolo_by_frame: dict[int, list] = {}
    for prompt, class_id in prompt_class_pairs:
        prompt_tracks, prompt_yolo = _run_one_prompt(
            predictor, args.video_folder, frame_paths, prompt, class_id, fps, W, H,
            args.score_threshold, out_dir, args.also_yolo,
            cfg.get("label_transfer", {}).get("sam_merge", {}))
        tracks.update(prompt_tracks)
        for frame_idx, rows in prompt_yolo.items():
            yolo_by_frame.setdefault(frame_idx, []).extend(rows)

    predictor.shutdown()

    for entry in tracks.values():
        entry["frames"].sort(key=lambda r: r["frame_idx"])

    merge_iou = args.merge_iou
    if merge_iou is None:
        merge_iou = float(cfg.get("label_transfer", {}).get("cross_class_iou", 0.85))
    n_dropped = 0
    if not args.no_merge and merge_iou > 0:
        tracks, n_dropped = merge_duplicate_observations(tracks, merge_iou)
        print(f"\nCross-prompt NMS: dropped {n_dropped} duplicate observation(s) "
              f"(IoU >= {merge_iou:.2f})")

    if args.also_yolo:
        yolo_by_frame = {}
        for key, entry in tracks.items():
            for obs in entry["frames"]:
                yolo_by_frame.setdefault(int(obs["frame_idx"]), []).append((entry["class_id"], obs))
        for frame_idx, rows in yolo_by_frame.items():
            with open(labels_dir / f"{frame_idx}.txt", "w") as f:
                for class_id, obs in rows:
                    f.write(f"{class_id} {obs['cx']:.6f} {obs['cy']:.6f} "
                            f"{obs['w']:.6f} {obs['h']:.6f}\n")

    frame_times_us = [int(round(index * 1e6 / fps)) for index in range(len(frame_paths))]
    payload = dict(fps=fps, width=W, height=H,
                   n_frames=len(frame_paths), frame_times_us=frame_times_us,
                   fps_provenance="camera.fps_original from config.yaml",
                   propagation_directions=["forward", "backward"],
                   merge_policy="continuity_then_confidence_then_receding_tiebreak",
                   prompts=[p for p, _ in prompt_class_pairs],
                   merge_iou=merge_iou, n_merged_duplicate_observations=n_dropped,
                   tracks=tracks)
    tracks_path = out_dir / "tracks.json"
    tracks_path.write_text(json.dumps(payload, indent=2))

    n_tracks = len(tracks)
    n_obs = sum(len(t["frames"]) for t in tracks.values())
    print(f"\n✓ {tracks_path}  ({n_tracks} track(s), {n_obs} box-observation(s) total)")
    if args.also_yolo:
        print(f"  YOLO/overlay (sanity check) -> {labels_dir}/ , {overlay_dir}/")
        print("  Inspect several overlays before label transfer; downstream event labels "
              "inherit any SAM3 annotation error.")
    print(f"\n  Next: python label_transfer.py --tracks {tracks_path} --events <events.h5> "
          f"--window-us 8340 --output <windows.json>")


if __name__ == "__main__":
    main()
