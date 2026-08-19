#!/usr/bin/env python3
"""
calibrate_event_camera.py — Event-Camera Intrinsic & Cross-Sensor Registration
==================================================================================
Implements paper Section III-C.4 (Event-Camera Intrinsic Calibration and
Cross-Sensor Registration): recovers K_event/D_event for the physical Triton2
sensor, and the RGB<->event spatial correspondence (homography H, with a
full stereo extrinsic (R,t) fallback), from a blinking-checkerboard capture.

WHY THIS VERSION DIFFERS FROM A NAIVE FIRST DRAFT
----------------------------------------------------
1. AUTO-DETECTED BLINK TRANSITIONS, not an assumed fixed period. A GIF's
   nominal per-frame duration is not a guarantee of real monitor/player
   timing — slicing a recording into fixed windows starting from t_min
   silently drifts out of alignment with the real transitions as playback
   jitter accumulates. Transitions are instead found from the event stream
   itself: a real blink produces a sharp, board-wide burst (every edge/
   corner fires within a few hundred microseconds of each other), which
   stands out clearly against the background rate in a coarse histogram —
   robust to whatever the display actually did, not to what the GIF says
   it should have done.

2. ONE FILE PER PHYSICAL POSE, not one long continuous recording. This is
   the SAME operational convention Section III-C's existing RGB chessboard
   calibration already uses (one photo per pose in chessboard/) — extended
   here to "one short event recording per pose" instead of inventing a
   fragile continuous-stream pose-segmentation heuristic. Filenames pair
   RGB and event captures for the SAME physical pose directly (pose01.h5 <->
   pose01.tiff), which is what makes correct point-pooling for Eq. 32
   (homography) and the stereo fallback possible at all — a single
   index like [0] cannot correctly pair two streams with unrelated
   detection counts and no shared clock.

3. MULTIPLE BLINKS PER POSE ARE AVERAGED, not treated as independent views.
   A held pose typically spans several blink cycles; treating each as its
   own Zhang's-method "view" both inflates the apparent sample size with
   near-duplicate, highly correlated observations (biasing the intrinsic
   estimate) and throws away the noise-averaging benefit of having several
   independent measurements of the exact same true corner positions.

4. CORNER POINTS ARE UNDISTORTED BEFORE HOMOGRAPHY FITTING. The RGB rig
   (20mm) and event rig (6mm) have very different fields of view and
   correspondingly different distortion profiles; fitting a homography
   directly on still-distorted pixel coordinates biases the fit, worst at
   the frame periphery. Points are undistorted with each camera's own
   freshly (or previously) computed K,D before H is estimated.

5. BOTH A HOMOGRAPHY AND A FULL STEREO EXTRINSIC ARE COMPUTED, with
   per-pose depth (via solvePnP) and per-pose reprojection error under
   both models reported side by side — exactly the diagnostic the paper
   text calls for to confirm whether the small-baseline/large-depth
   approximation actually holds over the range of distances one roadside
   scene spans, rather than assuming it does.

6. FAIL-LOUD at every stage with an actionable message (same convention as
   calibrate.py's `if found < 5: raise RuntimeError`, and
   linear16_to_srgb8.py's guard functions) instead of silently proceeding
   on too little data.

Usage:
  python calibrate_event_camera.py \\
      --event-dir event_calib/ --rgb-dir chessboard_matched/ \\
      --config config.yaml --debug-dir _debug_event_calib/

  Expects, for each pose N: event_calib/poseNN.h5  <->  chessboard_matched/poseNN.tiff
  (any matching stem works — files are paired by sorted-filename order, and
  the counts on both sides must match exactly).
"""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml


# ══════════════════════════════════════════════════════════════════
#  N1 · CONFIG
# ══════════════════════════════════════════════════════════════════

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ec_cfg(cfg):
    """Optional tunables under event_calibration: — sensible defaults if the
    section isn't present yet, same .get()-with-default convention used
    elsewhere in this project for non-required knobs (e.g.
    sam3_export_tracks.py's label_transfer.cross_class_iou)."""
    e = cfg.get("event_calibration", {})
    return dict(
        blink_bin_us=int(e.get("blink_bin_us", 2000)),
        blink_prominence=float(e.get("blink_prominence", 3.0)),
        blink_min_gap_us=int(e.get("blink_min_gap_us", 200_000)),
        frame_window_us=int(e.get("frame_window_us", 10_000)),
        blur_sigma=float(e.get("blur_sigma", 1.0)),
        min_events_for_attempt=int(e.get("min_events_for_attempt", 500)),
    )


# ══════════════════════════════════════════════════════════════════
#  N2 · LOAD EVENTS
# ══════════════════════════════════════════════════════════════════

def load_events_h5(path):
    with h5py.File(str(path), "r") as hf:
        return {k: hf[k][:] for k in ("x", "y", "t", "p")}


# ══════════════════════════════════════════════════════════════════
#  N3 · AUTO-DETECT BLINK TRANSITIONS  (see docstring point 1)
# ══════════════════════════════════════════════════════════════════

def detect_blink_transitions(events: dict, bin_us: int, prominence: float,
                             min_gap_us: int) -> np.ndarray:
    """Find timestamps where the GLOBAL event rate spikes sharply — the
    signature of every corner/edge on the board firing near-simultaneously
    right after an inversion. Robust to real display timing not matching
    the GIF's nominal per-frame duration."""
    t = events["t"]
    if len(t) == 0:
        return np.array([], dtype=np.int64)

    t_min, t_max = int(t.min()), int(t.max())
    n_bins = max(1, (t_max - t_min) // bin_us + 1)
    counts, edges = np.histogram(t, bins=n_bins, range=(t_min, t_max))

    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return np.array([], dtype=np.int64)
    background = np.median(nonzero)
    threshold = max(background * prominence, 10)

    candidate_bins = np.where(counts > threshold)[0]
    if len(candidate_bins) == 0:
        return np.array([], dtype=np.int64)

    bin_times = edges[candidate_bins].astype(np.int64)
    transitions = [int(bin_times[0])]
    for bt in bin_times[1:]:
        if bt - transitions[-1] > min_gap_us:
            transitions.append(int(bt))
    return np.array(transitions, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════
#  N4 · ACCUMULATE ONE EVENT FRAME PER TRANSITION
# ══════════════════════════════════════════════════════════════════

def accumulate_event_frame(events: dict, t_start: int, t_end: int,
                           width: int, height: int,
                           blur_sigma: float) -> tuple[np.ndarray, int]:
    """Rasterize events in [t_start, t_end) into an 8-bit count image.
    Returns (image, n_events) so the caller can skip near-empty windows
    before spending time on corner detection."""
    mask = (events["t"] >= t_start) & (events["t"] < t_end)
    x = events["x"][mask].astype(np.int64)
    y = events["y"][mask].astype(np.int64)
    n_events = len(x)

    img = np.zeros((height, width), dtype=np.float32)
    if n_events > 0:
        np.add.at(img, (y, x), 1.0)

    if img.max() > 0:
        img = np.clip(img / img.max() * 255.0, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # Real accumulated event edges are pixel-sparse/speckled compared to a
    # smooth grayscale photo — a light blur helps findChessboardCorners'
    # internal thresholding see continuous lines rather than broken ones.
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)

    return img, n_events


# ══════════════════════════════════════════════════════════════════
#  N5 · CORNER DETECTION  (shared by RGB and event frames)
# ══════════════════════════════════════════════════════════════════

CORNER_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK)


def find_corners_one(img: np.ndarray, board_size: tuple) -> np.ndarray | None:
    ret, corners = cv2.findChessboardCorners(img, board_size, flags=CORNER_FLAGS)
    if not ret:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), crit)


# ══════════════════════════════════════════════════════════════════
#  N6 · PER-POSE EVENT CORNER EXTRACTION  (auto-detect + average, see
#       docstring points 1 and 3)
# ══════════════════════════════════════════════════════════════════

def event_pose_corners(h5_path: Path, cfg_e: dict, width: int, height: int,
                       board_size: tuple, debug_dir: Path | None,
                       pose_name: str) -> tuple[np.ndarray | None, dict]:
    events = load_events_h5(h5_path)
    transitions = detect_blink_transitions(
        events, cfg_e["blink_bin_us"], cfg_e["blink_prominence"], cfg_e["blink_min_gap_us"])

    stats = dict(pose=pose_name, n_transitions_found=int(len(transitions)),
                 n_frames_attempted=0, n_frames_with_corners=0)

    if len(transitions) == 0:
        return None, stats

    per_transition_corners = []
    for i, t_start in enumerate(transitions):
        t_end = t_start + cfg_e["frame_window_us"]
        img, n_events = accumulate_event_frame(
            events, t_start, t_end, width, height, cfg_e["blur_sigma"])

        if n_events < cfg_e["min_events_for_attempt"]:
            continue  # too sparse, don't waste time on findChessboardCorners
        stats["n_frames_attempted"] += 1

        corners = find_corners_one(img, board_size)
        if corners is None:
            continue
        stats["n_frames_with_corners"] += 1
        per_transition_corners.append(corners)

        if debug_dir is not None:
            dbg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(dbg, board_size, corners, True)
            cv2.imwrite(str(debug_dir / f"{pose_name}_blink{i:03d}.png"), dbg)

    if not per_transition_corners:
        return None, stats

    # Average corner positions across all successful blink detections
    # for this pose — docstring point 3: several correlated views of the
    # same true position, not several independent views.
    stacked = np.stack(per_transition_corners, axis=0)  # [n_blinks, n_corners, 1, 2]
    averaged = stacked.mean(axis=0)
    stats["n_blinks_averaged"] = len(per_transition_corners)
    return averaged, stats


# ══════════════════════════════════════════════════════════════════
#  N7 · RGB POSE CORNER EXTRACTION
# ══════════════════════════════════════════════════════════════════

def rgb_pose_corners(img_path: Path, board_size: tuple) -> np.ndarray | None:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return find_corners_one(gray, board_size)


# ══════════════════════════════════════════════════════════════════
#  N8 · PAIR RGB <-> EVENT POSE FILES BY SORTED FILENAME  (see docstring 2)
# ══════════════════════════════════════════════════════════════════

def pair_pose_files(event_dir: Path, rgb_dir: Path) -> list[tuple[Path, Path]]:
    event_files = sorted(event_dir.glob("*.h5"))
    rgb_exts = ["*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg"]
    rgb_files = sorted(p for ext in rgb_exts for p in rgb_dir.glob(ext))

    if len(event_files) != len(rgb_files):
        raise RuntimeError(
            f"Pose count mismatch: {len(event_files)} event file(s) in {event_dir} "
            f"vs {len(rgb_files)} RGB file(s) in {rgb_dir}. Each physical pose needs "
            f"exactly one file on each side, captured while the board was in the "
            f"SAME position -- fix the capture set before calibrating, don't let "
            f"this script guess a pairing."
        )
    return list(zip(event_files, rgb_files))


# ══════════════════════════════════════════════════════════════════
#  N9 · UNDISTORT POINTS  (see docstring point 4)
# ══════════════════════════════════════════════════════════════════

def undistort_points(pts: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """cv2.undistortPoints returns NORMALIZED coordinates; re-project through
    K so the result is back in pixel space, comparable across cameras."""
    und = cv2.undistortPoints(pts, K, D)
    und_h = cv2.convertPointsToHomogeneous(und).reshape(-1, 3)
    pix = (K @ und_h.T).T
    pix = pix[:, :2] / pix[:, 2:3]
    return pix.reshape(-1, 1, 2).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
#  N10 · MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event-dir", required=True,
                    help="Folder of poseNN.h5 event recordings, one per physical pose")
    ap.add_argument("--rgb-dir", default=None,
                    help="Folder of poseNN.tiff RGB photos, SAME poses as --event-dir, "
                         "paired by sorted filename order. Omit together with "
                         "--event-only for an event-camera-only intrinsic run "
                         "(no RGB camera needed at all -- e.g. a day before the "
                         "RGB rig is available).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--debug-dir", default=None,
                    help="Save accumulated event frames with drawn corners here "
                         "for visual sanity-check -- open a few before trusting "
                         "the numeric result, same convention as sam3_export_tracks.py")
    ap.add_argument("--event-only", action="store_true",
                    help="Compute ONLY K_event/D_event from event-camera poses alone "
                         "(no RGB, no pairing, no homography/stereo). Zhang's method "
                         "needs many DIVERSE poses of ONE camera, not a cross-camera "
                         "session -- this can be run any day the event camera is "
                         "available by itself. Run again later WITHOUT this flag, "
                         "with --rgb-dir pointing at a same-session paired capture, "
                         "to add the cross-registration (H, R, t); the K_event/D_event "
                         "computed here will be reused rather than recomputed from "
                         "what may be a much smaller cross-registration-only pose set.")
    args = ap.parse_args()

    if not args.event_only and args.rgb_dir is None:
        raise RuntimeError(
            "--rgb-dir is required unless --event-only is set. Cross-sensor "
            "registration needs a SAME-SESSION paired capture (both cameras "
            "recording the identical physical pose at once) -- it cannot be "
            "assembled from an event-only session on one day and an RGB-only "
            "session on another, since there is no way to guarantee the board "
            "was in the exact same position both times. Use --event-only today "
            "if only the event camera is available; add --rgb-dir once both "
            "cameras can record the SAME poses together."
        )

    cfg = load_config(args.config)
    cfg_e = ec_cfg(cfg)
    W, H = int(cfg["camera"]["width"]), int(cfg["camera"]["height"])
    cols, rows = int(cfg["calibration"]["board_cols"]), int(cfg["calibration"]["board_rows"])
    square_m = float(cfg["calibration"]["square_size_m"])
    board_size = (cols, rows)
    calib_dir = Path(cfg["paths"]["calibration_dir"])

    debug_dir = None
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'━'*60}")
    print(f"  Event-Camera Calibration"
          f"{'  (INTRINSIC-ONLY, event camera alone)' if args.event_only else ''}")
    print(f"{'━'*60}")

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    # ══════════════════════════════════════════════════════════
    #  EVENT-ONLY MODE — no RGB, no pairing, intrinsics only
    # ══════════════════════════════════════════════════════════
    if args.event_only:
        event_files = sorted(Path(args.event_dir).glob("*.h5"))
        print(f"  {len(event_files)} event pose file(s) found\n")

        obj_pts_all, ev_pts_all, pose_names = [], [], []
        for ev_path in event_files:
            pose_name = ev_path.stem
            ev_corners, stats = event_pose_corners(
                ev_path, cfg_e, W, H, board_size, debug_dir, pose_name)
            if ev_corners is None:
                print(f"  [{pose_name}] NO corners found "
                      f"({stats['n_transitions_found']} transition(s) detected) — skipped")
                continue
            obj_pts_all.append(objp)
            ev_pts_all.append(ev_corners)
            pose_names.append(pose_name)
            print(f"  [{pose_name}] OK — {stats['n_blinks_averaged']} blink(s) averaged, "
                  f"{stats['n_frames_with_corners']}/{stats['n_frames_attempted']} attempted")

        n_valid = len(pose_names)
        if n_valid < 5:
            raise RuntimeError(
                f"Only {n_valid} valid pose(s) — need >= 5 for Zhang's method. "
                f"Capture more diverse poses (different angles/distances/tilts), "
                f"or check --debug-dir images for why detection is failing."
            )
        print(f"\n  {n_valid} valid pose(s) — computing K_event, D_event...")
        rms_ev, K_ev, D_ev, _, _ = cv2.calibrateCamera(
            obj_pts_all, ev_pts_all, (W, H), None, None)
        print(f"    RMS: {rms_ev:.4f} px  fx={K_ev[0,0]:.1f} fy={K_ev[1,1]:.1f} "
              f"cx={K_ev[0,2]:.1f} cy={K_ev[1,2]:.1f}")

        calib_dir.mkdir(parents=True, exist_ok=True)
        out_path = calib_dir / "event_camera_params.npz"
        np.savez(out_path, K_event=K_ev, D_event=D_ev)
        print(f"\n✓ Saved {out_path}  (K_event, D_event only — no H/R/t yet)")
        print(f"  Next: once the RGB rig is available, re-run this script WITHOUT "
              f"--event-only, with --rgb-dir pointing at a SAME-SESSION paired "
              f"capture, to add cross-sensor registration. These intrinsics will "
              f"be reused automatically rather than recomputed.")
        return

    # ══════════════════════════════════════════════════════════
    #  FULL MODE — paired RGB+event session, cross-registration
    # ══════════════════════════════════════════════════════════
    pairs = pair_pose_files(Path(args.event_dir), Path(args.rgb_dir))
    print(f"  {len(pairs)} pose(s) paired by filename\n")

    obj_pts_all, ev_pts_all, rgb_pts_all, pose_names = [], [], [], []
    all_stats = []

    for ev_path, rgb_path in pairs:
        pose_name = ev_path.stem
        ev_corners, stats = event_pose_corners(
            ev_path, cfg_e, W, H, board_size, debug_dir, pose_name)
        all_stats.append(stats)

        if ev_corners is None:
            print(f"  [{pose_name}] event: NO corners found "
                  f"({stats['n_transitions_found']} transition(s) detected) — skipped")
            continue

        rgb_corners = rgb_pose_corners(rgb_path, board_size)
        if rgb_corners is None:
            print(f"  [{pose_name}] event OK but RGB corners not found — skipped")
            continue

        obj_pts_all.append(objp)
        ev_pts_all.append(ev_corners)
        rgb_pts_all.append(rgb_corners)
        pose_names.append(pose_name)
        print(f"  [{pose_name}] OK — {stats['n_blinks_averaged']} blink(s) averaged, "
              f"{stats['n_frames_with_corners']}/{stats['n_frames_attempted']} attempted")

    n_valid = len(pose_names)
    if n_valid < 5:
        raise RuntimeError(
            f"Only {n_valid} valid pose(s) with corners on BOTH sides — need >= 5 "
            f"for Zhang's method (same threshold calibrate.py's RGB-only calibration "
            f"uses). Capture more poses, or check --debug-dir images for why "
            f"detection is failing on the ones you have."
        )
    print(f"\n  {n_valid}/{len(pairs)} pose(s) valid on both sides\n")

    # ── Event-camera intrinsics: reuse a prior --event-only run if present ──
    existing_path = calib_dir / "event_camera_params.npz"
    K_ev = D_ev = None
    if existing_path.exists():
        d = np.load(existing_path)
        if "K_event" in d and "D_event" in d:
            K_ev, D_ev = d["K_event"], d["D_event"]
            print(f"  Reusing cached K_event/D_event from {existing_path} "
                  f"(computed by an earlier --event-only run) rather than "
                  f"recomputing from this cross-registration session's "
                  f"{n_valid} pose(s), which may be too few/too similar to "
                  f"trust for intrinsics alone.")

    if K_ev is None:
        print("  No cached event intrinsics found — computing from THIS session "
              "(fine if this session already has >=5 diverse poses; if it was "
              "captured specifically as a small cross-registration session, "
              "consider running --event-only separately with more/more-diverse "
              "poses for a better intrinsic estimate).")
        rms_ev, K_ev, D_ev, _, _ = cv2.calibrateCamera(
            obj_pts_all, ev_pts_all, (W, H), None, None)
        print(f"    RMS: {rms_ev:.4f} px  fx={K_ev[0,0]:.1f} fy={K_ev[1,1]:.1f} "
              f"cx={K_ev[0,2]:.1f} cy={K_ev[1,2]:.1f}")

    # ── Load existing RGB intrinsics (from calibrate.py) ──────────
    rgb_params_path = calib_dir / "camera_params.npz"
    K_rgb = D_rgb = None
    if rgb_params_path.exists():
        d = np.load(rgb_params_path)
        K_rgb, D_rgb = d["K"], d["D"]
        print(f"    Loaded existing RGB intrinsics from {rgb_params_path}")
    else:
        print(f"    [warn] {rgb_params_path} not found — run calibrate.py's RGB "
              f"chessboard calibration first; falling back to calibrating K_rgb "
              f"HERE from --rgb-dir instead (less ideal: this reuses the SAME "
              f"pose images as the cross-registration step rather than a larger, "
              f"independent RGB-only pose set).")
        rms_rgb, K_rgb, D_rgb, _, _ = cv2.calibrateCamera(
            obj_pts_all, rgb_pts_all, (W, H), None, None)
        print(f"    (fallback) RGB RMS: {rms_rgb:.4f} px")

    # ── Undistort both point sets before cross-sensor fitting ─────
    ev_pts_und = [undistort_points(p, K_ev, D_ev) for p in ev_pts_all]
    rgb_pts_und = [undistort_points(p, K_rgb, D_rgb) for p in rgb_pts_all]

    # ── Homography (primary model, see paper Eq. 32 / Section III-C.4) ─
    print("\n  Computing homography H (RGB -> event, undistorted, all poses pooled)...")
    rgb_pool = np.concatenate(rgb_pts_und, axis=0).reshape(-1, 2)
    ev_pool = np.concatenate(ev_pts_und, axis=0).reshape(-1, 2)
    H_mat, inlier_mask = cv2.findHomography(rgb_pool, ev_pool, cv2.RANSAC, 5.0)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    print(f"    {n_inliers}/{len(rgb_pool)} inlier point(s)")

    # ── Stereo extrinsic fallback (see paper Section III-C.4) ──────
    print("  Computing stereo extrinsic (R, t) fallback...")
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    rms_stereo, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
        obj_pts_all, rgb_pts_all, ev_pts_all,
        K_rgb, D_rgb, K_ev, D_ev, (W, H),
        flags=stereo_flags)
    print(f"    Stereo RMS: {rms_stereo:.4f} px")
    print(f"    Baseline |t|: {np.linalg.norm(t):.4f} m  (sanity-check this against "
          f"the actual physical mounting offset between the two cameras)")

    # ── Per-pose depth + per-model reprojection error ──────────────
    print("\n  Per-pose depth and reprojection error (homography vs. stereo):")
    per_pose_report = []
    for i, name in enumerate(pose_names):
        ok, rvec, tvec = cv2.solvePnP(objp, rgb_pts_all[i], K_rgb, D_rgb)
        depth_m = float(tvec[2]) if ok else None

        rgb_u = rgb_pts_und[i].reshape(-1, 2)
        ev_u = ev_pts_und[i].reshape(-1, 2)

        rgb_h = np.concatenate([rgb_u, np.ones((len(rgb_u), 1))], axis=1)
        proj = (H_mat @ rgb_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        homog_err = float(np.linalg.norm(proj - ev_u, axis=1).mean())

        proj_ev, _ = cv2.projectPoints(objp, rvec, tvec, K_ev, D_ev)
        # rvec/tvec above are RGB-frame pose; compose with stereo (R,t) to
        # get the pose in the event camera's own frame for a fair reprojection.
        rvec_ev, _ = cv2.Rodrigues(R @ cv2.Rodrigues(rvec)[0])
        tvec_ev = R @ tvec + t
        proj_ev, _ = cv2.projectPoints(objp, rvec_ev, tvec_ev, K_ev, D_ev)
        stereo_err = float(np.linalg.norm(
            proj_ev.reshape(-1, 2) - ev_pts_all[i].reshape(-1, 2), axis=1).mean())

        per_pose_report.append(dict(
            pose=name, depth_m=depth_m,
            homography_reproj_err_px=homog_err,
            stereo_reproj_err_px=stereo_err,
        ))
        print(f"    [{name}] depth={depth_m:.2f}m  "
              f"H_err={homog_err:.2f}px  stereo_err={stereo_err:.2f}px")

    # ── Save (preserve K_event/D_event whether freshly computed or reused) ─
    calib_dir.mkdir(parents=True, exist_ok=True)
    np.savez(calib_dir / "event_camera_params.npz",
             K_event=K_ev, D_event=D_ev, H_rgb_to_event=H_mat, R=R, t=t)

    report_path = calib_dir / "event_calib_report.json"
    report_path.write_text(json.dumps(dict(
        n_poses_used=n_valid,
        rms_event_intrinsic_px=float(rms_ev),
        rms_stereo_px=float(rms_stereo),
        homography_inliers=n_inliers,
        homography_total_points=int(len(rgb_pool)),
        per_pose=per_pose_report,
        per_pose_detection_stats=all_stats,
    ), indent=2))

    print(f"\n✓ Saved:")
    print(f"  {calib_dir}/event_camera_params.npz  (K_event, D_event, H_rgb_to_event, R, t)")
    print(f"  {report_path}")
    if debug_dir is not None:
        print(f"  {debug_dir}/  ({sum(1 for _ in debug_dir.glob('*.png'))} debug image(s) "
              f"-- open a few, confirm corners actually land on the board)")
    print(f"\n  Check the per-pose homography vs. stereo error above: if they're close "
          f"across all depths tested, the paper's small-baseline homography "
          f"approximation is justified; if homography error grows sharply with depth, "
          f"use the saved (R,t) stereo extrinsic instead.")


if __name__ == "__main__":
    main()
