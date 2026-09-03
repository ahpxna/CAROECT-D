#!/usr/bin/env python3
"""
CAROECT-D — Calibration Script
================================
Run ONCE per camera setup to compute and save the calibration data that
preprocess.py consumes. Each function below produces one file:

  compute_dark_mean   -> dark_mean.npy       (used by preprocess N2)
  compute_gain_map    -> gain_map.npy        (used by preprocess N3)
  compute_wb_gains    -> wb_gains.npy        (used by preprocess N4)
  calibrate_camera    -> camera_params.npz   (used by preprocess N6: K, D, image_size)

Expected directory structure under calibration_dir/:
  dark/           10-20 TIFFs, lens cap on (same ISO/shutter as footage)
  flat/           10-20 TIFFs of a uniform scene (white wall / evenly lit monitor)
  gray_card.tiff  single TIFF of a gray card under the target light
  chessboard/     20+ images of the calibration board (JPG or PNG)

Usage:
  python calibrate.py
  python calibrate.py --config config.yaml
"""

import cv2
import numpy as np
import tifffile
import yaml
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def decode_transfer(rgb: np.ndarray, transfer: str) -> np.ndarray:
    """Convert values to linear scale for isolated compatibility tests.

    The aligned workflow supplies DaVinci-linearized SDR TIFFs and therefore
    uses only the ``linear`` branch.
    """
    if transfer == "linear":
        return rgb
    if transfer == "srgb":
        n = np.clip(rgb / 65535.0, 0.0, 1.0)
        lin = np.where(n <= 0.04045, n / 12.92, ((n + 0.055) / 1.055) ** 2.4)
        return (lin * 65535.0).astype(np.float32)
    raise ValueError(f"camera.input_transfer must be 'linear' or 'srgb', got {transfer!r}")


def read_tiff_linear(path: str | Path, transfer: str) -> np.ndarray:
    img = tifffile.imread(str(path)).astype(np.float32)
    if img.ndim == 3:
        return decode_transfer(img, transfer)
    if transfer != "linear":
        return decode_transfer(img[..., np.newaxis], transfer)[..., 0]
    return img


# ── Dark frame  -> dark_mean.npy ──────────────────────────────────
def compute_dark_mean(dark_dir: str, transfer: str = "linear") -> np.ndarray:
    """Average lens-cap frames to estimate the sensor's fixed noise/bias floor."""
    paths = sorted(Path(dark_dir).glob("*.tiff"))
    if not paths:
        raise FileNotFoundError(f"No .tiff files in {dark_dir}")
    stack = np.stack([read_tiff_linear(p, transfer) for p in paths])
    mean = np.mean(stack, axis=0)
    print(f"  [Dark]  {len(paths)} frames  |  baseline mean: {mean.mean():.1f}")
    return mean


# ── Flat field  -> gain_map.npy ───────────────────────────────────
def compute_gain_map(flat_dir: str, dark_mean: np.ndarray = None,
                     transfer: str = "linear") -> np.ndarray:
    """Per-pixel gain from a uniform scene = mean(flat) / flat. Removes vignetting."""
    paths = sorted(Path(flat_dir).glob("*.tiff"))
    if not paths:
        raise FileNotFoundError(f"No .tiff files in {flat_dir}")
    stack = np.stack([read_tiff_linear(p, transfer) for p in paths])
    flat = np.mean(stack, axis=0)
    flat = flat - dark_mean if dark_mean is not None else flat
    flat = np.clip(flat, 1.0, None)                 # avoid divide-by-zero only
    gain = np.mean(flat) / flat
    print(f"  [Flat]  {len(paths)} frames  |  gain range: {gain.min():.3f} - {gain.max():.3f}")
    return gain


# ── White balance  -> wb_gains.npy ────────────────────────────────
def compute_wb_gains(gray_card_path: str, roi=None, transfer: str = "linear") -> np.ndarray:
    """Per-channel gains from a gray card, normalized to the G channel."""
    img = read_tiff_linear(gray_card_path, transfer)
    patch = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] if roi else img
    r, g, b = patch[:, :, 0].mean(), patch[:, :, 1].mean(), patch[:, :, 2].mean()
    gains = np.array([g / r if r > 0 else 1.0, 1.0, g / b if b > 0 else 1.0], np.float32)
    print(f"  [WB]    R={gains[0]:.4f}  G={gains[1]:.4f}  B={gains[2]:.4f}")
    return gains


# ── Camera calibration  -> camera_params.npz ──────────────────────
def calibrate_camera(chess_dir: str, board_cols: int, board_rows: int, square_m: float):
    """
    Intrinsics K + distortion D from chessboard images.
    ALSO returns image_size (w, h) — preprocess N6 needs it to rescale K when the
    exported frames are a different resolution than these calibration images.
    """
    board_size = (board_cols, board_rows)
    objp = np.zeros((board_cols * board_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_m

    obj_pts, img_pts, image_size = [], [], None
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    paths = sorted(p for ext in exts for p in Path(chess_dir).glob(ext))

    found = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]                # (w, h)
        ret, corners = cv2.findChessboardCornersSB(
            gray,
            board_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        )

        if not ret:
            continue

        obj_pts.append(objp)
        img_pts.append(corners)
        found += 1

    if found < 5:
        raise RuntimeError(f"Only {found} valid boards — need >= 5.")

    rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, image_size, None, None)
    print(f"  [Calib] RMS: {rms:.4f} px  ({found}/{len(paths)} boards valid)  size={image_size}")
    print(f"          fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    return K, D, image_size


# ── Main ──────────────────────────────────────────────────────────
def run(cfg):
    calib_dir = Path(cfg["paths"]["calibration_dir"])
    calib_dir.mkdir(parents=True, exist_ok=True)
    c = cfg["calibration"]
    transfer = cfg.get("camera", {}).get("input_transfer", "linear")
    if transfer != "linear":
        raise ValueError(
            "Calibration references must follow the same DaVinci-linearized SDR "
            "export path as footage; camera.input_transfer must be 'linear'.")

    print(f"\n{'━'*45}\n  CAROECT-D Calibration\n{'━'*45}")
    print(f"  [Transfer] calibration TIFFs interpreted as {transfer!r}, same as footage")
    corrections = {}
    references = {}

    dark_mean = None
    dark_path = calib_dir / "dark"
    if dark_path.exists() and any(dark_path.glob("*.tiff")):
        dark_mean = compute_dark_mean(str(dark_path), transfer)
        np.save(calib_dir / "dark_mean.npy", dark_mean)
        references["dark"] = {
            "type": "dark_no_light",
            "path": str(dark_path),
            "frames": len(list(dark_path.glob("*.tiff"))),
        }
        corrections["residual_offset"] = {
            "valid": True,
            "artifact": "dark_mean.npy",
            "meaning": "residual offset diagnostic/correction from no-light frames",
        }
    else:
        print("  [Dark]  skipped (no frames)")

    flat_path = calib_dir / "flat"
    if flat_path.exists() and any(flat_path.glob("*.tiff")):
        np.save(calib_dir / "gain_map.npy", compute_gain_map(str(flat_path), dark_mean, transfer))
        references["flat_field"] = {
            "type": "homogeneous_nonzero_field",
            "path": str(flat_path),
            "frames": len(list(flat_path.glob("*.tiff"))),
        }
        corrections["relative_gain_prnu"] = {
            "valid": True,
            "artifact": "gain_map.npy",
            "meaning": "relative gain/PRNU from a homogeneous non-zero field",
        }
    else:
        print("  [Flat]  skipped (no frames)")

    gray_card = calib_dir / "gray_card.tiff"
    if gray_card.exists():
        roi = c.get("gray_card_roi")
        gains = compute_wb_gains(str(gray_card), roi, transfer)
        np.save(calib_dir / "wb_gains.npy", gains)
        references["gray_card"] = {
            "type": "measured_neutral_reference",
            "path": str(gray_card),
            "roi": roi,
        }
        corrections["white_balance"] = {
            "valid": True,
            "artifact": "wb_gains.npy",
            "meaning": "measured neutral-reference channel gains",
        }

        # ── Exposure normalization (reuses the SAME gray card shot) ──────
        # WHY: the same 18%-gray card on two different sessions reads
        # different pixel values purely because the camera's exposure
        # (ISO/aperture/shutter) differed between sessions — not because the
        # scene was actually brighter/darker. Left uncorrected, "brightness"
        # in the exported linear TIFF conflates real scene radiance with
        # whatever exposure the camera happened to use that day, so event
        # rate differences between sessions could come from exposure alone,
        # not the scene.
        # HOW: after WB, the gray patch's R/G/B are forced equal (that IS
        # what WB does) — that common value is this session's exposure
        # level. The FIRST calibration ever run for this project bootstraps
        # a fixed canonical target (exposure_target.npy, written once, never
        # overwritten again). Every later session computes its own
        # exposure_level.npy; preprocess.py multiplies every pixel by
        # (exposure_target / exposure_level) to match the reference session.
        img = read_tiff_linear(gray_card, transfer)
        patch = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] if roi else img
        wb_patch = patch * gains[np.newaxis, np.newaxis, :]
        exposure_level = float(wb_patch.mean())
        np.save(calib_dir / "exposure_level.npy", np.array(exposure_level, dtype=np.float32))

        target_path = calib_dir / "exposure_target.npy"
        measured_target = c.get("exposure_reference_level")
        if measured_target is not None:
            target = float(measured_target)
            np.save(target_path, np.array(target, dtype=np.float32))
            scalar = target / exposure_level if exposure_level > 0 else 1.0
            print(f"  [Exposure] {exposure_level:.1f}  (target={target:.1f}, "
                  f"this session's preprocess scalar will be x{scalar:.4f})")
            corrections["exposure_normalization"] = {
                "valid": True,
                "artifact": "exposure_level.npy",
                "target_artifact": "exposure_target.npy",
                "meaning": "normalization to an explicitly measured reference",
            }
        else:
            print("  [Exposure] diagnostic level saved; normalization remains invalid "
                  "until calibration.exposure_reference_level is explicitly provided.")
            corrections["exposure_normalization"] = {
                "valid": False,
                "artifact": "exposure_level.npy",
                "meaning": "diagnostic only; no measured reference target configured",
            }
    else:
        print("  [WB]    skipped (no gray_card.tiff)")
        print("  [Exposure] skipped (needs the same gray_card.tiff)")

    chess_path = calib_dir / "chessboard"
    if chess_path.exists():
        K, D, image_size = calibrate_camera(
            str(chess_path), c["board_cols"], c["board_rows"], c["square_size_m"])
        # image_size is saved so preprocess N6 can rescale K to the export resolution
        np.savez(calib_dir / "camera_params.npz", K=K, D=D, image_size=np.array(image_size))
    else:
        print("  [Calib] skipped (no chessboard dir)")

    levels_dir = calib_dir / c.get("linearity_levels_dir", "linearity")
    level_files = sorted(levels_dir.glob("*.tiff")) if levels_dir.exists() else []
    references["linearity"] = {
        "type": "multiple_nonzero_levels",
        "path": str(levels_dir),
        "frames": len(level_files),
        "valid": len(level_files) >= 3,
        "note": "Validation evidence only; no automatic correction is derived.",
    }

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_decode": c.get("acquisition", {}),
        "input_transfer": transfer,
        "working_primaries": cfg.get("camera", {}).get("working_primaries"),
        "references": references,
        "corrections": corrections,
    }
    manifest_path = calib_dir / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  [Manifest] {manifest_path}")

    print(f"\n✓ Calibration saved to: {calib_dir}/\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CAROECT-D: Compute calibration data")
    p.add_argument("--config", default="config.yaml")
    run(load_config(p.parse_args().config))
