#!/usr/bin/env python3
"""
CAROECT-D — Preprocessing v2 (dual-branch, geometry-shared, streaming)
======================================================================

WHAT CHANGED vs v1 (and why)
----------------------------
1. DUAL BRANCH. One run now produces BOTH outputs the project needs:
     branch A (--output-rgb): sRGB-encoded 8-bit TIFFs -> SAM3+DINO labeling
     branch B (--output):     linear luminance 16-bit TIFFs -> v2e / DVS-Voltmeter
2. GEOMETRY SHARED BY CONSTRUCTION. Undistort + resize (+ optional stabilize)
   run EXACTLY ONCE on a single shared RGB array; both branches derive from
   that same array. Identical geometry is guaranteed structurally — not by
   "calling the same function twice and hoping" — so SAM3 boxes/masks copy
   1:1 onto the event frames. (RGB->Y is a per-pixel LINEAR combination and
   INTER_AREA resize is a LINEAR operator, so Y-after-resize == resize-after-Y
   mathematically; doing geometry on RGB once is equivalent AND safer.)
3. STREAMING. v1 held every frame in RAM before writing (60 s @ 119.88 fps
   = 7,193 frames x ~11 MB — tens of GB, guaranteed OOM). v2 writes each
   frame as it is produced. Stabilization, which needs whole-clip knowledge,
   became a cheap two-pass design (pass 1 estimates tiny per-frame motion
   params; pass 2 applies them) instead of holding pixel data.
4. FORMULA-ORDER FIX. v1 ran the optional sRGB->linear input decode AFTER
   dark/flat/WB. Those corrections are linear-light math; running them on a
   nonlinear encoding is invalid. Decode now happens FIRST (N1.5). (No-op on
   the recommended linear export path, so no old data is invalidated.)

PIPELINE (all float32 until the final write of each branch)
-----------------------------------------------------------
   [DaVinci: 16-bit LINEAR Rec.2020 RGB TIFF - ONE export shared by all]
     N1   LOAD          uint16 -> float32
     N1.5 TRANSFER      srgb->linear decode IF input_transfer==srgb (else no-op)
     N2   DARK          - dark_mean          (no clip: negatives ride through)
     N3   FLAT-FIELD    x gain_map           (measured de-vignette)
     N4   WHITE BALANCE x wb_gains           (gray-card measured)
     N4.5 EXPOSURE      x (target/level)     (cross-session normalization)
     N5   UNDISTORT     cv2.remap, maps precomputed ONCE (== cv2.undistort math)
     N6   RESIZE        INTER_AREA -> event sensor resolution (1280x720)
     N6.5 STABILIZE     optional, two-pass, applied to the SHARED RGB
     ── branch split (photometric only from here; geometry is frozen) ──
     N7a  sRGB ENCODE   IEC 61966-2-1 curve -> 8-bit TIFF  [SAM3 branch]
     N7b  RGB -> Y      Rec.2020 luma -> (optional bilateral) -> 16-bit TIFF

TONE MAY DIFFER BETWEEN BRANCHES; GEOMETRY MUST NOT. sRGB encode and
bilateral denoise change VALUES, never positions — labels stay valid.

Usage:
  python preprocess.py --input davinci_tiffs/ --output out_y/ --output-rgb out_rgb/ --verify
  python preprocess.py --input davinci_tiffs/ --output out_y/          # event branch only
"""

import cv2
import numpy as np
import tifffile
import yaml
import argparse
import time
from pathlib import Path

from linear16_to_srgb8 import encode_srgb_u8


# ══════════════════════════════════════════════════════════════════
#  CONFIG + CALIBRATION
# ══════════════════════════════════════════════════════════════════

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_calibration(calib_dir: Path, cfg: dict) -> dict:
    """Load calibrate.py outputs, gated by config flags. A stage runs later
    only if its key is present in the returned dict."""
    pp = cfg["preprocessing"]
    calib, loaded = {}, []

    def maybe(name, flag, key):
        p = calib_dir / name
        if flag and p.exists():
            calib[key] = np.load(str(p))
            loaded.append(key)

    maybe("dark_mean.npy", pp["dark_correction"], "dark")   # -> N2
    maybe("gain_map.npy",  pp["flat_field"],      "gain")   # -> N3
    maybe("wb_gains.npy",  pp["white_balance"],   "wb")     # -> N4

    # N4.5 exposure normalization: needs BOTH this session's reading and the
    # project-wide canonical target (both from calibrate.py's gray card).
    lv, tg = calib_dir / "exposure_level.npy", calib_dir / "exposure_target.npy"
    if pp.get("exposure_normalize", False) and lv.exists() and tg.exists():
        level, target = float(np.load(str(lv))), float(np.load(str(tg)))
        calib["exposure_scalar"] = target / level if level > 0 else 1.0
        loaded.append(f"exposure(x{calib['exposure_scalar']:.4f})")

    cam = calib_dir / "camera_params.npz"                   # -> N5
    if pp["undistort"] and cam.exists():
        d = np.load(str(cam))
        calib["K"], calib["D"] = d["K"], d["D"]
        calib["calib_size"] = tuple(int(v) for v in d["image_size"]) if "image_size" in d else None
        loaded.append("K/D")

    print(f"  Calibration loaded: {', '.join(loaded) if loaded else 'none (raw passthrough)'}")
    return calib


# ══════════════════════════════════════════════════════════════════
#  N1 LOAD  +  N1.5 TRANSFER DECODE
# ══════════════════════════════════════════════════════════════════

def load_tiff(path) -> np.ndarray:
    """uint16 HxWx3 RGB TIFF -> float32 (values stay in 0..65535 scale)."""
    img = tifffile.imread(str(path))
    if img.dtype != np.uint16:
        raise ValueError(f"Expected uint16, got {img.dtype}: {Path(path).name}\n"
                         "  -> DaVinci Deliver: Format=TIFF, Codec=RGB 16-bit.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {img.shape}: {Path(path).name}")
    return img.astype(np.float32)


def decode_transfer(rgb: np.ndarray, transfer: str) -> np.ndarray:
    """
    N1.5 — bring the file's encoding to scene-linear BEFORE any physical
    correction (dark/flat/WB are linear-light math; order is not optional).
    'linear' (the recommended DaVinci export) is a no-op.
    NOTE: if you ever use 'srgb' here, the calibration TIFFs (dark/flat/
    gray-card) must have gone through the SAME encoding — capture them via
    the same export path as the footage or the corrections won't match.
    """
    if transfer == "linear":
        return rgb
    if transfer == "srgb":                       # IEC 61966-2-1 EOTF (decode)
        n = np.clip(rgb / 65535.0, 0.0, 1.0)
        lin = np.where(n <= 0.04045, n / 12.92, ((n + 0.055) / 1.055) ** 2.4)
        return (lin * 65535.0).astype(np.float32)
    raise ValueError(f"input_transfer must be 'linear' or 'srgb', got '{transfer}'")


# ══════════════════════════════════════════════════════════════════
#  N2-N4.5 PHYSICAL CORRECTIONS (all pure linear ops, all measured)
# ══════════════════════════════════════════════════════════════════

def apply_corrections(rgb: np.ndarray, calib: dict) -> np.ndarray:
    if "dark" in calib:
        rgb = rgb - calib["dark"]                # N2 (NO clip — keep linearity)
    if "gain" in calib:
        rgb = rgb * calib["gain"]                # N3 measured flat-field
    if "wb" in calib:
        rgb = rgb * calib["wb"][np.newaxis, np.newaxis, :]   # N4 gray-card WB
    if "exposure_scalar" in calib:
        rgb = rgb * calib["exposure_scalar"]     # N4.5 cross-session exposure
    return rgb


# ══════════════════════════════════════════════════════════════════
#  N5 UNDISTORT — maps precomputed ONCE, shared by every frame & branch
# ══════════════════════════════════════════════════════════════════

def _scale_K(K, calib_size, cur_size):
    """K is in pixels -> rescale if footage resolution != chessboard resolution.
    (D is dimensionless: no rescale needed.)"""
    if calib_size is None or tuple(calib_size) == tuple(cur_size):
        return K.astype(np.float64)
    (cw, ch), (w, h) = calib_size, cur_size
    Ks = K.astype(np.float64).copy()
    Ks[0, 0] *= w / cw; Ks[0, 2] *= w / cw
    Ks[1, 1] *= h / ch; Ks[1, 2] *= h / ch
    return Ks


def build_undistort_maps(calib: dict, frame_size, alpha: float):
    """
    cv2.initUndistortRectifyMap + cv2.remap(INTER_LINEAR) is EXACTLY what
    cv2.undistort does internally — same math, hoisted out of the loop so the
    mapping is computed once and is bit-identical for every frame and both
    branches. alpha=0 crops the black border the correction would leave
    (a hard fake edge = systematic fake events).
    """
    w, h = frame_size
    Ks = _scale_K(calib["K"], calib.get("calib_size"), (w, h))
    new_K, _ = cv2.getOptimalNewCameraMatrix(Ks, calib["D"], (w, h), alpha)
    m1, m2 = cv2.initUndistortRectifyMap(Ks, calib["D"], None, new_K, (w, h), cv2.CV_32FC1)
    return m1, m2


# ══════════════════════════════════════════════════════════════════
#  N6 RESIZE  (INTER_AREA: correct anti-aliasing downsample; linear operator)
# ══════════════════════════════════════════════════════════════════

def resize_rgb(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


# ══════════════════════════════════════════════════════════════════
#  N6.5 STABILIZE — optional, two-pass, OFF by default
# ══════════════════════════════════════════════════════════════════
#  POLICY (decided in project discussion — do not change casually):
#   * Prefer DaVinci's built-in stabilizer (Mode=Translation) BEFORE export;
#     one single export then feeds both branches -> geometry shared for free.
#   * If the trained model will be TESTED against a REAL event camera
#     (Triton2), do NOT stabilize at all: real event cameras cannot be
#     frame-warp stabilized, so training data should keep the same
#     vibration-induced event statistics the test data will have.
#   * This OpenCV fallback exists only for when DaVinci's tool is unusable.
#  Two-pass: pass 1 estimates per-frame (dx, dy, da) at OUTPUT geometry
#  (same undistort maps + resize -> what it measures is what gets warped);
#  pass 2 applies. Only a few floats per frame are kept — no pixel stacks.

def estimate_stab_transforms(tiffs, transfer, maps, out_size):
    w, h = out_size
    prev8, steps = None, []
    for p in tiffs:
        g = load_tiff(p)[:, :, 1]                       # green ~ cheap luma proxy
        if transfer == "srgb":                          # keep pass-1 geometry-only cheap:
            pass                                        # tracking is tone-insensitive enough
        if maps is not None:
            g = cv2.remap(g, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        g8 = np.clip(g / 256.0, 0, 255).astype(np.uint8)
        if prev8 is not None:
            pts = cv2.goodFeaturesToTrack(prev8, maxCorners=300, qualityLevel=0.01, minDistance=30)
            if pts is None or len(pts) < 4:
                steps.append((0.0, 0.0, 0.0))
            else:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev8, g8, pts, None)
                idx = np.where(st.ravel() == 1)[0]
                M, _ = cv2.estimateAffinePartial2D(pts[idx], nxt[idx], method=cv2.RANSAC)
                steps.append((float(M[0, 2]), float(M[1, 2]),
                              float(np.arctan2(M[1, 0], M[0, 0]))) if M is not None else (0.0, 0.0, 0.0))
        prev8 = g8

    def movavg(x, r=15):
        k = np.ones(2 * r + 1) / (2 * r + 1)
        return np.convolve(np.pad(x, (r, r), "edge"), k, "same")[r:-r]

    C = np.vstack([np.zeros(3), np.cumsum(np.array(steps), axis=0)])   # abs trajectory
    S = np.column_stack([movavg(C[:, i]) for i in range(3)])            # smoothed
    return S - C                                                        # per-frame warp


def apply_stab(rgb_small: np.ndarray, warp) -> np.ndarray:
    dx, dy, da = warp
    ca, sa = np.cos(da), np.sin(da)
    M = np.array([[ca, -sa, dx], [sa, ca, dy]], np.float32)
    return cv2.warpAffine(rgb_small, M, (rgb_small.shape[1], rgb_small.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ══════════════════════════════════════════════════════════════════
#  N7a sRGB ENCODE (SAM3 branch)  — IEC 61966-2-1, applied LAST
# ══════════════════════════════════════════════════════════════════
#  Why last: interpolation inside undistort/resize is a weighted AVERAGE of
#  neighboring pixels, and averaging is only physically correct on LINEAR
#  light (same reason DaVinci's "Apply resize transformations in: Linear").
#  So: geometry first (on linear), perceptual encode as the final step.
#  Why sRGB at all: SAM3/DINO were trained on millions of sRGB photos;
#  scene-linear frames look flat/dark to them. The curve is monotonic and
#  per-pixel -> changes values only, never positions -> labels unaffected.
#
#  encode_srgb_u8() now lives in linear16_to_srgb8.py (single source of
#  truth — used to be copy-pasted here AND in quick_tiff_to_sam3.py, which
#  is exactly the kind of silent-drift tech debt the project tries to avoid;
#  see that file's docstring for the full explanation, including why a
#  16-bit TIFF must never reach SAM3 unconverted).


# ══════════════════════════════════════════════════════════════════
#  N7b RGB -> Y (event branch)  — Rec.2020 luminance, from config
# ══════════════════════════════════════════════════════════════════
#  Coefficients MUST match the export gamut (project-verified: DaVinci
#  Input=Output=Rec.2020/Linear -> [0.2627, 0.6780, 0.0593] from BT.2020-2;
#  they are the Y row of the RGB->XYZ matrix, i.e. true luminance weights).

def rgb_to_luma(rgb: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return rgb[:, :, 0] * coeffs[0] + rgb[:, :, 1] * coeffs[1] + rgb[:, :, 2] * coeffs[2]


def bilateral_denoise(Y: np.ndarray, d: int, sigma_frac: float) -> np.ndarray:
    """Edge-preserving photometric smoothing (event branch only; OFF by
    default — prefer shooting low ISO). Changes values, not positions."""
    n = np.clip(Y / 65535.0, 0.0, 1.0).astype(np.float32)
    return cv2.bilateralFilter(n, d, sigma_frac, sigma_frac) * 65535.0


# ══════════════════════════════════════════════════════════════════
#  VERIFY IMAGE
# ══════════════════════════════════════════════════════════════════

def save_verify(raw_rgb, Y, srgb_u8, path):
    def to8(x):
        return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = Y.shape
    raw = cv2.resize(to8(raw_rgb[:, :, 1]), (w, h), interpolation=cv2.INTER_AREA)
    tiles = [raw, to8(Y)]
    labels = ["DaVinci export (G)", "Y linear (event)"]
    if srgb_u8 is not None:
        tiles.append(cv2.cvtColor(srgb_u8, cv2.COLOR_RGB2GRAY))
        labels.append("sRGB (SAM3)")
    canvas = np.zeros((h + 30, w * len(tiles) + 10 * (len(tiles) - 1)), np.uint8)
    for i, (t, lb) in enumerate(zip(tiles, labels)):
        x0 = i * (w + 10)
        canvas[30:, x0:x0 + w] = t
        cv2.putText(canvas, lb, (x0 + 5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 200, 1)
    cv2.imwrite(path, canvas)


# ══════════════════════════════════════════════════════════════════
#  MAIN — streaming loop
# ══════════════════════════════════════════════════════════════════

def process_sequence(args, cfg):
    in_dir = Path(args.input)
    out_y = Path(args.output); out_y.mkdir(parents=True, exist_ok=True)
    out_rgb = Path(args.output_rgb) if args.output_rgb else None
    if out_rgb:
        out_rgb.mkdir(parents=True, exist_ok=True)

    cam, pp = cfg["camera"], cfg["preprocessing"]
    W, H = cam["width"], cam["height"]
    coeffs = np.asarray(cam["luma_coeffs"], dtype=np.float32)
    transfer = cam.get("input_transfer", "linear")
    calib = load_calibration(Path(cfg["paths"]["calibration_dir"]), cfg)

    tiffs = sorted(set(list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff"))))
    if not tiffs:
        raise FileNotFoundError(f"No .tif/.tiff in {in_dir}")

    print(f"\n{'━'*64}")
    print(f"  CAROECT-D Preprocessing v2   ({len(tiffs)} frames)")
    print(f"  Y  (event) -> {out_y}")
    print(f"  RGB (SAM3) -> {out_rgb if out_rgb else '(disabled — no --output-rgb)'}")
    print(f"  transfer={transfer}  stab={'Y' if pp['stabilize'] else '-'}  "
          f"denoise={pp['denoise']}(Y-branch only)")
    print(f"  reminder: #TIFFs ≈ real_seconds x {cam['fps_original']} (capture fps, "
          f"real_seconds = đồng hồ thật của cảnh quay — KHÔNG phải theo timeline đã "
          f"relabel/kéo dài), NOT x {cam['fps_export']}")
    print(f"{'━'*64}\n")

    # Undistort maps: need the real frame size -> peek at frame 0 once.
    maps = None
    if "K" in calib:
        h0, w0 = load_tiff(tiffs[0]).shape[:2]
        maps = build_undistort_maps(calib, (w0, h0), pp.get("undistort_alpha", 0.0))

    warps = None
    if pp["stabilize"]:
        print("[stab] pass 1/2: estimating per-frame motion (streaming)...")
        warps = estimate_stab_transforms(tiffs, transfer, maps, (W, H))

    t0, first_raw, first_Y, first_srgb = time.time(), None, None, None
    for i, p in enumerate(tiffs):
        rgb = load_tiff(p)                                   # N1
        if i == 0:
            first_raw = rgb.copy()
        rgb = decode_transfer(rgb, transfer)                 # N1.5 (BEFORE corrections)
        rgb = apply_corrections(rgb, calib)                  # N2..N4.5 (linear ops)

        if maps is not None:                                 # N5 shared undistort
            rgb = cv2.remap(rgb, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
        rgb = resize_rgb(rgb, W, H)                          # N6 shared resize
        if warps is not None:                                # N6.5 shared stabilize
            rgb = apply_stab(rgb, warps[i])

        # ── branch split: geometry is FROZEN above this line ──
        if out_rgb is not None:                              # N7a SAM3 branch
            srgb = encode_srgb_u8(rgb)
            # 8-bit TIFF, not PNG/JPG: SAM3's own loader (io_utils.py IMAGE_EXTS) accepts
            # .tiff natively via PIL, so this is the SAME container family as the rest of
            # the pipeline (calibrate.py/preprocess.py/run_v2e.py/run_dvsvolt.py all speak
            # TIFF) — one less format to reason about, and no JPEG quantization on top of
            # the sRGB quantization we already do here. RGB order kept as-is (tifffile
            # writes RGB directly; no BGR swap needed, unlike cv2.imwrite).
            tifffile.imwrite(str(out_rgb / (p.stem + ".tiff")), srgb, photometric="rgb")
            if i == 0:
                first_srgb = srgb

        Y = rgb_to_luma(rgb, coeffs)                         # N7b event branch
        if pp["denoise"] == "bilateral":
            Y = bilateral_denoise(Y, pp["denoise_bilateral_d"], pp["denoise_bilateral_sigma"])
        elif pp["denoise"] not in ("none", None):
            raise ValueError(f"denoise must be 'none'|'bilateral', got {pp['denoise']}")
        tifffile.imwrite(str(out_y / (p.stem + ".tiff")),
                         np.clip(np.rint(Y), 0, 65535).astype(np.uint16))   # the ONLY uint16 clip
        # np.rint() before the cast: .astype(uint16) alone TRUNCATES (floors toward
        # zero), inconsistent with the round-explicit convention used everywhere
        # else float->int happens in this project (np.rint() in normalize_events(),
        # np.round() for the 8-bit PNG export in run_dvsvolt.py). This is the ONLY
        # quantization point in the whole event branch, so worth being consistent
        # even though the error is tiny (~0.5 DN out of 65535).
        if i == 0:
            first_Y = Y

        if (i + 1) % 100 == 0 or (i + 1) == len(tiffs):
            r = (i + 1) / (time.time() - t0)
            print(f"  [{i+1:>5}/{len(tiffs)}]  {r:.1f} fr/s  ETA {(len(tiffs)-i-1)/r:.0f}s")

    if args.verify and first_raw is not None:
        vp = str(out_y / "_verify_frame0.png")
        save_verify(first_raw, first_Y, first_srgb, vp)
        print(f"[verify] -> {vp}")

    print(f"\n✓ Done. {len(tiffs)} frames in {time.time()-t0:.1f}s")
    print(f"  Event branch: python run_v2e.py --input {out_y} --output data/events_v2e/<session>")
    if out_rgb:
        print(f"  SAM3  branch: feed {out_rgb} to the labeling pipeline\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CAROECT-D preprocessing v2 (dual-branch)")
    ap.add_argument("--input", required=True, help="DaVinci export dir (16-bit RGB TIFFs)")
    ap.add_argument("--output", required=True, help="OUT: linear luminance TIFFs (event branch)")
    ap.add_argument("--output-rgb", default=None, help="OUT: sRGB 8-bit TIFFs (SAM3 branch); omit to skip")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    process_sequence(args, load_config(args.config))
