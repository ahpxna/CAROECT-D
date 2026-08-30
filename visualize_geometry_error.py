#!/usr/bin/env python3
"""
Visualize the geometric error caused by omitted or incorrect undistortion.

The figures are intended for presentation slides. They show how far a SAM3
bounding box moves when transferred to the event grid and how that displacement
increases with radial distance r.

ILLUSTRATED PRINCIPLE
---------------------
A physical camera is not a perfect pinhole. The Brown-Conrady model is:
    x_distorted = x(1 + k1 r² + k2 r⁴ + k3 r⁶) + [2 p1 xy + p2(r²+2x²)]
    y_distorted = y(1 + k1 r² + k2 r⁴ + k3 r⁶) + [p1(r²+2y²) + 2 p2 xy]
    r = sqrt(x² + y²)   (normalized coordinates about principal point c_x,c_y)

A captured physical-camera image is distorted. When preprocess.py applies the
calibrated undistortion, SAM3 operates on a rectified image and its box has the
correct physical location. If this step is omitted, SAM3 returns coordinates on
the distorted image. Transferring those coordinates to the rectified event grid
then introduces exactly the local distortion-vector error.

Using the calibrated K and D from calibrate.py/camera_params.npz, this script:
  1. Draws a point grid with arrows from captured distorted pixels to their
     corresponding physically rectified positions. This is the displacement
     introduced by forgetting undistortion.
  2. Draws one user-selected bounding box, or a default box near the image edge,
     in both states: rectified/correct in green and distorted/incorrect in red.
  3. Plots displacement magnitude in pixels against radial distance r, exposing
     the k1 r² + k2 r⁴ + k3 r⁶ behavior described above.

DEMO MODE WITHOUT camera_params.npz
-----------------------------------
The script can construct a plausible illustrative K,D pair for a 6 mm lens on
an IMX636 sensor. These values are assumptions for illustrating shape only. They
must be replaced by parameters obtained from real chessboard data with
calibrate.py before reporting any quantitative result.

Usage:
  # Calibrated camera_params.npz plus a sample image:
  python visualize_geometry_error.py --camera-params calibration/camera_params.npz \
      --image sample_frame.png --output-dir slides/geometry --box 980 560 120 90

  # Standalone slide demo with assumed K,D and a generated background:
  python visualize_geometry_error.py --output-dir slides/geometry --demo
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════
#  N1 · K, D — calibrated values or explicit demo assumptions
# ══════════════════════════════════════════════════════════════════════════

def load_or_demo_camera_params(path, width, height):
    if path and Path(path).exists():
        d = np.load(path)
        K, D = d["K"].astype(np.float64), d["D"].astype(np.float64)
        calib_size = tuple(int(v) for v in d["image_size"]) if "image_size" in d else (width, height)
        if calib_size != (width, height):
            sx, sy = width / calib_size[0], height / calib_size[1]
            K = K.copy(); K[0, 0] *= sx; K[0, 2] *= sx; K[1, 1] *= sy; K[1, 2] *= sy
        print(f"[camera] Loaded calibrated K,D from {path}")
        return K, D, True

    print("[camera] ⚠ camera_params.npz is unavailable; using ASSUMED K,D for a "
          "6 mm/IMX636 demo. Do not report these illustrative values as measurements. "
          "Run calibrate.py on real chessboard images and pass --camera-params.")
    # Rough approximation: fx=fy ~= focal_mm / pixel_pitch_mm. With an IMX636
    # pixel pitch near 4.86 um and focal length 6 mm, f_px ~= 1235 px. Moderate
    # coefficients make the illustrative curve legible on a slide.
    f = 1235.0
    K = np.array([[f, 0, width / 2 + 5], [0, f, height / 2 - 3], [0, 0, 1]], dtype=np.float64)
    D = np.array([-0.18, 0.06, 0.0006, -0.0004, -0.01], dtype=np.float64)  # k1,k2,p1,p2,k3
    return K, D, False


# ══════════════════════════════════════════════════════════════════════════
#  N2 · FIGURE 1 — distortion displacement vector field
# ══════════════════════════════════════════════════════════════════════════

def distortion_vector_field(K, D, width, height, step=60):
    """Map each distorted grid point to its physically rectified pixel.

    Returns ``(pts_distorted, pts_undistorted)``. OpenCV returns normalized
    coordinates, so multiplying by K restores pixel coordinates.
    """
    xs = np.arange(step // 2, width, step, dtype=np.float64)
    ys = np.arange(step // 2, height, step, dtype=np.float64)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float64).reshape(-1, 1, 2)

    undist_norm = cv2.undistortPoints(grid, K, D)          # -> normalized ideal coords
    ones = np.ones((undist_norm.shape[0], 1, 1))
    homog = np.concatenate([undist_norm, ones], axis=-1)   # Nx1x3
    pts_undist = (K @ homog.reshape(-1, 3).T).T             # Nx3
    pts_undist = pts_undist[:, :2] / pts_undist[:, 2:3]

    return grid.reshape(-1, 2), pts_undist


def draw_vector_field(base_img, pts_distorted, pts_undistorted, out_path):
    img = base_img.copy()
    max_disp = 0.0
    for (dx, dy), (ux, uy) in zip(pts_distorted, pts_undistorted):
        disp = np.hypot(ux - dx, uy - dy)
        max_disp = max(max_disp, disp)
        p1 = (int(round(dx)), int(round(dy)))
        p2 = (int(round(ux)), int(round(uy)))
        cv2.circle(img, p1, 3, (0, 0, 255), -1)  # red: captured distorted pixel
        cv2.arrowedLine(img, p1, p2, (0, 255, 0), 1, tipLength=0.35)  # green: rectified target
    cv2.putText(img, "Red = captured distorted pixel  |  Green arrow = rectified position",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Maximum displacement in frame: {max_disp:.1f} px",
                (10, height_text_y(img)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    return max_disp


def height_text_y(img):
    return img.shape[0] - 15


# ══════════════════════════════════════════════════════════════════════════
#  N3 · FIGURE 2 — one bounding box, correct versus incorrect
# ══════════════════════════════════════════════════════════════════════════

def draw_box_comparison(base_img, K, D, box_xywh, out_path):
    x, y, w, h = box_xywh
    corners_distorted = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)

    pts = corners_distorted.reshape(-1, 1, 2)
    undist_norm = cv2.undistortPoints(pts, K, D)
    ones = np.ones((4, 1, 1))
    homog = np.concatenate([undist_norm, ones], axis=-1).reshape(-1, 3)
    corners_undistorted = (K @ homog.T).T
    corners_undistorted = corners_undistorted[:, :2] / corners_undistorted[:, 2:3]

    img = base_img.copy()
    poly_wrong = corners_distorted.astype(np.int32).reshape(-1, 1, 2)
    poly_right = corners_undistorted.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly_wrong], True, (0, 0, 255), 2)  # red: incorrect if undistortion is skipped
    cv2.polylines(img, [poly_right], True, (0, 255, 0), 2)  # green: rectified/correct

    center_wrong = corners_distorted.mean(axis=0)
    center_right = corners_undistorted.mean(axis=0)
    shift_px = float(np.hypot(*(center_right - center_wrong)))

    cv2.putText(img, "Red = incorrect (no undistortion)   Green = rectified/correct",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Box-center displacement: {shift_px:.1f} px",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    return shift_px


# ══════════════════════════════════════════════════════════════════════════
#  N4 · FIGURE 3 — radial displacement under the Brown-Conrady model
# ══════════════════════════════════════════════════════════════════════════

def plot_radial_error(K, D, width, height, out_path):
    cx, cy, f = K[0, 2], K[1, 2], (K[0, 0] + K[1, 1]) / 2
    r_max_px = np.hypot(max(cx, width - cx), max(cy, height - cy))
    r_px = np.linspace(0, r_max_px, 200)

    # Sample one ray from the optical center toward the image edge. This is
    # sufficient for the radial component, which depends on |r| rather than
    # direction and therefore uses the even powers r², r⁴, and r⁶.
    angle = np.deg2rad(30)
    xs = cx + r_px * np.cos(angle)
    ys = cy + r_px * np.sin(angle)
    xs = np.clip(xs, 0, width - 1); ys = np.clip(ys, 0, height - 1)
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float64)

    undist_norm = cv2.undistortPoints(pts, K, D)
    ones = np.ones((len(r_px), 1, 1))
    homog = np.concatenate([undist_norm, ones], axis=-1).reshape(-1, 3)
    pts_undist = (K @ homog.T).T
    pts_undist = pts_undist[:, :2] / pts_undist[:, 2:3]

    disp_px = np.hypot(pts_undist[:, 0] - xs, pts_undist[:, 1] - ys)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_px, disp_px, linewidth=2)
    ax.set_xlabel("Distance from optical center r (pixels)")
    ax.set_ylabel("Pixel displacement if undistortion is omitted")
    ax.set_title("Box displacement grows with k₁r²+k₂r⁴+k₃r⁶")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-params", default=None, help="calibration/camera_params.npz with calibrated K,D")
    ap.add_argument("--image", default=None, help="Sample PNG/JPG frame; omit for a generated demo background")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--box", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"),
                    help="Sample box as top-left pixel plus size. Defaults near the frame edge, "
                         "where distortion is strongest and easiest to see.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--demo", action="store_true",
                    help="Run with assumed K,D when camera parameters are absent and generate a "
                         "background when no image is supplied. Illustrative only.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        base = cv2.imread(args.image)
        if base is None:
            raise FileNotFoundError(args.image)
        height, width = base.shape[:2]
    else:
        if not args.demo:
            raise ValueError("Provide --image, or use --demo to generate an illustrative background.")
        width, height = args.width, args.height
        # A gray checker grid makes distortion visible through grid curvature;
        # it is often clearer than a photograph for explaining the principle.
        base = np.full((height, width, 3), 40, dtype=np.uint8)
        for gx in range(0, width, 40):
            cv2.line(base, (gx, 0), (gx, height), (80, 80, 80), 1)
        for gy in range(0, height, 40):
            cv2.line(base, (0, gy), (width, gy), (80, 80, 80), 1)

    K, D, is_real = load_or_demo_camera_params(args.camera_params, width, height)
    tag = "" if is_real else "  [DEMO — assumed K,D; see console warning]"

    pts_d, pts_u = distortion_vector_field(K, D, width, height)
    max_disp = draw_vector_field(base, pts_d, pts_u, out_dir / "1_vector_field.png")
    print(f"[1] Vector field -> {out_dir/'1_vector_field.png'}  "
          f"(maximum frame displacement: {max_disp:.1f} px){tag}")

    box = args.box or [int(width * 0.78), int(height * 0.78), int(width * 0.10), int(height * 0.12)]
    shift = draw_box_comparison(base, K, D, box, out_dir / "2_box_comparison.png")
    print(f"[2] Box comparison -> {out_dir/'2_box_comparison.png'}  "
          f"(box-center displacement: {shift:.1f} px){tag}")

    plot_radial_error(K, D, width, height, out_dir / "3_radial_error_curve.png")
    print(f"[3] Radial error curve -> {out_dir/'3_radial_error_curve.png'}{tag}")

    print(f"\n✓ Wrote three figures to {out_dir}/ for the 'why undistortion is required' slide.")
    if not is_real:
        print("  ⚠ Rerun with calibrated --camera-params from calibrate.py before reporting "
              "specific quantitative values; the current output is illustrative only.")


if __name__ == "__main__":
    main()
