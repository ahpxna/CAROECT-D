#!/usr/bin/env python3
"""
visualize_geometry_error.py — Vẽ minh hoạ cho slide thuyết trình: NẾU bỏ qua
undistort (hoặc geometry sai), bounding box từ SAM3 sẽ lệch bao nhiêu pixel
khi đắp sang lưới event, và độ lệch đó tăng theo bán kính r như thế nào.

NGUYÊN LÝ ĐANG MINH HOẠ (đã thống nhất trong phần lý thuyết project)
------------------------------------------------------------------------
Camera thật không phải lỗ kim hoàn hảo — công thức Brown-Conrady:
    x_distorted = x(1 + k1 r² + k2 r⁴ + k3 r⁶) + [2 p1 xy + p2(r²+2x²)]
    y_distorted = y(1 + k1 r² + k2 r⁴ + k3 r⁶) + [p1(r²+2y²) + 2 p2 xy]
    r = sqrt(x² + y²)   (toạ độ chuẩn hoá, gốc tại principal point c_x,c_y)

Ảnh THẬT quay ra từ camera LUÔN ở dạng "distorted" (vế trái). Nếu pipeline
undistort đúng (preprocess.py bước N5), SAM3 chạy trên ảnh đã nắn thẳng ->
box đúng vị trí vật lý. Nếu BỎ QUA bước này, SAM3 chạy trên ảnh còn méo ->
box lấy đúng toạ độ pixel trên ảnh méo đó, nhưng khi đắp sang lưới event
(vốn giả định hệ toạ độ đã thẳng, y hệt cách simulator dùng ảnh TIFF ĐàN
undistort ở nhánh preprocess) -> box bị LỆCH đúng bằng đúng vector méo tại
vị trí đó.

Script này dùng chính K, D đã calibrate (calibrate.py -> camera_params.npz)
để:
  1. Vẽ ảnh mẫu với LƯỚI các điểm, mỗi điểm có 1 mũi tên chỉ từ vị trí PIXEL
     THẬT trên ảnh (distorted, ảnh camera thật xuất ra) tới vị trí ĐÚNG về
     mặt vật lý (undistorted) — đây chính là "độ lệch nếu quên undistort".
  2. Vẽ 1 bounding box mẫu tại vị trí do người dùng chỉ định (hoặc mặc định
     gần rìa khung hình, nơi lỗi rõ nhất) ở CẢ HAI trạng thái: box "đúng"
     (sau undistort, viền xanh) và box "sai" (nếu skip undistort, viền đỏ)
     chồng lên ảnh thật.
  3. Vẽ biểu đồ độ lớn lệch (pixel) theo bán kính r — đúng đường cong
     k1 r² + k2 r⁴ + k3 r⁶ đã giải thích bằng công thức trước đó.

NẾU CHƯA CÓ camera_params.npz (m nói chưa lấy data)
------------------------------------------------------
Script tự tạo một bộ (K, D) MẪU hợp lý cho ống kính 6mm trên sensor IMX636
(khớp thông số focal length trong CAROECT-D_outline.pdf) để DEMO ngay hôm
nay — có in cảnh báo rõ ràng đây là số liệu giả định, PHẢI thay bằng
camera_params.npz thật (chạy calibrate.py) trước khi dùng số liệu này để
báo cáo định lượng thật.

Usage:
  # Có camera_params.npz thật + 1 ảnh mẫu:
  python visualize_geometry_error.py --camera-params calibration/camera_params.npz \
      --image sample_frame.png --output-dir slides/geometry --box 980 560 120 90

  # Chưa có gì, chỉ muốn demo nhanh cho slide (dùng K,D giả định + ảnh trắng):
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
#  N1 · K, D — thật (camera_params.npz) hoặc giả định demo
# ══════════════════════════════════════════════════════════════════════════

def load_or_demo_camera_params(path, width, height):
    if path and Path(path).exists():
        d = np.load(path)
        K, D = d["K"].astype(np.float64), d["D"].astype(np.float64)
        calib_size = tuple(int(v) for v in d["image_size"]) if "image_size" in d else (width, height)
        if calib_size != (width, height):
            sx, sy = width / calib_size[0], height / calib_size[1]
            K = K.copy(); K[0, 0] *= sx; K[0, 2] *= sx; K[1, 1] *= sy; K[1, 2] *= sy
        print(f"[camera] Dùng K,D THẬT từ {path}")
        return K, D, True

    print("[camera] ⚠ KHÔNG có camera_params.npz — dùng K,D GIẢ ĐỊNH cho lens 6mm/IMX636 "
          "(chỉ để demo hình dạng minh hoạ, KHÔNG dùng số liệu này để báo cáo định lượng thật. "
          "Chạy calibrate.py với ảnh chessboard thật rồi truyền --camera-params để có số đúng.)")
    # Xấp xỉ thô: fx=fy ~ focal_mm / pixel_pitch_mm, IMX636 pixel pitch ~ 4.86um,
    # focal 6mm -> f_px ~ 6000/4.86 ~ 1235 px. k1 âm vừa phải (pincushion nhẹ,
    # điển hình lens công nghiệp góc hẹp) để đường cong minh hoạ rõ ràng trên slide.
    f = 1235.0
    K = np.array([[f, 0, width / 2 + 5], [0, f, height / 2 - 3], [0, 0, 1]], dtype=np.float64)
    D = np.array([-0.18, 0.06, 0.0006, -0.0004, -0.01], dtype=np.float64)  # k1,k2,p1,p2,k3
    return K, D, False


# ══════════════════════════════════════════════════════════════════════════
#  N2 · MINH HOẠ 1 — trường vector độ lệch (distortion vector field)
# ══════════════════════════════════════════════════════════════════════════

def distortion_vector_field(K, D, width, height, step=60):
    """Với mỗi điểm lưới trên ảnh THẬT (distorted), tính vị trí ĐÚNG vật lý
    (undistorted) tương ứng — trả về (pts_distorted, pts_undistorted).
    cv2.undistortPoints trả normalized coords -> nhân lại K để ra pixel."""
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
        cv2.circle(img, p1, 3, (0, 0, 255), -1)          # đỏ = vị trí pixel THẬT (distorted)
        cv2.arrowedLine(img, p1, p2, (0, 255, 0), 1, tipLength=0.35)  # xanh = tới vị trí ĐÚNG
    cv2.putText(img, "Do (red)=vi tri pixel that tren anh camera  |  "
                     "Mui ten xanh -> vi tri DUNG neu da undistort",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Do lech pixel lon nhat trong khung hinh: {max_disp:.1f}px",
                (10, height_text_y(img)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    return max_disp


def height_text_y(img):
    return img.shape[0] - 15


# ══════════════════════════════════════════════════════════════════════════
#  N3 · MINH HOẠ 2 — 1 bounding box cụ thể, đúng vs sai
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
    cv2.polylines(img, [poly_wrong], True, (0, 0, 255), 2)    # đỏ = box SAI nếu skip undistort
    cv2.polylines(img, [poly_right], True, (0, 255, 0), 2)    # xanh = box ĐÚNG

    center_wrong = corners_distorted.mean(axis=0)
    center_right = corners_undistorted.mean(axis=0)
    shift_px = float(np.hypot(*(center_right - center_wrong)))

    cv2.putText(img, "Do = box SAI (bo qua undistort)   Xanh = box DUNG (da undistort)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Lech tam box: {shift_px:.1f} px",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    return shift_px


# ══════════════════════════════════════════════════════════════════════════
#  N4 · MINH HOẠ 3 — biểu đồ độ lệch theo bán kính r (đúng công thức Brown-Conrady)
# ══════════════════════════════════════════════════════════════════════════

def plot_radial_error(K, D, width, height, out_path):
    cx, cy, f = K[0, 2], K[1, 2], (K[0, 0] + K[1, 1]) / 2
    r_max_px = np.hypot(max(cx, width - cx), max(cy, height - cy))
    r_px = np.linspace(0, r_max_px, 200)

    # Lấy mẫu dọc theo 1 tia từ tâm ra rìa (đủ đại diện vì méo xuyên tâm chỉ
    # phụ thuộc |r|, không phụ thuộc hướng — đúng như đã giải thích lý do
    # dùng mũ chẵn r²,r⁴,r⁶ trong phần lý thuyết).
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
    ax.set_xlabel("Khoảng cách tới tâm quang học r (pixel)")
    ax.set_ylabel("Độ lệch pixel nếu bỏ qua undistort")
    ax.set_title("Độ lệch box tăng theo r — đúng dạng k₁r²+k₂r⁴+k₃r⁶")
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
    ap.add_argument("--camera-params", default=None, help="calibration/camera_params.npz (K,D thật)")
    ap.add_argument("--image", default=None, help="1 frame mẫu (PNG/JPG); bỏ trống -> nền xám demo")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--box", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"),
                    help="Box mẫu (pixel, góc trên-trái). Mặc định: gần rìa khung hình, "
                         "nơi méo mạnh nhất, để lỗi nhìn rõ trên slide.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--demo", action="store_true",
                    help="Chạy full demo (K,D giả định nếu chưa có --camera-params, nền xám nếu "
                         "chưa có --image) — dùng ngay hôm nay không cần chờ data thật")
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
            raise ValueError("Cần --image, hoặc dùng --demo để tự tạo ảnh nền minh hoạ")
        width, height = args.width, args.height
        # Ảnh nền lưới ca-rô xám — giúp NHÌN THẤY méo qua độ cong của lưới,
        # trực quan hơn cả ảnh thật cho slide giải thích nguyên lý.
        base = np.full((height, width, 3), 40, dtype=np.uint8)
        for gx in range(0, width, 40):
            cv2.line(base, (gx, 0), (gx, height), (80, 80, 80), 1)
        for gy in range(0, height, 40):
            cv2.line(base, (0, gy), (width, gy), (80, 80, 80), 1)

    K, D, is_real = load_or_demo_camera_params(args.camera_params, width, height)
    tag = "" if is_real else "  [DEMO — K,D giả định, xem cảnh báo console]"

    pts_d, pts_u = distortion_vector_field(K, D, width, height)
    max_disp = draw_vector_field(base, pts_d, pts_u, out_dir / "1_vector_field.png")
    print(f"[1] Vector field -> {out_dir/'1_vector_field.png'}  "
          f"(lệch pixel lớn nhất trong khung hình: {max_disp:.1f}px){tag}")

    box = args.box or [int(width * 0.78), int(height * 0.78), int(width * 0.10), int(height * 0.12)]
    shift = draw_box_comparison(base, K, D, box, out_dir / "2_box_comparison.png")
    print(f"[2] Box comparison -> {out_dir/'2_box_comparison.png'}  "
          f"(lệch tâm box: {shift:.1f}px){tag}")

    plot_radial_error(K, D, width, height, out_dir / "3_radial_error_curve.png")
    print(f"[3] Radial error curve -> {out_dir/'3_radial_error_curve.png'}{tag}")

    print(f"\n✓ 3 ảnh trong {out_dir}/ — dùng trực tiếp cho slide 'vì sao undistort là bắt buộc'.")
    if not is_real:
        print("  ⚠ NHỚ chạy lại với --camera-params calibration/camera_params.npz thật (từ "
              "calibrate.py) trước khi dùng SỐ LIỆU cụ thể (không phải hình minh hoạ) để báo cáo.")


if __name__ == "__main__":
    main()
