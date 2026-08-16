#!/usr/bin/env python3
"""
linear16_to_srgb8.py — Single source of truth cho bước 16-bit linear -> 8-bit
sRGB (nhánh SAM3). Trước đây encode_srgb_u8() bị COPY-PASTE giống hệt nhau ở
2 nơi (preprocess.py N7a + quick_tiff_to_sam3.py) — tech debt thật: sửa 1 chỗ
quên chỗ kia thì 2 nhánh lệch nhau âm thầm (dataset chính thức vs quick-run
sẽ ra màu khác nhau mà không ai biết cho tới khi so pixel). File này gộp lại
làm MỘT, cả hai import từ đây.

Đặt tên linear16_to_srgb8.py (không phải "16_to_8_bits.py" như đề xuất ban
đầu) vì Python không cho định danh module bắt đầu bằng chữ số — file "16_to_
8_bits.py" VẪN tồn tại được trên đĩa, nhưng `import` nó từ script khác sẽ
lỗi cú pháp (không thể viết `import 16_to_8_bits`). Tên này giữ đúng ý nghĩa
(linear 16-bit -> sRGB 8-bit) mà vẫn là identifier hợp lệ.

CÂU HỎI ĐÃ ĐẶT: "nếu không dùng encode_srgb_u8() của mình, SAM3 có tự áp
gamma/xuống 8-bit một cách không kiểm soát được không?"
---------------------------------------------------------------------------
CÓ, nếu lỡ đưa thẳng TIFF 16-bit vào SAM3 (bỏ qua bước convert này). SAM3's
io_utils.py load ảnh bằng PIL: Image.open(...).convert("RGB"). Với TIFF
16-bit, Pillow đọc ảnh ở mode "I;16" (hoặc "I") — convert("RGB") trên mode
đó KHÔNG áp đường cong sRGB nào cả, chỉ làm phép chia tuyến tính thô
(thường >> 8, tức chia 256, không phải chia 257 hay áp gamma 1/2.4 đúng
chuẩn). Kết quả: ảnh ra rất tối/sai màu, và quan trọng hơn — không kiểm
soát được (phụ thuộc version Pillow, có thể đổi khác giữa các máy). Đây
chính xác là kịch bản cần tránh.

=> guard_reject_16bit() dưới đây là hàng rào chặn kịch bản đó: mọi script
đưa ảnh vào SAM3 (test_sam3.py, sam3_video_to_labels.py,
sam3_export_tracks.py) gọi hàm này trước khi load — nếu phát hiện TIFF
16-bit lẫn vào (quên chạy preprocess.py/quick_tiff_to_sam3.py, hoặc trỏ
nhầm thư mục), báo lỗi RÕ RÀNG và DỪNG NGAY thay vì để SAM3 âm thầm ăn dữ
liệu sai (đúng nguyên tắc "fail loud" đã dùng xuyên suốt project — xem các
marker VERIFY-1..4 ở evs_recorder.cpp, và cách raw_to_events.py/raw_to_video.py
từ chối đoán encoding thay vì render rác một cách hợp lý).
"""

from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════════
#  ENCODE — 16-bit linear (0..65535) -> 8-bit sRGB (IEC 61966-2-1)
# ══════════════════════════════════════════════════════════════════
#  Nguồn gốc: preprocess.py N7a (comment gốc giữ nguyên ở đó). Áp SAU CÙNG
#  trong pipeline chính thức (sau undistort/resize, vốn phải chạy trên ảnh
#  linear vì nội suy là trung bình có trọng số — chỉ đúng vật lý trên ánh
#  sáng tuyến tính). Đường cong đơn điệu, per-pixel -> chỉ đổi giá trị,
#  không đổi vị trí -> label không bị ảnh hưởng bởi thứ tự áp bước này.

def encode_srgb_u8(rgb_lin: np.ndarray) -> np.ndarray:
    """rgb_lin: float array, thang 0..65535 (đúng range TIFF 16-bit uint16
    đọc trực tiếp, coi là linear — khớp input_transfer=linear mặc định của
    project trong config.yaml). Trả về uint8 HxWx3."""
    n = np.clip(rgb_lin / 65535.0, 0.0, 1.0)
    s = np.where(n <= 0.0031308, n * 12.92, 1.055 * np.power(n, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
#  GUARD — chặn 16-bit lọt vào SAM3 (xem giải thích ở docstring đầu file)
# ══════════════════════════════════════════════════════════════════

class Unexpected16BitInputError(RuntimeError):
    pass


def guard_reject_16bit(img: np.ndarray, source: str = "<array>") -> None:
    """Gọi TRƯỚC khi đưa 1 ảnh đã-load vào SAM3. Raise nếu ảnh không phải
    uint8 HxWx3 — dấu hiệu đây là TIFF 16-bit linear (hoặc single-channel Y)
    lọt qua chưa được encode_srgb_u8(), tức bị SAM3 tự xử lý ngoài tầm kiểm
    soát của project (xem docstring). KHÔNG tự sửa/convert giùm — bắt người
    dùng chạy lại preprocess.py/quick_tiff_to_sam3.py cho đúng bước, đúng
    nguyên tắc fail-loud của project."""
    if img.dtype != np.uint8:
        raise Unexpected16BitInputError(
            f"{source}: dtype={img.dtype}, cần uint8. Đây rất có thể là TIFF "
            "16-bit linear CHƯA qua encode_srgb_u8() (preprocess.py --output-rgb "
            "hoặc quick_tiff_to_sam3.py) bị đưa thẳng vào SAM3. Nếu để lọt, "
            "PIL's Image.open(...).convert('RGB') của io_utils.py sẽ tự làm "
            "phép chia tuyến tính thô (không áp gamma đúng) -> ảnh sai màu, "
            "không kiểm soát được. Chạy lại bước convert trước, đừng trỏ SAM3 "
            "thẳng vào thư mục export DaVinci.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise Unexpected16BitInputError(
            f"{source}: shape={img.shape}, cần HxWx3 RGB 8-bit.")


def guard_reject_16bit_path(path) -> None:
    """Kiểm tra nhanh 1 file TIFF trên đĩa TRƯỚC khi load full — đọc header
    qua tifffile (rẻ hơn PIL cho việc chỉ hỏi dtype), fail loud nếu 16-bit.
    Dùng ở các script duyệt cả thư mục frame (sam3_video_to_labels.py,
    sam3_export_tracks.py) để bắt lỗi sớm, trước khi SAM3 kịp chạy."""
    import tifffile
    p = Path(path)
    with tifffile.TiffFile(str(p)) as tf:
        dtype = tf.series[0].dtype
    if dtype != np.uint8:
        raise Unexpected16BitInputError(
            f"{p}: dtype={dtype}, cần uint8. TIFF 16-bit linear CHƯA convert "
            "lọt vào thư mục frame cho SAM3 — xem giải thích trong "
            "linear16_to_srgb8.py. Chạy lại preprocess.py --output-rgb hoặc "
            "quick_tiff_to_sam3.py cho đúng thư mục này trước.")
