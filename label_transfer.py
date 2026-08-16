#!/usr/bin/env python3
"""
label_transfer.py — MẢNH TRUNG TÂM còn thiếu của CAROECT-D: ánh xạ nhãn SAM3
(track box theo frame RGB) sang event stream (voxel/window theo timestamp).

TẠI SAO FILE NÀY LÀ MẢNH GHÉP QUAN TRỌNG NHẤT
------------------------------------------------
Toàn bộ lý do preprocess.py giữ hình học "shared by construction" (undistort +
resize làm 1 lần duy nhất, dùng chung cho cả nhánh SAM3 lẫn nhánh simulator)
là để phép ánh xạ không gian giữa toạ độ RGB và toạ độ event trở thành MA TRẬN
ĐƠN VỊ — không cần warp gì thêm:

    Ω_RGB = Ω_EVS = {(x,y) | 0<=x<1280, 0<=y<720}

Nhưng "không gian khớp" chỉ là điều kiện cần. Còn "THỜI GIAN khớp" là việc của
CHÍNH FILE NÀY: track box chỉ tồn tại tại các mốc thời gian rời rạc của frame
RGB (T_TIFF = {t0,t1,...,tN}, cách nhau ~1/fps_original), trong khi event có
timestamp liên tục (microsecond). Với mỗi event e_k=(x_k,y_k,t_k,p_k), ta cần:

  1. Tìm 2 frame liền kề t_i <= t_k < t_{i+1} của MỖI track đang "sống" quanh t_k.
  2. Nội suy tuyến tính vị trí box tại đúng t_k:
         α = (t_k - t_i) / (t_{i+1} - t_i)
         B_j(t_k) = (1-α)·B_j(t_i) + α·B_j(t_{i+1})
  3. Event thuộc object j nếu (x_k,y_k) ∈ B_j(t_k). Nếu event rơi vào NHIỀU box
     chồng nhau (occlusion) -> gán cho box có diện tích NHỎ NHẤT tại t_k (quy
     ước: object ở gần camera hơn / nằm trên thường chiếm box nhỏ hơn trong
     ảnh phối cảnh roadside — đây là heuristic, không phải định lý; ghi log
     tỷ lệ occlusion để biết heuristic này ảnh hưởng bao nhiêu % dữ liệu).
  4. Event không rơi vào box nào -> background.

VÌ SAO OUTPUT LÀ "WINDOW-LEVEL" CHỨ KHÔNG PHẢI PER-EVENT
-----------------------------------------------------------
Baseline detector cho event data (RVT, RED, YOLOv8 trên event-frame — đúng 3
model eTraM dùng làm baseline) đều học trên một REPRESENTATION đã gộp theo
cửa sổ thời gian cố định (voxel grid / event count image), không học trực
tiếp trên từng event rời rạc. Nên ground-truth hữu ích nhất là 1 box list cho
MỖI CỬA SỔ [t_start, t_end) — đây là windows.json, thứ build_event_dataset.py
tiêu thụ trực tiếp. Hàm nội suy per-event (assign_events_to_tracks) vẫn được
giữ và implement đầy đủ — dùng để (a) tính thống kê occlusion in ra console,
(b) tuỳ chọn xuất nhãn per-event thật (--per-event-labels) cho ai cần độ chi
tiết pixel-level (ví dụ để visualize cho hội đồng), nhưng KHÔNG bắt buộc cho
luồng train chính.

GIỚI HẠN — không tự động giải quyết
-------------------------------------
- Không extrapolate: event trước frame đầu tiên hoặc sau frame cuối cùng track
  xuất hiện KHÔNG được gán nhãn (object có thể đã ra khỏi khung hình).
- --max-gap-frames chặn nội suy xuyên qua khoảng track bị mất quá lâu (occlusion
  dài) — mặc định 5 frame (~42ms @119.88fps); vượt ngưỡng này track coi như
  "mất dấu", 2 đoạn trước/sau KHÔNG nối bằng đường thẳng.
- CLOCK ALIGNMENT với event THẬT: tracks.json's t=0 là mốc bắt đầu clip RGB;
  events.h5's t (khi từ raw_to_events.py / legacy/cevt_to_events.py, tức 2
  camera vật lý riêng) là đồng hồ CỦA RIÊNG camera event, không có quan hệ nào
  đảm bảo với t=0 của RGB. File này GIẢ ĐỊNH track frame_idx=0 khớp với
  events['t'].min() + --time-offset-us (mặc định offset=0 -- một giả định,
  không phải số đo). Sai giả định này -> mọi event lệch hẳn ra ngoài range
  track -> gán nhãn background 100% một cách ÂM THẦM, không exception nào cả.
  Luôn chạy --stats lần đầu trên data thật và kiểm tra tỉ lệ foreground có hợp
  lý với cảnh quay không. (Không áp dụng cho events.h5 từ run_v2e.py/
  run_dvsvolt.py -- t ở đó sinh trực tiếp từ chính frame_idx của track nên
  luôn khớp đúng theo construction, không cần offset.)

Usage:
  python label_transfer.py --tracks tracks.json --events events.h5 \
      --window-us 8342 --max-gap-frames 5 --output windows.json --stats

  # Event THẬT (2 camera riêng) -- luôn kiểm tra --stats trước khi tin windows.json,
  # và set --time-offset-us nếu biết độ lệch trigger giữa 2 camera:
  python label_transfer.py --tracks tracks.json --events events_real.h5 \
      --output windows.json --stats --time-offset-us 0

  # Xuất thêm nhãn per-event thật (để visualize / debug), KHÔNG cần cho train:
  python label_transfer.py --tracks tracks.json --events events.h5 \
      --output windows.json --per-event-labels event_labels.h5
"""

import argparse
import bisect
import json
from pathlib import Path

import h5py
import numpy as np


# ══════════════════════════════════════════════════════════════════════════
#  N1 · LOAD
# ══════════════════════════════════════════════════════════════════════════

def load_tracks(path: str) -> dict:
    payload = json.loads(Path(path).read_text())
    base_dir = Path(path).parent
    tracks = {}
    for numeric_id, (obj_id, entry) in enumerate(payload["tracks"].items()):
        frames = sorted(entry["frames"], key=lambda r: r["t_us"])
        tracks[obj_id] = dict(
            numeric_id=numeric_id,
            class_id=entry["class_id"],
            class_name=entry.get("class_name"),
            base_dir=base_dir,
            frame_idx=np.array([r.get("frame_idx", -1) for r in frames], dtype=np.int64),
            t=np.array([r["t_us"] for r in frames], dtype=np.int64),
            cx=np.array([r["cx"] for r in frames], dtype=np.float64),
            cy=np.array([r["cy"] for r in frames], dtype=np.float64),
            w=np.array([r["w"] for r in frames], dtype=np.float64),
            h=np.array([r["h"] for r in frames], dtype=np.float64),
            mask_path=[r.get("mask_path") for r in frames],
        )
    return payload["fps"], payload["width"], payload["height"], tracks


def load_events_h5(path: str):
    """Same schema as run_v2e.py / run_dvsvolt.py / raw_to_events.py output.
    Returns (data, attrs) -- attrs is used by main() to tell simulated events
    (attrs['simulator'] present, written by run_v2e.py/run_dvsvolt.py) apart
    from real recorded events (no such key), which matters for the
    --time-offset-us clock-alignment warning below."""
    with h5py.File(str(path), "r") as hf:
        data = {k: hf[k][:] for k in ("x", "y", "t", "p")}
        attrs = dict(hf.attrs)
    return data, attrs


# ══════════════════════════════════════════════════════════════════════════
#  N2 · NỘI SUY TUYẾN TÍNH — chính là công thức α ở module docstring
# ══════════════════════════════════════════════════════════════════════════

def _find_bracket(t: np.ndarray, t_query: float, max_gap_us: float):
    """MỘT nơi duy nhất tìm cặp bracket [idx, idx+1] bao quanh t_query + alpha.

    Trước đây interpolate_box() và _active_obs_index() (mask) mỗi hàm tự có
    1 bản logic bracket riêng gần-giống-nhưng-không-giống-hệt — box có nhánh
    fallback np.searchsorted khi bisect_right không bao đúng t_query (hay gặp
    khi 2 observation trùng timestamp), mask thì không có nhánh đó -> có ca
    box nội suy được nhưng mask lại fail, và tệ hơn nữa là NẾU cả hai đều
    thành công thì cũng không có gì đảm bảo 2 alpha tính ra giống nhau. Gộp về
    đây để box và mask LUÔN dùng chung đúng 1 alpha cho cùng 1 t_query.

    Trả về (idx, alpha) hoặc (None, None) nếu:
      - t_query nằm ngoài [t_first, t_last] của track (không extrapolate), hoặc
      - track chỉ có 1 quan sát (không đủ để định nghĩa 1 khoảng), hoặc
      - khoảng [t[idx], t[idx+1]] bao t_query lớn hơn max_gap_us (occlusion
        dài -> không nối 2 đoạn track bằng 1 đường thẳng/mask giả)."""
    if len(t) < 2 or t_query < t[0] or t_query > t[-1]:
        return None, None

    idx = bisect.bisect_right(t, t_query) - 1
    idx = min(max(idx, 0), len(t) - 2)
    t_i, t_ip1 = t[idx], t[idx + 1]
    if t_query < t_i or t_query > t_ip1:
        # t_query nằm giữa 2 khoảng bisect không trực tiếp bao — quét lại tuyến tính
        # (mảng track nhỏ, N observations mỗi track thường vài trăm -> O(N) chấp nhận được)
        idx = np.searchsorted(t, t_query, side="right") - 1
        idx = min(max(idx, 0), len(t) - 2)
        t_i, t_ip1 = t[idx], t[idx + 1]
        if not (t_i <= t_query <= t_ip1):
            return None, None

    if (t_ip1 - t_i) > max_gap_us:
        return None, None  # gap quá lớn -> track coi như mất dấu trong khoảng này

    alpha = 0.0 if t_ip1 == t_i else (t_query - t_i) / (t_ip1 - t_i)
    return idx, alpha


def interpolate_box(track: dict, t_query: float, max_gap_us: float):
    """Trả về (cx,cy,w,h) nội suy tại t_query theo alpha từ _find_bracket(),
    hoặc None nếu track không bao t_query / gap quá lớn (xem _find_bracket)."""
    idx, alpha = _find_bracket(track["t"], t_query, max_gap_us)
    if idx is None:
        return None
    cx = (1 - alpha) * track["cx"][idx] + alpha * track["cx"][idx + 1]
    cy = (1 - alpha) * track["cy"][idx] + alpha * track["cy"][idx + 1]
    w = (1 - alpha) * track["w"][idx] + alpha * track["w"][idx + 1]
    h = (1 - alpha) * track["h"][idx] + alpha * track["h"][idx + 1]
    return cx, cy, w, h


# ══════════════════════════════════════════════════════════════════════════
#  N3 · WINDOW-LEVEL LABELS  (output chính, build_event_dataset.py tiêu thụ)
# ══════════════════════════════════════════════════════════════════════════

def build_windows(t_origin_us: float, duration_us: float, window_us: float):
    n = int(np.ceil(duration_us / window_us))
    windows = []
    for i in range(n):
        t0 = t_origin_us + i * window_us
        t1 = t_origin_us + min((i + 1) * window_us, duration_us)
        windows.append((t0, t1, 0.5 * (t0 + t1)))  # (start, end, center)
    return windows


def _clip_normalized_box(cx: float, cy: float, w: float, h: float):
    x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x1, y1 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if x1 <= x0 or y1 <= y0:
        return None
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0


def label_windows(tracks: dict, windows: list, max_gap_us: float,
                   width: int, height: int) -> list:
    """Với mỗi window, nội suy box của MỌI track tại thời điểm TRUNG TÂM cửa sổ
    (xấp xỉ hợp lý khi window nhỏ hơn nhiều so với tốc độ vật chuyển động qua
    khung hình — nếu window quá lớn, xe có thể đi hết cả window, box trung
    tâm không đại diện tốt -> giảm --window-us)."""
    out = []
    for (t0, t1, tc) in windows:
        boxes = []
        for track_id, track in tracks.items():
            r = interpolate_box(track, tc, max_gap_us)
            if r is None:
                continue
            cx, cy, w, h = r
            clipped = _clip_normalized_box(cx, cy, w, h)
            if clipped is None:
                continue  # tâm box đã ra khỏi khung hình -> bỏ qua window này
            cx, cy, w, h = clipped
            # windows.json stores ONE mask_path per box (COCO/YOLO export consumes it
            # as a single PNG, not a blend) -- so here we pick whichever of the two
            # bracketing observations is GEOMETRICALLY CLOSER to the window center,
            # using the same (idx, alpha) bracket the box interpolation just used
            # (not the old always-floor _active_obs_index, which silently froze the
            # mask at the earlier frame even when alpha was e.g. 0.99). Real linear
            # blending of both masks happens at the per-event level instead, in
            # _mask_contains() below via assign_events_to_tracks() -- that is the
            # granularity where "blend of two discretely-observed masks" in the
            # paper actually applies (per exact event timestamp t_k), not here.
            idx, alpha = _active_obs_bracket(track, tc, max_gap_us)
            mask_path = None
            if idx is not None:
                chosen_idx = idx if alpha < 0.5 else idx + 1
                if chosen_idx < len(track["mask_path"]):
                    mask_path = track["mask_path"][chosen_idx]
            boxes.append(dict(track_id=track_id, class_id=track["class_id"],
                              cx=cx, cy=cy, w=w, h=h, mask_path=mask_path))
        out.append(dict(t_start_us=t0, t_end_us=t1, t_center_us=tc, boxes=boxes))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  N4 · (TÙY CHỌN) PER-EVENT LABEL — công thức L(e_k) đầy đủ + occlusion stats
# ══════════════════════════════════════════════════════════════════════════

def _load_mask(track: dict, obs_idx: int, mask_cache: dict):
    """Load 1 mask PNG (grayscale, >0 = foreground) tại obs_idx, cache theo path.
    Trả về mảng bool 2D hoặc None nếu thiếu mask_path / đọc file lỗi."""
    rel = track["mask_path"][obs_idx] if obs_idx < len(track["mask_path"]) else None
    if not rel:
        return None
    key = str(track["base_dir"] / rel)
    if key not in mask_cache:
        import cv2
        mask = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        mask_cache[key] = None if mask is None else (mask > 0)
    return mask_cache[key]


def _mask_contains(track: dict, idx: int, alpha: float, x_px: int, y_px: int,
                    mask_cache: dict, thresh: float = 0.5) -> bool | None:
    """Blend TUYẾN TÍNH mask[idx] và mask[idx+1] theo alpha, đúng câu paper
    ("linear blend of two discretely-observed masks") — thay vì trước đây
    chỉ lấy nguyên mask[idx] (nearest-left), không hề đụng tới idx+1.
    idx, alpha PHẢI đến từ cùng _find_bracket() dùng cho box (xem
    _active_obs_bracket) để box và mask luôn khớp nhau tại đúng 1 t_query.

    Trả về None nếu KHÔNG mask nào đọc được (giữ nguyên hành vi cũ: caller
    fallback về box-only khi None). Nếu chỉ 1 trong 2 mask đọc được, dùng
    nguyên mask đó (không coi phần thiếu là 0 — tránh kéo blended value
    xuống thấp giả tạo do lỗi đọc file, không phải do object thật sự
    biến mất khỏi mask)."""
    m_lo = _load_mask(track, idx, mask_cache)
    m_hi = _load_mask(track, idx + 1, mask_cache)
    if m_lo is None and m_hi is None:
        return None

    def _val(mask):
        if (mask is None or y_px < 0 or x_px < 0
                or y_px >= mask.shape[0] or x_px >= mask.shape[1]):
            return None
        return float(mask[y_px, x_px])

    v_lo, v_hi = _val(m_lo), _val(m_hi)
    if v_lo is None and v_hi is None:
        return None
    if v_lo is None:
        return v_hi > thresh
    if v_hi is None:
        return v_lo > thresh
    blended = (1.0 - alpha) * v_lo + alpha * v_hi
    return blended > thresh


def _active_obs_bracket(track: dict, t_query: float, max_gap_us: float):
    """Alias mỏng qua _find_bracket() — mask giờ dùng CHUNG bracket-finder với
    box (interpolate_box), thay vì có bản logic riêng như _active_obs_index()
    cũ (thiếu nhánh fallback np.searchsorted mà box có -> có ca box nội suy
    được nhưng mask lại fail, xem comment trong _find_bracket)."""
    return _find_bracket(track["t"], t_query, max_gap_us)


def assign_events_to_tracks(events: dict, tracks: dict, max_gap_us: float,
                             width: int, height: int, use_masks: bool = True,
                             report_every: int = 2_000_000):
    """Trả về (label_track_id[uint32,-1=background], label_class_id[int16,-1=bg],
    n_occluded) — occlusion resolved bằng "box nhỏ nhất tại t_k thắng" (heuristic,
    xem cảnh báo trong module docstring).

    ĐỘ PHỨC TẠP: với mỗi event, quét qua mọi track đang "sống" quanh t_k để tìm
    box chứa (x,y). Với N_events lớn (chục triệu) và N_tracks vừa phải (roadside
    site thường < 50 object cùng lúc), đây vẫn khả thi (~N_events × N_tracks),
    nhưng CHẬM nếu N_tracks lớn — script này ưu tiên ĐÚNG hơn NHANH; nếu cần
    tăng tốc, thay bằng spatial index (ví dụ interval tree theo track t-range)."""
    n = len(events["t"])
    label_track = np.full(n, -1, dtype=np.int64)
    label_class = np.full(n, -1, dtype=np.int16)
    track_ids_sorted = sorted(tracks.keys())
    n_multi_match = 0

    x, y, t = events["x"].astype(np.float64), events["y"].astype(np.float64), events["t"]
    mask_cache = {}

    # Cache nội suy theo timestamp lặp lại nhiều lần (nhiều event trong cùng
    # microsecond) để đỡ tính lại — key = (track_id, t_k)
    last_t = None
    cached = {}
    for k in range(n):
        tk = t[k]
        if tk != last_t:
            cached = {}
            last_t = tk
        xk_n, yk_n = x[k] / width, y[k] / height  # normalize để so với cx,cy,w,h chuẩn hoá

        matches = []
        for tid in track_ids_sorted:
            if tid in cached:
                r = cached[tid]
            else:
                r = interpolate_box(tracks[tid], float(tk), max_gap_us)
                cached[tid] = r
            if r is None:
                continue
            cx, cy, w, h = r
            in_box = (cx - w / 2) <= xk_n <= (cx + w / 2) and (cy - h / 2) <= yk_n <= (cy + h / 2)
            if not in_box:
                continue
            in_object = True
            if use_masks:
                # THIS is the per-event granularity the paper's "linear blend of two
                # discretely-observed masks" describes: idx/alpha come from the same
                # bracket-finder used for the box, so mask and box are evaluated at
                # the exact same t_k, and _mask_contains() blends mask[idx] and
                # mask[idx+1] by alpha instead of freezing at the earlier frame.
                idx, alpha = _active_obs_bracket(tracks[tid], float(tk), max_gap_us)
                if idx is not None:
                    in_mask = _mask_contains(tracks[tid], idx, alpha, int(round(x[k])), int(round(y[k])), mask_cache)
                    if in_mask is not None:
                        in_object = in_mask
            if in_object:
                matches.append((tid, w * h))

        if matches:
            if len(matches) > 1:
                n_multi_match += 1
            tid, _area = min(matches, key=lambda m: m[1])  # box nhỏ nhất thắng
            label_track[k] = tracks[tid]["numeric_id"]
            label_class[k] = tracks[tid]["class_id"]

        if (k + 1) % report_every == 0:
            print(f"    [assign_events] {k+1:,}/{n:,} events processed...")

    return label_track, label_class, n_multi_match


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True, help="tracks.json (từ sam3_export_tracks.py)")
    ap.add_argument("--events", required=True, help="events.h5 (sim hoặc real, cùng schema x,y,t,p)")
    ap.add_argument("--output", required=True, help="windows.json (nhãn theo cửa sổ)")
    ap.add_argument("--window-us", type=float, default=None,
                    help="Độ dài cửa sổ (µs). Mặc định = 1e6/fps của tracks.json "
                         "(tức đúng bằng 1 frame RGB gốc — an toàn nhất).")
    ap.add_argument("--max-gap-frames", type=int, default=5,
                    help="Số frame RGB tối đa được phép 'mất dấu' track trước khi coi là "
                         "occlusion dài, không nội suy xuyên qua (default 5 ~42ms@119.88fps)")
    ap.add_argument("--stats", action="store_true", help="In thống kê chi tiết")
    ap.add_argument("--per-event-labels", default=None,
                    help="(tuỳ chọn, KHÔNG cần cho train) Xuất thêm nhãn per-event thật vào "
                         "file h5 này — dùng để visualize/debug độ chính xác pixel-level")
    ap.add_argument("--no-masks", action="store_true",
                    help="Không dùng mask PNG trong tracks.json; fallback về box-only.")
    ap.add_argument("--time-offset-us", type=float, default=0.0,
                    help="Offset (µs) cộng vào events['t'].min() để định nghĩa mốc mà "
                         "track frame_idx=0 (t=0 trong tracks.json) THỰC SỰ xảy ra trên "
                         "đồng hồ của camera event — CÙNG QUY ƯỚC với calibrate_simulator.py's "
                         "--time-offset-us. Mặc định 0.0 nghĩa là 'frame 0 khớp đúng event "
                         "đầu tiên', chỉ ĐÚNG khi 2 camera trigger cùng lúc thật sự (hoặc khi "
                         "events.h5 tới từ run_v2e.py/run_dvsvolt.py, vốn sinh t từ CHÍNH "
                         "frame_idx của track nên khớp tuyệt đối theo construction). Với event "
                         "THẬT (raw_to_events.py / legacy/cevt_to_events.py) đây là 2 camera "
                         "vật lý riêng, 2 đồng hồ riêng — offset=0 là GIẢ ĐỊNH, không phải đo "
                         "đạc; xem cảnh báo in ra khi chạy nếu chưa set cờ này.")
    args = ap.parse_args()

    fps, W, H, tracks = load_tracks(args.tracks)
    events, events_attrs = load_events_h5(args.events)
    n_events = len(events["t"])
    if n_events == 0:
        raise ValueError(f"{args.events} có 0 event — không có gì để gán nhãn")

    # ── CLOCK ALIGNMENT ──────────────────────────────────────────────────
    # tracks.json's t is 0-based (frame_idx * 1e6/fps, relative to the RGB
    # clip's own start). events.h5's t is either (a) 0-based too, BY
    # CONSTRUCTION, when it came from run_v2e.py/run_dvsvolt.py -- those
    # scripts derive every event's t directly from the same TIFF frame
    # index the track was built from, so no alignment is needed or possible
    # to get wrong -- or (b) an independent physical event camera's own
    # clock (raw_to_events.py: real sensor µs; legacy/cevt_to_events.py:
    # device/host buffer time), which has NO guaranteed relationship to the
    # RGB camera's t=0 whatsoever. Shifting tracks['t'] by
    # (events['t'].min() + --time-offset-us) is the only place this
    # assumption enters the file; every comparison below (_find_bracket,
    # build_windows) works in events.h5's own absolute time frame after
    # this point.
    is_simulated = "simulator" in events_attrs
    frame0_event_t = float(events["t"].min()) + args.time_offset_us
    if not is_simulated:
        tag = "[warning]" if args.time_offset_us == 0.0 else "[info]"
        print(f"\n{tag} events.h5 có nguồn THẬT (không có attrs['simulator']) — 2 camera vật "
              f"lý riêng, 2 đồng hồ riêng. Đang giả định track frame_idx=0 xảy ra tại "
              f"events['t'].min() + --time-offset-us = {frame0_event_t:.0f} µs "
              f"(--time-offset-us={args.time_offset_us:.0f}).")
        if args.time_offset_us == 0.0:
            print("  Đây là GIẢ ĐỊNH mặc định (offset=0), KHÔNG phải số đo — nếu 2 camera "
                  "không trigger đồng thời thật sự (network/USB/GigE latency, jitter phần "
                  "cứng khi start), MỌI event sẽ bị gán sai nhãn (thường là tất cả rơi ra "
                  "ngoài [t_first, t_last] của track -> background 100%, không có warning "
                  "nào khác ngoài warning này). Cách kiểm tra nhanh: chạy với --stats, nếu "
                  "'event background' ~100% mà cảnh biết chắc có object suốt clip, đó là "
                  "dấu hiệu offset sai, không phải model tệ. Đo offset thật cần 1 sự kiện "
                  "đồng bộ nhìn thấy được ở CẢ HAI luồng (vd đèn flash/clapperboard xuất "
                  "hiện trong cả TIFF lẫn event) rồi trừ chênh lệch t giữa 2 luồng cho sự "
                  "kiện đó — project chưa có script tự động làm việc này.")
    for track in tracks.values():
        track["t"] = track["t"] + frame0_event_t

    t_origin_us = float(events["t"].min())
    duration_us = float(events["t"].max() - events["t"].min())
    window_us = args.window_us or (1e6 / fps)
    max_gap_us = args.max_gap_frames * (1e6 / fps)

    print(f"tracks: {len(tracks)}  |  events: {n_events:,} ({duration_us/1e6:.2f}s)  |  "
          f"window={window_us:.0f}µs  max_gap={max_gap_us:.0f}µs")

    windows = build_windows(t_origin_us, duration_us, window_us)
    labeled = label_windows(tracks, windows, max_gap_us, W, H)

    n_with_box = sum(1 for w in labeled if w["boxes"])
    n_boxes_total = sum(len(w["boxes"]) for w in labeled)
    print(f"windows: {len(labeled)}  |  {n_with_box} có >=1 box  |  "
          f"{n_boxes_total} box-observation(s) tổng")

    Path(args.output).write_text(json.dumps(
        dict(fps=fps, width=W, height=H, t_origin_us=t_origin_us, window_us=window_us,
             max_gap_us=max_gap_us, tracks_base_dir=str(Path(args.tracks).resolve().parent),
             windows=labeled), indent=2))
    print(f"✓ {args.output}")

    if args.stats or args.per_event_labels:
        print("\nĐang gán nhãn per-event (dùng cho --stats và/hoặc --per-event-labels)...")
        label_track, label_class, n_multi = assign_events_to_tracks(
            events, tracks, max_gap_us, W, H, use_masks=not args.no_masks)
        n_fg = int(np.sum(label_class >= 0))
        print(f"\n{'='*60}\nTHỐNG KÊ PER-EVENT LABEL ASSIGNMENT\n{'='*60}")
        print(f"  event thuộc object (foreground) : {n_fg:,} / {n_events:,} "
              f"({100*n_fg/n_events:.1f}%)")
        print(f"  event background                : {n_events - n_fg:,} "
              f"({100*(n_events-n_fg)/n_events:.1f}%)")
        print(f"  event rơi vào >1 box chồng nhau  : {n_multi:,} "
              f"({100*n_multi/n_events:.2f}%, resolved bằng box nhỏ nhất — "
              f"heuristic, xem cảnh báo trong module docstring)")

        if args.per_event_labels:
            with h5py.File(args.per_event_labels, "w") as f:
                f.create_dataset("track_id", data=label_track, compression="gzip")
                f.create_dataset("class_id", data=label_class, compression="gzip")
                f.attrs["source_events"] = str(args.events)
                f.attrs["source_tracks"] = str(args.tracks)
                f.attrs["track_id_map_json"] = json.dumps(
                    {v["numeric_id"]: k for k, v in tracks.items()}, ensure_ascii=False)
                f.attrs["assignment_rule"] = "Eq E.5: interpolate track box at each t_k; optional mask point-in-mask; smallest box resolves overlaps."
                f.attrs["uses_masks"] = bool(not args.no_masks)
                f.attrs["note"] = "Index-aligned với x,y,t,p trong file events.h5 gốc. -1 = background."
            print(f"\n  Per-event labels -> {args.per_event_labels}")


if __name__ == "__main__":
    main()
