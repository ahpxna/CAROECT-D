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

Usage:
  python label_transfer.py --tracks tracks.json --events events.h5 \
      --window-us 8342 --max-gap-frames 5 --output windows.json --stats

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


def load_events_h5(path: str) -> dict:
    """Same schema as run_v2e.py / run_dvsvolt.py / cevt_to_events.py output."""
    with h5py.File(str(path), "r") as hf:
        return {k: hf[k][:] for k in ("x", "y", "t", "p")}


# ══════════════════════════════════════════════════════════════════════════
#  N2 · NỘI SUY TUYẾN TÍNH — chính là công thức α ở module docstring
# ══════════════════════════════════════════════════════════════════════════

def interpolate_box(track: dict, t_query: float, max_gap_us: float):
    """Trả về (cx,cy,w,h) nội suy tại t_query, hoặc None nếu:
      - t_query nằm ngoài [t_first, t_last] của track (không extrapolate), hoặc
      - t_query rơi vào 1 khoảng gap giữa 2 quan sát liên tiếp lớn hơn max_gap_us
        (occlusion dài -> không nối 2 đoạn track bằng 1 đường thẳng giả)."""
    t = track["t"]
    if len(t) < 1 or t_query < t[0] or t_query > t[-1]:
        return None
    idx = bisect.bisect_right(t, t_query) - 1
    idx = min(max(idx, 0), len(t) - 2) if len(t) > 1 else 0
    if len(t) == 1:
        return None  # 1 điểm duy nhất -> không đủ để nội suy khoảng thời gian nào

    t_i, t_ip1 = t[idx], t[idx + 1]
    if t_query < t_i or t_query > t_ip1:
        # t_query nằm giữa 2 khoảng bisect không trực tiếp bao — quét lại tuyến tính
        # (mảng track nhỏ, N observations mỗi track thường vài trăm -> O(N) chấp nhận được)
        idx = np.searchsorted(t, t_query, side="right") - 1
        idx = min(max(idx, 0), len(t) - 2)
        t_i, t_ip1 = t[idx], t[idx + 1]
        if not (t_i <= t_query <= t_ip1):
            return None

    if (t_ip1 - t_i) > max_gap_us:
        return None  # gap quá lớn -> track coi như mất dấu trong khoảng này

    alpha = 0.0 if t_ip1 == t_i else (t_query - t_i) / (t_ip1 - t_i)
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
            obs_idx = _active_obs_index(track, tc, max_gap_us)
            mask_path = None
            if obs_idx is not None and obs_idx < len(track["mask_path"]):
                mask_path = track["mask_path"][obs_idx]
            boxes.append(dict(track_id=track_id, class_id=track["class_id"],
                              cx=cx, cy=cy, w=w, h=h, mask_path=mask_path))
        out.append(dict(t_start_us=t0, t_end_us=t1, t_center_us=tc, boxes=boxes))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  N4 · (TÙY CHỌN) PER-EVENT LABEL — công thức L(e_k) đầy đủ + occlusion stats
# ══════════════════════════════════════════════════════════════════════════

def _mask_contains(track: dict, obs_idx: int, x_px: int, y_px: int, mask_cache: dict) -> bool | None:
    rel = track["mask_path"][obs_idx] if obs_idx < len(track["mask_path"]) else None
    if not rel:
        return None
    path = track["base_dir"] / rel
    key = str(path)
    if key not in mask_cache:
        import cv2
        mask = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask_cache[key] = None
        else:
            mask_cache[key] = mask > 0
    mask = mask_cache[key]
    if mask is None or y_px < 0 or x_px < 0 or y_px >= mask.shape[0] or x_px >= mask.shape[1]:
        return None
    return bool(mask[y_px, x_px])


def _active_obs_index(track: dict, t_query: float, max_gap_us: float):
    t = track["t"]
    if len(t) < 2 or t_query < t[0] or t_query > t[-1]:
        return None
    idx = bisect.bisect_right(t, t_query) - 1
    idx = min(max(idx, 0), len(t) - 2)
    if (t[idx + 1] - t[idx]) > max_gap_us:
        return None
    if not (t[idx] <= t_query <= t[idx + 1]):
        return None
    return idx


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
                obs_idx = _active_obs_index(tracks[tid], float(tk), max_gap_us)
                if obs_idx is not None:
                    in_mask = _mask_contains(tracks[tid], obs_idx, int(round(x[k])), int(round(y[k])), mask_cache)
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
    args = ap.parse_args()

    fps, W, H, tracks = load_tracks(args.tracks)
    events = load_events_h5(args.events)
    n_events = len(events["t"])
    if n_events == 0:
        raise ValueError(f"{args.events} có 0 event — không có gì để gán nhãn")

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
