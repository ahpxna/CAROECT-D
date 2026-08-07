#!/usr/bin/env python3
"""
xypt_to_h5.py — Đọc file .cevt dạng MỚI (magic "CAROXYP1", ghi bởi
evs_recorder.cpp --flat-xypt) thành events.h5 theo đúng schema chung
(x uint16, y uint16, t int64 µs, p uint8 1=ON/0=OFF) — dùng chung với
run_v2e.py / run_dvsvolt.py / cevt_to_events.py.

VÌ SAO FILE NÀY ĐƠN GIẢN HƠN HẲN cevt_to_events.py
------------------------------------------------------
cevt_to_events.py phải "đoán" định dạng (dense CD-frame / XYPT / EVT3.0 word
stream) vì evs_recorder.cpp bản CŨ ghi payload mà C++ side không biết chắc
Arena đã trả về format nào (--output-format-value chỉ là YÊU CẦU, có thể bị
camera từ chối âm thầm). Với --flat-xypt + --strict-xypt (khuyên dùng cùng
nhau — xem evs_recorder.cpp), việc XYPT có xảy ra hay không được XÁC NHẬN
NGAY LÚC GHI (recording abort nếu không xác nhận được), nên file luôn chắc
chắn là mảng {x,y,t,p} float32 liên tục — không cần đoán gì nữa.

⚠ CẢNH BÁO ĐỘ CHÍNH XÁC TIMESTAMP (đọc trước khi tin số liệu µs)
---------------------------------------------------------------------
LucidXYTPPixel.t là float32 (32-bit). float32 chỉ biểu diễn số nguyên CHÍNH
XÁC tới 2^24 = 16,777,216. Nếu t là microsecond tuyệt đối tính từ lúc bắt
đầu record, thì SAU ~16.7 GIÂY quay, các timestamp microsecond sẽ bắt đầu bị
làm tròn (2 event cách nhau vài µs có thể bị gán CÙNG 1 giá trị t sau khi ép
kiểu). Với recording dài hơn ~17s, ĐỪNG coi t là chính xác tuyệt đối tới µs —
script này in ra thống kê khoảng cách giữa các timestamp liên tiếp (dt) để
giúp phát hiện hiện tượng "trùng t hàng loạt" này. Nếu thấy nhiều, đây là giới
hạn PHẦN CỨNG/FIRMWARE (Arena SDK trả t dạng float32), không phải lỗi script
— cần báo lại cho LUCID hoặc chuyển sang đọc EVT3.0 word-stream gốc (có
timestamp uint64) nếu cần độ chính xác µs cho recording dài.

Script này luôn ghi `t` trong H5 dạng int64 microsecond, nhưng nếu nguồn XYPT
đã là float32 absolute-us quá dài thì precision đã mất trước khi Python đọc
được. Vì vậy H5 sẽ có attrs["timestamp_precision_status"] để các bước sau
(đặc biệt calibrate_simulator.py) tự từ chối timing calibration khi status
không còn an toàn.

Usage:
  python xypt_to_h5.py --input session01.cevt --output session01_real.h5
  python xypt_to_h5.py --input session01.cevt --inspect-only
"""

import argparse
import struct
from pathlib import Path

import h5py
import numpy as np

FILE_HEADER_FMT = "<8sQIIIQ"
FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_FMT)   # 36 bytes
MAGIC = b"CAROXYP1"
XYPT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("t", "<f4"), ("p", "<f4")])  # 16 B/event


def read_header(f):
    raw = f.read(FILE_HEADER_SIZE)
    if len(raw) < FILE_HEADER_SIZE:
        raise ValueError("File too short for FileHeader — not a valid recording")
    magic, pixel_format, bpp, w, h, _reserved = struct.unpack(FILE_HEADER_FMT, raw)
    magic_s = magic.split(b"\x00")[0]
    if magic_s != MAGIC:
        raise ValueError(
            f"magic={magic_s!r}, expected {MAGIC!r}. Nếu đây là file ghi bởi evs_recorder.cpp "
            f"KHÔNG dùng --flat-xypt (magic sẽ là 'CAROEVT1'), dùng cevt_to_events.py thay vì "
            f"script này.")
    return dict(pixel_format=pixel_format, bpp=bpp, width=w, height=h)


def load_flat_xypt(path: str):
    with open(path, "rb") as f:
        hdr = read_header(f)
        remaining = f.read()

    n_full = len(remaining) // XYPT_DTYPE.itemsize
    tail_bytes = len(remaining) - n_full * XYPT_DTYPE.itemsize
    if tail_bytes:
        print(f"  [info] {tail_bytes} byte(s) thừa ở cuối file (không đủ 1 event 16-byte) — "
              f"bỏ qua, thường do recording bị ngắt giữa chừng lúc ghi buffer cuối cùng.")

    arr = np.frombuffer(remaining, dtype=XYPT_DTYPE, count=n_full)
    return hdr, arr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help=".cevt ghi bởi evs_recorder.cpp --flat-xypt")
    ap.add_argument("--output", default=None, help="events.h5 (mặc định: cùng tên, đổi .h5)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--t-unit", choices=["us", "s", "auto"], default="auto",
                    help="Đơn vị của t trong file — 'auto' so khớp với event rate hợp lý "
                         "(vài trăm kev/s tới vài chục Mev/s) để đoán us vs s")
    ap.add_argument("--source", default="triton2_real")
    ap.add_argument("--inspect-only", action="store_true", help="Chỉ in thống kê, không ghi .h5")
    ap.add_argument("--max-zero-dt-frac", type=float, default=0.01,
                    help="Ngưỡng dt=0 fraction để coi timestamp degraded (default 1%).")
    ap.add_argument("--require-precise-t", action="store_true",
                    help="Fail thay vì chỉ cảnh báo nếu timestamp_precision_status != precise.")
    args = ap.parse_args()

    print(f"Đọc {args.input} ...")
    hdr, arr = load_flat_xypt(args.input)
    n = len(arr)
    if n == 0:
        raise ValueError("0 event decode được — file rỗng hoặc bị hỏng")
    print(f"  FileHeader: pixelFormat=0x{hdr['pixel_format']:x} bpp={hdr['bpp']} "
          f"w={hdr['width']} h={hdr['height']}")
    print(f"  {n:,} event(s) decode được ({n * XYPT_DTYPE.itemsize / 1e6:.1f} MB payload)")

    # ── Kiểm tra p convention thực tế (0/1 hay -1/1 hay khác) ──────────────
    p_unique = np.unique(arr["p"])
    print(f"  Giá trị p duy nhất trong file: {p_unique}")
    if set(np.round(p_unique).astype(int).tolist()) == {0, 1}:
        p_on = (arr["p"] > 0.5)
    elif set(np.round(p_unique).astype(int).tolist()) == {-1, 1}:
        p_on = (arr["p"] > 0)
        print("  -> phát hiện convention {-1,+1}, map về {0,1} (1=ON)")
    else:
        print(f"  ⚠ Convention p lạ ({p_unique}) — mặc định coi p>0 là ON, TỰ KIỂM TRA lại "
              f"bằng cách vẽ vài event lên ảnh trước khi tin kết quả.")
        p_on = (arr["p"] > 0)

    # ── Suy luận đơn vị t nếu auto ───────────────────────────────────────
    t_raw = arr["t"].astype(np.float64)
    t_span = float(t_raw.max() - t_raw.min())
    t_unit = args.t_unit
    if t_unit == "auto":
        # Coi t là µs trước: event rate hợp lý là ~1e3..1e8 ev/s cho roadside traffic.
        rate_as_us = n / (t_span / 1e6) if t_span > 0 else 0
        rate_as_s = n / t_span if t_span > 0 else 0
        # Chọn cách hiểu nào cho ra rate "hợp lý" hơn (trong khoảng 1e2..1e8 ev/s)
        plausible_us = 1e2 <= rate_as_us <= 1e8
        plausible_s = 1e2 <= rate_as_s <= 1e8
        if plausible_us and not plausible_s:
            t_unit = "us"
        elif plausible_s and not plausible_us:
            t_unit = "s"
        else:
            t_unit = "us"  # mặc định theo comment trong evs_recorder.cpp ("REAL microsecond")
        print(f"  [auto t-unit] rate nếu t=µs: {rate_as_us/1e6:.3f} Mev/s  |  "
              f"rate nếu t=s: {rate_as_s/1e6:.3f} Mev/s  ->  chọn t-unit='{t_unit}'")

    t_us = t_raw if t_unit == "us" else t_raw * 1e6

    # ── Cảnh báo mất chính xác float32 nếu recording dài ───────────────────
    is_long_recording = (t_span > 16_000_000) if t_unit == "us" else (t_span > 16.0)
    if is_long_recording:
        print(f"  ⚠ Recording dài (~{t_span/1e6 if t_unit=='us' else t_span:.1f}s) — t gốc là "
              f"float32, mất precision nguyên sau ~16.7s (xem cảnh báo trong module "
              f"docstring). Kiểm tra dt dưới đây có nhiều giá trị TRÙNG bất thường không.")

    order = np.argsort(t_us, kind="stable")
    t_us_sorted = t_us[order]
    n_out_of_order = int(np.sum(np.diff(t_us) < 0))
    if n_out_of_order:
        print(f"  [info] {n_out_of_order:,} event không theo thứ tự thời gian trong file gốc "
              f"— đã sort lại (bình thường nếu ring buffer nhiều slot flush không hoàn toàn "
              f"tuần tự; label_transfer.py cần t đã sort).")

    dt = np.diff(t_us_sorted)
    dt_pos = dt[dt > 0]
    n_zero_dt = int(np.sum(dt == 0))
    zero_dt_frac = n_zero_dt / max(n - 1, 1)
    timestamp_precision_status = "precise"
    timestamp_precision_notes = []
    if is_long_recording:
        timestamp_precision_status = "float32_long_recording"
        timestamp_precision_notes.append("source XYPT.t is float32 and span exceeds ~16s")
    if zero_dt_frac > args.max_zero_dt_frac:
        timestamp_precision_status = "degraded_zero_dt"
        timestamp_precision_notes.append(
            f"dt=0 fraction {zero_dt_frac:.4f} exceeds threshold {args.max_zero_dt_frac:.4f}")
    print(f"  dt giữa các event liên tiếp (sau sort): median={np.median(dt_pos) if len(dt_pos) else 0:.2f}µs  "
          f"  dt=0 (trùng timestamp): {n_zero_dt:,} ({100*n_zero_dt/max(n-1,1):.2f}%)")
    print(f"  timestamp_precision_status: {timestamp_precision_status}")
    if timestamp_precision_notes:
        for note in timestamp_precision_notes:
            print(f"    - {note}")
    if args.require_precise_t and timestamp_precision_status != "precise":
        raise RuntimeError("Timestamp precision không đạt --require-precise-t. "
                           "Quay clip ngắn hơn, giảm span, hoặc record raw EVT3/Bpe64 "
                           "rồi decode bằng converter giữ uint64 timestamp.")

    x = np.clip(np.round(arr["x"][order]), 0, args.width - 1).astype(np.uint16)
    y = np.clip(np.round(arr["y"][order]), 0, args.height - 1).astype(np.uint16)
    t_final = np.round(t_us_sorted).astype(np.int64)
    p_final = p_on[order].astype(np.uint8)

    print(f"\n  x range: [{x.min()}, {x.max()}]   y range: [{y.min()}, {y.max()}]")
    print(f"  ON: {p_final.sum():,} ({100*p_final.sum()/n:.1f}%)   "
          f"OFF: {n-p_final.sum():,} ({100*(n-p_final.sum())/n:.1f}%)")
    print(f"  duration: {(t_final[-1]-t_final[0])/1e6:.2f}s   "
          f"rate: {n/((t_final[-1]-t_final[0])/1e6):.0f} ev/s" if t_final[-1] > t_final[0] else "")

    if args.inspect_only:
        return

    out = args.output or str(Path(args.input).with_suffix(".h5"))
    with h5py.File(out, "w") as f:
        f.create_dataset("x", data=x, compression="gzip", compression_opts=4)
        f.create_dataset("y", data=y, compression="gzip", compression_opts=4)
        f.create_dataset("t", data=t_final, compression="gzip", compression_opts=4)
        f.create_dataset("p", data=p_final, compression="gzip", compression_opts=4)
        f.attrs["n_events"] = n
        f.attrs["width"] = args.width
        f.attrs["height"] = args.height
        f.attrs["source"] = args.source
        f.attrs["t_unit"] = "microseconds"
        f.attrs["p_convention"] = "1=ON, 0=OFF"
        f.attrs["source_file"] = Path(args.input).name
        f.attrs["container_format"] = "CAROXYP1 (flat, no per-buffer framing)"
        f.attrs["timestamp_dtype"] = "int64_microseconds"
        f.attrs["timestamp_precision_status"] = timestamp_precision_status
        f.attrs["timestamp_zero_dt_fraction"] = float(zero_dt_frac)
        f.attrs["timestamp_precision_note"] = "; ".join(timestamp_precision_notes)
    print(f"\n✓ {out}")


if __name__ == "__main__":
    main()
