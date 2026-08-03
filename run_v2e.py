#!/usr/bin/env python3
"""
CAROECT-D — v2e driver (16-bit path, KHÔNG sửa source v2e)
==========================================================

WHY THIS FILE EXISTS
--------------------
v2e CLI đọc input qua reader 8-bit của nó → nghiền nát dynamic range 16-bit
linear mà preprocess.py giữ gìn. Thay vì fork v2e, driver này gọi thẳng class
EventEmulator của v2e như một LIBRARY và tự đút frame float 16-bit-precision
vào. Physics của v2e không dính một dòng diff nào → paper vẫn cite
"v2e [Hu et al. 2021]" sạch sẽ.

2 MODES
-------
  --mode lib  (DEFAULT)  16-bit precision, gọi EventEmulator trực tiếp.
                         Nếu import/API fail (version drift) → báo lỗi to,
                         kèm hướng dẫn; KHÔNG âm thầm rớt xuống 8-bit.
  --mode cli  (FALLBACK) chạy binary v2e.py gốc qua subprocess trên video
                         lossless 8-bit dựng từ TIFF. Chạy được ngay trong
                         mọi version, đổi lại mất precision dưới 8-bit.

DATA FLOW
---------
  linear luminance TIFFs (preprocess.py, 16-bit, 1280x720)
      │
      ├─[lib]─ N1 FRAMES → N2 LOAD f32(0..255) → N3 EMULATOR → N4 LOOP ─┐
      │                                                                  │
      └─[cli]─ C1 PNG8 → C2 ffmpeg lossless AVI → C3 v2e.py subprocess ─┤
                                                                         ▼
                                        N5 NORMALIZE (tự suy cột t,x,y,p)
                                                                         ▼
                                        N6 events.h5  (x,y,t[µs],p{0,1})
                                          + params.json (mọi tham số + git hash)

Usage (chạy TRONG env v2e):
  conda run -n v2e python run_v2e.py --input data/processed/site01 --output data/events_v2e/site01
  conda run -n v2e python run_v2e.py ... --limit 120        # smoke test
  conda run -n v2e python run_v2e.py ... --mode cli         # fallback 8-bit
"""

import sys
import json
import time
import inspect
import argparse
import subprocess
import shutil
from pathlib import Path

import numpy as np
import tifffile
import yaml

try:
    import h5py
except ImportError:
    h5py = None


# ══════════════════════════════════════════════════════════════════════════
#  NODE 0 · CONFIG
# ══════════════════════════════════════════════════════════════════════════

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def git_hash(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════
#  NODE 1 · FRAME LIST + TIMESTAMPS
# ══════════════════════════════════════════════════════════════════════════
#  in   : --input dir (output của preprocess.py), fps_original từ config
#  out  : (paths, t_us)  → N2/N4
#  what : liệt kê TIFF theo tên; timestamp t_us[i] = round(i · 1e6 / fps).
#  why  : fps PHẢI là fps_original (119.88) — capture thật — chứ không phải
#         fps_export (29.98) của DaVinci; sai cái này timestamp lệch 4×
#         (đúng cái bẫy đã ghi trong config.yaml).
# ══════════════════════════════════════════════════════════════════════════

def list_frames(input_dir: Path, fps: float, limit: int | None):
    paths = sorted(set(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))))
    if not paths:
        raise FileNotFoundError(f"Không thấy .tif/.tiff trong {input_dir}")
    if limit:
        paths = paths[:limit]
    t_us = np.array([round(i * 1e6 / fps) for i in range(len(paths))], dtype=np.int64)
    return paths, t_us


# ══════════════════════════════════════════════════════════════════════════
#  NODE 2 · LOAD 16-bit → float32 (0..255)
# ══════════════════════════════════════════════════════════════════════════
#  in   : path 1 TIFF 16-bit grayscale  ← N1
#  out  : frame f32 H×W, dải 0..255 NHƯNG có giá trị lẻ (sub-8bit)  → N4
#  what : chia 257.0 (65535/257 = 255.0 chính xác) để đổi thang 16-bit về
#         "đơn vị DN 0..255" mà model v2e (hàm lin_log, ngưỡng 20 DN) được
#         thiết kế quanh nó.
#  why  : đây là mấu chốt của cả file — giữ nguyên precision 16-bit (float,
#         không làm tròn) trong khi vẫn nói đúng "ngôn ngữ đơn vị" của v2e.
#         Ép uint8 mới là thứ giết dynamic range, còn float 0..255 thì không.
# ══════════════════════════════════════════════════════════════════════════

def load_frame_f32(path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim != 2:
        raise ValueError(f"{Path(path).name}: cần grayscale HxW, gặp shape {img.shape} "
                         "(đây phải là OUTPUT của preprocess.py, không phải TIFF DaVinci)")
    return img.astype(np.float32) / 257.0


# ══════════════════════════════════════════════════════════════════════════
#  NODE 3 · EMULATOR (lib mode)
# ══════════════════════════════════════════════════════════════════════════
#  in   : repo v2e, params từ config, seed  ← N0
#  out  : EventEmulator instance  → N4
#  what : import v2ecore.emulator.EventEmulator, lọc kwargs qua
#         inspect.signature để sống sót qua khác biệt version (arg nào
#         emulator không nhận thì bỏ + in ra cho m biết).
#  why  : KHÔNG sửa source v2e; mọi physics param đi qua constructor —
#         tức là qua config.yaml — nên paper report được đầy đủ.
#  VERIFY-1: tên module/class/method (v2ecore.emulator, EventEmulator,
#         generate_events(frame, t_seconds)) là theo docs/tutorial v2e.
#         Nếu ImportError/AttributeError → chạy `--mode cli` trước, rồi gửi
#         t output của:
#           conda run -n v2e python -c "import v2ecore.emulator as e; import inspect; print(inspect.signature(e.EventEmulator.__init__)); print([m for m in dir(e.EventEmulator) if 'event' in m.lower()])"
# ══════════════════════════════════════════════════════════════════════════

def build_emulator(repo: Path, v: dict, seed: int):
    sys.path.insert(0, str(repo))
    from v2ecore.emulator import EventEmulator          # VERIFY-1
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    want = dict(
        pos_thres=v["pos_thres"],
        neg_thres=v["neg_thres"],
        sigma_thres=v["sigma_thres"],
        cutoff_hz=v["cutoff_hz"],
        leak_rate_hz=v["leak_rate_hz"],
        shot_noise_rate_hz=v["shot_noise_rate_hz"],
        refractory_period_s=v.get("refractory_period", 1e-3),
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    sig = inspect.signature(EventEmulator.__init__)
    kwargs = {k: w for k, w in want.items() if k in sig.parameters}
    dropped = sorted(set(want) - set(kwargs))
    if dropped:
        print(f"  [lib] EventEmulator version này không nhận {dropped} → bỏ qua "
              "(nếu là param quan trọng, xem VERIFY-1 trong file)")
    print(f"  [lib] EventEmulator({', '.join(f'{k}={v}' for k, v in kwargs.items())})")
    return EventEmulator(**kwargs)


# ══════════════════════════════════════════════════════════════════════════
#  NODE 4 · SIMULATION LOOP (lib mode)
# ══════════════════════════════════════════════════════════════════════════
#  in   : emulator ← N3, frames+t_us ← N1/N2
#  out  : mảng event Nx4 (thứ tự cột do v2e quyết — N5 sẽ tự suy)  → N5
#  what : stream từng frame từ đĩa (KHÔNG preload — 7200 frame 720p float
#         ≈ 26GB RAM), gọi generate_events(frame, t_giây). Call đầu trả
#         None (frame baseline) — bình thường.
# ══════════════════════════════════════════════════════════════════════════

def run_lib(repo: Path, paths, t_us, v: dict, seed: int):
    em = build_emulator(repo, v, seed)
    chunks = []
    t0 = time.time()
    for i, (p, t) in enumerate(zip(paths, t_us)):
        fr = load_frame_f32(p)
        ev = em.generate_events(fr, float(t) / 1e6)     # VERIFY-1: t tính bằng GIÂY
        if ev is not None and len(ev) > 0:
            chunks.append(np.asarray(ev))
        if (i + 1) % 200 == 0 or (i + 1) == len(paths):
            r = (i + 1) / (time.time() - t0)
            print(f"  [lib] {i+1:>6}/{len(paths)}  {r:.1f} fr/s  "
                  f"events: {sum(len(c) for c in chunks):,}")
    if not chunks:
        raise RuntimeError("v2e không sinh event nào — input phẳng quá hoặc threshold cao quá?")
    return np.concatenate(chunks, axis=0)


# ══════════════════════════════════════════════════════════════════════════
#  C1–C4 · CLI FALLBACK (8-bit, chạy binary v2e.py gốc)
# ══════════════════════════════════════════════════════════════════════════
#  what : C1 quantize TIFF→PNG 8-bit (MẤT precision — fallback thôi)
#         C2 ffmpeg đóng thành AVI lossless (FFV1) @ fps_original
#         C3 subprocess v2e.py với đầy đủ flags từ config (kèm seed)
#         C4 mở .h5 v2e xuất ra, lấy dataset Nx4 đầu tiên → N5
#  why  : đường này chỉ dùng flags CLI đã document → gần như chắc chắn chạy
#         được với mọi version v2e, làm lưới an toàn cho lib mode.
# ══════════════════════════════════════════════════════════════════════════

def run_cli(repo: Path, paths, v: dict, seed: int, fps: float,
            width: int, height: int, workdir: Path):
    import cv2
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Thiếu ffmpeg (sudo apt install ffmpeg) — cần cho cli mode.")

    # C1 — PNG 8-bit
    png_dir = workdir / "png8"
    png_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [cli] C1: quantize {len(paths)} frames → PNG 8-bit (mất sub-8bit precision)")
    for i, p in enumerate(paths):
        u8 = np.clip(np.round(tifffile.imread(str(p)).astype(np.float32) / 257.0),
                     0, 255).astype(np.uint8)
        cv2.imwrite(str(png_dir / f"{i:06d}.png"), u8)

    # C2 — lossless video
    avi = workdir / "frames.avi"
    print(f"  [cli] C2: ffmpeg FFV1 @ {fps}fps → {avi.name}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", f"{fps}", "-i", str(png_dir / "%06d.png"),
                    "-c:v", "ffv1", str(avi)], check=True)

    # C3 — v2e.py (flags theo docs; nếu argparse chê flag lạ → python v2e.py -h)
    out_h5 = "events_v2e_raw.h5"
    cmd = [sys.executable, str(repo / "v2e.py"),
           "--input", str(avi),
           "--output_folder", str(workdir),
           "--input_frame_rate", f"{fps}",
           "--disable_slomo",
           "--pos_thres", str(v["pos_thres"]),
           "--neg_thres", str(v["neg_thres"]),
           "--sigma_thres", str(v["sigma_thres"]),
           "--cutoff_hz", str(v["cutoff_hz"]),
           "--leak_rate_hz", str(v["leak_rate_hz"]),
           "--shot_noise_rate_hz", str(v["shot_noise_rate_hz"]),
           "--refractory_period", str(v.get("refractory_period", 1e-3)),
           "--dvs_emulator_seed", str(seed),
           "--output_width", str(width),
           "--output_height", str(height),
           "--dvs_h5", out_h5,
           "--no_preview"]
    print("  [cli] C3: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))

    # C4 — đọc h5 của v2e
    if h5py is None:
        raise RuntimeError("pip install h5py trong env v2e")
    h5path = workdir / out_h5
    found = {}
    with h5py.File(h5path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2 \
               and obj.shape[1] == 4 and "arr" not in found:
                found["arr"], found["name"] = obj[...], name
        f.visititems(visit)
    if "arr" not in found:
        raise RuntimeError(f"Không thấy dataset Nx4 trong {h5path} — gửi t `h5ls -r {h5path}`")
    print(f"  [cli] C4: dataset '{found['name']}'  ({len(found['arr']):,} events)")
    return found["arr"]


# ══════════════════════════════════════════════════════════════════════════
#  NODE 5 · NORMALIZE — tự suy cột (t,x,y,p) từ dải giá trị
# ══════════════════════════════════════════════════════════════════════════
#  in   : Nx4, thứ tự cột KHÔNG biết trước (tùy version/simulator)
#  out  : dict x(u16), y(u16), t(u64 µs, sorted), p(u8 0/1)  → N6
#  what : p = cột chỉ chứa {-1,0,1};  t = cột đơn điệu tăng, range lớn nhất;
#         x/y = 2 cột còn lại (cột nào max ≥ height thì là x, vì 1280 > 720).
#         Đơn vị t: nếu range << duration_kỳ_vọng(µs) → đang là GIÂY → ×1e6.
#  why  : v2e trả (t,x,y,p) t=giây; file txt DVS-Voltmeter fmt %1.0f;
#         thay vì TIN tài liệu từng version, suy từ chính data → driver
#         sống sót qua version drift. In kết quả suy ra cho m soi.
#  NOTE : cùng một hàm này được copy sang run_dvsvolt.py — CỐ Ý duplicate
#         để mỗi file tự chạy độc lập không cần module chung.
# ══════════════════════════════════════════════════════════════════════════

def normalize_events(arr, width: int, height: int, dur_us_hint: float):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"Cần Nx4, gặp {a.shape}")
    cols = set(range(4))

    p_col = next((c for c in cols
                  if set(np.unique(a[:, c]).astype(np.int64).tolist()) <= {-1, 0, 1}), None)
    if p_col is None:
        raise ValueError("Không tìm được cột polarity (giá trị ngoài {-1,0,1})")
    cols.discard(p_col)

    t_col, best = None, -1.0
    for c in cols:
        d = np.diff(a[:, c])
        if d.size == 0 or d.min() >= 0:
            r = a[:, c].max() - a[:, c].min()
            if r > best:
                best, t_col = r, c
    if t_col is None:                                   # không cột nào monotonic (lạ)
        t_col = max(cols, key=lambda c: a[:, c].max())
    cols.discard(t_col)

    c1, c2 = sorted(cols)
    if a[:, c1].max() >= height > a[:, c2].max():
        x_col, y_col = c1, c2
    elif a[:, c2].max() >= height > a[:, c1].max():
        x_col, y_col = c2, c1
    else:
        x_col, y_col = c1, c2                           # mơ hồ → giữ thứ tự, tự soi log

    t = a[:, t_col].copy()
    unit = "µs"
    if dur_us_hint and (t.max() - t.min()) < dur_us_hint / 1000.0:
        t *= 1e6
        unit = "s→µs"

    p = np.where(a[:, p_col] > 0, 1, 0).astype(np.uint8)
    order = np.argsort(t, kind="stable")

    print(f"  [norm] suy cột: t=c{t_col}({unit})  x=c{x_col}  y=c{y_col}  p=c{p_col}"
          f"   |  {len(t):,} events, {t.max()/1e6 - t.min()/1e6:.2f}s")
    return dict(x=a[order, x_col].astype(np.uint16),
                y=a[order, y_col].astype(np.uint16),
                t=t[order].astype(np.uint64),
                p=p[order])


# ══════════════════════════════════════════════════════════════════════════
#  NODE 6 · WRITE UNIFIED H5
# ══════════════════════════════════════════════════════════════════════════
#  in   : dict từ N5 + metadata
#  out  : events.h5 (schema GIỐNG read_evt3.py: x,y,t,p gzip) + params.json
#  why  : v2e / DVS-Voltmeter / event thật từ Triton2 → cùng MỘT format,
#         code train + code so sánh sim-real chỉ viết một đường đọc.
#         attrs chứa đủ params + git hash → mỗi file tự khai nguồn gốc.
# ══════════════════════════════════════════════════════════════════════════

def write_h5(out_dir: Path, ev: dict, attrs: dict):
    if h5py is None:
        raise RuntimeError("pip install h5py")
    out_dir.mkdir(parents=True, exist_ok=True)
    h5path = out_dir / "events.h5"
    with h5py.File(h5path, "w") as f:
        for k in ("x", "y", "t", "p"):
            f.create_dataset(k, data=ev[k], compression="gzip")
        f.attrs["n_events"] = len(ev["t"])
        for k, v in attrs.items():
            f.attrs[k] = v if isinstance(v, (int, float, str)) else json.dumps(v)
    (out_dir / "params.json").write_text(json.dumps(attrs, indent=2))
    print(f"  [save] {h5path}  ({len(ev['t']):,} events)  + params.json")
    return h5path


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="CAROECT-D: linear TIFFs → v2e events (unified h5)")
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["lib", "cli"], default="lib",
                    help="lib = 16-bit qua EventEmulator (default) | cli = v2e.py gốc, 8-bit")
    ap.add_argument("--limit", type=int, default=None, help="chỉ N frame đầu (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cam, v, sim = cfg["camera"], cfg["v2e"], cfg.get("simulator", {})
    seed = int(sim.get("seed", 42))
    fps = float(cam["fps_original"])                     # 119.88 — KHÔNG phải fps_export
    W, H = int(cam["width"]), int(cam["height"])
    repo = Path(cfg["paths"].get("v2e_repo", "~/caroect_sim/v2e")).expanduser()
    if not repo.exists():
        raise FileNotFoundError(f"v2e repo chưa có ở {repo} — chạy ./setup_sim.sh trước")

    in_dir, out_dir = Path(args.input), Path(args.output)
    paths, t_us = list_frames(in_dir, fps, args.limit)
    dur_us = float(len(paths)) * 1e6 / fps

    print(f"\n{'━'*60}\n  v2e driver  |  mode={args.mode}  |  {len(paths)} frames "
          f"({dur_us/1e6:.2f}s @ {fps}fps)\n  {in_dir} → {out_dir}\n{'━'*60}")

    if args.mode == "lib":
        raw = run_lib(repo, paths, t_us, v, seed)
    else:
        raw = run_cli(repo, paths, v, seed, fps, W, H, out_dir / "_work_cli")

    ev = normalize_events(raw, W, H, dur_us)
    attrs = dict(simulator="v2e", mode=args.mode, git_commit=git_hash(repo),
                 seed=seed, fps=fps, width=W, height=H,
                 source=str(in_dir), params=v)
    write_h5(out_dir, ev, attrs)
    print(f"\n✓ Xong. So sánh với event thật:  python read_evt3.py --input <site>.raw --stats\n")


if __name__ == "__main__":
    main()
