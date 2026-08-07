#!/usr/bin/env python3
"""
CAROECT-D — DVS-Voltmeter driver (16-bit path, KHÔNG sửa source repo)
=====================================================================

WHY THIS FILE EXISTS
--------------------
main.py gốc của DVS-Voltmeter đọc ảnh bằng cv2.IMREAD_GRAYSCALE → ép về
8-bit, giết dynamic range 16-bit linear của preprocess. File này THAY THẾ
main.py của nó (chứ không sửa): tự load 16-bit, gọi thẳng class simulator
trong src/simulator.py. Physics (mô hình Brownian-motion-with-drift, tham số
k1..k6) giữ nguyên 100% → paper cite "DVS-Voltmeter [Lin et al. ECCV 2022]".

2 MODES
-------
  --mode lib  (DEFAULT)  16-bit precision + seed được (reproducible).
                         API của repo là research-code không document —
                         driver tự dò class/method (VERIFY-2/3); nếu dò
                         fail → lỗi to kèm lệnh để m gửi t signature.
  --mode cli  (FALLBACK) chạy main.py GỐC qua subprocess trên PNG 8-bit
                         + info.txt đúng format của nó. Chắc chắn chạy,
                         nhưng 8-bit và KHÔNG seed được (main gốc không
                         có seed — xem notes).

DATA FLOW
---------
  linear luminance TIFFs (16-bit, 1280x720, 119.88fps)
      │
      ├─[lib]─ N1 FRAMES → N2 LOAD f32(0..255) → N3 IMPORT SIM → N4 LOOP ─┐
      │                                                                    │
      └─[cli]─ C1 PNG8 + info.txt → C2 main.py subprocess → C3 gom .txt ──┤
                                                                           ▼
                                          N5 NORMALIZE (tự suy cột t,x,y,p)
                                                                           ▼
                                          N6 events.h5 (x,y,t[µs],p{0,1})

TỐC ĐỘ (đọc trước khi than máy chậm):
  Repo gốc chạy CPU. Paper đo ~15.5 ms/cặp frame ở 346×260 → ở 1280×720
  (~10.2× pixel) ≈ 160 ms/cặp → clip 60s @119.88fps (7192 cặp) ≈ 19 PHÚT.
  Cách chữa KHÔNG cần sửa source: chạy nhiều clip song song (mỗi clip 1
  process), vì các clip độc lập hoàn toàn.

Usage (chạy TRONG env dvsvolt):
  conda run -n dvsvolt python run_dvsvolt.py --input data/processed/site01 --output data/events_dvsvolt/site01
  conda run -n dvsvolt python run_dvsvolt.py ... --limit 120    # smoke test
  conda run -n dvsvolt python run_dvsvolt.py ... --mode cli     # fallback
"""
from __future__ import annotations
# ^ BẮT BUỘC cho env dvsvolt (Python 3.9, cố tình giữ cũ để tương thích
#   OpenCV 4.5.1). list_frames() dùng "int | None" (PEP 604 union), chỉ
#   chạy native được từ Python 3.10+. Dòng import này khiến type hint được
#   coi là STRING (không đánh giá lúc chạy) — tương thích ngược tới 3.7.

import sys
import json
import time
import inspect
import argparse
import importlib
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
#  NODE 0 · CONFIG  (+ helpers dùng chung — copy có chủ đích từ run_v2e.py)
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
#  NODE 1 · FRAME LIST + TIMESTAMPS (µs)
# ══════════════════════════════════════════════════════════════════════════
#  DVS-Voltmeter nói chuyện bằng MICRO-GIÂY (info.txt của nó là µs) →
#  t_us[i] = round(i · 1e6 / 119.88). fps_original, không phải fps_export.
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
#  ⚠ ĐƠN VỊ QUAN TRỌNG: k1..k6 được calibrate trên ảnh 8-bit (DN 0..255).
#  Trong model, L xuất hiện ở μ = k1/(L+k2)·k_dL + k4 + k5·L và
#  σ = k3/(L+k2)·√L + k6 — tức là GIÁ TRỊ TUYỆT ĐỐI của L có nghĩa.
#  Nếu đút thẳng 0..65535 vào, toàn bộ k sai thang → physics sai.
#  Chia 257.0 giữ nguyên precision 16-bit (float lẻ) nhưng đúng thang DN
#  mà bộ k DVS346/DVS240 được sinh ra cho.
# ══════════════════════════════════════════════════════════════════════════

def load_frame_f32(path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim != 2:
        raise ValueError(f"{Path(path).name}: cần grayscale HxW (output preprocess.py), "
                         f"gặp shape {img.shape}")
    return img.astype(np.float32) / 257.0


# ══════════════════════════════════════════════════════════════════════════
#  NODE 3 · IMPORT SIMULATOR (lib mode) — dò class trong repo research-code
# ══════════════════════════════════════════════════════════════════════════
#  what : import src.simulator (VERIFY-2), tìm class (ưu tiên tên EventSim),
#         construct với EasyDict cfg giống config.py của nó (SENSOR.K,
#         SENSOR.CAMERA_TYPE) — thử vài chữ ký constructor phổ biến.
#  why  : repo nghiên cứu không có API ổn định; dò + fail-loud tốt hơn là
#         hardcode đại rồi chết khó hiểu.
#  VERIFY-2: nếu lỗi ở đây → gửi t output của:
#    conda run -n dvsvolt python - <<'PY'
#    import sys; sys.path.insert(0, "<repo>")
#    import src.simulator as s, inspect
#    print([n for n,o in vars(s).items() if inspect.isclass(o)])
#    print(inspect.signature([o for n,o in vars(s).items() if inspect.isclass(o)][0].__init__))
#    PY
# ══════════════════════════════════════════════════════════════════════════

def import_sim_class(repo: Path):
    sys.path.insert(0, str(repo))
    mod = None
    for name in ("src.simulator", "simulator", "src.event_sim"):
        try:
            mod = importlib.import_module(name)
            break
        except ImportError:
            continue
    if mod is None:
        raise RuntimeError(f"Không import được simulator từ {repo} — repo đã clone chưa? "
                           "(./setup_sim.sh)  Nếu rồi: gửi t output `ls " + str(repo) + "/src`")
    for cn in ("EventSim", "Simulator", "EventSimulator", "DVSVoltmeter"):
        if hasattr(mod, cn):
            return mod, getattr(mod, cn)
    own = [o for n, o in vars(mod).items()
           if inspect.isclass(o) and o.__module__ == mod.__name__]
    if len(own) == 1:
        return mod, own[0]
    raise RuntimeError(f"Nhiều class trong {mod.__name__}: {[c.__name__ for c in own]} "
                       "— xem VERIFY-2 trong file này")


def construct_sim(cls, K, camera_type, width, height):
    from easydict import EasyDict
    cfg = EasyDict(dict(
        SENSOR=dict(CAMERA_TYPE=camera_type, K=list(map(float, K))),
        DIR=dict(IN_PATH="", OUT_PATH=""),
        Width=width, Height=height,
    ))
    trials = [(cfg,), (list(map(float, K)),), (list(map(float, K)), camera_type), tuple()]
    last = None
    for args in trials:
        try:
            sim = cls(*args)
            print(f"  [lib] {cls.__name__}{inspect.signature(cls.__init__)} ← khớp {len(args)} arg")
            return sim
        except TypeError as e:
            last = e
    raise RuntimeError(f"Constructor {cls.__name__} không khớp cách gọi nào "
                       f"(lỗi cuối: {last}) — xem VERIFY-2")


# ══════════════════════════════════════════════════════════════════════════
#  NODE 4 · SIMULATION LOOP (lib mode) — dò chữ ký method 1 lần, rồi stream
# ══════════════════════════════════════════════════════════════════════════
#  what : model của nó tích phân GIỮA 2 frame → chữ ký tự nhiên là
#         (f_prev, f_cur, t_prev, t_cur). Một số bản viết kiểu stream
#         (f, t) có state bên trong. Dò cả hai ở cặp đầu (VERIFY-3),
#         khóa lại, rồi stream từ đĩa (không preload — 26GB RAM đó m).
#  seed : np.random.seed + torch.manual_seed TRƯỚC khi sim — cái mà main.py
#         gốc không làm được (điểm cộng reproducibility của lib mode).
# ══════════════════════════════════════════════════════════════════════════

def _resolve_caller(sim, f0, f1, t0, t1):
    for m in ("generate_events", "simulate", "__call__", "run"):
        fn = getattr(sim, m, None)
        if fn is None or not callable(fn):
            continue
        try:
            ev = fn(f0, f1, int(t0), int(t1))
            print(f"  [lib] method: .{m}(f_prev, f_cur, t_prev, t_cur)")
            return ("pair", fn, ev)
        except TypeError:
            pass
        try:
            fn(f0, int(t0))                      # nạp baseline
            ev = fn(f1, int(t1))
            print(f"  [lib] method: .{m}(frame, t)  [stream/stateful]")
            return ("stream", fn, ev)
        except TypeError:
            continue
    raise RuntimeError("Không khớp được method sinh event nào — xem VERIFY-2 "
                       "(gửi t signature là t khớp trong 1 phút)")


def run_lib(repo: Path, paths, t_us, K, camera_type, width, height, seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    _, cls = import_sim_class(repo)
    sim = construct_sim(cls, K, camera_type, width, height)

    chunks, style, fn = [], None, None
    prev = load_frame_f32(paths[0])
    t0w = time.time()
    for i in range(1, len(paths)):
        cur = load_frame_f32(paths[i])
        if fn is None:
            style, fn, ev = _resolve_caller(sim, prev, cur, t_us[i-1], t_us[i])
        elif style == "pair":
            ev = fn(prev, cur, int(t_us[i-1]), int(t_us[i]))
        else:
            ev = fn(cur, int(t_us[i]))
        if ev is not None and len(ev) > 0:
            chunks.append(np.asarray(ev))
        prev = cur
        if i % 200 == 0 or i == len(paths) - 1:
            r = i / (time.time() - t0w)
            eta = (len(paths) - 1 - i) / r if r > 0 else 0
            print(f"  [lib] {i:>6}/{len(paths)-1} cặp  {r:.1f} cặp/s  ETA {eta/60:.1f} phút  "
                  f"events: {sum(len(c) for c in chunks):,}")
    if not chunks:
        raise RuntimeError("DVS-Voltmeter không sinh event nào — kiểm tra input/k params")
    return np.concatenate(chunks, axis=0)


# ══════════════════════════════════════════════════════════════════════════
#  C1–C3 · CLI FALLBACK — chạy main.py GỐC, không đụng 1 dòng
# ══════════════════════════════════════════════════════════════════════════
#  C1: xuất PNG 8-bit + info.txt ("<abs_path> <t_us>") theo layout
#      input_dir/<session>/  mà main.py gốc duyệt (VERIFY-4: nếu output
#      rỗng → thử --input_dir trỏ THẲNG vào folder session, driver tự in
#      gợi ý).
#  C2: subprocess main.py với các flag ĐÃ XÁC NHẬN từ source:
#      --input_dir --output_dir --camera_type --model_para k1..k6
#  C3: gom mọi *.txt nó xuất (np.savetxt fmt='%1.0f') → concat → N5.
#  ⚠ main gốc KHÔNG có seed → cli mode không reproducible bит-perfect.
# ══════════════════════════════════════════════════════════════════════════

def run_cli(repo: Path, paths, t_us, K, camera_type, workdir: Path, env_name: str):
    import cv2
    session = "session"
    in_dir = workdir / "in" / session
    out_dir = workdir / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [cli] C1: {len(paths)} frames → PNG 8-bit + info.txt (mất sub-8bit precision)")
    lines = []
    for i, (p, t) in enumerate(zip(paths, t_us)):
        u8 = np.clip(np.round(tifffile.imread(str(p)).astype(np.float32) / 257.0),
                     0, 255).astype(np.uint8)
        fp = in_dir / f"{i:06d}.png"
        cv2.imwrite(str(fp), u8)
        lines.append(f"{fp.resolve()} {int(t)}")
    (in_dir / "info.txt").write_text("\n".join(lines) + "\n")

    conda = shutil.which("conda") or "conda"
    cmd = [conda, "run", "-n", env_name, "python", "main.py",
           "--input_dir", str((workdir / "in").resolve()),
           "--output_dir", str(out_dir.resolve()),
           "--camera_type", camera_type,
           "--model_para"] + [str(float(k)) for k in K]
    print("  [cli] C2: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))

    txts = sorted(out_dir.rglob("*.txt"))
    if not txts:
        raise RuntimeError("main.py không xuất .txt nào. VERIFY-4: thử sửa --input_dir "
                           f"trỏ thẳng {workdir/'in'/session} rồi chạy lại; hoặc gửi t "
                           "output `ls -R` của workdir để t khớp layout.")
    arrs = [np.loadtxt(str(t)) for t in txts]
    arrs = [a.reshape(-1, 4) for a in arrs if a.size]
    print(f"  [cli] C3: gom {len(txts)} file .txt → {sum(len(a) for a in arrs):,} events")
    return np.concatenate(arrs, axis=0)


# ══════════════════════════════════════════════════════════════════════════
#  NODE 5 · NORMALIZE — (bản sao có chủ đích từ run_v2e.py, xem note bên đó)
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
    if t_col is None:
        t_col = max(cols, key=lambda c: a[:, c].max())
    cols.discard(t_col)

    c1, c2 = sorted(cols)
    if a[:, c1].max() >= height > a[:, c2].max():
        x_col, y_col = c1, c2
    elif a[:, c2].max() >= height > a[:, c1].max():
        x_col, y_col = c2, c1
    else:
        x_col, y_col = c1, c2

    t = a[:, t_col].copy()
    unit = "µs"
    if dur_us_hint and (t.max() - t.min()) < dur_us_hint / 1000.0:
        t *= 1e6
        unit = "s→µs"

    p = np.where(a[:, p_col] > 0, 1, 0).astype(np.uint8)
    order = np.argsort(t, kind="stable")

    print(f"  [norm] suy cột: t=c{t_col}({unit})  x=c{x_col}  y=c{y_col}  p=c{p_col}"
          f"   |  {len(t):,} events, {(t.max()-t.min())/1e6:.2f}s")
    x = np.clip(np.rint(a[order, x_col]), 0, width - 1).astype(np.uint16)
    y = np.clip(np.rint(a[order, y_col]), 0, height - 1).astype(np.uint16)
    return dict(x=x,
                y=y,
                t=t[order].astype(np.uint64),
                p=p[order])


# ══════════════════════════════════════════════════════════════════════════
#  NODE 6 · WRITE UNIFIED H5 (schema = read_evt3.py: x,y,t,p + attrs)
# ══════════════════════════════════════════════════════════════════════════

def write_h5(out_dir: Path, ev: dict, attrs: dict):
    if h5py is None:
        raise RuntimeError("pip install h5py trong env dvsvolt")
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
    ap = argparse.ArgumentParser(description="CAROECT-D: linear TIFFs → DVS-Voltmeter events (unified h5)")
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["lib", "cli"], default="lib",
                    help="lib = 16-bit + seed (default) | cli = main.py gốc, 8-bit, no seed")
    ap.add_argument("--limit", type=int, default=None, help="chỉ N frame đầu (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cam, dv, sim = cfg["camera"], cfg["dvs_voltmeter"], cfg.get("simulator", {})
    seed = int(sim.get("seed", 42))
    fps = float(cam["fps_original"])
    W, H = int(cam["width"]), int(cam["height"])
    K = dv["k"]
    camera_type = dv.get("camera_type", "DVS346")
    if len(K) != 6:
        raise ValueError(f"dvs_voltmeter.k phải đúng 6 số (k1..k6), đang có {len(K)}")
    repo = Path(cfg["paths"].get("dvsvolt_repo", "~/caroect_sim/DVS-Voltmeter")).expanduser()
    if not repo.exists():
        raise FileNotFoundError(f"DVS-Voltmeter repo chưa có ở {repo} — chạy ./setup_sim.sh trước")

    in_dir, out_dir = Path(args.input), Path(args.output)
    paths, t_us = list_frames(in_dir, fps, args.limit)
    dur_us = float(len(paths)) * 1e6 / fps

    print(f"\n{'━'*60}\n  DVS-Voltmeter driver  |  mode={args.mode}  |  {camera_type} "
          f"k={K}\n  {len(paths)} frames ({dur_us/1e6:.2f}s @ {fps}fps)   "
          f"{in_dir} → {out_dir}\n{'━'*60}")

    if args.mode == "lib":
        raw = run_lib(repo, paths, t_us, K, camera_type, W, H, seed)
    else:
        env_name = sim.get("envs", {}).get("dvsvolt", "dvsvolt")
        raw = run_cli(repo, paths, t_us, K, camera_type, out_dir / "_work_cli", env_name)

    ev = normalize_events(raw, W, H, dur_us)
    attrs = dict(simulator="dvs_voltmeter", mode=args.mode, git_commit=git_hash(repo),
                 seed=(seed if args.mode == "lib" else "N/A (cli mode không seed được)"),
                 fps=fps, width=W, height=H, camera_type=camera_type,
                 source=str(in_dir), params=dict(k=list(map(float, K))))
    write_h5(out_dir, ev, attrs)
    print(f"\n✓ Xong. Chạy nhiều clip song song để bù tốc độ CPU (xem header file).\n")


if __name__ == "__main__":
    main()
