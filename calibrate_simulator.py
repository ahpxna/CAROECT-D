#!/usr/bin/env python3
"""
calibrate_simulator.py — Script NGOÀI để calibrate v2e / DVS-Voltmeter theo event thật
=========================================================================================

TẠI SAO FILE NÀY TỒN TẠI
-------------------------
v2e và DVS-Voltmeter đều là simulator THUẦN TÚY: cho video/tham số vào, chúng sinh
event giả — bản thân chúng không biết, và không có cơ chế nào để tự biết, event đó
có giống event camera thật hay không. So khớp và tinh chỉnh tham số là việc của một
lớp NGOÀI 2 repo — đúng nguyên tắc "Loại 1 — physics: KHÔNG fork" đã chốt trong scope
(sửa k1..k6 hay pos_thres/neg_thres là chỉnh THAM SỐ đầu vào của model đã công bố,
không phải sửa code model — nên không tốn thêm 1 paper con nào để bảo vệ).

INPUT CẦN CÓ TRƯỚC
-------------------
  1. events_real.h5   — từ cevt_to_events.py, ghi bằng evs_recorder.cpp, quay CÙNG
                         cảnh CÙNG lúc với clip TIFF ở mục 2 (rig Nikon+Triton2 gắn
                         cứng, side-by-side — điều kiện bắt buộc để so sánh có nghĩa).
  2. 1 thư mục TIFF    — output nhánh Y (16-bit linear) của preprocess.py, cùng cảnh
                         cùng lúc với (1).

QUY TRÌNH (coordinate descent — CHỈNH 1 THAM SỐ MỘT LẦN, đúng khuyến nghị 2 paper:
DVS-Voltmeter nói k1 nhạy nhất nên chỉnh trước, rồi k2; v2e thì pos/neg_thres trước)
-----------------------------------------------------------------------------------
  Với mỗi giá trị ứng viên trong --search:
    1. Tạo config.yaml tạm, chỉ đổi đúng 1 tham số --param, giữ mọi thứ khác nguyên.
    2. Gọi run_dvsvolt.py (hoặc run_v2e.py khi có) trên clip TIFF, --limit N khung
       hình đầu (mặc định 120 ~1s ở 119.88fps) để search nhanh, không chờ 19 phút/clip.
    3. Cắt events_real.h5 về đúng cùng khoảng thời gian [0, N/fps] để so sánh công bằng
       (giả định 2 camera bắt đầu ghi cùng lúc — đúng bản chất rig side-by-side).
    4. Tính sai số = |log(rate_sim / rate_real)| + 0.5 * |on_frac_sim - on_frac_real|
       (log-ratio cho rate vì lệch bậc quan trọng hơn lệch tuyến tính; ON/OFF split
       là tín hiệu phụ, trọng số thấp hơn).
  In bảng candidate -> sai số, đề xuất giá trị tốt nhất. KHÔNG tự ghi đè config.yaml —
  phải chạy lại với --apply mới ghi (có backup .bak trước khi ghi).

GIỚI HẠN ĐÃ BIẾT — ĐỌC TRƯỚC KHI TIN KẾT QUẢ
----------------------------------------------
Nếu events_real.h5 có attrs["decode_method_counts"] cho thấy phần lớn record là
"dense" (không phải "xypt" — xem cevt_to_events.py), thì timestamp trong file đó là
BỊA từ frame_index, không phải thời điểm event thật. Trong trường hợp đó:
  - Metric RATE và ON/OFF SPLIT (dùng ở đây) vẫn tin được — chỉ cần đếm event.
  - Không tin được PHÂN PHỐI khoảng-cách-thời-gian (τ) giữa các event — nên script
    này CHỈ tune nhóm tham số ảnh hưởng SỐ LƯỢNG event: k1,k2,k4,k5 (DVS-Voltmeter)
    hoặc pos_thres,neg_thres (v2e). KHÔNG dùng script này để tune k3,k6 (nhóm
    noise/jitter thời gian) — cần XYPT thật (bật mặc định ở evs_recorder từ giờ).
Nếu decode_method_counts cho thấy "xypt" chiếm đa số, script in thêm so sánh phân
phối inter-event-interval (Wasserstein 1D, công thức min-cost 1 chiều — xem PDF scope
mục Q2) để bạn có căn cứ chỉnh k3/k6 bằng tay (chưa tự động hoá bước đó).

TÙY CHỌN — OPTUNA (tự động hoá search thay vì tự liệt kê --search)
--------------------------------------------------------------------
⚠ DEPENDENCY MỚI, CHƯA CÓ Ở ĐÂU KHÁC TRONG PROJECT: `pip install optuna` — cài
trong env đang chạy CHÍNH FILE NÀY (host/base env), KHÔNG phải trong env
`dvsvolt`/`v2e` (2 env đó chỉ được gọi qua subprocess `conda run`, không cần
optuna). Chỉ cần cài nếu dùng cờ --optuna; nếu không, code chạy y như cũ với
--search (không import optuna, không lỗi thiếu package).

Chỗ dùng optuna trong file này (tìm bằng cách grep "OPTUNA:" trong code):
  - `run_optuna_search()` — tạo 1 optuna.study, sampler mặc định TPE
    (Tree-structured Parzen Estimator — Bayesian, học từ trial trước để chọn
    trial sau, hiệu quả hơn grid rời rạc khi cần dò một khoảng liên tục).
  - `objective(trial)` closure bên trong nó — mỗi trial gọi lại ĐÚNG
    `run_candidate()` + `score_candidate()` đã dùng cho --search, nên 2 chế
    độ (grid thủ công vs optuna tự động) cho kết quả CÙNG một thang đo, so
    được với nhau.
  - Kết quả `study.best_params[param]` / `study.best_value` được in ra bảng
    CÙNG FORMAT với --search để không phải học đọc kết quả kiểu mới.

Dùng --optuna khi: muốn dò 1 khoảng liên tục (vd pos_thres trong [0.1, 0.4])
mà không biết chia mốc nào hợp lý. Dùng --search khi: đã có vài giá trị nghi
ngờ cụ thể (vd preset DVS346 vs DVS240) muốn so trực tiếp — --search KHÔNG bị
thay thế, vẫn là default.

Usage:
  # Grid thủ công (mặc định, không cần cài optuna)
  python calibrate_simulator.py --real events_real.h5 --sim-input data/processed/site01 \\
      --simulator dvsvolt --param k1 --search 3.0 4.0 5.3 6.5 8.0 --limit 120

  python calibrate_simulator.py --real events_real.h5 --sim-input data/processed/site01 \\
      --simulator dvsvolt --param k1 --search 3.0 4.0 5.3 6.5 8.0 --apply   # ghi lại config.yaml

  python calibrate_simulator.py --real events_real.h5 --sim-input data/processed/site01 \\
      --simulator v2e --param pos_thres --search 0.15 0.20 0.25 0.30 --limit 120

  # Optuna tự động (cần: pip install optuna)
  python calibrate_simulator.py --real events_real.h5 --sim-input data/processed/site01 \\
      --simulator v2e --param pos_thres --optuna --low 0.10 --high 0.40 --n-trials 25
"""

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    import h5py
except ImportError:
    h5py = None

from measure_event_rate import load_events_h5, analyze  # reuse, don't duplicate


# ══════════════════════════════════════════════════════════════════════════
#  Tham số hợp lệ theo simulator + tọa độ trong config.yaml
# ══════════════════════════════════════════════════════════════════════════

DVSVOLT_K_INDEX = {"k1": 0, "k2": 1, "k3": 2, "k4": 3, "k5": 4, "k6": 5}
DVSVOLT_RATE_SAFE = {"k1", "k2", "k4", "k5"}   # ảnh hưởng SỐ LƯỢNG event — tin được cả khi t bịa
DVSVOLT_TIMING_ONLY = {"k3", "k6"}              # ảnh hưởng NOISE/JITTER thời gian — cần XYPT thật

V2E_RATE_SAFE = {"pos_thres", "neg_thres", "refractory_period"}
V2E_TIMING_ONLY = {"sigma_thres", "cutoff_hz", "leak_rate_hz", "shot_noise_rate_hz"}

RUNNERS = {
    # simulator -> (module/script name, conda env key trong config.simulator.envs)
    "dvsvolt": ("run_dvsvolt.py", "dvsvolt"),
    "v2e":     ("run_v2e.py", "v2e"),
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════
#  N1 · Chuẩn bị 1 config tạm với đúng 1 tham số bị đổi
# ══════════════════════════════════════════════════════════════════════════

def make_candidate_config(base_cfg: dict, simulator: str, param: str, value: float) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if simulator == "dvsvolt":
        idx = DVSVOLT_K_INDEX[param]
        cfg["dvs_voltmeter"]["k"][idx] = float(value)
    elif simulator == "v2e":
        cfg["v2e"][param] = float(value)
    else:
        raise ValueError(f"Unknown simulator: {simulator}")
    return cfg


# ══════════════════════════════════════════════════════════════════════════
#  N2 · Chạy 1 candidate — subprocess conda run -n <env> python run_*.py
# ══════════════════════════════════════════════════════════════════════════

def run_candidate(simulator: str, sim_input: Path, work_dir: Path, cfg: dict,
                   limit: int, tag: str) -> Path:
    script, env_key = RUNNERS[simulator]
    if not Path(script).exists():
        raise FileNotFoundError(
            f"{script} không tồn tại trong thư mục hiện tại — chưa viết file này thì "
            f"chưa calibrate được simulator='{simulator}'. Các simulator khác vẫn dùng "
            f"được script này bình thường.")

    cand_dir = work_dir / tag
    cand_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cand_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    out_dir = cand_dir / "out"
    env_name = cfg.get("simulator", {}).get("envs", {}).get(env_key, env_key)
    conda = shutil.which("conda") or "conda"
    cmd = [conda, "run", "-n", env_name, "python", script,
           "--input", str(sim_input), "--output", str(out_dir),
           "--config", str(cfg_path), "--limit", str(limit)]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"{script} thất bại cho candidate {tag} (xem log trên).")

    h5path = out_dir / "events.h5"
    if not h5path.exists():
        raise FileNotFoundError(f"{script} chạy xong nhưng không thấy {h5path}")
    return h5path


# ══════════════════════════════════════════════════════════════════════════
#  N3 · So sánh sim vs real trong cùng cửa sổ thời gian
# ══════════════════════════════════════════════════════════════════════════

def slice_real_to_window(real_h5: Path, duration_s: float, sensor_w: int, sensor_h: int):
    """Cắt events_real.h5 về [t_min, t_min + duration_s] rồi trả về cùng dict thống
    kê mà analyze() trả — tự viết một bản ghi tạm .h5 để tái dùng analyze() nguyên
    vẹn thay vì viết lại logic thống kê (tránh 2 nơi tính rate khác công thức)."""
    ev = load_events_h5(real_h5)
    if len(ev["t"]) == 0:
        raise ValueError(f"{real_h5} rỗng — không so sánh được.")
    t0 = ev["t"].min()
    mask = ev["t"] <= (t0 + duration_s * 1e6)
    sliced = {k: v[mask] for k, v in ev.items()}

    tmp_path = real_h5.parent / f"_tmp_slice_{int(duration_s*1000)}ms.h5"
    with h5py.File(tmp_path, "w") as hf:
        for k in ("x", "y", "t", "p"):
            hf.create_dataset(k, data=sliced[k])
    stats = analyze(str(tmp_path), "real(sliced)", sensor_w, sensor_h, window_s=max(0.1, duration_s / 5))
    tmp_path.unlink(missing_ok=True)
    if stats is None:
        raise ValueError("Cửa sổ real quá ngắn/rỗng sau khi cắt — kiểm tra duration_s và "
                          "offset thời gian giữa 2 camera (có thể chưa bắt đầu ghi cùng lúc).")
    return stats


def run_optuna_search(simulator: str, param: str, low: float, high: float, n_trials: int,
                       base_cfg: dict, sim_input: Path, work_dir: Path, limit: int,
                       W: int, H: int, duration_s: float, real_stats: dict):
    """OPTUNA: thay vòng lặp --search cố định bằng Bayesian search (TPE) trong
    khoảng liên tục [low, high]. Dùng LẠI run_candidate()/score_candidate() —
    cùng thang đo với chế độ --search, chỉ khác cách chọn giá trị tiếp theo.
    Trả về list rows CÙNG format (value, rate_hz, on_frac, err) để phần in
    bảng/apply phía main() không cần biết grid hay optuna đã tạo ra nó."""
    try:
        import optuna  # OPTUNA: lazy import — chỉ cần khi --optuna được dùng
    except ImportError as e:
        raise ImportError(
            "Thiếu package optuna. Cài bằng: pip install optuna  (cài trong env đang "
            "chạy calibrate_simulator.py — KHÔNG phải env dvsvolt/v2e, 2 env đó chỉ được "
            "gọi qua subprocess).") from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # OPTUNA: khỏi spam log trial
    rows = []

    def objective(trial: "optuna.Trial") -> float:         # OPTUNA: hàm mục tiêu, 1 trial = 1 candidate
        value = trial.suggest_float(param, low, high)
        tag = f"optuna_trial{trial.number}"
        cfg = make_candidate_config(base_cfg, simulator, param, value)
        try:
            h5path = run_candidate(simulator, sim_input, work_dir, cfg, limit, tag)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [trial {trial.number}] [skip] {e}")
            raise optuna.TrialPruned()
        sim_stats = analyze(str(h5path), f"sim(trial{trial.number})", W, H,
                            window_s=max(0.1, duration_s / 5))
        if sim_stats is None:
            raise optuna.TrialPruned()
        err = score_candidate(sim_stats, real_stats)
        rows.append((value, sim_stats["rate_hz"], sim_stats["n_on"] / max(sim_stats["n"], 1), err))
        print(f"  [trial {trial.number}] {param}={value:.5f}  rate={sim_stats['rate_hz']:,.0f} ev/s  "
              f"error={err:.4f}")
        return err

    # OPTUNA: TPESampler = Bayesian, học phân bố trial tốt/xấu trước đó để đề
    # xuất trial sau — hiệu quả hơn random/grid khi n_trials nhỏ (search tốn
    # thời gian vì mỗi trial là 1 lần chạy conda run thật, không phải hàm rẻ).
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    print(f"\n[optuna] best {param} = {study.best_params[param]:.5f}  "
          f"(error={study.best_value:.4f}, {len(study.trials)} trial(s))")
    return rows


def score_candidate(sim_stats: dict, real_stats: dict) -> float:
    """Sai số thấp = khớp tốt. log-ratio cho rate (lệch bậc quan trọng hơn lệch
    tuyến tính), + 0.5x lệch ON/OFF split (tín hiệu phụ)."""
    rate_term = abs(np.log(max(sim_stats["rate_hz"], 1e-6) / max(real_stats["rate_hz"], 1e-6)))
    sim_on_frac = sim_stats["n_on"] / max(sim_stats["n"], 1)
    real_on_frac = real_stats["n_on"] / max(real_stats["n"], 1)
    onoff_term = abs(sim_on_frac - real_on_frac)
    return rate_term + 0.5 * onoff_term


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="events_real.h5 (từ cevt_to_events.py)")
    ap.add_argument("--sim-input", required=True,
                    help="Thư mục TIFF 16-bit (output nhánh Y của preprocess.py), cùng "
                         "cảnh cùng lúc với --real")
    ap.add_argument("--simulator", required=True, choices=["dvsvolt", "v2e"])
    ap.add_argument("--param", required=True,
                    help="dvsvolt: k1..k6 | v2e: pos_thres,neg_thres,sigma_thres,"
                         "cutoff_hz,leak_rate_hz,shot_noise_rate_hz")
    ap.add_argument("--search", type=float, nargs="+",
                    help="Danh sách giá trị ứng viên cho --param (default nếu không dùng --optuna)")
    ap.add_argument("--optuna", action="store_true",
                    help="OPTUNA: tự động search bằng TPE trong [--low, --high] thay vì "
                         "liệt kê --search tay. Cần: pip install optuna")
    ap.add_argument("--low", type=float, help="Cận dưới (bắt buộc nếu --optuna)")
    ap.add_argument("--high", type=float, help="Cận trên (bắt buộc nếu --optuna)")
    ap.add_argument("--n-trials", type=int, default=20, help="Số trial optuna (default 20)")
    ap.add_argument("--limit", type=int, default=120,
                    help="Số frame đầu dùng để search (mặc định 120 ~1s @119.88fps)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--work-dir", default="_calib_work")
    ap.add_argument("--apply", action="store_true",
                    help="Ghi giá trị tốt nhất vào config.yaml (có backup .bak trước khi ghi)")
    args = ap.parse_args()

    if h5py is None:
        raise ImportError("pip install h5py")

    if args.optuna:
        if args.low is None or args.high is None:
            raise ValueError("--optuna cần cả --low và --high")
    elif not args.search:
        raise ValueError("Cần --search <giá trị...> (hoặc dùng --optuna --low --high)")

    rate_safe = DVSVOLT_RATE_SAFE if args.simulator == "dvsvolt" else V2E_RATE_SAFE
    timing_only = DVSVOLT_TIMING_ONLY if args.simulator == "dvsvolt" else V2E_TIMING_ONLY
    valid_params = rate_safe | timing_only
    if args.param not in valid_params:
        raise ValueError(f"--param '{args.param}' không hợp lệ cho simulator="
                          f"{args.simulator}. Hợp lệ: {sorted(valid_params)}")
    if args.param in timing_only:
        print(f"[warning] '{args.param}' thuộc nhóm ảnh hưởng NOISE/JITTER thời gian, "
              "không đáng tin nếu events_real.h5 dùng timestamp bịa (decode_method=dense). "
              "Kiểm tra attrs['decode_method_counts'] của file trước khi tin kết quả này.\n")

    base_cfg = load_config(args.config)
    fps = float(base_cfg["camera"]["fps_original"])
    W, H = int(base_cfg["camera"]["width"]), int(base_cfg["camera"]["height"])
    duration_s = args.limit / fps

    real_path = Path(args.real)
    with h5py.File(real_path, "r") as hf:
        decode_counts = hf.attrs.get("decode_method_counts", "unknown")
    print(f"events_real.h5 decode_method_counts = {decode_counts}\n")

    real_stats = slice_real_to_window(real_path, duration_s, W, H)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    mode_desc = (f"optuna TPE, {args.n_trials} trial(s) in [{args.low}, {args.high}]"
                 if args.optuna else f"{len(args.search)} candidate(s) (grid)")
    print(f"\n{'='*72}\nCalibrating {args.simulator}.{args.param}  |  {mode_desc}  |  "
          f"window={duration_s:.2f}s ({args.limit} frames @ {fps}fps)\n{'='*72}")

    if args.optuna:
        # OPTUNA path — xem run_optuna_search() để biết chính xác optuna được
        # dùng ở đâu (TPESampler + study.optimize), phần còn lại (bảng kết
        # quả, --apply) dùng chung với grid path bên dưới.
        rows = run_optuna_search(args.simulator, args.param, args.low, args.high,
                                  args.n_trials, base_cfg, Path(args.sim_input), work_dir,
                                  args.limit, W, H, duration_s, real_stats)
    else:
        rows = []
        for value in args.search:
            tag = f"{args.param}_{value}".replace(".", "p")
            print(f"\n-- candidate {args.param}={value} --")
            cfg = make_candidate_config(base_cfg, args.simulator, args.param, value)
            try:
                h5path = run_candidate(args.simulator, Path(args.sim_input), work_dir, cfg,
                                        args.limit, tag)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  [skip] {e}")
                continue
            sim_stats = analyze(str(h5path), f"sim({value})", W, H, window_s=max(0.1, duration_s / 5))
            if sim_stats is None:
                print(f"  [skip] candidate sinh 0 event")
                continue
            err = score_candidate(sim_stats, real_stats)
            rows.append((value, sim_stats["rate_hz"], sim_stats["n_on"] / max(sim_stats["n"], 1), err))
            print(f"  rate={sim_stats['rate_hz']:,.0f} ev/s  on_frac={sim_stats['n_on']/max(sim_stats['n'],1):.3f}  "
                  f"error={err:.4f}")

    if not rows:
        print("\nKhông có candidate nào chạy được — xem log lỗi ở trên.")
        sys.exit(1)

    print(f"\n{'='*72}\nKẾT QUẢ  (real: rate={real_stats['rate_hz']:,.0f} ev/s  "
          f"on_frac={real_stats['n_on']/max(real_stats['n'],1):.3f})\n{'='*72}")
    print(f"{'value':>10}  {'rate_hz':>12}  {'on_frac':>8}  {'error':>8}")
    for value, rate, on_frac, err in sorted(rows, key=lambda r: r[3]):
        print(f"{value:>10}  {rate:>12,.0f}  {on_frac:>8.3f}  {err:>8.4f}")

    best_value, best_rate, best_on_frac, best_err = min(rows, key=lambda r: r[3])
    print(f"\n→ Tốt nhất: {args.param} = {best_value}  (error={best_err:.4f})")

    if args.apply:
        cfg_path = Path(args.config)
        backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
        shutil.copy(cfg_path, backup)
        final_cfg = make_candidate_config(base_cfg, args.simulator, args.param, best_value)
        with open(cfg_path, "w") as f:
            yaml.safe_dump(final_cfg, f, sort_keys=False, allow_unicode=True)
        print(f"✓ Đã ghi {args.param}={best_value} vào {cfg_path}  (backup: {backup})")
    else:
        print("Chưa ghi config.yaml — chạy lại với --apply để ghi giá trị tốt nhất "
              f"({args.param}={best_value}).")


if __name__ == "__main__":
    main()
