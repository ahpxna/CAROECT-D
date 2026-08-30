#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  CAROECT-D — setup_sim.sh
#  ALL-IN-ONE: Miniconda → 2 conda envs → clone v2e + DVS-Voltmeter
#              → version lock → (optional) esim_torch → (optional) smoke test
# ═══════════════════════════════════════════════════════════════════
#  Usage:
#    ./setup_sim.sh                                  # setup env + repos
#    ./setup_sim.sh --test data/processed/site01     # setup + smoke test (120 frames)
#    ./setup_sim.sh --with-esim                      # + esim_torch (requires nvcc)
#    SIM_ROOT=/data/sim ./setup_sim.sh               # choose another repo/env root
#
#  Design rules:
#   * IDEMPOTENT — reruns are safe; existing components are reused.
#   * Do not modify simulator sources. The drivers (run_v2e.py and
#     run_dvsvolt.py) bypass repository I/O while retaining simulator physics.
#   * Write VERSIONS.txt (git hashes + environment locks) for methodology use.
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ── S0 · PATHS & ARGS ──────────────────────────────────────────────
SIM_ROOT="${SIM_ROOT:-$HOME/caroect_sim}"
ENV_V2E="v2e"
ENV_DVS="dvsvolt"
WITH_ESIM=0
TEST_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-esim) WITH_ESIM=1; shift ;;
    --test)      TEST_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done
mkdir -p "$SIM_ROOT"
echo "[S0] SIM_ROOT = $SIM_ROOT"

# ── S1 · CONDA ─────────────────────────────────────────────────────
#  Install Miniconda when absent, then source conda.sh directly. Non-interactive
#  scripts do not necessarily source ~/.bashrc, so this avoids the common
#  "conda: command not found" / "run 'conda init'" failure.
if ! command -v conda >/dev/null 2>&1; then
  if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
    echo "[S1] Miniconda not found; downloading and installing silently..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  fi
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "[S1] conda OK: $(conda --version)   base=$CONDA_BASE"

# ── S2 · CUDA DETECTION → choose a pytorch-cuda tag ────────────────
#  Read "CUDA Version: X.Y" from nvidia-smi. This is the version supported by
#  the driver, so the pytorch-cuda wheel must not exceed it:
#  >=12.1 -> 12.1 | 11.8-12.0 -> 11.8 | unavailable -> CPU.
CUDA_VER="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)"
PT_CUDA=""
if [ -n "$CUDA_VER" ]; then
  MAJ="${CUDA_VER%%.*}"; MIN="${CUDA_VER##*.}"
  if   [ "$MAJ" -ge 13 ]; then PT_CUDA="12.1"
  elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 1 ]; then PT_CUDA="12.1"
  elif [ "$MAJ" -eq 12 ]; then PT_CUDA="11.8"
  elif [ "$MAJ" -eq 11 ] && [ "$MIN" -ge 8 ]; then PT_CUDA="11.8"
  fi
fi
if [ -n "$PT_CUDA" ]; then
  echo "[S2] GPU driver CUDA $CUDA_VER; installing pytorch-cuda=$PT_CUDA"
else
  echo "[S2] ⚠ NVIDIA GPU not detected; v2e will run on CPU and may be very slow. Check nvidia-smi."
fi

# ── S3 · ENV v2e  (python 3.10 + PyTorch-CUDA + v2e editable) ──────
#  The v2e README specifies Python 3.10. Install Torch from the pytorch conda
#  channel and install the repository editable so its source remains inspectable.
if ! conda env list | grep -qE "^${ENV_V2E}[[:space:]]"; then
  conda create -y -n "$ENV_V2E" python=3.10
fi
if ! conda run -n "$ENV_V2E" python -c "import torch" >/dev/null 2>&1; then
  if [ -n "$PT_CUDA" ]; then
    conda install -y -n "$ENV_V2E" pytorch torchvision pytorch-cuda="$PT_CUDA" -c pytorch -c nvidia
  else
    conda install -y -n "$ENV_V2E" pytorch torchvision cpuonly -c pytorch
  fi
fi
[ -d "$SIM_ROOT/v2e" ] || git clone https://github.com/SensorsINI/v2e "$SIM_ROOT/v2e"
conda run -n "$ENV_V2E" python -m pip install -q -e "$SIM_ROOT/v2e"
conda run -n "$ENV_V2E" python -m pip install -q tifffile h5py pyyaml
echo "[S3] env '$ENV_V2E' ready (Torch CUDA: $(conda run -n "$ENV_V2E" python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null || echo '?'))"

# ── S4 · ENV dvsvolt (Python 3.9 + legacy pins + CPU Torch) ─────────
#  DVS-Voltmeter pins opencv-python==4.5.1.48, whose wheel is available only
#  through Python 3.9, and requires numpy<2 for its older ABI. CPU Torch is
#  sufficient for the original repository and avoids the larger CUDA package.
if ! conda env list | grep -qE "^${ENV_DVS}[[:space:]]"; then
  conda create -y -n "$ENV_DVS" python=3.9
fi
if ! conda run -n "$ENV_DVS" python -c "import torch, easydict, cv2" >/dev/null 2>&1; then
  conda run -n "$ENV_DVS" python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  conda run -n "$ENV_DVS" python -m pip install -q easydict==1.9 opencv-python==4.5.1.48 tqdm==4.49.0 "numpy>=1.20.1,<2" tifffile h5py pyyaml
fi
[ -d "$SIM_ROOT/DVS-Voltmeter" ] || git clone https://github.com/Lynn0306/DVS-Voltmeter "$SIM_ROOT/DVS-Voltmeter"
echo "[S4] env '$ENV_DVS' ready"

# ── S5 · VERSION LOCK (for the paper methodology) ─────────────────
{
  echo "date: $(date -Iseconds)"
  echo "v2e_commit: $(git -C "$SIM_ROOT/v2e" rev-parse HEAD)"
  echo "dvs_voltmeter_commit: $(git -C "$SIM_ROOT/DVS-Voltmeter" rev-parse HEAD)"
  echo "cuda_driver: ${CUDA_VER:-none}   pytorch_cuda: ${PT_CUDA:-cpu}"
} > "$SIM_ROOT/VERSIONS.txt"
conda env export -n "$ENV_V2E"  > "$SIM_ROOT/env_v2e.lock.yml"    || true
conda env export -n "$ENV_DVS"  > "$SIM_ROOT/env_dvsvolt.lock.yml" || true
echo "[S5] VERSIONS.txt + env locks → $SIM_ROOT/"

# ── S6 · (OPTIONAL) esim_torch — fast third GPU simulator ──────────
#  Building the CUDA extension requires nvcc with the same major version as
#  Torch. A build failure must not block the two primary simulators.
if [ "$WITH_ESIM" -eq 1 ]; then
  [ -d "$SIM_ROOT/rpg_vid2e" ] || git clone https://github.com/uzh-rpg/rpg_vid2e "$SIM_ROOT/rpg_vid2e"
  echo "[S6] building esim_torch (requires nvcc)..."
  conda run -n "$ENV_V2E" python -m pip install "$SIM_ROOT/rpg_vid2e/esim_torch/" \
    && echo "[S6] esim_torch OK" \
    || echo "[S6] ⚠ esim_torch build failed; continuing because v2e and DVS-Voltmeter are unaffected"
fi

# ── S7 · (OPTIONAL) SMOKE TEST — first 120 frames through both simulators ─
if [ -n "$TEST_DIR" ]; then
  echo "[S7] smoke-test input: $TEST_DIR"
  conda run -n "$ENV_V2E" python run_v2e.py      --input "$TEST_DIR" --output "$SIM_ROOT/_smoke/v2e"      --limit 120
  conda run -n "$ENV_DVS" python run_dvsvolt.py  --input "$TEST_DIR" --output "$SIM_ROOT/_smoke/dvsvolt"  --limit 120
  echo "[S7] smoke test passed; inspect $SIM_ROOT/_smoke/*/events.h5"
fi

echo ""
echo "✓ Setup complete. Full-run examples:"
echo "  conda run -n $ENV_V2E  python run_v2e.py     --input data/processed/<s> --output data/events_v2e/<s>"
echo "  conda run -n $ENV_DVS  python run_dvsvolt.py --input data/processed/<s> --output data/events_dvsvolt/<s>"
