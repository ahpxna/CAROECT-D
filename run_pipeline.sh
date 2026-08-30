#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  CAROECT-D paper-aligned pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════
#  Current acquisition is sparse Metavision RAW:
#    evs_recorder.cpp -> .raw -> raw_to_events.py -> events.h5.
#  Arena .cevt remains only under legacy/ for already-recorded data and is not
#  a current acquisition method.
#
#  Usage:
#    ./run_pipeline.sh sim <davinci_tiff_dir> <session> <split> <default|calibrated>
#        Builds v2e and DVS-Voltmeter under disjoint simulator/condition/window roots.
#
#    ./run_pipeline.sh real  <site.raw | site.cevt> <session_name>
#        New recordings must be .raw. The .cevt branch is legacy conversion only.
#
#    ./run_pipeline.sh calibrate <events_real.h5> <processed_tiff_dir> <simulator> <param> <values...>
#        # Example: ... v2e pos_thres 0.15 0.2 0.25 0.3
#
#    ./run_pipeline.sh calibrate-eq30 <events_real.h5> <gray_gradient_tiff_dir> [v2e|dvsvolt]
#    ./run_pipeline.sh calibrate-camera event-only <event_pose_dir> [debug_dir]
#    ./run_pipeline.sh calibrate-camera register <event_pose_dir> <rgb_pose_dir> [debug_dir]
#
#    ./run_pipeline.sh train  <dataset_root>
#    ./run_pipeline.sh eval   <weights.pt> <test_data.yaml> [baseline.pt]
#
#  Optional radiometric corrections require a validated calibration manifest.
#  v2e/DVS-Voltmeter must be installed by setup_sim.sh; SAM3 must be installed.
#
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

CMD="${1:-}"
CFG="${CFG:-config.yaml}"
V2E_ENV="${V2E_ENV:-v2e}"
DVS_ENV="${DVS_ENV:-dvsvolt}"
LABEL_STATS="${LABEL_STATS:-0}"
EXPORT_COCO="${EXPORT_COCO:-1}"
# Explicit detector-window ablation. These values are independent of capture
# and simulation FPS; every variant uses the same ending timestamps and labels.
WINDOWS_US="${WINDOWS_US:-8340 16700 33300 50000}"
# Legacy .cevt conversion has no silent fallback FPS. Set REAL_FPS only when a
# known value is needed for already-recorded legacy data.
REAL_FPS="${REAL_FPS:-}"

run_py() {
  python "$@"
}

run_env() {
  local env_name="$1"
  shift
  if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    conda run -n "$env_name" python "$@"
  else
    echo "[warning] conda environment '$env_name' was not found; using current Python."
    python "$@"
  fi
}
 
usage() {
  echo "Usage: ./run_pipeline.sh {sim|real|calibrate|calibrate-eq30|calibrate-camera|train|eval} ..."
  echo "See the header comments for command arguments."
  exit 1
}
 
case "$CMD" in
 
  sim)
    DAVINCI_INPUT="${2:?<davinci_tiff_dir>}"
    SESSION="${3:?<session_name>}"
    SPLIT="${4:-train}"
    CONDITION="${5:-calibrated}"
    if [ "$CONDITION" != "default" ] && [ "$CONDITION" != "calibrated" ]; then
      echo "Condition must be 'default' or 'calibrated'." >&2
      exit 1
    fi
    PROCESSED="data/processed/${SESSION}"
    EVENTS_V2E="data/events/v2e/${CONDITION}/${SESSION}"
    EVENTS_DVS="data/events/dvsvolt/${CONDITION}/${SESSION}"
    SAM3_OUT="data/sam3/${SESSION}"
    WINDOWS_ROOT="data/windows/${CONDITION}/${SESSION}"
    DATASET_BASE="data/datasets"
 
    RGB_OUT="data/rgb/${SESSION}"
 
    echo "═══ [1/7] Preprocess (linear Y for simulation, sRGB for SAM3) ══"
    run_py preprocess.py --input "${DAVINCI_INPUT}" --output "${PROCESSED}" \
      --output-rgb "${RGB_OUT}" --verify --config "$CFG"
 
    echo "═══ [2/7] v2e event simulation ════════════════════════"
    run_env "$V2E_ENV" run_v2e.py --input "${PROCESSED}" --output "${EVENTS_V2E}" --config "$CFG"
 
    echo "═══ [3/7] DVS-Voltmeter stochastic simulation ═════════"
    run_env "$DVS_ENV" run_dvsvolt.py --input "${PROCESSED}" --output "${EVENTS_DVS}" --config "$CFG"
 
    echo "═══ [4/7] SAM3 -> tracks.json + masks ════════════════"
    run_py sam3_export_tracks.py "${RGB_OUT}" --all-classes \
      --output-dir "${SAM3_OUT}" --config "$CFG" --also-yolo
 
    echo "═══ [5/7] Causal label transfer for each detector window ═══"
    mkdir -p "$WINDOWS_ROOT"
    STATS_FLAG=()
    if [ "$LABEL_STATS" = "1" ]; then
      STATS_FLAG=(--stats)
    fi
    for WINDOW_US in $WINDOWS_US; do
      TAG="$(printf 'dt_%05dus' "$WINDOW_US")"
      run_py label_transfer.py --tracks "${SAM3_OUT}/tracks.json" \
        --events "${EVENTS_V2E}/events.h5" \
        --window-us "$WINDOW_US" --output "${WINDOWS_ROOT}/v2e_${TAG}.json" \
        "${STATS_FLAG[@]}"
      run_py label_transfer.py --tracks "${SAM3_OUT}/tracks.json" \
        --events "${EVENTS_DVS}/events.h5" \
        --window-us "$WINDOW_US" --output "${WINDOWS_ROOT}/dvsvolt_${TAG}.json" \
        "${STATS_FLAG[@]}"
    done
 
    echo "═══ [6/7] Build isolated dataset variants (split=${SPLIT}) ═══"
    COCO_FLAG=()
    if [ "$EXPORT_COCO" = "1" ]; then
      COCO_FLAG=(--export-coco)
    fi
    for WINDOW_US in $WINDOWS_US; do
      TAG="$(printf 'dt_%05dus' "$WINDOW_US")"
      REPRESENTATION="${DATASET_BASE}/representations/${TAG}.json"
      for SIMULATOR_NAME in v2e dvsvolt; do
        if [ "$SIMULATOR_NAME" = "v2e" ]; then
          EVENT_ROOT="$EVENTS_V2E"
        else
          EVENT_ROOT="$EVENTS_DVS"
        fi
        ROOT="${DATASET_BASE}/${SIMULATOR_NAME}/${CONDITION}/${TAG}"
        mkdir -p "$ROOT"
        cp "$CFG" "$ROOT/effective_config.yaml"
        git rev-parse HEAD > "$ROOT/git_revision.txt"
        if [ -f calibration/calibration_manifest.json ]; then
          cp calibration/calibration_manifest.json "$ROOT/calibration_manifest.json"
        fi
        run_py build_event_dataset.py --events "${EVENT_ROOT}/events.h5" \
          --windows "${WINDOWS_ROOT}/${SIMULATOR_NAME}_${TAG}.json" \
          --output "$ROOT" --split "$SPLIT" --site-id "$SESSION" \
          --representation "$REPRESENTATION" "${COCO_FLAG[@]}"
      done
    done
 
    echo "═══ [7/7] Done ════════════════════════════════════════"
    echo ""
    echo "✓ $SESSION -> $DATASET_BASE/<simulator>/$CONDITION/<window>/$SPLIT"
    ;;
 
  real)
    INFILE="${2:?<site.cevt | site.raw>}"
    SESSION="${3:?<session_name>}"
    OUT="data/events_real/${SESSION}.h5"
    mkdir -p "$(dirname "$OUT")"

    case "$INFILE" in
      *.raw)
        echo "═══ Real event: .raw (Metavision, evs_recorder.cpp, SPARSE) -> events.h5 ═══"
        echo "  Real per-event µs timestamps — no --fps needed, no continuity check needed"
        echo "  (that diagnostic is only meaningful for the dense/.cevt path)."
        run_py raw_to_events.py "${INFILE}" --output "${OUT}"
        ;;
      *.cevt)
        echo "═══ Real event: .cevt (Arena, CAROEVT1/2, DENSE accumulated, LEGACY) -> events.h5 ═══"
        echo "  [note] Current evs_recorder.cpp writes Metavision .raw, not .cevt."
        echo "  This branch only converts previously recorded legacy data."
        echo "  Diagnose legacy timestamp provenance before conversion:"
        echo "    python legacy/cevt_to_events.py ${INFILE} --debug-time-continuity"
        FPS_FLAG=()
        if [ -n "${REAL_FPS:-}" ]; then
          FPS_FLAG=(--fps "${REAL_FPS}")
        fi
        run_py legacy/cevt_to_events.py "${INFILE}" --output-h5 "${OUT}" "${FPS_FLAG[@]}"
        ;;
      *)
        echo "Unsupported input suffix: ${INFILE} (expected .raw or legacy .cevt)" >&2
        exit 1
        ;;
    esac

    echo ""
    echo "✓ ${OUT}"
    echo "  Calibrate with: ./run_pipeline.sh calibrate ${OUT} <processed_tiff_dir> ..."
    echo "  Or build a real test set with scene-specific tracks.json and windows.json."
    ;;
 
  calibrate)
    EVENTS_REAL="${2:?<events_real.h5>}"
    PROCESSED="${3:?<processed_tiff_dir>}"
    SIMULATOR="${4:?<simulator: v2e|dvsvolt>}"
    PARAM="${5:?<param>}"
    shift 5
    if [ "$#" -eq 0 ]; then
      echo "At least one search value is required for '${PARAM}'." >&2
      exit 1
    fi
    echo "═══ Closed-loop simulator calibration (${SIMULATOR}.${PARAM}) ══"
    run_py calibrate_simulator.py --real "${EVENTS_REAL}" \
      --sim-input "${PROCESSED}" --simulator "${SIMULATOR}" \
      --param "${PARAM}" --search "$@" --config "$CFG" --apply
    echo ""
    echo "✓ Updated ${SIMULATOR}.${PARAM} in ${CFG}"
    echo "  Re-run the calibrated simulation condition with the new value."
    ;;
 
  calibrate-eq30)
    EVENTS_REAL="${2:?<events_real.h5>}"
    GRAY_TIFF_DIR="${3:?<gray_gradient_tiff_dir>}"
    SIMULATOR="${4:-v2e}"
    echo "═══ Physical Eq.30 calibration (${SIMULATOR}) ═════════"
    run_py calibrate_simulator.py --eq30 --real "${EVENTS_REAL}" \
      --sim-input "${GRAY_TIFF_DIR}" --simulator "${SIMULATOR}" \
      --config "$CFG" --apply
    echo ""
    echo "✓ Eq.30 report -> _calib_work/eq30_estimate.json"
    ;;

  calibrate-camera)
    MODE="${2:?<event-only|register>}"
    EVENT_POSES="${3:?<event_pose_dir>}"
    if [ "$MODE" = "event-only" ]; then
      DEBUG_DIR="${4:-calibration/event_camera_debug}"
      echo "═══ Physical event-camera intrinsic calibration ═══════"
      run_py calibrate_event_camera.py --event-only --event-dir "$EVENT_POSES" \
        --debug-dir "$DEBUG_DIR" --config "$CFG"
    elif [ "$MODE" = "register" ]; then
      RGB_POSES="${4:?<rgb_pose_dir>}"
      DEBUG_DIR="${5:-calibration/event_rgb_registration_debug}"
      echo "═══ Physical RGB/event registration calibration ══════"
      run_py calibrate_event_camera.py --event-dir "$EVENT_POSES" \
        --rgb-dir "$RGB_POSES" --debug-dir "$DEBUG_DIR" --config "$CFG"
    else
      echo "calibrate-camera mode must be 'event-only' or 'register'." >&2
      exit 1
    fi
    echo "Event-camera parameters: calibration/event_camera_params.npz"
    if [ -f calibration/event_calib_report.json ]; then
      echo "Registration report: calibration/event_calib_report.json"
    fi
    ;;
 
  train)
    DATASET_ROOT="${2:?<dataset_root>}"
    MIXED_FLAG="${3:-}"
    if [[ "$DATASET_ROOT" == *"/mixed/"* ]] && [ "$MIXED_FLAG" != "--mixed" ]; then
      echo "A mixed condition root requires the explicit --mixed opt-in." >&2
      exit 1
    fi
    if [ ! -f "$DATASET_ROOT/dataset_manifest.json" ]; then
      echo "Missing traceable dataset manifest: $DATASET_ROOT/dataset_manifest.json" >&2
      exit 1
    fi
    echo "═══ Train YOLO ═══════════════════════════════════════"
    run_py train_event_yolo.py --data "${DATASET_ROOT}/data.yaml"
    ;;
 
  eval)
    WEIGHTS="${2:?<weights.pt>}"
    DATA="${3:?<test_data.yaml>}"
    BASELINE="${4:-}"
    echo "═══ Eval ═════════════════════════════════════════════"
    if [ -n "$BASELINE" ]; then
      run_py eval_event_yolo.py --weights "${WEIGHTS}" --data "${DATA}" \
        --baseline-weights "${BASELINE}"
    else
      run_py eval_event_yolo.py --weights "${WEIGHTS}" --data "${DATA}"
    fi
    ;;
 
  *)
    usage
    ;;
esac
 
