#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  CAROECT-D — run_pipeline.sh — FULL orchestrator (thay pipeline.sh cũ)
# ═══════════════════════════════════════════════════════════════════
#  pipeline.sh cũ dừng ở bước events.h5 (chỉ preprocess + v2e). Bản này
#  chạy hết: preprocess -> simulate(v2e+dvsvolt) -> SAM3 tracks -> label transfer ->
#  build dataset -> train -> eval. record_evs.py/read_evt3.py (Metavision
#  SDK, .raw EVT3.0) KHÔNG còn dùng — nhánh event thật giờ đi qua
#  evs_recorder.cpp (Arena SDK, EVT3.0 + EventFormatSize bắt buộc, KHÔNG có
#  XYPT — node đó đã xác nhận không tồn tại trên firmware TRT009S-E, xem
#  comment đầu evs_recorder.cpp) -> cevt_to_events.py --output-h5.
#  xypt_to_h5.py KHÔNG còn dùng nữa (chết theo XYPT).
#
#  Usage:
#    ./run_pipeline.sh sim   <davinci_tiff_dir> <session_name> <split>
#        # preprocess -> simulate (v2e+dvsvolt) -> SAM3 all classes -> label transfer
#        # -> build dataset (2 simulator variants, cho split train|val|test)
#
#    ./run_pipeline.sh real  <site.cevt> <session_name>
#        # cevt_to_events.py --output-h5 -> events_real.h5 (dùng cho
#        # calibrate_simulator.py HOẶC làm real test set qua
#        # build_event_dataset.py thủ công). Trước khi tin số t cho việc
#        # calibrate, chạy chẩn đoán liên tục thời gian trước:
#        #   python cevt_to_events.py <site.cevt> --debug-time-continuity
#        # nếu nó cho thấy TIME_LOW/TIME_HIGH nối liền qua các record thay
#        # vì reset mỗi record, set REAL_EVT3_CONTINUOUS_TIME=1 rồi chạy lại.
#
#    ./run_pipeline.sh calibrate <events_real.h5> <processed_tiff_dir> <simulator> <param> <values...>
#        # vd: ./run_pipeline.sh calibrate data/events_real/site01.h5 data/processed/site01 v2e pos_thres 0.15 0.2 0.25 0.3
#
#    ./run_pipeline.sh calibrate-eq23 <events_real.h5> <gray_gradient_tiff_dir> [v2e|dvsvolt]
#        # physical C_real = ΔlogL / Nbar(ΔlogL); --apply maps directly to v2e thresholds
#
#    ./run_pipeline.sh train  <dataset_root>
#    ./run_pipeline.sh eval   <weights.pt> <test_data.yaml> [baseline.pt]
#
#  Giả định: python calibrate.py đã chạy (calibration/*.npy, camera_params.npz
#  tồn tại), v2e/DVS-Voltmeter đã setup_sim.sh, SAM3 đã cài (facebookresearch/sam3).
#
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
 
CMD="${1:-}"
CFG="${CFG:-config.yaml}"
V2E_ENV="${V2E_ENV:-v2e}"
DVS_ENV="${DVS_ENV:-dvsvolt}"
LABEL_STATS="${LABEL_STATS:-0}"
DENSE_LABELS="${DENSE_LABELS:-0}"   # xem review: mặc định TẮT vì O(n_events x n_tracks) Python thuần, có thể mất hàng giờ trên site thật. Bật tay khi cần debug occlusion: LABEL_STATS=1 DENSE_LABELS=1 ./run_pipeline.sh sim ...
EXPORT_COCO="${EXPORT_COCO:-1}"
REAL_FPS="${REAL_FPS:-30.0}"                              # cevt_to_events.py --fps: chỉ dùng khi 1 record fallback về dense-frame; EVT3.0 record dùng t thật giải mã từ payload
REAL_EVT3_CONTINUOUS_TIME="${REAL_EVT3_CONTINUOUS_TIME:-0}"  # xem cevt_to_events.py --debug-time-continuity trước khi bật cái này
 
run_py() {
  python "$@"
}
 
run_env() {
  local env_name="$1"
  shift
  if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    conda run -n "$env_name" python "$@"
  else
    echo "[warn] conda env '$env_name' không thấy; chạy bằng python hiện tại."
    python "$@"
  fi
}
 
usage() {
  echo "Usage: ./run_pipeline.sh {sim|real|calibrate|calibrate-eq23|train|eval} ..."
  echo "  Xem comment đầu file này để biết đúng cú pháp từng lệnh."
  exit 1
}
 
case "$CMD" in
 
  sim)
    DAVINCI_INPUT="${2:?<davinci_tiff_dir>}"
    SESSION="${3:?<session_name>}"
    SPLIT="${4:-train}"
    PROCESSED="data/processed/${SESSION}"
    EVENTS_V2E="data/events_v2e/${SESSION}"
    EVENTS_DVS="data/events_dvsvolt/${SESSION}"
    SAM3_OUT="data/sam3/${SESSION}"
    WINDOWS_V2E="data/windows/${SESSION}_v2e.json"
    WINDOWS_DVS="data/windows/${SESSION}_dvsvolt.json"
    EVENT_LABELS_V2E="data/event_labels/${SESSION}_v2e.h5"
    EVENT_LABELS_DVS="data/event_labels/${SESSION}_dvsvolt.h5"
    DATASET="data/dataset"
 
    RGB_OUT="data/rgb/${SESSION}"
 
    echo "═══ [1/7] Preprocess (dual branch: Y-linear cho sim, sRGB cho SAM3) ══"
    run_py preprocess.py --input "${DAVINCI_INPUT}" --output "${PROCESSED}" \
      --output-rgb "${RGB_OUT}" --verify --config "$CFG"
 
    echo "═══ [2/7] v2e event simulation ════════════════════════"
    run_env "$V2E_ENV" run_v2e.py --input "${PROCESSED}" --output "${EVENTS_V2E}" --config "$CFG"
 
    echo "═══ [3/7] DVS-Voltmeter stochastic simulation ═════════"
    run_env "$DVS_ENV" run_dvsvolt.py --input "${PROCESSED}" --output "${EVENTS_DVS}" --config "$CFG"
 
    echo "═══ [4/7] SAM3 -> tracks.json + masks ════════════════"
    run_py sam3_export_tracks.py "${RGB_OUT}" --all-classes \
      --output-dir "${SAM3_OUT}" --config "$CFG" --also-yolo
 
    echo "═══ [5/7] Label transfer (v2e + dvsvolt) ═════════════"
    mkdir -p "$(dirname "${WINDOWS_V2E}")"
    STATS_FLAG=()
    if [ "$LABEL_STATS" = "1" ]; then
      STATS_FLAG=(--stats)
    fi
    DENSE_V2E_FLAG=()
    DENSE_DVS_FLAG=()
    if [ "$DENSE_LABELS" = "1" ]; then
      mkdir -p "$(dirname "${EVENT_LABELS_V2E}")"
      DENSE_V2E_FLAG=(--per-event-labels "${EVENT_LABELS_V2E}")
      DENSE_DVS_FLAG=(--per-event-labels "${EVENT_LABELS_DVS}")
    fi
    run_py label_transfer.py --tracks "${SAM3_OUT}/tracks.json" \
      --events "${EVENTS_V2E}/events.h5" --output "${WINDOWS_V2E}" \
      "${STATS_FLAG[@]}" "${DENSE_V2E_FLAG[@]}"
    run_py label_transfer.py --tracks "${SAM3_OUT}/tracks.json" \
      --events "${EVENTS_DVS}/events.h5" --output "${WINDOWS_DVS}" \
      "${STATS_FLAG[@]}" "${DENSE_DVS_FLAG[@]}"
 
    echo "═══ [6/7] Build dataset variants (split=${SPLIT}) ═════"
    COCO_FLAG=()
    if [ "$EXPORT_COCO" = "1" ]; then
      COCO_FLAG=(--export-coco)
    fi
    run_py build_event_dataset.py --events "${EVENTS_V2E}/events.h5" \
      --windows "${WINDOWS_V2E}" --output "${DATASET}" --split "${SPLIT}" \
      --site-id "${SESSION}_v2e" "${COCO_FLAG[@]}"
    run_py build_event_dataset.py --events "${EVENTS_DVS}/events.h5" \
      --windows "${WINDOWS_DVS}" --output "${DATASET}" --split "${SPLIT}" \
      --site-id "${SESSION}_dvsvolt" "${COCO_FLAG[@]}"
 
    echo "═══ [7/7] Done ════════════════════════════════════════"
    echo ""
    echo "✓ ${SESSION} -> ${DATASET}/${SPLIT}/  (chạy lại cho các site khác, "
    echo "  --split khác, để dựng đủ train/val/test theo policy đang test)"
    ;;
 
  real)
    CEVT="${2:?<site.cevt>}"
    SESSION="${3:?<session_name>}"
    OUT="data/events_real/${SESSION}.h5"
    mkdir -p "$(dirname "$OUT")"
    echo "═══ Real event: .cevt (CAROEVT1, EVT3.0) -> events.h5 ═══"
    echo "  [reminder] chưa chạy chẩn đoán liên tục t? Chạy trước:"
    echo "    python cevt_to_events.py ${CEVT} --debug-time-continuity"
    CONT_FLAG=()
    if [ "$REAL_EVT3_CONTINUOUS_TIME" = "1" ]; then
      CONT_FLAG=(--evt3-continuous-time)
    fi
    run_py cevt_to_events.py "${CEVT}" --output-h5 "${OUT}" --fps "${REAL_FPS}" "${CONT_FLAG[@]}"
    echo ""
    echo "✓ ${OUT}"
    echo "  Dùng cho: ./run_pipeline.sh calibrate ${OUT} <processed_tiff_dir>"
    echo "       hoặc: dùng làm real test set qua build_event_dataset.py thủ công"
    echo "             (cần tracks.json + windows.json riêng cho scene đó)"
    ;;
 
  calibrate)
    EVENTS_REAL="${2:?<events_real.h5>}"
    PROCESSED="${3:?<processed_tiff_dir>}"
    SIMULATOR="${4:?<simulator: v2e|dvsvolt>}"
    PARAM="${5:?<param>}"
    shift 5
    if [ "$#" -eq 0 ]; then
      echo "Cần ít nhất 1 giá trị search cho param '${PARAM}'." >&2
      exit 1
    fi
    echo "═══ Closed-loop simulator calibration (${SIMULATOR}.${PARAM}) ══"
    run_py calibrate_simulator.py --real "${EVENTS_REAL}" \
      --sim-input "${PROCESSED}" --simulator "${SIMULATOR}" \
      --param "${PARAM}" --search "$@" --config "$CFG" --apply
    echo ""
    echo "✓ ${CFG} đã cập nhật ${SIMULATOR}.${PARAM}"
    echo "  Chạy lại ./run_pipeline.sh sim ... để dùng θ mới calibrate"
    ;;
 
  calibrate-eq23)
    EVENTS_REAL="${2:?<events_real.h5>}"
    GRAY_TIFF_DIR="${3:?<gray_gradient_tiff_dir>}"
    SIMULATOR="${4:-v2e}"
    echo "═══ Physical Eq.23 calibration (${SIMULATOR}) ═════════"
    run_py calibrate_simulator.py --eq23 --real "${EVENTS_REAL}" \
      --sim-input "${GRAY_TIFF_DIR}" --simulator "${SIMULATOR}" \
      --config "$CFG" --apply
    echo ""
    echo "✓ Eq.23 report -> _calib_work/eq23_estimate.json"
    ;;
 
  train)
    DATASET_ROOT="${2:?<dataset_root>}"
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
 