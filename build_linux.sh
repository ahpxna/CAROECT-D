#!/usr/bin/env bash
# ============================================================================
# build_linux.sh -- build evs_recorder.cpp (Metavision-based; renamed from
# evs_recorder_mv.cpp / build_mv.sh once this became the ONLY recorder — the
# old Arena-SDK evs_recorder.cpp that wrote .cevt is gone, see project notes)
#
# Uses EXACTLY the flag combination that was proven to link and run with
# probe_metavision.cpp on this machine. Nothing here is guessed:
#
#   * Headers come from a source checkout of OpenEB 4.6.2, because the
#     LUCID-bundled Metavision libraries are built from openeb 4.6.2 (confirmed
#     via `strings libmetavision_hal.so.4.6.2` -> /builds/openeb/ + version
#     4.6.2) and shipped WITHOUT headers. Matching the exact tag avoids ABI
#     mismatch with the prebuilt .so files.
#
#   * libprotobuf.so.23 is NOT in Ubuntu 24.04 (which ships protobuf 3.21 as
#     .so.32). libmetavision_sdk_driver.so hard-links against .so.23, so the
#     ABI-correct .so.23 is extracted from a focal-era .deb and kept beside
#     the Metavision libs. Installing libprotobuf-dev from noble does NOT
#     fix this -- it is a different soname with different symbols.
#
#   * OpenCV is needed because libmetavision_sdk_driver/core reference
#     cv::Mat in their public surface (CDFrameGenerator etc).
#
# Usage:  bash build_mv.sh
# ============================================================================
set -euo pipefail

ARENA_DIR="${ARENA_DIR:-$HOME/ArenaSDK_Linux_x64}"
MV_DIR="$ARENA_DIR/Metavision"
OPENEB_SRC="${OPENEB_SRC:-$HOME/openeb-4.6.2}"
SRC="${1:-evs_recorder.cpp}"
OUT="${OUT:-evs_recorder}"

# ---- preflight -------------------------------------------------------------
[ -f "$SRC" ] || { echo "ERROR: source not found: $SRC"; exit 1; }
[ -d "$MV_DIR/lib" ] || { echo "ERROR: not found: $MV_DIR/lib"; exit 1; }

if [ ! -d "$OPENEB_SRC" ]; then
    echo "OpenEB 4.6.2 headers not found at $OPENEB_SRC"
    echo "Fetching them (shallow clone, headers only usage):"
    git clone --depth 1 --branch 4.6.2 https://github.com/prophesee-ai/openeb.git "$OPENEB_SRC"
fi

# libprotobuf.so.23 shim -----------------------------------------------------
if [ ! -f "$MV_DIR/lib/libprotobuf.so.23" ]; then
    echo "libprotobuf.so.23 missing from $MV_DIR/lib -- fetching focal build..."
    TMPD=$(mktemp -d)
    ( cd "$TMPD"
      wget -q http://archive.ubuntu.com/ubuntu/pool/main/p/protobuf/libprotobuf23_3.12.4-1ubuntu7_amd64.deb
      dpkg -x libprotobuf23_3.12.4-1ubuntu7_amd64.deb pb )
    cp "$TMPD"/pb/usr/lib/x86_64-linux-gnu/libprotobuf.so.23* "$MV_DIR/lib/"
    rm -rf "$TMPD"
    echo "  -> installed into $MV_DIR/lib"
fi

# ---- flags -----------------------------------------------------------------
MV_INC=(
    -I"$OPENEB_SRC/hal/cpp/include"
    -I"$OPENEB_SRC/sdk/modules/base/cpp/include"
    -I"$OPENEB_SRC/sdk/modules/core/cpp/include"
    -I"$OPENEB_SRC/sdk/modules/driver/cpp/include"
    -I/usr/include/opencv4
)

MV_LIB=(
    -L"$MV_DIR/lib"
    -Wl,-rpath,"$MV_DIR/lib"
    -lmetavision_sdk_driver
    -lmetavision_sdk_base
    -lmetavision_hal
    "$MV_DIR/lib/libprotobuf.so.23"
)

echo "Compiling $SRC -> $OUT ..."
g++ -std=c++17 -O2 -g -pthread "$SRC" \
    "${MV_INC[@]}" "${MV_LIB[@]}" \
    $(pkg-config --libs opencv4) \
    -o "$OUT"

echo ""
echo "OK -> ./$OUT"
echo ""
echo "Run with (these MUST be exported, the plugin is not auto-discovered):"
echo ""
echo "  export MV_HAL_PLUGIN_PATH=$MV_DIR/lib/metavision/hal/plugins:$MV_DIR/hal_plugin"
echo "  export LD_LIBRARY_PATH=$MV_DIR/lib:\$LD_LIBRARY_PATH"
echo ""
echo "  ./$OUT --info"
echo "  ./$OUT --output run01.raw --duration 60"
echo ""
echo "Close ArenaView / evs_recorder first -- only one process can hold the"
echo "GigE Vision control channel at a time."
