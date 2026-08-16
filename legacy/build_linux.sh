#!/usr/bin/env bash
# build_linux.sh — Build evs_recorder on Ubuntu/Linux with Arena SDK
# Usage: bash build_linux.sh [/path/to/ArenaSDK]
# If no path given, tries common install locations automatically.
set -e

# ── 1. Find Arena SDK ─────────────────────────────────────────────
if [ -n "$1" ]; then
    ARENA="$1"
else
    # Common locations LUCID installers use on Ubuntu
    for candidate in \
        "$HOME/ArenaSDK" \
        "$HOME/ArenaSDK_Linux_x64" \
        "/opt/ArenaSDK" \
        "/opt/lucid/ArenaSDK" \
        "$(dirname "$0")/../ArenaSDK" \
        "$(dirname "$0")/ArenaSDK"
    do
        if [ -d "$candidate/lib64" ] && [ -d "$candidate/include" ]; then
            ARENA="$candidate"
            break
        fi
    done
fi

if [ -z "$ARENA" ] || [ ! -d "$ARENA/lib64" ]; then
    echo "ERROR: Cannot find Arena SDK. Pass the path explicitly:"
    echo "  bash build_linux.sh /path/to/ArenaSDK"
    echo ""
    echo "The folder should contain:  include/  lib64/  GenICam/"
    exit 1
fi

echo "Arena SDK found: $ARENA"

# ── 2. Locate GenICam lib dir ─────────────────────────────────────
GENICAM_INC="$ARENA/GenICam/library/CPP/include"
GENICAM_LIB=""
for candidate in \
    "$ARENA/GenICam/library/lib/Linux64_x64" \
    "$ARENA/GenICam/library/lib/Linux64" \
    "$ARENA/GenICam/library/CPP/lib/Linux64_x64" \
    "$ARENA/GenICam/library/CPP/lib/Linux64" \
    "$ARENA/GenICam/library/CPP/lib"
do
    if [ -d "$candidate" ] && ls "$candidate"/libGCBase*.so &>/dev/null; then
        GENICAM_LIB="$candidate"
        break
    fi
done

if [ -z "$GENICAM_LIB" ]; then
    echo "ERROR: Cannot find GenICam .so files under $ARENA/GenICam/"
    echo "Expected:  libGCBase_*.so  libGenApi_*.so  etc."
    echo "Actual contents:"
    find "$ARENA/GenICam" -name "*.so" 2>/dev/null | head -20
    exit 1
fi

echo "GenICam lib:    $GENICAM_LIB"
echo ""
echo "Contents of lib64/:"
ls "$ARENA/lib64/" 2>/dev/null || echo "  (empty or missing)"
echo ""

# ── 3. Discover .so names ─────────────────────────────────────────
# Find the Arena core library first — it might have different names on Linux
ARENA_CORE_LIB=""
for name in libarena.so libArena.so libArena_v140.so libArena_gcc54.so; do
    if [ -f "$ARENA/lib64/$name" ]; then
        ARENA_CORE_LIB="${name#lib}"; ARENA_CORE_LIB="${ARENA_CORE_LIB%.so}"
        break
    fi
done
# Also try versioned symlinks (both cases)
if [ -z "$ARENA_CORE_LIB" ]; then
    found=$(ls "$ARENA/lib64/libarena"*.so* "$ARENA/lib64/libArena"*.so* 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        base=$(basename "$found")
        ARENA_CORE_LIB="${base#lib}"
        ARENA_CORE_LIB="${ARENA_CORE_LIB%.so*}"
    fi
fi

if [ -z "$ARENA_CORE_LIB" ]; then
    echo "ERROR: Cannot find libArena*.so in $ARENA/lib64/"
    echo "Files there:"
    ls "$ARENA/lib64/" 2>/dev/null
    exit 1
fi
echo "Arena core lib: -l$ARENA_CORE_LIB"

arena_libs="-l$ARENA_CORE_LIB"
for lib in \
    libGCBase_gcc54_v3_3_LUCID.so \
    libGenApi_gcc54_v3_3_LUCID.so \
    libLog_gcc54_v3_3_LUCID.so \
    liblog4cpp_gcc54_v3_3_LUCID.so \
    libMathParser_gcc54_v3_3_LUCID.so \
    libNodeMapData_gcc54_v3_3_LUCID.so \
    libResUsageStat_gcc54_v3_3_LUCID.so \
    libXmlParser_gcc54_v3_3_LUCID.so
do
    found_in=""
    if   [ -f "$ARENA/lib64/$lib" ];  then found_in="$ARENA/lib64"
    elif [ -f "$GENICAM_LIB/$lib" ]; then found_in="$GENICAM_LIB"
    fi
    if [ -n "$found_in" ]; then
        flag="${lib#lib}"; flag="${flag%.so}"
        arena_libs="$arena_libs -l$flag"
    fi
done

echo "Link flags:     $arena_libs"
echo ""

# ── 4. Compile ────────────────────────────────────────────────────
# -pthread is required on Linux for std::thread
# -Wl,-rpath bakes the library search path INTO the binary so you don't
# need to export LD_LIBRARY_PATH every time you run it.
echo "Compiling evs_recorder.cpp ..."

g++ -std=c++14 -O2 -pthread \
    -I "$ARENA/include" \
    -I "$GENICAM_INC" \
    evs_recorder.cpp \
    -L "$ARENA/lib64" \
    -L "$GENICAM_LIB" \
    $arena_libs \
    -Wl,-rpath,"$ARENA/lib64" \
    -Wl,-rpath,"$GENICAM_LIB" \
    -o evs_recorder

echo ""
echo "✓ Build successful -> ./evs_recorder"
echo ""
echo "Test (camera must be connected and ArenaView closed):"
echo "  ./evs_recorder --list-event-formats"
echo ""
echo "Record 10 seconds (--strict-xypt/--flat-xypt no longer exist - XYPT is dead on"
echo "this firmware; --event-format-size is REQUIRED so the camera cannot silently"
echo "reuse a leftover value from a previous session):"
echo "  ./evs_recorder --output test.cevt --event-format EVT3_0 --event-format-size Bpe16 --duration 10"
echo ""
echo "Confirm the timestamp source before trusting any timing:"
echo "  ./evs_recorder --output test.cevt --event-format EVT3_0 --event-format-size Bpe16 \\"
echo "                 --debug-buffers 20 --duration 5"
echo "  python cevt_to_events.py test.cevt --debug-time-continuity"
echo ""
echo "If you get 'error while loading shared libraries' despite -rpath, run once:"
echo "  export LD_LIBRARY_PATH=$ARENA/lib64:$GENICAM_LIB:\$LD_LIBRARY_PATH"
echo "  (add to ~/.bashrc to make permanent)"
