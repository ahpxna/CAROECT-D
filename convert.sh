#!/bin/bash

# Load the dedicated conversion environment to avoid dependency conflicts.
source ~/convert_env.sh

INPUT_RAW="$1"
OUTPUT_BASE="${2:-recording}"

if [ -z "$INPUT_RAW" ]; then
    echo "Usage: ./convert.sh <source.raw> <output_name>"
    exit 1
fi

echo "--- Converting to H5 ---"
/usr/bin/python3 raw_to_events.py "$INPUT_RAW" --output "${OUTPUT_BASE}.h5"

echo "--- Converting to MP4 ---"
/usr/bin/python3 raw_to_video.py "$INPUT_RAW" --output "${OUTPUT_BASE}.mp4"

echo "=== DONE ==="
