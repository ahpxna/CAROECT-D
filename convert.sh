#!/bin/bash

# 1. Dọn sạch môi trường để tránh xung đột
source ~/convert_env.sh

INPUT_RAW="$1"
OUTPUT_BASE="${2:-recording}"

if [ -z "$INPUT_RAW" ]; then
    echo "Cách dùng: ./convert.sh <file_goc.raw> <ten_ket_qua>"
    exit 1
fi

echo "--- Đang convert sang H5 ---"
/usr/bin/python3 raw_to_events.py "$INPUT_RAW" --output "${OUTPUT_BASE}.h5"

echo "--- Đang convert sang MP4 ---"
/usr/bin/python3 raw_to_video.py "$INPUT_RAW" --output "${OUTPUT_BASE}.mp4"

echo "=== XONG! ==="
