#!/usr/bin/env bash
# Backward-compatible wrapper for the full CAROECT-D orchestrator.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: ./pipeline.sh <davinci_tiff_dir> <session_name> [split]" >&2
  echo "Delegates to: ./run_pipeline.sh sim <davinci_tiff_dir> <session_name> [split]" >&2
  exit 1
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/run_pipeline.sh" sim "$@"
