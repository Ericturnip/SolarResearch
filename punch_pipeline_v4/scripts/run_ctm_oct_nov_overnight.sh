#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p outputs/logs
LOG="outputs/logs/ctm_oct_nov_p25_$(date +%Y%m%d_%H%M%S).log"

CMD=(
  python scripts/process_ctm_oct_nov_p25.py
  --start-date 2025-10-01
  --end-date 2025-11-30
  --output-root outputs/ctm_hourly_p25
  --cache-root outputs/_ctm_download_cache
  --hour-workers 4
  --download-workers 8
  --retries 4
  --timeout 60
)

echo "[run] ${CMD[*]}"
echo "[log] $LOG"

if command -v caffeinate >/dev/null 2>&1; then
  PYTHONUNBUFFERED=1 caffeinate -dimsu "${CMD[@]}" 2>&1 | tee "$LOG"
else
  PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee "$LOG"
fi
