#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

START_DATE="${START_DATE:-2025-09-01}"
END_DATE="${END_DATE:-2025-09-30}"
CTM_HOUR_WORKERS="${CTM_HOUR_WORKERS:-4}"
PTM_HOUR_WORKERS="${PTM_HOUR_WORKERS:-6}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
RETRIES="${RETRIES:-4}"
TIMEOUT="${TIMEOUT:-60}"

mkdir -p outputs/logs
LOG="outputs/logs/ctm_ptm_${START_DATE}_to_${END_DATE}_$(date +%Y%m%d_%H%M%S).log"

run_pipeline() {
  local product="$1"
  local layer="$2"
  local output_root="$3"
  local hour_workers="$4"

  echo "[run-${product}] ${START_DATE}..${END_DATE} layer=${layer}"
  python scripts/process_product_oct_nov_p25.py \
    --product "$product" \
    --layer "$layer" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --output-root "$output_root" \
    --cache-root outputs/_download_cache \
    --hour-workers "$hour_workers" \
    --download-workers "$DOWNLOAD_WORKERS" \
    --retries "$RETRIES" \
    --timeout "$TIMEOUT"

  echo "[sanitize-${product}] $output_root"
  python scripts/sanitize_tomography_txt_fixed_width.py "$output_root" --min-age-seconds 0
}

main() {
  echo "[start] CTM + PTM overnight run ${START_DATE}..${END_DATE}"
  echo "[log] $LOG"
  echo "[settings] CTM_HOUR_WORKERS=$CTM_HOUR_WORKERS PTM_HOUR_WORKERS=$PTM_HOUR_WORKERS DOWNLOAD_WORKERS=$DOWNLOAD_WORKERS"

  run_pipeline CTM brightness outputs/ctm_hourly_p25 "$CTM_HOUR_WORKERS"
  run_pipeline PTM Polar_pB outputs/ptm_hourly_p25 "$PTM_HOUR_WORKERS"

  echo "[done] CTM + PTM complete ${START_DATE}..${END_DATE}"
}

CAFFEINATE_PID=""
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
  CAFFEINATE_PID="$!"
  trap '[[ -n "$CAFFEINATE_PID" ]] && kill "$CAFFEINATE_PID" 2>/dev/null || true' EXIT
fi

PYTHONUNBUFFERED=1 main 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
