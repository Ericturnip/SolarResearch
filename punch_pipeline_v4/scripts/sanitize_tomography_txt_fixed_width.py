from __future__ import annotations

import argparse
import time
from pathlib import Path


BRIGHTNESS_MIN = -9999.99
BRIGHTNESS_MAX = 99999.99


def parse_fixed_width_row(line: str) -> tuple[float, float, float] | None:
    if len(line) < 26:
        return None
    try:
        ra = float(line[4:10])
        dec = float(line[11:17])
        brightness = float(line[17:25])
    except ValueError:
        return None
    if not line[25].isspace():
        return None
    return ra, dec, brightness


def row_fits_fixed_width(line: str) -> bool:
    parsed = parse_fixed_width_row(line)
    if parsed is None:
        return False
    _ra, _dec, brightness = parsed
    return BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX


def sanitize_file(path: Path, *, dry_run: bool, drop_zero: bool) -> tuple[int, int, int]:
    lines = path.read_text().splitlines(keepends=True)
    kept: list[str] = []
    removed_overflow = 0
    removed_malformed = 0
    removed_zero = 0

    for line in lines:
        if not line.startswith("L3"):
            kept.append(line)
            continue
        parsed = parse_fixed_width_row(line)
        if parsed is None:
            removed_malformed += 1
            continue
        _ra, _dec, brightness = parsed
        if brightness < BRIGHTNESS_MIN or brightness > BRIGHTNESS_MAX:
            removed_overflow += 1
            continue
        if drop_zero and brightness == 0.0:
            removed_zero += 1
            continue
        kept.append(line)

    removed = removed_overflow + removed_malformed + removed_zero
    if removed and not dry_run:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text("".join(kept))
        tmp_path.replace(path)
    return removed_overflow, removed_malformed, removed_zero


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove tomography TXT rows whose brightness cannot fit the fixed-width F8.2 field."
    )
    parser.add_argument("root", nargs="?", default="outputs/ctm_hourly_p25")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--min-age-seconds",
        type=float,
        default=30.0,
        help="Skip files modified more recently than this, useful while the processor is still running.",
    )
    parser.add_argument("--keep-zero", action="store_true", help="Keep rows whose printed brightness is 0.00")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    now = time.time()
    total_files = 0
    changed_files = 0
    skipped_recent = 0
    total_overflow = 0
    total_malformed = 0
    total_zero = 0

    for path in sorted(root.glob("**/*.txt")):
        total_files += 1
        if args.min_age_seconds > 0 and now - path.stat().st_mtime < args.min_age_seconds:
            skipped_recent += 1
            continue
        overflow, malformed, zero = sanitize_file(path, dry_run=args.dry_run, drop_zero=not args.keep_zero)
        if overflow or malformed or zero:
            changed_files += 1
            total_overflow += overflow
            total_malformed += malformed
            total_zero += zero
            action = "would-clean" if args.dry_run else "cleaned"
            print(f"[{action}] {path} overflow={overflow} malformed={malformed} zero={zero}")

    print(
        f"[summary] files={total_files} changed={changed_files} "
        f"overflow_removed={total_overflow} malformed_removed={total_malformed} "
        f"zero_removed={total_zero} "
        f"skipped_recent={skipped_recent} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
