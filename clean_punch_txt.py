#!/usr/bin/env python3
"""
Auto-fix PUNCH TXT metadata in the CURRENT DIRECTORY.

For every *.txt file:
  - Read the first L3 row
  - Extract its ISO timestamp
  - Rewrite ONLY the first header line (YYYY DOY.fraction)
  - Rename file to: PUNCH_L3_CIM_<YYYYMMDDhhmmss>_CLEANED.txt
  - Do NOT modify any L3 data rows
"""

from pathlib import Path
from datetime import datetime

L3_PREFIX = "L3"


def parse_iso_time(s: str) -> datetime:
    return datetime.fromisoformat(s)


def year_doy_fraction_string(dt: datetime) -> str:
    year = dt.year
    doy = int(dt.strftime("%j"))
    seconds = (
        dt.hour * 3600
        + dt.minute * 60
        + dt.second
        + dt.microsecond / 1e6
    )
    frac = seconds / 86400.0
    return f"{year} {doy + frac:.8f}"


def find_first_l3_time(lines):
    for line in lines:
        if line.startswith(L3_PREFIX):
            iso = line.strip().split()[-1]
            return parse_iso_time(iso)
    raise RuntimeError("No L3 rows found")


def process_file(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines(True)

    if not lines:
        raise RuntimeError("Empty file")

    dt = find_first_l3_time(lines)
    ts14 = dt.strftime("%Y%m%d%H%M%S")
    new_header = year_doy_fraction_string(dt)

    # ---- rewrite ONLY the first line ----
    first = lines[0]
    newline = "\n"
    if first.endswith("\r\n"):
        newline = "\r\n"
    elif first.endswith("\r"):
        newline = "\r"

    old_header = lines[0].strip()
    lines[0] = new_header + newline
    path.write_text("".join(lines), encoding="utf-8")

    # ---- rename file ----
    new_name = f"PUNCH_L3_CIM_{ts14}_CLEANED.txt"
    target = path.with_name(new_name)

    if target.exists() and target.resolve() != path.resolve():
        i = 1
        while True:
            candidate = path.with_name(
                f"PUNCH_L3_CIM_{ts14}_CLEANED__dup{i}.txt"
            )
            if not candidate.exists():
                target = candidate
                break
            i += 1

    path.rename(target)

    print(f"[OK] {path.name}")
    print(f"     header: {old_header} → {new_header}")
    print(f"     renamed → {target.name}")


def main():
    txt_files = sorted(Path(".").glob("*.txt"))
    if not txt_files:
        print("No .txt files found in current directory")
        return

    for p in txt_files:
        try:
            process_file(p)
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")


if __name__ == "__main__":
    main()