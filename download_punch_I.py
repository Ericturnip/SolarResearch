#!/usr/bin/env python3
import os
import argparse
from punchbowl import PunchData   # ✅ correct import

def fetch_punch_I(start_time, end_time, instrument, version, out_dir):
    """
    Download PUNCH Level 3 total-brightness (Stokes I) data
    for the specified instrument and date range.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Create the PunchData object for Level 3, Stokes I
    pd = PunchData(level=3, instrument=instrument, version=version, product="I")

    print(f"Fetching PUNCH {instrument} Level 3 I data from {start_time} to {end_time} ...")
    files = pd.fetch(start_time, end_time, path=out_dir)
    print(f"✅ Downloaded {len(files)} files to {out_dir}")
    for f in files:
        print("  " + f)

def main():
    parser = argparse.ArgumentParser(description="Download PUNCH Level 3 Stokes I data via punchbowl.")
    parser.add_argument("--start", required=True, help="Start time, e.g. 2025-10-10T00:00:00Z")
    parser.add_argument("--end",   required=True, help="End time, e.g. 2025-10-10T23:59:59Z")
    parser.add_argument("--instrument", required=True, help="Instrument code, e.g. PAM, PTM, PSM")
    parser.add_argument("--version", default="v02", help="Version string, e.g. v02")
    parser.add_argument("--outdir", default="./PUNCH_L3_I", help="Output directory")
    args = parser.parse_args()

    fetch_punch_I(args.start, args.end, args.instrument, args.version, args.outdir)

if __name__ == "__main__":
    main()