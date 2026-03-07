#!/usr/bin/env python3
"""
Quick diagnostic checker for PUNCH L3 TXT outputs.
"""

import glob
import numpy as np
import os

def run_diagnostics():
    txt_files = sorted(glob.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in the current directory.")
        return

    print(f"Found {len(txt_files)} text files. Running diagnostics...\n")

    for f_path in txt_files:
        print(f"--- {os.path.basename(f_path)} ---")
        
        with open(f_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            print("  [ERROR] File is entirely empty!\n")
            continue
            
        # 1. Check Header
        header = lines[0].strip()
        header_parts = header.split()
        if len(header_parts) == 2 and header_parts[0].isdigit():
            print(f"  [OK] Header: {header}")
        else:
            print(f"  [WARN] Suspicious header format: {header}")
            
        # 2. Extract Data
        data_lines = [l for l in lines[1:] if l.strip()]
        if not data_lines:
            print("  [WARN] File has a header but no data rows.\n")
            continue
            
        if not data_lines[0].startswith("L3"):
            print("  [WARN] First data row doesn't start with 'L3'. Check formatting.")
        
        # 3. Analyze Values
        try:
            # Columns are: [0]L3, [1]RA, [2]DEC, [3]VALUE, [4]TIME
            vals = []
            for row in data_lines:
                parts = row.split()
                if len(parts) >= 4:
                    vals.append(float(parts[3]))
            
            vals = np.array(vals)
            valid_vals = vals[np.isfinite(vals)]
            
            print(f"  Total Data Rows: {len(data_lines)}")
            
            if len(valid_vals) > 0:
                v_min = np.min(valid_vals)
                v_med = np.median(valid_vals)
                v_max = np.max(valid_vals)
                print(f"  Value Stats   -> Min: {v_min:8.2f} | Med: {v_med:8.2f} | Max: {v_max:8.2f}")
            
            nans = len(vals) - len(valid_vals)
            if nans > 0:
                print(f"  [WARN] Found {nans} NaN or Inf values in the output column!")
                
        except Exception as e:
            print(f"  [ERROR] Could not parse data values properly: {e}")
            
        print("")

if __name__ == "__main__":
    run_diagnostics()