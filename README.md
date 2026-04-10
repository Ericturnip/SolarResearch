# PUNCH Data Processing Pipeline

This repository contains a suite of Python tools designed to process Level-3 (L3) FITS data from the PUNCH (Polarimeter to Unify the Corona and Heliosphere) mission. The pipeline ingests raw 2D Coronal Imager (CIM) and 3D Polarized Imager (PIM) datacubes, performs spatial and temporal median filtering, cleans out telemetry errors and background artifacts, and exports formatted ASCII text files ready for 3D coronal tomography.

## Directory Structure

The repository is organized by instrument (`CIM` vs. `PIM`) and workflow type (`Single Pass` vs. `Double Pass`).

### 1. CIM (Coronal Imager)
These scripts process standard 2D unpolarized intensity data.

* **Single Pass**
  * `cim_median_filter.py`: A fast, single-pass script that reads CIM FITS files, applies a 1°x1° spatial median filter, masks out edge-padding and invalid coordinates, and writes a text file for every frame.
* **Double Pass** *(Recommended for high-fidelity background removal)*
  * `punch_build_background.py`: The "Pass 1" master script. Consolidates an hour of data into a single coarse background map by applying spatial binning and a configurable temporal percentile stack (e.g., p30).
  * `individual_image_binning.py`: The "Pass 2" script. Takes the hourly background map generated above and uses it to aggressively clean the line-of-sight data for *every individual 8-minute frame*, outputting a stack of clean text files.

### 2. PIM (Polarized Imager)
These scripts process 3D MZP (Minus, Zero, Plus) polarized datacubes. They mathematically isolate the M, Z, and P layers to properly compute Stokes Q and U before calculating Polarization Brightness (pB) to avoid noise-rectification bias.

* **Single Pass**
  * `pim_median_filter.py`: A single-pass script that spatially filters the individual MZP layers, computes Q, U, and pB, runs diagnostics on negative Stokes values, and outputs per-frame text files.
* **Double Pass**
  * `pim_build_background.py`: The "Pass 1" script for polarized data. Generates a robust 1-hour temporal background map from the 3D PIM datacubes.
  * `pim_image_binning.py`: The "Pass 2" script. Cleans individual PIM frames against the hourly baseline map and outputs high-cadence text files.

### 3. Diagnostics and Fixes
Utility scripts to evaluate the health of the data and fix minor formatting bugs.

* `minmap_metrics.py`: Computes statistical diagnostics from a binned text file, reporting median background S10, outlier fractions (>2, >10, >50 S10), and adjacent-pixel smoothness to detect unmasked stray light or stars.
* `clean_punch_txt.py`: Auto-fixes formatting errors in the output `.txt` files by rewriting the time headers without touching the underlying L3 science data.

### 4. Plotting
* `plot_punch.py`: Reads the output text files into a Pandas DataFrame and generates a 2D scatter-plot heatmap of the S10 brightness in celestial coordinates (RA/Dec). The color scale is clipped to highlight solar wind structures and hide residual star spikes.

## Standard Workflows

**Single Pass Execution (Fast Export)**
Use this to rapidly extract spatially binned data without temporal background subtraction.
```bash
# Navigate to the single pass directory
cd CIM/Single\ Pass/
python cim_median_filter.py /path/to/fits/*.fits