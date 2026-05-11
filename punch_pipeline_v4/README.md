# PUNCH Pipeline v4 — Modular Scaffolding README

## Overview

PUNCH Pipeline v4 is a modular rewrite of our earlier processing scripts used for PUNCH Level 3 data products. The objective is to support current and future products through a shared framework while isolating product-specific logic into adapters.

This system is designed to handle:

* 1° × 1° median-filter sky binning
* One-hour composite minimum maps
* TXT export for tomography / legacy downstream tools
* Product-specific layer handling (brightness, pB, raw polarizers, etc.)
* Easier extension when new PUNCH products appear

Instead of rewriting scripts every time a new FITS product arrives, we only implement or update a single adapter.

---

# Design Philosophy

We separate responsibilities into four layers:

## Adapters

Responsible for reading FITS files and exposing usable science layers.

## Processing

Shared numerical operations:

* binning
* compositing
* unit conversion

## Writers

Export outputs to standardized TXT files.

## Scripts

Command-line entry points for users.

This means:

* New product → usually only add/update one adapter
* Shared science logic remains reusable
* Easier debugging and maintenance

---

# Repository Layout

```text
punch_pipeline_v4/

src/punch_pipeline_v4/
│
├── adapters/
│   ├── base.py
│   ├── registry.py
│   ├── cim.py
│   ├── ctm.py
│   ├── pim.py
│   ├── ptm.py
│   ├── pam.py
│   └── cam.py
│
├── processing/
│   ├── binning.py
│   ├── composite.py
│   └── units.py
│
├── writers/
│   └── ascii.py
│
├── models.py
│
scripts/
│   ├── run_median_filter.py
│   └── run_hourly_composite.py
```

---

# Supported Products

## Direct Brightness Products

These contain one directly usable brightness layer.

Products:

* CIM
* CTM
* CAM

Typical workflow:

```text
load brightness layer
→ convert to S10
→ median-bin to 1° grid
→ export
```

---

## Processed Polarization Products

These already contain derived polarization layers.

Products:

* PTM
* PAM

Typical layers:

* Polar_B
* Polar_pB
* Polar_pBp

Usually we use:

```text
Polar_pB
```

---

## Raw Polarizer Product

Product:

* PIM

PIM contains raw analyzer images:

* Polar_M
* Polar_Z
* Polar_P

It does **not** contain direct pB.

The adapter reconstructs pB using the old proven equations:

```text
median-bin M
median-bin Z
median-bin P

Q = (4/3)Z − (2/3)(P + M)
U = (2/√3)P − (2/√3)M

pB = √(Q² + U²)
```

This reproduces legacy behavior.

---

# Core Data Models

## LayerFrame

Represents one 2D image before binning.

Fields include:

* product
* layer_name
* data
* timestamp
* solar_wcs
* radec_wcs
* native_unit
* metadata

---

## BinnedMap

Represents a 1° × 1° sky grid after processing.

Fields include:

* values
* hpln_centers
* hplt_centers
* timestamp
* time_map
* unit
* metadata

---

# Adapter System

Every product adapter implements:

## Required Methods

```python
list_layers()
load_layer()
```

## Optional Method

```python
make_binned_map(...)
```

Used for products requiring special science logic.

Example:

* PIM pB must be formed from three raw analyzer channels.

This lets the scripts remain generic.

---

# Median Filter Script

## Entry Point

```bash
python scripts/run_median_filter.py
```

## What It Does

1. Load product adapter
2. Ask adapter for custom binned map if needed
3. Otherwise load normal layer
4. Convert to S10
5. Apply masks / filtering
6. Export TXT file

---

# Hourly Composite Script

## Entry Point

```bash
python scripts/run_hourly_composite.py
```

## What It Does

1. Load all matching files in time window
2. Bin each frame to 1° × 1°
3. Stack maps
4. Compute temporal statistic (usually nanmin)
5. Preserve timestamp of source frame
6. Export TXT

---

# Why We Use nanmin

For tomography background estimation, we usually want:

```text
lowest brightness seen in each sky bin during the hour
```

This suppresses transient bright structures.

Implementation:

```python
np.nanmin(stack, axis=0)
```

Zeros can optionally be removed first:

```bash
--drop-zero-before-stat
```

---

# Units

## Native FITS Units

Many products report brightness in:

```text
2.009e+07 W/(m2 sr)
```

## Converted Units

We convert to S10 using legacy convention:

```text
S10 = native / 4.5e-16
```

---

# TXT Output Format

Example:

```text
2025 311.00000000
L3  189.20 -54.72   36.03 2025-11-07T00:00:00
```

Meaning:

Line 1:

* year
* day-of-year

Each data row:

* L3 tag
* longitude
* latitude
* brightness (S10)
* timestamp of contributing frame

---

# Example Commands

---

## CIM Median Filter

```bash
python scripts/run_median_filter.py \
  --input PUNCH_L3_CIM_xxx.fits \
  --product CIM \
  --layer brightness \
  --convert-to-s10
```

---

## CTM One-Hour Composite

```bash
python scripts/run_hourly_composite.py \
  --input-glob "PUNCH_L3_CTM_20251104*.fits" \
  --product CTM \
  --layer brightness \
  --composite-method nanmin \
  --convert-to-s10 \
  --drop-zero-before-stat
```

---

## PTM pB

```bash
python scripts/run_median_filter.py \
  --input PUNCH_L3_PTM_xxx.fits \
  --product PTM \
  --layer Polar_pB
```

---

## PAM One-Hour pB Composite

```bash
python scripts/run_hourly_composite.py \
  --input-glob "PUNCH_L3_PAM_20251105*.fits" \
  --product PAM \
  --layer Polar_pB \
  --composite-method nanmin
```

---

## PIM Reconstructed pB

```bash
python scripts/run_median_filter.py \
  --input PUNCH_L3_PIM_xxx.fits \
  --product PIM \
  --layer Polar_pB
```

Even though PIM has no direct pB layer, the adapter computes it automatically.

---

# How To Add Future Products

Suppose a new product arrives:

```text
XYZ
```

Create:

```text
adapters/xyz.py
```

Implement:

```python
class XYZAdapter(ProductAdapter):
```

Then register it in:

```text
registry.py
```

Usually no script rewrite is required.

---

# Why This Is Better Than Older Scripts

Older code was typically:

* monolithic
* product-specific
* harder to debug
* difficult to extend

v4 is:

* modular
* reusable
* scientifically transparent
* easier to compare products
* future-proof

---

# Current Scientific Recommendations

## CIM / CTM

Use:

* brightness layer
* one-hour nanmin composite

## PTM / PAM

Use:

* Polar_pB
* optionally positive-only filtering

## PIM

Use:

* adapter-generated Polar_pB

based on the legacy M/Z/P equations.

---

# Troubleshooting

## Too Many Zero Rows

Use:

```bash
--drop-zero-bins
```

## Negative Values

Use:

```bash
--positive-only
```

## Sparse Output

Try:

```bash
--no-drop-zero-bins
```

or inspect source product quality.

---

# Final Notes

This scaffolding is meant to allow rapid testing of new PUNCH products without rewriting science code each time.

Products mainly differ in:

* how layers are read
* how derived quantities are formed

Everything else should remain shared.

That is the core success of Pipeline v4.
