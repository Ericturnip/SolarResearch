# Pipeline Flow

The production path is:

```text
NASA directory --> FITS download --> product adapter --> 1 degree sky binning --> hourly p25 target --> nearest real sample --> fixed-width TXT
```

## Product Adapter

Adapters live in `src/punch_pipeline_v4/adapters/`.

Simple products load one image layer. Products like PIM can implement their own
binning path when the requested layer is derived from several planes.

## Binning

Each FITS frame is mapped into a common sky grid. The default bin size is
`1.0` degree.

The per-frame binned value is a median of source pixels inside each sky bin.

## Hourly Map

Files are grouped by the UTC hour in the filename. For each output pixel, the
pipeline collects all valid binned values in that hour.

Zeros are treated as missing before the p25 target is computed.

The TXT value is the real sample nearest to the p25 target. Ties go to the
earlier file because `argmin` returns the first match in the sorted hour.

## Output

One TXT file is written per hour:

```text
outputs/<product>_hourly_p25/YYYY/MM/DD/PUNCH_L3_PRODUCT_YYYYMMDDHH_layer_p25_COMPOSITE.txt
```
