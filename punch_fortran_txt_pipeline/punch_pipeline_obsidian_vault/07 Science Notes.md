# Science Notes

## Brightness Scale

The pipeline converts native PUNCH brightness to S10 with the coefficient in:

```text
src/punch_pipeline_v4/processing/units.py
```

If a PI says the old CIM tomography program used a different brightness
correction, compare this coefficient and the old preprocessing step first.

## p25 Versus Old Maps

The current hourly map is not a mean map and not a strict minimum map.

It uses p25 as a target and writes the nearest real sample to that target. This
reduces the chance that one bad low value controls the hour, while still biasing
toward lower brightness than a median.

## Radial Brightness Correction

This repo does not add a radial falloff correction unless it is added explicitly
in code. If an older workflow added brightness back before tomography, outputs
from this pipeline may have more negative values.

That difference should be tested against one known-good CIM hour from the older
pipeline.
