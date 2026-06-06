# Products And Layers

| Product | Layer to request | Notes |
| --- | --- | --- |
| CTM | `brightness` | Main product used for tomography TXT runs. |
| PTM | `Polar_pB` | Available only when NASA has PTM folders for the date. |
| CIM | `brightness` | Useful for comparison with older PI workflows. |
| PIM | `Polar_pB` | Derived from M, Z, and P polarizer planes. |
| PAM | `Polar_pB` | Same three-plane style as PTM. |

## PIM pB Calculation

The adapter bins the raw polarizer planes first, then computes:

```text
Q = (4 / 3) Z - (2 / 3) (P + M)
U = (2 / sqrt(3)) P - (2 / sqrt(3)) M
pB = sqrt(Q^2 + U^2)
```

This avoids mixing raw pixels from different bins during the polarizer math.

## Adding A Product

1. Add an adapter in `src/punch_pipeline_v4/adapters/`.
2. Register it in `src/punch_pipeline_v4/adapters/registry.py`.
3. Add the default layer in `scripts/process_product_oct_nov_p25.py`.
4. Run a one-hour test before a month-long run.
