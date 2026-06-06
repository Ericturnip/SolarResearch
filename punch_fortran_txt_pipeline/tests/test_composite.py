import numpy as np
from astropy.time import Time
from punch_pipeline_v4.models import BinnedMap
from punch_pipeline_v4.processing.composite import composite_binned_maps


def make_map(values, t):
    arr = np.array(values, dtype=float)
    return BinnedMap(
        product="TEST",
        layer_name="brightness",
        values=arr,
        hpln_centers=np.zeros_like(arr),
        hplt_centers=np.zeros_like(arr),
        timestamp=Time(t),
        time_map=np.full(arr.shape, t, dtype="<U30"),
    )


def test_min_ignores_zero():
    a = make_map([[0, 20]], "2025-01-01T00:00:00")
    b = make_map([[10, 30]], "2025-01-01T00:30:00")
    result = composite_binned_maps([a, b], method="nanmin", drop_zero_before_stat=True)
    assert result.binned_map.values[0, 0] == 10
    assert result.binned_map.values[0, 1] == 20


def test_percentile_writes_nearest_real_sample_with_matching_time():
    a = make_map([[10, 100]], "2025-01-01T00:00:00")
    b = make_map([[20, 400]], "2025-01-01T00:10:00")
    c = make_map([[30, 900]], "2025-01-01T00:20:00")

    result = composite_binned_maps(
        [a, b, c],
        method="percentile",
        percentile=25,
        drop_zero_before_stat=True,
    )

    assert result.binned_map.values[0, 0] == 10
    assert result.binned_map.time_map[0, 0] == "2025-01-01T00:00:00"
    assert result.binned_map.values[0, 1] == 100
    assert result.binned_map.time_map[0, 1] == "2025-01-01T00:00:00"
