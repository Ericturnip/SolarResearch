import numpy as np
from punch_pipeline_v4.processing.units import native_to_s10, s10_valid_mask


def test_native_to_s10():
    assert native_to_s10(4.5e-16) == 1.0


def test_zero_rejected_by_default():
    vals = np.array([0.0, 1.0, np.nan, 3000.0, -600.0, 5.0])
    mask = s10_valid_mask(vals)
    assert mask.tolist() == [False, True, False, False, False, True]
