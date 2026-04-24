from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SecondPassConfig:
    """Placeholder config for second-pass cleaning logic.

    Fill this once the current second-pass code is uploaded.
    """

    threshold: float | None = None
    coverage_min: float | None = None
    drop_zero_bins: bool = True


def run_second_pass_on_txt(input_path: str | Path, output_path: str | Path, config: SecondPassConfig) -> dict:
    """Placeholder second-pass stage.

    The v4 design keeps this stage distinct from median binning and hourly
    compositing. Once the existing second-pass code is available, port its logic
    here without changing the first-pass modules.
    """
    raise NotImplementedError("Second-pass logic pending upload of current working code.")
