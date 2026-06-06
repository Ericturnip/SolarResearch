#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from urllib.parse import urljoin
from urllib.request import urlopen, urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from punch_pipeline_v4.adapters.registry import get_adapter
from punch_pipeline_v4.processing.binning import median_bin_layer_frame
from punch_pipeline_v4.writers.ascii import write_tomography_txt


DEFAULT_URL = "https://umbra.nascom.nasa.gov/punch/3/PTM/2025/10/24/"
DEFAULT_VMIN = -100.0
DEFAULT_VMAX = 500.0


@dataclass(frozen=True)
class ProcessedSample:
    fits_path: Path
    txt_path: Path
    plot_path: Path
    timestamp: str
    rows_written: int
    finite_bins: int
    zero_bins: int
    value_min: float
    value_max: float


def list_ptm_files(index_url: str) -> list[str]:
    with urlopen(index_url) as response:
        html = response.read().decode("utf-8", errors="replace")

    names = sorted(set(re.findall(r'href="([^"]*PUNCH_L3_PTM_[^"]*\.fits)"', html)))
    if not names:
        raise RuntimeError(f"No PTM FITS links found at {index_url}")
    return names


def select_evenly_spaced(names: list[str], count: int) -> list[str]:
    if count >= len(names):
        return names
    indices = np.linspace(0, len(names) - 1, count, dtype=int)
    return [names[int(i)] for i in indices]


def download_file(index_url: str, name: str, fits_dir: Path) -> Path:
    fits_dir.mkdir(parents=True, exist_ok=True)
    path = fits_dir / Path(name).name
    if path.exists() and path.stat().st_size > 0:
        print(f"[download] keeping existing {path}")
        return path

    url = urljoin(index_url, name)
    print(f"[download] {url}")
    urlretrieve(url, path)
    return path


def plot_binned_map(bmap, plot_path: Path, title: str) -> None:
    values = np.asarray(bmap.values, dtype=np.float64)
    mask = np.isfinite(values) & (values != 0.0)
    ra, dec = _coords_for_plot(bmap)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(
        ra[mask.ravel()],
        dec[mask.ravel()],
        c=values.ravel()[mask.ravel()],
        cmap="plasma",
        vmin=DEFAULT_VMIN,
        vmax=DEFAULT_VMAX,
        s=18,
        marker="s",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Brightness (S10 Units)")
    ax.set_title(title)
    ax.set_xlabel("Right Ascension (RA) [deg]")
    ax.set_ylabel("Declination (DEC) [deg]")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _coords_for_plot(bmap):
    hpln = bmap.hpln_centers.ravel()
    hplt = bmap.hplt_centers.ravel()
    if bmap.solar_wcs is not None and bmap.radec_wcs is not None:
        px, py = bmap.solar_wcs.world_to_pixel_values(hpln, hplt)
        ra, dec = bmap.radec_wcs.pixel_to_world_values(px, py)
        return np.asarray(ra), np.asarray(dec)
    return hpln, hplt


def process_one(fits_path: Path, output_dir: Path, bin_size_deg: float) -> ProcessedSample:
    adapter = get_adapter("PTM")
    frame = adapter.load_layer(fits_path, "Polar_pB")
    bmap, _ = median_bin_layer_frame(frame, bin_size_deg=bin_size_deg, convert_to_s10=True)
    bmap.unit = "S10"

    values = np.asarray(bmap.values, dtype=np.float64)
    finite = np.isfinite(values)
    zero_bins = finite & (values == 0.0)
    nonzero = finite & (values != 0.0)

    stem = f"PUNCH_L3_PTM_{bmap.timestamp.strftime('%Y%m%d%H%M%S')}_{bmap.layer_name}_BIN_drop_zeros"
    txt_path = output_dir / "txt" / f"{stem}.txt"
    plot_path = output_dir / "plots" / f"{stem}_plot_m100_500.png"

    report = write_tomography_txt(
        bmap,
        txt_path,
        values_are_s10=False,
        drop_zero_bins=True,
    )
    plot_binned_map(
        bmap,
        plot_path,
        title=f"PTM Polar_pB {bmap.timestamp.isot} (zeros dropped, color -100..500 S10)",
    )

    return ProcessedSample(
        fits_path=fits_path,
        txt_path=txt_path,
        plot_path=plot_path,
        timestamp=bmap.timestamp.isot,
        rows_written=int(report["rows_written"]),
        finite_bins=int(finite.sum()),
        zero_bins=int(zero_bins.sum()),
        value_min=float(np.nanmin(values[nonzero])) if np.any(nonzero) else float("nan"),
        value_max=float(np.nanmax(values[nonzero])) if np.any(nonzero) else float("nan"),
    )


def write_comparison_plot(samples: list[ProcessedSample], output_dir: Path) -> Path:
    image_paths = [sample.plot_path for sample in samples]
    images = [plt.imread(path) for path in image_paths]
    fig, axes = plt.subplots(len(images), 1, figsize=(10, 4 * len(images)))
    if len(images) == 1:
        axes = [axes]
    for ax, image, sample in zip(axes, images, samples):
        ax.imshow(image)
        ax.set_title(Path(sample.fits_path).stem)
        ax.axis("off")
    out_path = output_dir / "PTM_20251024_five_sample_plots.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PTM samples, median-bin Polar_pB, drop zero bins, and save plots."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "ptm_20251024_samples"))
    parser.add_argument("--bin-size-deg", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    fits_dir = output_dir / "fits"

    names = list_ptm_files(args.url)
    selected = select_evenly_spaced(names, args.count)
    print(f"[select] {len(selected)} of {len(names)} files")
    for name in selected:
        print(f"  {Path(name).name}")

    samples: list[ProcessedSample] = []
    for name in selected:
        fits_path = download_file(args.url, name, fits_dir)
        sample = process_one(fits_path, output_dir, args.bin_size_deg)
        samples.append(sample)
        print(
            "[processed] "
            f"{sample.timestamp} rows={sample.rows_written} "
            f"finite_bins={sample.finite_bins} zero_bins={sample.zero_bins} "
            f"range={sample.value_min:.2f}..{sample.value_max:.2f} "
            f"plot={sample.plot_path}"
        )

    comparison = write_comparison_plot(samples, output_dir)
    print(f"[comparison] {comparison}")


if __name__ == "__main__":
    main()
