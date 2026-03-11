#!/usr/bin/env python3
"""
Download MERRA-2 monthly data and compute T2M/TQV climatology.

Step 1: Download all monthly granules to a local directory.
Step 2: Compute monthly climatology (mean per calendar month over 1980-2022).
Step 3: Save as a single NetCDF file.
"""

import argparse
import time
from pathlib import Path

import earthaccess
import numpy as np
import xarray as xr

DOI = "10.5067/5ESKGQTZG7FO"  # M2IMNXASM v5.12.4


def download_granules(download_dir):
    """Download all monthly granules to download_dir. Returns list of local paths."""
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    earthaccess.login()
    granules = earthaccess.search_data(
        doi=DOI, temporal=("1980-01-01", "2022-12-31")
    )
    print(f"Found {len(granules)} granules")

    paths = earthaccess.download(granules, str(download_dir))
    print(f"Downloaded {len(paths)} files to {download_dir}")
    return sorted(Path(p) for p in paths)


def build_climatology(download_dir, output_path):
    """Load downloaded files, compute monthly climatology, save."""
    download_dir = Path(download_dir)
    files = sorted(download_dir.glob("*.nc4"))
    if not files:
        raise FileNotFoundError(f"No .nc4 files found in {download_dir}")

    print(f"Processing {len(files)} files from {download_dir}")
    t0 = time.time()

    datasets = []
    for i, f in enumerate(files):
        ds = xr.open_dataset(f, engine="h5netcdf")
        ds = ds[["T2M", "TQV"]].sel(lat=slice(-86, -39))
        datasets.append(ds.load())
        ds.close()
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(files)} ({time.time() - t0:.1f}s)")

    print(f"All files loaded in {time.time() - t0:.1f}s")

    ds = xr.concat(datasets, dim="time")
    ds = ds.assign_coords(
        lat=ds.lat.round(5),
        lon=ds.lon.round(5),
    )

    climatology = ds.groupby("time.month").mean()
    climatology.encoding.pop("unlimited_dims", None)
    print(f"Climatology shape: {dict(climatology.dims)}")
    print(f"Variables: {list(climatology.data_vars)}")

    output_path = Path(output_path)
    climatology.to_netcdf(output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved to {output_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-dir",
        default="/home/espg/software/antarctic_AR_catalogs/M2IMNXASM_monthly",
        help="Directory to download raw monthly granules into.",
    )
    parser.add_argument(
        "--output",
        default="/home/espg/software/antarctic_AR_catalogs/MERRA2_monthly_climatology.nc",
        help="Output path for the climatology NetCDF.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use if files are already downloaded).",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_granules(args.download_dir)

    build_climatology(args.download_dir, args.output)


if __name__ == "__main__":
    main()
