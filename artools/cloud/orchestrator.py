"""
Orchestrator for cloud-based storm attribute computation via Lithops.

This module coordinates the full workflow:
  1. Load storm catalog
  2. Build/load granule index (earthaccess, cached)
  3. Load static data (AIS mask, cell areas, climatology)
  4. Map storms to granule URLs
  5. Dispatch workers via Lithops
  6. Collect results and save
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..loading_utils import load_ais, load_cell_areas
from .aggregation_registry import AGGREGATION_SPECS, MERRA2_COLLECTIONS
from .auth import get_gesdisc_s3_credentials
from .catalog import build_granule_index, map_storm_to_granules

logger = logging.getLogger(__name__)


def _load_climatology(granule_index, s3_credentials=None):
    """
    Load the MERRA-2 monthly climatology dataset.

    This is needed for anomaly computations. The climatology is loaded once
    and distributed to all workers as shared data.

    Parameters
    ----------
    granule_index : dict
        The granule index (unused here, but available for future use).
    s3_credentials : dict, optional
        S3 credentials for direct access. If None, uses earthaccess.

    Returns
    -------
    xr.Dataset or None
        Monthly climatology dataset, or None if unavailable.
    """
    import earthaccess

    try:
        earthaccess.login()
        doi = MERRA2_COLLECTIONS["climatology"]["doi"]
        granules = earthaccess.search_data(
            doi=doi,
            temporal=("1980-01-01", "2022-12-31"),
        )
        if not granules:
            logger.warning("No climatology granules found")
            return None

        pointers = earthaccess.open(granules, show_progress=False)
        datasets = [
            xr.open_dataset(p, engine="h5netcdf").sel(lat=slice(-86, -39))
            for p in pointers
        ]
        ds = xr.concat(datasets, dim="time")
        ds = ds.assign_coords(
            lat=ds.lat.round(5),
            lon=ds.lon.round(5),
        )
        # Compute the monthly climatology (average over years per month)
        climatology = ds.groupby("time.month").mean()
        return climatology
    except Exception as e:
        logger.warning("Failed to load climatology: %s", e)
        return None


def run_cloud_attributes(
    catalog_path,
    lithops_config=None,
    output_path=None,
    max_resident_timesteps=10,
    granule_cache_path=None,
    aggregation_specs=None,
    local_mode=False,
    static_data_path=None,
    limit=None,
):
    """
    Main entry point for cloud-based storm attribute computation.

    Parameters
    ----------
    catalog_path : str or Path
        Path to HDF5 storm catalog (e.g., from antarctic_AR_catalogs).
    lithops_config : dict, optional
        Lithops configuration override. If None, uses default lithops config.
    output_path : str or Path, optional
        Where to save results. Defaults to catalog_path with '_cloud_attributes' suffix.
    max_resident_timesteps : int
        Max number of timesteps to hold in memory simultaneously per worker.
        Higher values increase throughput; lower values reduce memory. Default 10.
    granule_cache_path : str or Path, optional
        Path to cache the granule index JSON. Avoids re-querying earthaccess.
    aggregation_specs : list of AggregationSpec, optional
        Aggregation specs to compute. Defaults to AGGREGATION_SPECS.
    local_mode : bool
        If True, use Lithops LocalhostExecutor for testing. Default False.
    static_data_path : str or Path, optional
        Local directory containing AIS mask and cell area files.
        If None, downloads from HuggingFace.

    Returns
    -------
    pd.DataFrame
        Storm catalog with computed attribute columns appended.
    """
    import lithops

    t_start = time.time()

    catalog_path = Path(catalog_path)
    if output_path is None:
        output_path = catalog_path.with_name(
            catalog_path.stem + "_cloud_attributes.h5"
        )
    if aggregation_specs is None:
        aggregation_specs = AGGREGATION_SPECS
    if granule_cache_path is None:
        granule_cache_path = catalog_path.parent / "granule_cache.json"

    # 1. Load storm catalog
    logger.info("Loading storm catalog from %s", catalog_path)
    storms = pd.read_hdf(catalog_path)
    if limit:
        storms = storms.iloc[:limit]
    logger.info("Loaded %d storms", len(storms))

    # 2. Build or load cached granule index
    t0 = time.time()
    logger.info("Building granule index...")
    granule_index = build_granule_index(cache_path=granule_cache_path)
    logger.info("Granule index ready (%.1fs)", time.time() - t0)

    # 3. Load static data — must .compute() so data is in-memory for serialization
    t0 = time.time()
    logger.info("Loading static data (AIS mask, cell areas)...")
    ais_mask = load_ais(load_path=static_data_path).compute()
    cell_areas = load_cell_areas(load_path=static_data_path).compute()

    logger.info("Loading climatology...")
    climatology = _load_climatology(granule_index)
    logger.info("Static data ready (%.1fs)", time.time() - t0)

    # 4. Build per-storm payloads
    t0 = time.time()
    logger.info("Building per-storm payloads...")
    payloads = []
    total_granules = 0
    for idx, row in storms.iterrows():
        storm_da = row.data_array
        urls = map_storm_to_granules(storm_da, granule_index)
        n_granules = sum(len(v) for v in urls.values())
        total_granules += n_granules
        payloads.append({
            "storm_payload": {
                "storm_id": idx,
                "storm_mask": storm_da,
                "granule_urls": urls,
                "aggregation_specs": aggregation_specs,
                "max_resident_timesteps": max_resident_timesteps,
            }
        })
    logger.info(
        "Built %d payloads (%d total granules, %.1f avg per storm) in %.1fs",
        len(payloads), total_granules, total_granules / len(payloads),
        time.time() - t0,
    )

    # 5. Dispatch via Lithops
    if local_mode:
        fexec = lithops.LocalhostExecutor(config=lithops_config)
    else:
        fexec = lithops.FunctionExecutor(config=lithops_config)

    from .worker import process_storm

    # Shared data: static arrays + S3 credentials
    logger.info("Fetching S3 credentials...")
    creds = get_gesdisc_s3_credentials()
    shared_data = {
        "ais_mask": ais_mask,
        "cell_areas": cell_areas,
        "climatology": climatology,
        "s3_credentials": creds,
    }

    for p in payloads:
        p["storm_payload"].update(shared_data)

    logger.info("Dispatching %d storms via Lithops...", len(payloads))
    futures = fexec.map(process_storm, payloads)
    results = fexec.get_result()

    _log_execution_stats(futures, len(payloads), t_start)

    # 6. Assemble output DataFrame
    logger.info("Assembling results...")
    results_df = pd.DataFrame(results, index=storms.index)
    output = pd.concat([storms, results_df], axis=1)

    # 7. Save
    logger.info("Saving results to %s", output_path)
    output.to_hdf(str(output_path), key="catalog")

    t_total = time.time() - t_start
    logger.info(
        "=== COMPLETE === %d storms processed in %.1fs wall time (%.2f storms/s)",
        len(storms), t_total, len(storms) / t_total,
    )

    return output


def _log_execution_stats(futures, n_storms, t_run_start):
    """Log comprehensive timing, throughput, and cost stats from Lithops futures."""
    wall = time.time() - t_run_start
    logger.info("--- Execution Stats ---")
    logger.info("  Total wall time: %.1fs", wall)
    logger.info("  Storms processed: %d", n_storms)
    logger.info("  Throughput: %.2f storms/s", n_storms / wall if wall else 0)

    # Extract per-worker stats from futures
    worker_times = []
    worker_exec_times = []
    for f in futures:
        s = getattr(f, "stats", {})
        if "worker_func_exec_time" in s:
            worker_times.append(s["worker_func_exec_time"])
        if "worker_exec_time" in s:
            worker_exec_times.append(s["worker_exec_time"])

    if not worker_times:
        return

    agg_compute = sum(worker_times)

    logger.info("  Aggregate compute time: %.1fs", agg_compute)
    logger.info(
        "  Worker func time: avg=%.1fs, min=%.1fs, max=%.1fs",
        np.mean(worker_times), min(worker_times), max(worker_times),
    )
    if worker_exec_times:
        logger.info(
            "  Worker total time (incl. overhead): avg=%.1fs, max=%.1fs",
            np.mean(worker_exec_times), max(worker_exec_times),
        )

    # Cost estimate: Lambda pricing is $0.0000166667 per GB-second
    # worker_exec_time is the billed duration (includes init overhead)
    billed_seconds = sum(worker_exec_times) if worker_exec_times else agg_compute
    gb_seconds = billed_seconds * 2.0  # 2048 MB = 2 GB
    cost_usd = gb_seconds * 0.0000166667
    logger.info(
        "  Estimated Lambda cost: $%.4f (%.1f GB-seconds @ 2GB)",
        cost_usd, gb_seconds,
    )
