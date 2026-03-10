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
        ds = xr.open_mfdataset(pointers)
        # Subset to Antarctic region and round coords
        ds = ds.sel(lat=slice(-86, -39))
        ds = ds.assign_coords(
            lat=ds.lat.round(5),
            lon=ds.lon.round(5),
        )
        # Compute the monthly climatology (average over years per month)
        climatology = ds.groupby("time.month").mean()
        return climatology.compute()
    except Exception as e:
        logger.warning("Failed to load climatology: %s", e)
        return None


def run_cloud_attributes(
    catalog_path,
    lithops_config=None,
    output_path=None,
    batch_size=None,
    max_resident_timesteps=10,
    granule_cache_path=None,
    aggregation_specs=None,
    local_mode=False,
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
    batch_size : int, optional
        Process storms in batches of this size (for credential refresh).
        If None, processes all at once.
    max_resident_timesteps : int
        Max number of timesteps to hold in memory simultaneously per worker.
        Higher values increase throughput; lower values reduce memory. Default 10.
    granule_cache_path : str or Path, optional
        Path to cache the granule index JSON. Avoids re-querying earthaccess.
    aggregation_specs : list of AggregationSpec, optional
        Aggregation specs to compute. Defaults to AGGREGATION_SPECS.
    local_mode : bool
        If True, use Lithops LocalhostExecutor for testing. Default False.

    Returns
    -------
    pd.DataFrame
        Storm catalog with computed attribute columns appended.
    """
    import lithops

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
    logger.info("Loaded %d storms", len(storms))

    # 2. Build or load cached granule index
    logger.info("Building granule index...")
    granule_index = build_granule_index(cache_path=granule_cache_path)

    # 3. Load static data
    logger.info("Loading static data (AIS mask, cell areas)...")
    ais_mask = load_ais()
    cell_areas = load_cell_areas()

    logger.info("Loading climatology...")
    climatology = _load_climatology(granule_index)

    # 4. Build per-storm payloads
    logger.info("Building per-storm payloads...")
    payloads = []
    for idx, row in storms.iterrows():
        storm_da = row.data_array
        urls = map_storm_to_granules(storm_da, granule_index)
        payloads.append({
            "storm_id": idx,
            "storm_mask": storm_da,
            "granule_urls": urls,
            "aggregation_specs": aggregation_specs,
            "max_resident_timesteps": max_resident_timesteps,
        })

    # 5. Dispatch via Lithops
    logger.info("Dispatching %d storms via Lithops...", len(payloads))

    if local_mode:
        fexec = lithops.LocalhostExecutor(config=lithops_config)
    else:
        fexec = lithops.FunctionExecutor(config=lithops_config)

    # Import worker function
    from .worker import process_storm

    # Shared data passed to all workers (Lithops stores in cloud storage)
    shared_data = {
        "ais_mask": ais_mask,
        "cell_areas": cell_areas,
        "climatology": climatology,
    }

    if batch_size:
        all_results = []
        n_batches = (len(payloads) + batch_size - 1) // batch_size

        for i, batch_start in enumerate(range(0, len(payloads), batch_size)):
            batch = payloads[batch_start : batch_start + batch_size]

            # Refresh credentials per batch
            logger.info(
                "Batch %d/%d: dispatching %d storms, refreshing credentials...",
                i + 1, n_batches, len(batch),
            )
            creds = get_gesdisc_s3_credentials()
            shared_data["s3_credentials"] = creds

            # Merge shared data into each payload
            batch_with_shared = [{**p, **shared_data} for p in batch]

            fexec.map(process_storm, batch_with_shared)
            batch_results = fexec.get_result()
            all_results.extend(batch_results)

        results = all_results
    else:
        creds = get_gesdisc_s3_credentials()
        shared_data["s3_credentials"] = creds

        # Merge shared data into each payload
        payloads_with_shared = [{**p, **shared_data} for p in payloads]

        fexec.map(process_storm, payloads_with_shared)
        results = fexec.get_result()

    # 6. Assemble output DataFrame
    logger.info("Assembling results...")
    results_df = pd.DataFrame(results, index=storms.index)
    output = pd.concat([storms, results_df], axis=1)

    # 7. Save
    logger.info("Saving results to %s", output_path)
    output.to_hdf(str(output_path), key="catalog")

    return output
