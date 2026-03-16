"""
Orchestrator for cloud-based storm attribute computation via AWS Lambda.

This module coordinates the full workflow:
  1. Load storm catalog
  2. Build/load granule index (earthaccess, cached)
  3. Get temporary S3 credentials for MERRA-2 access
  4. Dispatch one Lambda invocation per storm
  5. Collect results and save
"""

import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregation_registry import AGGREGATION_SPECS
from .auth import get_gesdisc_s3_credentials
from .catalog import build_granule_index, map_storm_to_granules

logger = logging.getLogger(__name__)


def _serialize_dataarray(da):
    """Serialize an xr.DataArray to a JSON-safe dict (no pickle).

    Encodes values as base64 numpy bytes and coordinates as lists,
    avoiding cross-version pickle compatibility issues.
    """
    return {
        "values_b64": base64.b64encode(np.ascontiguousarray(da.values).tobytes()).decode(),
        "dtype": str(da.dtype),
        "shape": list(da.shape),
        "time": [t.isoformat() for t in pd.DatetimeIndex(da.coords["time"].values)],
        "lat": da.coords["lat"].values.tolist(),
        "lon": da.coords["lon"].values.tolist(),
    }


def run_cloud_attributes(
    catalog_path,
    function_name="ar-worker",
    output_path=None,
    max_resident_timesteps=10,
    granule_cache_path=None,
    max_workers=1000,
    limit=None,
    region="us-west-2",
):
    """
    Main entry point for cloud-based storm attribute computation.

    Parameters
    ----------
    catalog_path : str or Path
        Path to HDF5 storm catalog.
    function_name : str
        Lambda function name. Default "ar-worker".
    output_path : str or Path, optional
        Where to save results. Defaults to <catalog>_cloud_attributes.h5.
    max_resident_timesteps : int
        Max timesteps to hold in memory per worker. Default 10.
    granule_cache_path : str or Path, optional
        Path to cache the granule index JSON.
    max_workers : int
        Max concurrent Lambda invocations. Default 1000.
    limit : int, optional
        Only process the first N storms (for testing).
    region : str
        AWS region. Default "us-west-2".

    Returns
    -------
    pd.DataFrame
        Storm catalog with computed attribute columns appended.
    """
    import boto3

    t_start = time.time()

    catalog_path = Path(catalog_path)
    if output_path is None:
        output_path = catalog_path.with_name(
            catalog_path.stem + "_cloud_attributes.h5"
        )
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

    # 3. Get temporary S3 credentials for MERRA-2 access
    logger.info("Fetching S3 credentials...")
    creds = get_gesdisc_s3_credentials()

    # 4. Build per-storm events
    t0 = time.time()
    logger.info("Building per-storm payloads...")
    events = []
    total_granules = 0
    for idx, row in storms.iterrows():
        storm_da = row.data_array
        urls = map_storm_to_granules(storm_da, granule_index)
        total_granules += sum(len(v) for v in urls.values())
        event = {
            "storm_id": int(idx),
            "storm_mask": _serialize_dataarray(storm_da),
            "granule_urls": urls,
            "s3_credentials": creds,
            "max_resident_timesteps": max_resident_timesteps,
        }
        events.append((idx, event))
    logger.info(
        "Built %d payloads (%d total granules, %.1f avg) in %.1fs",
        len(events), total_granules, total_granules / len(events),
        time.time() - t0,
    )

    # 5. Dispatch via Lambda
    from botocore.config import Config as BotoConfig
    boto_config = BotoConfig(
        max_pool_connections=max_workers,
        read_timeout=900,  # match Lambda's max timeout
    )
    lambda_client = boto3.client("lambda", region_name=region, config=boto_config)
    logger.info(
        "Dispatching %d storms to Lambda (max %d concurrent)...",
        len(events), max_workers,
    )

    results = {}
    errors = []
    billed_durations_ms = []
    bytes_read_per_storm = []

    def invoke_one(idx, event):
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            LogType="Tail",
            Payload=json.dumps(event).encode(),
        )
        payload = json.loads(response["Payload"].read())
        if "FunctionError" in response:
            raise RuntimeError(
                f"Storm {idx}: {payload.get('errorMessage', payload)}"
            )

        # Parse billed duration from Lambda log tail
        billed_ms = _parse_billed_duration(response.get("LogResult", ""))
        return idx, payload, billed_ms

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(invoke_one, idx, event): idx
            for idx, event in events
        }
        logger.info("All %d invocations submitted.", len(futures))

        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                idx, result, billed_ms = future.result()
                storm_bytes = result.pop("_bytes_read", None)
                results[idx] = result
                if billed_ms is not None:
                    billed_durations_ms.append(billed_ms)
                if storm_bytes is not None:
                    bytes_read_per_storm.append(storm_bytes)
            except Exception as e:
                errors.append(str(e))
                logger.error("Error: %s", e)

            if completed % 100 == 0 or completed == len(events):
                elapsed = time.time() - t_start
                logger.info(
                    "  Progress: %d/%d storms (%.1fs, %.1f storms/s)",
                    completed, len(events), elapsed, completed / elapsed,
                )

    if errors:
        logger.warning("%d storms failed", len(errors))

    # 6. Assemble output
    logger.info("Assembling results...")
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.index = results_df.index.astype(storms.index.dtype)
    output = pd.concat([storms, results_df], axis=1)

    # 7. Save
    logger.info("Saving results to %s", output_path)
    output.to_hdf(str(output_path), key="catalog")

    # 8. Stats
    t_total = time.time() - t_start
    _log_stats(len(results), len(storms), t_total, billed_durations_ms,
               bytes_read_per_storm, memory_mb=2048)

    return output


def _parse_billed_duration(log_result_b64):
    """Extract billed duration in ms from Lambda's base64-encoded log tail."""
    import re

    if not log_result_b64:
        return None
    try:
        log_text = base64.b64decode(log_result_b64).decode("utf-8", errors="replace")
        match = re.search(r"Billed Duration: (\d+) ms", log_text)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None


def _log_stats(n_success, n_total, wall_time, billed_durations_ms,
               bytes_read_per_storm, memory_mb):
    """Log execution stats and cost estimate."""
    logger.info("--- Execution Stats ---")
    logger.info("  Wall time: %.1fs", wall_time)
    logger.info("  Storms: %d/%d succeeded", n_success, n_total)
    if wall_time > 0:
        logger.info("  Throughput: %.2f storms/s", n_success / wall_time)

    if bytes_read_per_storm:
        total_bytes = np.sum(bytes_read_per_storm)
        total_gb = total_bytes / (1024 ** 3)
        avg_mb = np.mean(bytes_read_per_storm) / (1024 ** 2)
        logger.info(
            "  Data read: %.2f GB total (%.1f MB avg per storm)",
            total_gb, avg_mb,
        )

    if not billed_durations_ms:
        return

    durations = np.array(billed_durations_ms) / 1000.0  # to seconds
    logger.info(
        "  Billed duration: avg=%.1fs, min=%.1fs, max=%.1fs, total=%.0fs",
        np.mean(durations), np.min(durations), np.max(durations),
        np.sum(durations),
    )

    # Lambda pricing: $0.0000166667 per GB-second
    gb_seconds = np.sum(durations) * (memory_mb / 1024.0)
    cost_usd = gb_seconds * 0.0000166667
    # Request pricing: $0.20 per 1M requests
    request_cost = len(billed_durations_ms) * 0.0000002
    total_cost = cost_usd + request_cost

    logger.info(
        "  Estimated cost: $%.4f (%.1f GB-s compute + $%.4f requests)",
        total_cost, gb_seconds, request_cost,
    )
