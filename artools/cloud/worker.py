"""
AWS Lambda worker for processing a single storm.

This module contains the Lambda handler and the core processing function.
Static data (AIS mask, cell areas, climatology) is loaded from disk at
cold start and cached for subsequent invocations.

The handler receives a storm's binary mask and granule URLs, loads
shared data from the container image, and returns computed attributes.
"""

import base64
import logging
import os

import numpy as np
import s3fs
import xarray as xr

from .accumulators import TEMPORAL_REDUCERS
from .aggregation_registry import AGGREGATION_SPECS, MERRA2_COLLECTIONS
from .spatial_functions import SPATIAL_FUNCTIONS, _apply_mask

logger = logging.getLogger(__name__)

def _default_static_data_dir():
    """Return the static data directory, checking both container and layer paths."""
    # Layer deployment: files are extracted to /opt/
    layer_path = "/opt/static_data"
    # Container deployment: files are in the task root
    container_path = "/var/task/static_data"
    if os.path.isdir(layer_path):
        return layer_path
    return container_path

STATIC_DATA_DIR = os.environ.get("STATIC_DATA_DIR") or _default_static_data_dir()

# Module-level cache for static data (persists across warm Lambda invocations)
_static_data = {}


def _load_static_data():
    """Load static data from disk on first invocation (cold start)."""
    if _static_data:
        return _static_data

    logger.info("Cold start: loading static data from %s", STATIC_DATA_DIR)

    # AIS mask — boolean mask of Antarctic Ice Sheet grid cells
    ais_ds = xr.open_dataset(
        os.path.join(STATIC_DATA_DIR, "AIS_Full_basins_Zwally_MERRA2grid_new.nc"),
        engine="h5netcdf",
    )
    ais_mask = (ais_ds.Zwallybasins > 0).sel(lat=slice(-86, -39))
    ais_mask = ais_mask.assign_coords(
        lat=ais_mask.lat.round(5), lon=ais_mask.lon.round(5)
    ).load()

    # Cell areas — area in m² of each MERRA-2 grid cell
    areas_ds = xr.open_dataset(
        os.path.join(STATIC_DATA_DIR, "MERRA2_gridarea.nc"),
        engine="h5netcdf",
    )
    cell_areas = areas_ds.cell_area
    cell_areas = cell_areas.assign_coords(
        lat=cell_areas.lat.round(5), lon=cell_areas.lon.round(5)
    ).load()

    # Monthly climatology for anomaly computations
    climatology = xr.open_dataset(
        os.path.join(STATIC_DATA_DIR, "MERRA2_monthly_climatology.nc"),
        engine="h5netcdf",
    ).load()

    _static_data["ais_mask"] = ais_mask
    _static_data["cell_areas"] = cell_areas
    _static_data["climatology"] = climatology

    logger.info("Static data loaded")
    return _static_data


def _deserialize_dataarray(d):
    """Reconstruct an xr.DataArray from the JSON-safe dict format."""
    import pandas as pd

    values = np.frombuffer(
        base64.b64decode(d["values_b64"]), dtype=d["dtype"]
    ).reshape(d["shape"])
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.DatetimeIndex(d["time"]),
            "lat": np.array(d["lat"], dtype=np.float64),
            "lon": np.array(d["lon"], dtype=np.float64),
        },
    )


def lambda_handler(event, context):
    """AWS Lambda entry point."""
    storm_da = _deserialize_dataarray(event["storm_mask"])
    static = _load_static_data()

    payload = {
        "storm_mask": storm_da,
        "ais_mask": static["ais_mask"],
        "cell_areas": static["cell_areas"],
        "climatology": static["climatology"],
        "granule_urls": event["granule_urls"],
        "s3_credentials": event.get("s3_credentials", {}),
        "aggregation_specs": AGGREGATION_SPECS,
        "max_resident_timesteps": event.get("max_resident_timesteps", 10),
    }

    result = process_storm(payload)

    # Convert numpy types to JSON-serializable Python types
    return {
        k: float(v) if v is not None else None
        for k, v in result.items()
    }


# ---------------------------------------------------------------------------
# Self-contained copies of utility functions from attribute_utils.py.
# Copied here to avoid pulling in st_dbscan/sklearn on Lambda workers.
# ---------------------------------------------------------------------------


def _align_storm_coords(storm_da, reference_da):
    """
    Snap a storm DataArray's floating-point coordinates to perfectly match
    a reference grid (e.g., area_da or ais_da), preventing downstream
    AlignmentErrors from floating point errors.
    """
    clean_lat = reference_da.sel(lat=storm_da.lat, method="nearest").lat
    clean_lon = reference_da.sel(lon=storm_da.lon, method="nearest").lon
    return storm_da.assign_coords(lat=clean_lat, lon=clean_lon)


def _augment_storm_da(storm_da):
    """
    For any grid cell which had AR conditions, extend AR conditions to all
    grid cells 24 hours later. Used for precipitation attribution.
    """
    import pandas as pd

    start = storm_da.time.values[0]
    end = storm_da.time.values[-1] + np.timedelta64(1, "D")
    full_dates = pd.date_range(start, end, freq="3h")

    unincluded_times = set(np.array(full_dates)) - set(storm_da.time.values)

    unincluded_array = np.zeros(
        (len(unincluded_times), storm_da.shape[1], storm_da.shape[2])
    )
    unincluded_coords = {
        "time": np.array(list(unincluded_times)),
        "lat": storm_da.lat.values,
        "lon": storm_da.lon.values,
    }
    unincluded_da = xr.DataArray(unincluded_array, coords=unincluded_coords)

    augmented_da = xr.concat([storm_da, unincluded_da], dim="time")
    augmented_da = augmented_da.rolling(time=8, min_periods=1).max()

    return augmented_da


def _make_s3fs(credentials):
    """Create an s3fs filesystem from NASA Earthdata S3 credentials."""
    return s3fs.S3FileSystem(
        key=credentials["accessKeyId"],
        secret=credentials["secretAccessKey"],
        token=credentials["sessionToken"],
        client_kwargs={"region_name": "us-west-2"},
    )


def _open_merra2(url, fs, half_hour=False, session=None):
    """
    Open a single MERRA-2 granule as an xarray Dataset.

    Parameters
    ----------
    url : str
        S3 or HTTPS URL to the granule.
    fs : s3fs.S3FileSystem or None
        Authenticated S3 filesystem (for S3 URLs).
    half_hour : bool
        If True, shift time coordinates by -30 minutes.
    session : requests.Session or None
        Authenticated earthdata session (for HTTPS URLs).

    Returns
    -------
    xr.Dataset
    """
    if url.startswith("s3://"):
        fileobj = fs.open(url, cache_type="blockcache", block_size=1 * 1024 * 1024)
    else:
        # HTTPS URL — download to temp file with authenticated session
        import tempfile
        import shutil

        resp = session.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".nc4", delete=False)
        shutil.copyfileobj(resp.raw, tmp)
        tmp.close()
        fileobj = tmp.name
    ds = xr.open_dataset(fileobj, engine="h5netcdf")

    # Round coordinates to avoid floating-point mismatch
    ds = ds.assign_coords(
        lat=ds.lat.round(5),
        lon=ds.lon.round(5),
    )

    if half_hour:
        ds = ds.assign_coords(time=ds.time - np.timedelta64(30, "m"))

    return ds


def _find_first_landfall(storm_da, ais_mask):
    """
    Find the first timestep at which the storm intersects the AIS.

    Returns None if the storm never makes landfall.
    """
    storm_aligned = _align_storm_coords(storm_da, ais_mask)
    ais_subset = ais_mask.sel(lat=storm_aligned.lat, lon=storm_aligned.lon)
    storm_ais = storm_aligned.where(ais_subset, 0)

    landfall_times = storm_da.time[storm_ais.any(dim=["lat", "lon"])]
    if len(landfall_times) == 0:
        return None
    return landfall_times.values[0]


def _overlapping_times(ds, storm_da, half_hour=False):
    """
    Get timesteps from the dataset that overlap with the storm mask.

    For standard (3-hourly) data, selects times where hour % 3 == 0.
    Returns sorted array of overlapping numpy datetime64 values.
    """
    ds_times = ds.time.values
    storm_times = set(storm_da.time.values)

    # Filter to 3-hourly cadence if needed
    overlap = np.array(sorted(t for t in ds_times if t in storm_times))
    return overlap


def _get_precip_times(ds, augmented_da):
    """Get timesteps from precip dataset that overlap with augmented mask."""
    ds_times = ds.time.values
    aug_times = set(augmented_da.time.values)
    return np.array(sorted(t for t in ds_times if t in aug_times))


def process_storm(storm_payload):
    """
    Process a single storm and return summary statistics.

    Parameters
    ----------
    storm_payload : dict
        Must contain:
        - storm_mask: xr.DataArray — binary storm mask (time, lat, lon)
        - granule_urls: {collection_key: [s3_url, ...]}
        - s3_credentials: {accessKeyId, secretAccessKey, sessionToken}
        - ais_mask: xr.DataArray — AIS binary mask
        - cell_areas: xr.DataArray — grid cell areas
        - climatology: xr.Dataset — precomputed monthly climatology
        - aggregation_specs: list of AggregationSpec
        - max_resident_timesteps: int — batch size for timestep loading

    Returns
    -------
    dict
        {metric_name: scalar_value} for all aggregation specs.
    """
    storm_da = storm_payload["storm_mask"]
    ais_mask = storm_payload["ais_mask"]
    cell_areas = storm_payload["cell_areas"]
    climatology = storm_payload["climatology"]
    specs = storm_payload["aggregation_specs"]
    max_ts = storm_payload.get("max_resident_timesteps", 10)

    # Align storm coordinates to reference grid
    storm_da = _align_storm_coords(storm_da, cell_areas)

    # Set up data access — S3 for Lambda, HTTPS session for local testing
    creds = storm_payload.get("s3_credentials") or {}
    session = None
    fs = None
    if creds.get("accessKeyId"):
        fs = _make_s3fs(creds)
    # Check if any URL is HTTPS — if so, need an earthdata session
    all_urls = [u for urls in storm_payload["granule_urls"].values() for u in urls]
    if any(u.startswith("https://") for u in all_urls):
        import earthaccess
        auth = earthaccess.login()
        session = auth.get_session()

    # Subset static data to storm's spatial extent
    ais_subset = ais_mask.sel(lat=storm_da.lat, lon=storm_da.lon)
    areas_subset = cell_areas.sel(lat=storm_da.lat, lon=storm_da.lon)

    # Determine first landfall timestep
    first_landfall = _find_first_landfall(storm_da, ais_mask)

    # Build augmented mask for precipitation (24h lookahead)
    augmented_da = _augment_storm_da(storm_da)

    # Initialize accumulators for each spec
    accumulators = {}
    for spec in specs:
        accumulators[spec.output_name] = TEMPORAL_REDUCERS[spec.temporal_reducer]()

    # Group specs by collection
    specs_by_collection = {}
    for spec in specs:
        specs_by_collection.setdefault(spec.collection, []).append(spec)

    # Process each collection
    for collection_key, collection_specs in specs_by_collection.items():
        urls = storm_payload["granule_urls"].get(collection_key, [])
        if not urls:
            continue

        half_hour = MERRA2_COLLECTIONS[collection_key]["half_hour"]

        # Separate precip and non-precip specs
        precip_specs = [s for s in collection_specs if s.params.get("precip")]
        regular_specs = [s for s in collection_specs if not s.params.get("precip")]

        for url in urls:
            try:
                ds = _open_merra2(url, fs, half_hour=half_hour, session=session)
            except Exception as e:
                logger.warning("Failed to open %s: %s", url, e)
                continue

            # --- Process regular (non-precip) specs ---
            if regular_specs:
                timesteps = _overlapping_times(ds, storm_da, half_hour=half_hour)

                # Process timesteps in batches
                for batch_start in range(0, len(timesteps), max_ts):
                    batch_times = timesteps[batch_start : batch_start + max_ts]

                    # Pre-load all needed variables in a single .compute()
                    needed_vars = list({s.variable for s in regular_specs})
                    try:
                        batch_ds = ds[needed_vars].sel(
                            lat=storm_da.lat,
                            lon=storm_da.lon,
                            time=batch_times,
                        ).compute()
                    except Exception as e:
                        logger.warning(
                            "Failed to load variables from %s: %s", url, e,
                        )
                        batch_ds = None

                    # Derive raw and anomaly versions from the loaded batch
                    loaded_vars = {}
                    if batch_ds is not None:
                        for spec in regular_specs:
                            var_key = (spec.variable, spec.is_anomaly)
                            if var_key in loaded_vars:
                                continue
                            var_da = batch_ds[spec.variable]
                            if spec.is_anomaly and climatology is not None:
                                clim_var = climatology[spec.variable]
                                var_da = var_da.groupby("time.month") - clim_var
                                var_da = var_da.drop_vars("month")
                            loaded_vars[var_key] = var_da

                    # Compute spatial aggregations per timestep
                    for t in batch_times:
                        storm_t = storm_da.sel(time=t)

                        for spec in regular_specs:
                            # Skip non-landfall timesteps for first_landfall reducers
                            if spec.temporal_reducer == "first_landfall":
                                if first_landfall is None or t != first_landfall:
                                    continue

                            var_key = (spec.variable, spec.is_anomaly)
                            var_batch = loaded_vars.get(var_key)
                            if var_batch is None:
                                continue

                            var_t = var_batch.sel(time=t)
                            if spec.negate:
                                var_t = -var_t

                            combined_mask = _apply_mask(
                                storm_t, ais_subset, spec.mask,
                            )

                            spatial_func = SPATIAL_FUNCTIONS[spec.spatial_func]
                            value = spatial_func(var_t, combined_mask, areas_subset)
                            accumulators[spec.output_name].update(value)

                    # Free batch memory
                    del loaded_vars

            # --- Process precipitation specs ---
            if precip_specs:
                _process_precip_from_granule(
                    ds, precip_specs, augmented_da, ais_subset,
                    areas_subset, accumulators, half_hour,
                )

            # Free file memory
            del ds

    return {name: acc.finalize() for name, acc in accumulators.items()}


def _process_precip_from_granule(
    ds, precip_specs, augmented_da, ais_subset, areas_subset,
    accumulators, half_hour,
):
    """
    Handle precipitation variables from a single granule.

    Precipitation requires special treatment:
    - Uses augmented storm mask (24h lookahead)
    - Raw data is in kg/m²/s, needs conversion to per-3-hour amounts
    - PRECCU + PRECLS = total rainfall
    - PRECSN = snowfall
    """
    precip_vars = ["PRECLS", "PRECCU", "PRECSN"]

    # Check which precip vars are in this dataset
    available_vars = [v for v in precip_vars if v in ds.data_vars]
    if not available_vars:
        return

    # Get overlapping times with augmented mask
    timesteps = _get_precip_times(ds, augmented_da)
    if len(timesteps) == 0:
        return

    try:
        # Load precip variables, subset spatially
        precip_ds = ds[available_vars].sel(
            lat=augmented_da.lat,
            lon=augmented_da.lon,
            time=timesteps,
        )
        # Convert from kg/m²/s to kg/m² per 3-hour block
        precip_ds = (precip_ds * 3600).resample(time="3h").sum()
        precip_ds = precip_ds.assign_coords(
            lat=precip_ds.lat.round(5),
            lon=precip_ds.lon.round(5),
        )
        precip_ds = precip_ds.compute()
    except Exception as e:
        logger.warning("Failed to process precip: %s", e)
        return

    # Compute derived variables
    if "PRECCU" in precip_ds and "PRECLS" in precip_ds:
        precip_ds["_rainfall"] = precip_ds["PRECCU"] + precip_ds["PRECLS"]

    # Process each timestep
    resampled_times = precip_ds.time.values
    for t in resampled_times:
        if t not in augmented_da.time.values:
            continue

        aug_t = augmented_da.sel(time=t)

        for spec in precip_specs:
            if spec.variable not in precip_ds:
                continue

            var_t = precip_ds[spec.variable].sel(time=t)
            combined_mask = _apply_mask(aug_t, ais_subset, spec.mask)

            spatial_func = SPATIAL_FUNCTIONS[spec.spatial_func]
            value = spatial_func(var_t, combined_mask, areas_subset)
            accumulators[spec.output_name].update(value)
