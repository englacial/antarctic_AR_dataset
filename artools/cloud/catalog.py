"""
Granule catalog builder for MERRA-2 S3 access.

Queries earthaccess ONCE for all MERRA-2 granules across all collections,
builds a {collection: {date_str: s3_url}} index, and caches it to disk.
Then maps individual storms to their required granule URLs.
"""

import json
import logging
from pathlib import Path

import earthaccess
import numpy as np

from .aggregation_registry import MERRA2_COLLECTIONS

logger = logging.getLogger(__name__)


def _extract_date_from_granule(granule) -> str:
    """Extract YYYYMMDD date string from an earthaccess granule object."""
    # earthaccess granules have temporal metadata; extract the start date
    umm = granule.get("umm", granule)
    try:
        # Try the standard UMM temporal extent
        temporal = umm["TemporalExtent"]["RangeDateTime"]
        begin = temporal["BeginningDateTime"]  # e.g., '2020-01-15T00:00:00.000Z'
        return begin[:10].replace("-", "")
    except (KeyError, TypeError):
        pass

    # Fallback: extract from data links (filename contains date)
    for link in granule.data_links():
        # MERRA-2 filenames contain YYYYMMDD as the second-to-last dot segment
        # e.g., MERRA2_400.tavg1_2d_slv_Nx.20200115.nc4
        parts = link.split(".")
        for part in parts:
            if len(part) == 8 and part.isdigit():
                return part

    raise ValueError(f"Could not extract date from granule: {granule}")


def _extract_s3_url(granule) -> str:
    """Extract the S3 URL from an earthaccess granule object."""
    links = granule.data_links(access="direct")
    if links:
        return links[0]
    # Fallback to any available link
    links = granule.data_links()
    if links:
        return links[0]
    raise ValueError(f"No data links found for granule: {granule}")


def build_granule_index(
    time_range=("1980-01-01", "2022-12-31"),
    collections=None,
    cache_path=None,
):
    """
    Query earthaccess for all MERRA-2 granules and build a date-indexed catalog.

    Parameters
    ----------
    time_range : tuple of str
        (start_date, end_date) in YYYY-MM-DD format.
    collections : list of str, optional
        Collection keys to query. Defaults to all non-climatology collections.
    cache_path : str or Path, optional
        Path to cache the index as JSON. If the cache exists and is non-empty,
        it is loaded instead of re-querying.

    Returns
    -------
    dict
        {collection_key: {date_str: s3_url}}
    """
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            logger.info("Loading cached granule index from %s", cache_path)
            with open(cache_path) as f:
                return json.load(f)

    if collections is None:
        # All collections except climatology (which is handled separately)
        collections = [k for k in MERRA2_COLLECTIONS if k != "climatology"]

    earthaccess.login()

    index = {}
    for collection_key in collections:
        meta = MERRA2_COLLECTIONS[collection_key]
        doi = meta["doi"]

        logger.info(
            "Querying earthaccess for %s (DOI: %s) over %s",
            collection_key, doi, time_range,
        )
        granules = earthaccess.search_data(
            doi=doi,
            temporal=time_range,
            count=-1,  # return all results
        )
        logger.info("  Found %d granules for %s", len(granules), collection_key)

        collection_index = {}
        for granule in granules:
            try:
                date_str = _extract_date_from_granule(granule)
                s3_url = _extract_s3_url(granule)
                collection_index[date_str] = s3_url
            except ValueError as e:
                logger.warning("Skipping granule: %s", e)

        index[collection_key] = collection_index

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(index, f)
        logger.info("Cached granule index to %s", cache_path)

    return index


def map_storm_to_granules(storm_da, granule_index, include_precip_lookahead=True):
    """
    Map a storm's time range to the required S3 granule URLs per collection.

    Parameters
    ----------
    storm_da : xr.DataArray
        Storm binary mask with time dimension.
    granule_index : dict
        {collection_key: {date_str: s3_url}} from build_granule_index().
    include_precip_lookahead : bool
        If True, extend precip collection URLs by +24 hours for the
        augmented storm mask.

    Returns
    -------
    dict
        {collection_key: [s3_url, ...]} with URLs ordered chronologically.
    """
    storm_dates = np.unique(storm_da.time.dt.strftime("%Y%m%d").values)

    # For precip, extend by one day beyond the last storm date
    if include_precip_lookahead:
        last_date = np.max(storm_da.time.values)
        extended_end = last_date + np.timedelta64(1, "D")
        # Generate all dates from storm start to extended end
        all_dates = np.arange(
            np.min(storm_da.time.values).astype("datetime64[D]"),
            extended_end.astype("datetime64[D]") + np.timedelta64(1, "D"),
            dtype="datetime64[D]",
        )
        precip_dates = np.array(
            [np.datetime_as_string(d, unit="D").replace("-", "") for d in all_dates]
        )
    else:
        precip_dates = storm_dates

    result = {}
    for collection_key, date_index in granule_index.items():
        # Use extended dates for precip collection, regular for others
        dates_needed = precip_dates if collection_key == "VFLXQV_PRECIP" else storm_dates

        urls = []
        for date_str in dates_needed:
            if date_str in date_index:
                urls.append(date_index[date_str])

        result[collection_key] = urls

    return result
