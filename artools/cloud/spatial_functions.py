"""
Per-timestep spatial aggregation functions.

Each function operates on a single timestep: it takes a variable DataArray
(already subsetted to the storm's spatial extent), a binary storm mask for
that timestep, cell areas, and an optional spatial mask (AIS, ocean, etc.),
and returns a scalar value or tuple for the temporal accumulator.

The math here is identical to attribute_utils.py, just decomposed to
operate on a single timestep rather than the full spatiotemporal cube.
"""

import numpy as np
import xarray as xr


def _apply_mask(storm_mask_t, ais_mask_subset, mask_type):
    """
    Apply the appropriate spatial mask to the storm mask for one timestep.

    Parameters
    ----------
    storm_mask_t : xr.DataArray
        Binary storm mask for one timestep (lat, lon).
    ais_mask_subset : xr.DataArray
        AIS mask subsetted to storm extent.
    mask_type : str
        "ais" = storm ∩ AIS, "ocean" = storm ∩ ¬AIS, "full" = storm only.

    Returns
    -------
    xr.DataArray
        Combined binary mask for one timestep.
    """
    if mask_type == "ais":
        return storm_mask_t.where(ais_mask_subset, 0)
    elif mask_type == "ocean":
        ocean_mask = ~ais_mask_subset
        return storm_mask_t.where(ocean_mask, 0)
    else:  # "full"
        return storm_mask_t


def spatial_max(var_t, combined_mask, cell_areas):
    """Max value under the masked footprint for one timestep."""
    masked = var_t * combined_mask
    vals = masked.values[combined_mask.values > 0]
    if len(vals) == 0:
        return np.nan
    return float(np.nanmax(vals))


def spatial_min(var_t, combined_mask, cell_areas):
    """Min value under the masked footprint for one timestep."""
    masked = var_t.where(combined_mask > 0)
    vals = masked.values[~np.isnan(masked.values)]
    if len(vals) == 0:
        return np.nan
    return float(np.nanmin(vals))


def spatial_weighted_sum(var_t, combined_mask, cell_areas):
    """Area-weighted sum under the masked footprint for one timestep."""
    masked = var_t * combined_mask
    weighted = masked * cell_areas
    val = float(weighted.sum().values)
    return val


def spatial_weighted_mean_parts(var_t, combined_mask, cell_areas):
    """
    Returns (weighted_sum, weight_sum) for one timestep.

    The WeightedMeanAccumulator accumulates these across timesteps
    and divides at finalization.
    """
    masked_var = var_t * combined_mask
    weights = cell_areas * combined_mask
    weighted_sum = float((masked_var * cell_areas).sum().values)
    weight_sum = float(weights.sum().values)
    return (weighted_sum, weight_sum)


def spatial_max_gradient(var_t, combined_mask, cell_areas):
    """
    Max pressure gradient magnitude under the masked footprint.

    Computes spatial derivatives in lat/lon, converts to physical
    distance using Earth's radius, and returns the max gradient magnitude.
    """
    if (combined_mask == 0).all().values:
        return np.nan

    # Convert to radians for differentiation
    rads = var_t.assign_coords(
        lon=np.radians(var_t.lon),
        lat=np.radians(var_t.lat),
    )
    r = 6378  # Earth radius in km
    lat_partials = rads.differentiate("lat") / r
    lon_partials = rads.differentiate("lon") / (np.sin(rads.lat) * r)

    magnitude = np.sqrt(lon_partials**2 + lat_partials**2)
    grad_vals = magnitude.values * combined_mask.values
    nonzero = grad_vals[combined_mask.values > 0]
    if len(nonzero) == 0:
        return np.nan
    return float(np.nanmax(nonzero))


def spatial_min_level_then_weighted_mean(var_t, combined_mask, cell_areas):
    """
    For 3D variables (e.g., OMEGA with lev dimension): take min over levels,
    then compute area-weighted mean over the masked footprint.

    Returns (weighted_sum, weight_sum) tuple for WeightedMeanAccumulator
    or FirstLandfallCapture.
    """
    # Collapse vertical levels by taking min
    if "lev" in var_t.dims:
        var_2d = var_t.min("lev")
    else:
        var_2d = var_t

    weights = cell_areas * combined_mask
    weight_sum = float(weights.sum().values)
    if weight_sum == 0:
        return (np.nan, np.nan)

    weighted_sum = float((var_2d * combined_mask * cell_areas).sum().values)
    return (weighted_sum, weight_sum)


SPATIAL_FUNCTIONS = {
    "max": spatial_max,
    "min": spatial_min,
    "weighted_sum": spatial_weighted_sum,
    "weighted_mean": spatial_weighted_mean_parts,
    "max_gradient": spatial_max_gradient,
    "min_over_levels": spatial_min_level_then_weighted_mean,
}
