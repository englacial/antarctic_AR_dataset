"""
Declarative aggregation registry for storm summary statistics.

Each AggregationSpec defines a single output metric by specifying:
  - Which MERRA-2 variable and collection to read
  - A spatial function (per-timestep) from spatial_functions.py
  - A temporal reducer (cross-timestep) from accumulators.py
  - Which spatial mask to apply (AIS, ocean, or full footprint)
  - Whether to compute anomalies, negate the variable, etc.

Adding a new metric requires only adding a new AggregationSpec entry —
no code changes needed.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AggregationSpec:
    """Declarative specification of a storm summary metric."""

    output_name: str
    """Column name in the output DataFrame."""

    variable: str
    """MERRA-2 variable name (e.g., 'T2M', 'VFLXQV', 'PRECSN')."""

    collection: str
    """MERRA-2 collection key (e.g., 'T2M_TQV_SLP'). Must match a key in MERRA2_COLLECTIONS."""

    spatial_func: str
    """Key into SPATIAL_FUNCTIONS registry (e.g., 'max', 'weighted_sum')."""

    temporal_reducer: str
    """Key into TEMPORAL_REDUCERS registry (e.g., 'max', 'sum', 'first_landfall')."""

    mask: str = "ais"
    """Spatial mask: 'ais' (AIS only), 'ocean' (not AIS), 'full' (whole storm footprint)."""

    is_anomaly: bool = False
    """If True, subtract monthly climatology before aggregation."""

    negate: bool = False
    """If True, negate the variable (e.g., for southward flux stored as negative)."""

    params: dict = field(default_factory=dict)
    """Extra parameters. 'precip': True triggers augmented mask (24h lookahead)."""


# MERRA-2 collection metadata: DOIs and time-offset flags
MERRA2_COLLECTIONS = {
    "T2M_TQV_SLP": {"doi": "10.5067/3Z173KIE2TPD", "half_hour": False},
    "VFLXQV_PRECIP": {"doi": "10.5067/Q5GVUVUIVGO7", "half_hour": True},
    "V850": {"doi": "10.5067/VJAFPLI1CSIV", "half_hour": True},
    "OMEGA": {"doi": "10.5067/QBZ6MG944HW0", "half_hour": False},
    "climatology": {"doi": "10.5067/5ESKGQTZG7FO", "half_hour": False},
}


# ---------------------------------------------------------------------------
# Full aggregation registry
# ---------------------------------------------------------------------------

AGGREGATION_SPECS = [
    # --- T2M_TQV_SLP collection ---
    AggregationSpec(
        "max_T2m_ais", "T2M", "T2M_TQV_SLP",
        "max", "max", mask="ais",
    ),
    AggregationSpec(
        "max_T2M_anomaly_ais", "T2M", "T2M_TQV_SLP",
        "max", "max", mask="ais", is_anomaly=True,
    ),
    AggregationSpec(
        "max_IWV_ais", "TQV", "T2M_TQV_SLP",
        "max", "max", mask="ais",
    ),
    AggregationSpec(
        "max_IWV_anomaly_ais", "TQV", "T2M_TQV_SLP",
        "max", "max", mask="ais", is_anomaly=True,
    ),
    AggregationSpec(
        "min_SLP", "SLP", "T2M_TQV_SLP",
        "min", "first_landfall", mask="ocean",
    ),
    AggregationSpec(
        "max_SLP_gradient", "SLP", "T2M_TQV_SLP",
        "max_gradient", "first_landfall", mask="ocean",
    ),

    # --- VFLXQV_PRECIP collection (non-precip vars) ---
    AggregationSpec(
        "avg_vIVT_ais", "VFLXQV", "VFLXQV_PRECIP",
        "weighted_mean", "weighted_mean", mask="ais", negate=True,
    ),
    AggregationSpec(
        "max_vIVT_ais", "VFLXQV", "VFLXQV_PRECIP",
        "max", "max", mask="ais", negate=True,
    ),
    AggregationSpec(
        "max_vIVT", "VFLXQV", "VFLXQV_PRECIP",
        "max", "max", mask="full", negate=True,
    ),

    # --- VFLXQV_PRECIP collection (precip vars — uses augmented mask) ---
    AggregationSpec(
        "cumulative_rainfall_ais", "_rainfall", "VFLXQV_PRECIP",
        "weighted_sum", "sum", mask="ais",
        params={"precip": True},
    ),
    AggregationSpec(
        "cumulative_snowfall_ais", "PRECSN", "VFLXQV_PRECIP",
        "weighted_sum", "sum", mask="ais",
        params={"precip": True},
    ),

    # --- V850 collection ---
    AggregationSpec(
        "max_landfalling_v850hPa", "V850", "V850",
        "max", "first_landfall", mask="ocean", negate=True,
    ),
    AggregationSpec(
        "avg_landfalling_v850hPa", "V850", "V850",
        "weighted_mean", "first_landfall", mask="ocean", negate=True,
    ),

    # --- OMEGA collection ---
    AggregationSpec(
        "avg_landfalling_minomega", "OMEGA", "OMEGA",
        "min_over_levels", "first_landfall", mask="ais",
    ),
]
