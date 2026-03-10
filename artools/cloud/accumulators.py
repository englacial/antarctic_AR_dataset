"""
Temporal accumulator classes for cross-timestep reduction.

Each accumulator tracks running state across sequential timestep processing,
then finalizes to a scalar result. These are the temporal reduction half of
the (spatial_function, temporal_reducer) decomposition.
"""

import numpy as np


class MaxAccumulator:
    """Running maximum across timesteps."""

    def __init__(self):
        self.value = -np.inf

    def update(self, val):
        if val is not None and not np.isnan(val):
            self.value = max(self.value, val)

    def finalize(self):
        return float(self.value) if self.value != -np.inf else np.nan


class MinAccumulator:
    """Running minimum across timesteps."""

    def __init__(self):
        self.value = np.inf

    def update(self, val):
        if val is not None and not np.isnan(val):
            self.value = min(self.value, val)

    def finalize(self):
        return float(self.value) if self.value != np.inf else np.nan


class SumAccumulator:
    """Running sum across timesteps."""

    def __init__(self):
        self.value = 0.0
        self.has_data = False

    def update(self, val):
        if val is not None and not np.isnan(val):
            self.value += val
            self.has_data = True

    def finalize(self):
        return float(self.value) if self.has_data else np.nan


class WeightedMeanAccumulator:
    """
    Running weighted mean across timesteps.

    Spatial functions paired with this accumulator return (weighted_sum, weight_sum)
    tuples. The final result is the ratio.
    """

    def __init__(self):
        self.weighted_sum = 0.0
        self.weight_sum = 0.0

    def update(self, val):
        if val is None:
            return
        weighted_sum, weight_sum = val
        if not np.isnan(weighted_sum) and not np.isnan(weight_sum):
            self.weighted_sum += weighted_sum
            self.weight_sum += weight_sum

    def finalize(self):
        if self.weight_sum > 0:
            return float(self.weighted_sum / self.weight_sum)
        return np.nan


class FirstLandfallCapture:
    """
    Captures a value only at the first landfall timestep.

    For metrics like min_SLP and max_SLPgrad that are defined at the moment
    of first landfall only. The worker passes the first_landfall timestamp
    and this accumulator only fires once.
    """

    def __init__(self):
        self.value = None

    def update(self, val):
        # Only accept the first non-None value
        if self.value is None and val is not None:
            self.value = val

    def finalize(self):
        if self.value is None:
            return np.nan
        # Handle tuple values from weighted_mean spatial funcs
        if isinstance(self.value, tuple):
            weighted_sum, weight_sum = self.value
            if weight_sum > 0:
                return float(weighted_sum / weight_sum)
            return np.nan
        return float(self.value)


TEMPORAL_REDUCERS = {
    "max": MaxAccumulator,
    "min": MinAccumulator,
    "sum": SumAccumulator,
    "weighted_mean": WeightedMeanAccumulator,
    "first_landfall": FirstLandfallCapture,
}
