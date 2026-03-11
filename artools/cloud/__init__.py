"""
Cloud module for storm attribute computation via AWS Lambda.

This module provides a compute backend using AWS Lambda for computing storm
summary statistics in parallel. It complements the existing local
(compute_attributes.py) and Ray (compute_attributes_streaming.py) workflows.
"""

from .aggregation_registry import AGGREGATION_SPECS, AggregationSpec

def run_cloud_attributes(*args, **kwargs):
    """Lazy import to avoid pulling in heavy deps on Lambda workers."""
    from .orchestrator import run_cloud_attributes as _run
    return _run(*args, **kwargs)

__all__ = [
    "run_cloud_attributes",
    "AGGREGATION_SPECS",
    "AggregationSpec",
]
