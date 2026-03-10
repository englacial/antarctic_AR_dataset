"""
Cloud/serverless module for storm attribute computation via Lithops.

This module provides an alternative compute backend using AWS Lambda (via Lithops)
for computing storm summary statistics in parallel. It complements the existing
local (compute_attributes.py) and Ray (compute_attributes_streaming.py) workflows.
"""

from .orchestrator import run_cloud_attributes
from .aggregation_registry import AGGREGATION_SPECS, AggregationSpec

__all__ = [
    "run_cloud_attributes",
    "AGGREGATION_SPECS",
    "AggregationSpec",
]
