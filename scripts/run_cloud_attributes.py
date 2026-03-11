#!/usr/bin/env -S uv run --project . --extra cloud
"""
CLI entry point for cloud-based storm attribute computation.

Usage:
    uv run scripts/run_cloud_attributes.py /path/to/catalog.h5 \
        --granule-cache granule_cache.json

    # Test with a single storm
    uv run scripts/run_cloud_attributes.py /path/to/catalog.h5 --limit 1
"""

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(
        description="Compute storm attributes using AWS Lambda.",
    )
    parser.add_argument(
        "catalog_path",
        help="Path to HDF5 storm catalog.",
    )
    parser.add_argument(
        "--output", "-o",
        dest="output_path",
        default=None,
        help="Output path for results HDF5. Default: <catalog>_cloud_attributes.h5",
    )
    parser.add_argument(
        "--function-name",
        default="ar-worker",
        help="Lambda function name (default: ar-worker).",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=10,
        help="Max timesteps to hold in memory per worker (default: 10).",
    )
    parser.add_argument(
        "--granule-cache",
        default=None,
        help="Path to cache the granule index JSON.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1000,
        help="Max concurrent Lambda invocations (default: 1000).",
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region (default: us-west-2).",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Only process the first N storms (for testing).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    from artools.cloud.orchestrator import run_cloud_attributes

    result = run_cloud_attributes(
        catalog_path=args.catalog_path,
        function_name=args.function_name,
        output_path=args.output_path,
        max_resident_timesteps=args.max_timesteps,
        granule_cache_path=args.granule_cache,
        max_workers=args.max_workers,
        limit=args.limit,
        region=args.region,
    )

    print(f"\nDone. Computed attributes for {len(result)} storms.")
    print(f"Output columns: {list(result.columns)}")


if __name__ == "__main__":
    main()
