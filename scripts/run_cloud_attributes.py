#!/usr/bin/env -S uv run --project . --extra cloud
"""
CLI entry point for cloud-based storm attribute computation.

Usage:
    # Full run on Lambda
    uv run scripts/run_cloud_attributes.py /path/to/catalog.h5 \
        --config deployment/lithops/lithops_config.yaml \
        --static-data /path/to/antarctic_AR_catalogs \
        --granule-cache granule_cache.json
"""

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(
        description="Compute storm attributes using Lithops cloud backend.",
    )
    parser.add_argument(
        "catalog_path",
        help="Path to HDF5 storm catalog (e.g., from antarctic_AR_catalogs).",
    )
    parser.add_argument(
        "--output", "-o",
        dest="output_path",
        default=None,
        help="Output path for results HDF5. Default: <catalog>_cloud_attributes.h5",
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
        "--config",
        default=None,
        help="Path to lithops_config.yaml override.",
    )
    parser.add_argument(
        "--static-data",
        default=None,
        help="Path to local directory with AIS mask and cell area files. "
             "If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Only process the first N storms (for testing).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally using Lithops LocalhostExecutor (for testing).",
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

    # Load lithops config from file if specified
    lithops_config = None
    if args.config:
        import yaml
        with open(args.config) as f:
            lithops_config = yaml.safe_load(f)

    from artools.cloud.orchestrator import run_cloud_attributes

    result = run_cloud_attributes(
        catalog_path=args.catalog_path,
        lithops_config=lithops_config,
        output_path=args.output_path,
        max_resident_timesteps=args.max_timesteps,
        granule_cache_path=args.granule_cache,
        local_mode=args.local,
        static_data_path=args.static_data,
        limit=args.limit,
    )

    print(f"\nDone. Computed attributes for {len(result)} storms.")
    print(f"Output columns: {list(result.columns)}")


if __name__ == "__main__":
    main()
