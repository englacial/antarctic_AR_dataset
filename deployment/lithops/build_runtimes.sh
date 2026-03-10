#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Build from repo root so artools/ is available in Docker context
cd "$REPO_ROOT"

echo "Building x86_64 runtime..."
lithops runtime build \
    -f deployment/lithops/Dockerfile.x86_64 \
    -b aws_lambda \
    ar-worker-x86

echo "Building arm64 runtime..."
lithops runtime build \
    -f deployment/lithops/Dockerfile.arm64 \
    -b aws_lambda \
    ar-worker-arm64

echo "Both runtimes built and pushed to ECR."
echo ""
echo "To list deployed runtimes:"
echo "  lithops runtime list -b aws_lambda"
