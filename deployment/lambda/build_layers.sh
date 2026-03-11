#!/bin/bash
#
# Build and deploy the AR worker Lambda function using Lambda layers
# instead of a Docker container image.
#
# Usage:
#   ./build_layers.sh [STATIC_DATA_DIR]
#
# Arguments:
#   STATIC_DATA_DIR  Path to static NetCDF files (default: /home/espg/software/antarctic_AR_catalogs)
#
# This script:
#   1. Installs Python dependencies into a layer zip (~214 MB unzipped)
#   2. Packages static data files into a second layer zip (~8 MB)
#   3. Packages artools/ as the function code zip
#   4. Publishes both layers and creates/updates the Lambda function
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - pip (for installing packages)
#   - zip
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATIC_SRC="${1:-/home/espg/software/antarctic_AR_catalogs}"

AWS_REGION="us-west-2"
FUNCTION_NAME="ar-worker"
ROLE_ARN="arn:aws:iam::429435741471:role/lithops-ismip6-executor"
RUNTIME="python3.13"
S3_BUCKET="lithops-us-west-2-l2ic"  # for uploading layers >50 MB

# --------------------------------------------------------------------------
# 1. Create a temporary build directory
# --------------------------------------------------------------------------
BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ar-worker-build.XXXXXXXXXX")
echo "Build directory: $BUILD_DIR"

cleanup() {
    echo "Cleaning up $BUILD_DIR..."
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

# --------------------------------------------------------------------------
# 2. Build the Python dependencies layer
# --------------------------------------------------------------------------
echo ""
echo "=== Building dependencies layer ==="

DEPS_DIR="$BUILD_DIR/deps-layer"
mkdir -p "$DEPS_DIR/python"

# Single-pass install with all deps. --only-binary=:all: still resolves
# pure-python wheels (py3-none-any) which are compatible with any platform.
pip install \
    --target "$DEPS_DIR/python" \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.13 \
    --only-binary=:all: \
    numpy pandas xarray h5netcdf h5py fsspec s3fs earthaccess

# Strip .pyc files and __pycache__ to reduce size
find "$DEPS_DIR/python" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$DEPS_DIR/python" -name "*.pyc" -delete 2>/dev/null || true
# Keep .dist-info (needed by importlib.metadata at import time for numpy et al.)
# but strip bulky files within them that aren't needed at runtime
find "$DEPS_DIR/python" -path "*dist-info/RECORD" -delete 2>/dev/null || true
find "$DEPS_DIR/python" -path "*dist-info/LICENSE*" -delete 2>/dev/null || true
find "$DEPS_DIR/python" -path "*dist-info/WHEEL" -delete 2>/dev/null || true
# Strip numpy tests and f2py (saves ~30 MB)
rm -rf "$DEPS_DIR/python/numpy/tests" 2>/dev/null || true
rm -rf "$DEPS_DIR/python/numpy/f2py" 2>/dev/null || true
rm -rf "$DEPS_DIR/python/numpy/_pyinstaller" 2>/dev/null || true
# Strip pandas tests
rm -rf "$DEPS_DIR/python/pandas/tests" 2>/dev/null || true

DEPS_ZIP="$BUILD_DIR/deps-layer.zip"
echo "Packaging dependencies layer..."
(cd "$DEPS_DIR" && zip -r -q "$DEPS_ZIP" python/)

DEPS_SIZE=$(du -sh "$DEPS_DIR/python" | cut -f1)
DEPS_ZIP_SIZE=$(du -sh "$DEPS_ZIP" | cut -f1)
echo "Dependencies layer: $DEPS_SIZE unzipped, $DEPS_ZIP_SIZE zipped"

# --------------------------------------------------------------------------
# 3. Build the static data layer
# --------------------------------------------------------------------------
echo ""
echo "=== Building static data layer ==="

STATIC_DIR="$BUILD_DIR/static-layer"
mkdir -p "$STATIC_DIR/static_data"

cp "$STATIC_SRC/AIS_Full_basins_Zwally_MERRA2grid_new.nc" "$STATIC_DIR/static_data/"
cp "$STATIC_SRC/MERRA2_gridarea.nc" "$STATIC_DIR/static_data/"
cp "$STATIC_SRC/MERRA2_monthly_climatology.nc" "$STATIC_DIR/static_data/"

STATIC_ZIP="$BUILD_DIR/static-data-layer.zip"
echo "Packaging static data layer..."
(cd "$STATIC_DIR" && zip -r -q "$STATIC_ZIP" static_data/)

STATIC_SIZE=$(du -sh "$STATIC_DIR/static_data" | cut -f1)
STATIC_ZIP_SIZE=$(du -sh "$STATIC_ZIP" | cut -f1)
echo "Static data layer: $STATIC_SIZE unzipped, $STATIC_ZIP_SIZE zipped"

# --------------------------------------------------------------------------
# 4. Build the function code zip
# --------------------------------------------------------------------------
echo ""
echo "=== Building function code zip ==="

CODE_DIR="$BUILD_DIR/function-code"
mkdir -p "$CODE_DIR/artools/cloud"

# Copy only the files needed by the worker (not orchestrator/catalog/auth
# which are orchestrator-side only, but include them for completeness since
# __init__.py may reference them with try/except guards)
cp "$REPO_ROOT/artools/__init__.py" "$CODE_DIR/artools/"
cp "$REPO_ROOT/artools/cloud/__init__.py" "$CODE_DIR/artools/cloud/"
cp "$REPO_ROOT/artools/cloud/worker.py" "$CODE_DIR/artools/cloud/"
cp "$REPO_ROOT/artools/cloud/accumulators.py" "$CODE_DIR/artools/cloud/"
cp "$REPO_ROOT/artools/cloud/spatial_functions.py" "$CODE_DIR/artools/cloud/"
cp "$REPO_ROOT/artools/cloud/aggregation_registry.py" "$CODE_DIR/artools/cloud/"

CODE_ZIP="$BUILD_DIR/function-code.zip"
(cd "$CODE_DIR" && zip -r -q "$CODE_ZIP" artools/)

CODE_ZIP_SIZE=$(du -sh "$CODE_ZIP" | cut -f1)
echo "Function code zip: $CODE_ZIP_SIZE"

# --------------------------------------------------------------------------
# 5. Publish Lambda layers
# --------------------------------------------------------------------------
echo ""
echo "=== Publishing Lambda layers ==="

echo "Uploading dependencies layer to S3 (too large for direct upload)..."
S3_DEPS_KEY="lambda-layers/ar-worker-deps.zip"
aws s3 cp "$DEPS_ZIP" "s3://$S3_BUCKET/$S3_DEPS_KEY" --region "$AWS_REGION"

echo "Publishing dependencies layer..."
DEPS_LAYER_ARN=$(aws lambda publish-layer-version \
    --layer-name ar-worker-deps \
    --description "Python dependencies for AR worker (numpy, pandas, xarray, etc.)" \
    --compatible-runtimes "$RUNTIME" \
    --compatible-architectures x86_64 \
    --content "S3Bucket=$S3_BUCKET,S3Key=$S3_DEPS_KEY" \
    --region "$AWS_REGION" \
    --query 'LayerVersionArn' \
    --output text)
echo "Dependencies layer ARN: $DEPS_LAYER_ARN"

echo "Publishing static data layer..."
STATIC_LAYER_ARN=$(aws lambda publish-layer-version \
    --layer-name ar-worker-static-data \
    --description "Static data files (AIS mask, cell areas, climatology) for AR worker" \
    --compatible-runtimes "$RUNTIME" \
    --compatible-architectures x86_64 \
    --zip-file "fileb://$STATIC_ZIP" \
    --region "$AWS_REGION" \
    --query 'LayerVersionArn' \
    --output text)
echo "Static data layer ARN: $STATIC_LAYER_ARN"

# --------------------------------------------------------------------------
# 6. Create or update the Lambda function
# --------------------------------------------------------------------------
echo ""
echo "=== Deploying Lambda function ==="

# Check if function already exists
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    # Check if the existing function is an Image-based deployment (Docker).
    # Lambda doesn't allow switching package types in-place, so we must
    # delete and recreate when migrating from Image to Zip.
    EXISTING_PKG=$(aws lambda get-function \
        --function-name "$FUNCTION_NAME" \
        --region "$AWS_REGION" \
        --query 'Configuration.PackageType' \
        --output text)

    if [ "$EXISTING_PKG" = "Image" ]; then
        echo "Existing function uses Image packaging — deleting to recreate as Zip..."
        aws lambda delete-function \
            --function-name "$FUNCTION_NAME" \
            --region "$AWS_REGION"
        echo "Deleted. Recreating..."
    fi
fi

# Create or update
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "Updating existing function '$FUNCTION_NAME'..."

    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb://$CODE_ZIP" \
        --region "$AWS_REGION" \
        --query 'FunctionArn' \
        --output text

    echo "Waiting for function update to complete..."
    aws lambda wait function-updated \
        --function-name "$FUNCTION_NAME" \
        --region "$AWS_REGION"

    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --layers "$DEPS_LAYER_ARN" "$STATIC_LAYER_ARN" \
        --runtime "$RUNTIME" \
        --handler "artools.cloud.worker.lambda_handler" \
        --environment "Variables={STATIC_DATA_DIR=/opt/static_data}" \
        --memory-size 2048 \
        --timeout 900 \
        --region "$AWS_REGION" \
        --query 'FunctionArn' \
        --output text
else
    echo "Creating new function '$FUNCTION_NAME'..."

    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime "$RUNTIME" \
        --handler "artools.cloud.worker.lambda_handler" \
        --role "$ROLE_ARN" \
        --zip-file "fileb://$CODE_ZIP" \
        --layers "$DEPS_LAYER_ARN" "$STATIC_LAYER_ARN" \
        --environment "Variables={STATIC_DATA_DIR=/opt/static_data}" \
        --memory-size 2048 \
        --timeout 900 \
        --architectures x86_64 \
        --region "$AWS_REGION" \
        --query 'FunctionArn' \
        --output text
fi

echo ""
echo "=== Deployment complete ==="
echo "Function:     $FUNCTION_NAME"
echo "Runtime:      $RUNTIME"
echo "Deps layer:   $DEPS_LAYER_ARN"
echo "Static layer: $STATIC_LAYER_ARN"
echo "Handler:      artools.cloud.worker.lambda_handler"
echo "STATIC_DATA_DIR=/opt/static_data"
