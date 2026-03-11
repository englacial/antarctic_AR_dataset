#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATIC_SRC="${1:-/home/espg/software/antarctic_AR_catalogs}"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="us-west-2"
ECR_REPO="ar-worker"

# Copy static data files into build context
echo "Copying static data from $STATIC_SRC..."
mkdir -p "$REPO_ROOT/static_data"
cp "$STATIC_SRC/AIS_Full_basins_Zwally_MERRA2grid_new.nc" "$REPO_ROOT/static_data/"
cp "$STATIC_SRC/MERRA2_gridarea.nc" "$REPO_ROOT/static_data/"
cp "$STATIC_SRC/MERRA2_monthly_climatology.nc" "$REPO_ROOT/static_data/"

# Authenticate to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    podman login --username AWS --password-stdin \
    "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Ensure ECR repository exists
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" 2>/dev/null || \
    aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION"

# Build and push
ARCH="${2:-x86_64}"
IMAGE_TAG="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$ARCH"

echo "Building $ARCH image..."
podman build -t "$IMAGE_TAG" \
    -f "$SCRIPT_DIR/Dockerfile.$ARCH" \
    "$REPO_ROOT"

echo "Pushing to ECR..."
podman push "$IMAGE_TAG"

# Clean up static data copy
rm -rf "$REPO_ROOT/static_data"

echo ""
echo "Image pushed: $IMAGE_TAG"
echo ""
echo "To create/update the Lambda function:"
echo "  aws lambda create-function \\"
echo "    --function-name ar-worker \\"
echo "    --package-type Image \\"
echo "    --code ImageUri=$IMAGE_TAG \\"
echo "    --role arn:aws:iam::${AWS_ACCOUNT_ID}:role/lithops-ismip6-executor \\"
echo "    --memory-size 2048 \\"
echo "    --timeout 900 \\"
echo "    --region $AWS_REGION"
