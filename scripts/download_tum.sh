#!/bin/bash
# Download TUM RGB-D dataset sequences for visual odometry testing
# Usage: ./scripts/download_tum.sh [sequence]
#
# Available sequences (smallest first):
#   fr3_nostructure  - 0.21 GB, 16s  (challenging - minimal features)
#   fr1_rpy          - 0.42 GB, 28s  (rotation testing)
#   fr1_xyz          - 0.47 GB, 30s  (translation testing, recommended)
#   fr1_desk         - 0.58 GB, 23s  (office scene)

set -e

SEQUENCE="${1:-fr1_xyz}"
DATA_DIR="data/tum"

declare -A URLS
URLS["fr1_xyz"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
URLS["fr1_rpy"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_rpy.tgz"
URLS["fr1_desk"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
URLS["fr3_nostructure"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_nostructure_notexture_far.tgz"

declare -A SIZES
SIZES["fr1_xyz"]="0.47 GB"
SIZES["fr1_rpy"]="0.42 GB"
SIZES["fr1_desk"]="0.58 GB"
SIZES["fr3_nostructure"]="0.21 GB"

if [[ -z "${URLS[$SEQUENCE]}" ]]; then
    echo "Unknown sequence: $SEQUENCE"
    echo "Available: fr1_xyz (recommended), fr1_rpy, fr1_desk, fr3_nostructure"
    exit 1
fi

URL="${URLS[$SEQUENCE]}"
SIZE="${SIZES[$SEQUENCE]}"
FILENAME=$(basename "$URL")
DIRNAME="${FILENAME%.tgz}"

echo "Downloading TUM RGB-D: $SEQUENCE ($SIZE)"
echo "URL: $URL"
echo ""

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [[ -d "$DIRNAME" ]]; then
    echo "Dataset already exists: $DATA_DIR/$DIRNAME"
    echo "To re-download, remove it first: rm -rf $DATA_DIR/$DIRNAME"
    exit 0
fi

echo "Downloading..."
wget --progress=bar:force "$URL" -O "$FILENAME"

echo "Extracting..."
tar -xzf "$FILENAME"
rm "$FILENAME"

echo ""
echo "Done! Dataset extracted to: $DATA_DIR/$DIRNAME"
echo ""
echo "To run visual odometry:"
echo "  ./build/dev/visual_odometry --images $DATA_DIR/$DIRNAME/rgb --camera data/tum_camera.yaml"
echo ""
echo "Note: You may need to create data/tum_camera.yaml with TUM Freiburg1 intrinsics:"
echo "  fx: 517.3  fy: 516.5  cx: 318.6  cy: 255.3"
