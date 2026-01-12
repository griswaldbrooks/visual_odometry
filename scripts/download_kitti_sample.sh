#!/bin/bash
# Download KITTI odometry sample data
#
# Note: Full dataset requires registration at https://www.cvlibs.net/datasets/kitti/
# This script provides instructions and downloads ground truth poses (publicly available)

set -e

DATA_DIR="$(dirname "$0")/../data/kitti"
mkdir -p "$DATA_DIR/poses"
mkdir -p "$DATA_DIR/sequences/00/image_0"

echo "=== KITTI Odometry Sample Setup ==="
echo ""
echo "1. Ground truth poses (downloading now)..."

# Ground truth poses are available without registration
curl -L "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip" -o /tmp/kitti_poses.zip
unzip -o /tmp/kitti_poses.zip -d /tmp/kitti_poses
cp /tmp/kitti_poses/dataset/poses/00.txt "$DATA_DIR/poses/"
rm -rf /tmp/kitti_poses /tmp/kitti_poses.zip

echo "   ✓ Downloaded poses/00.txt"
echo ""
echo "2. Image data requires manual download:"
echo "   - Go to: https://www.cvlibs.net/datasets/kitti/eval_odometry.php"
echo "   - Download: 'odometry data set (grayscale)'"
echo "   - Extract sequence 00 to: $DATA_DIR/sequences/00/"
echo ""
echo "3. Alternatively, for quick testing, place any sequential images in:"
echo "   $DATA_DIR/sequences/00/image_0/"
echo "   Named as: 000000.png, 000001.png, etc."
echo ""
echo "Done! Ground truth poses are ready."
