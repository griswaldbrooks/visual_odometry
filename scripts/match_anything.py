#!/usr/bin/env python3
"""MatchAnything feature matcher - subprocess interface for C++ VO pipeline.

Usage:
    python match_anything.py <image1_path> <image2_path> [--threshold 0.2]

Outputs JSON to stdout:
    {"keypoints0": [[x,y], ...], "keypoints1": [[x,y], ...], "scores": [...]}
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def load_model():
    """Load MatchAnything model and processor (cached after first call)."""
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching

    model_id = "zju-community/matchanything_eloftr"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForKeypointMatching.from_pretrained(model_id)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return processor, model, device


def match_images(image1_path: str, image2_path: str, threshold: float = 0.2) -> dict:
    """Match features between two images.

    Args:
        image1_path: Path to first image.
        image2_path: Path to second image.
        threshold: Confidence threshold for matches.

    Returns:
        Dictionary with keypoints0, keypoints1, and scores.
    """
    from transformers.image_utils import load_image

    # Load images
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # Load model (will be cached by transformers)
    processor, model, device = load_model()

    # Process images
    inputs = processor([image1, image2], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    image_sizes = [[(image1.height, image1.width), (image2.height, image2.width)]]
    results = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=threshold)

    # Convert to serializable format
    result = results[0]
    return {
        "keypoints0": result["keypoints0"].cpu().numpy().tolist(),
        "keypoints1": result["keypoints1"].cpu().numpy().tolist(),
        "scores": result["matching_scores"].cpu().numpy().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="MatchAnything feature matcher")
    parser.add_argument("image1", type=str, help="Path to first image")
    parser.add_argument("image2", type=str, help="Path to second image")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Confidence threshold (default: 0.2)")
    args = parser.parse_args()

    # Validate inputs
    if not Path(args.image1).exists():
        print(json.dumps({"error": f"Image not found: {args.image1}"}))
        sys.exit(1)
    if not Path(args.image2).exists():
        print(json.dumps({"error": f"Image not found: {args.image2}"}))
        sys.exit(1)

    try:
        result = match_images(args.image1, args.image2, args.threshold)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
