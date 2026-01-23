#!/usr/bin/env python3
"""
Benchmark comparison script for visual odometry matchers.

Runs benchmarks for ORB and LightGlue matchers and compares their performance.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare performance of ORB and LightGlue matchers"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to image directory for benchmarking",
    )
    parser.add_argument(
        "--camera",
        type=str,
        help="Path to camera intrinsics YAML (for accuracy metrics)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="build/dev/benchmark_matchers",
        help="Path to benchmark_matchers executable (default: build/dev/benchmark_matchers)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to store benchmark results (default: benchmark_results)",
    )
    parser.add_argument(
        "--matchers",
        type=str,
        nargs="+",
        default=["orb", "lightglue"],
        help="List of matchers to benchmark (default: orb lightglue)",
    )
    return parser.parse_args()


def run_benchmark(
    benchmark_exe: Path,
    matcher: str,
    images_dir: Path,
    output_file: Path,
    camera_yaml: Optional[Path] = None,
) -> Dict:
    """Run benchmark for a single matcher and return results."""
    print(f"\n{'='*60}")
    print(f"Running benchmark: {matcher.upper()}")
    print(f"{'='*60}")

    cmd = [
        str(benchmark_exe),
        "--images",
        str(images_dir),
        "--matcher",
        matcher,
        "--output",
        str(output_file),
    ]

    if camera_yaml:
        cmd.extend(["--camera", str(camera_yaml)])

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark for {matcher}: {e}", file=sys.stderr)
        return {}
    except FileNotFoundError:
        print(
            f"Error: Benchmark executable not found at {benchmark_exe}",
            file=sys.stderr,
        )
        print("Please build the project first: pixi run build", file=sys.stderr)
        return {}

    # Load results
    try:
        with open(output_file) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading results from {output_file}: {e}", file=sys.stderr)
        return {}


def print_comparison_table(results: List[Dict]) -> None:
    """Print markdown comparison table."""
    if not results:
        print("No results to display")
        return

    print(f"\n{'='*60}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*60}\n")

    # Determine which columns to show based on available data
    has_accuracy = any(r.get("total_inliers", 0) > 0 for r in results)

    # Print header
    if has_accuracy:
        print("| Matcher    | Avg Time (ms) | Avg Matches | Avg Inliers | Inlier Ratio |")
        print("|------------|---------------|-------------|-------------|--------------|")
    else:
        print("| Matcher    | Avg Time (ms) | Avg Matches |")
        print("|------------|---------------|-------------|")

    # Print rows
    for result in results:
        matcher = result.get("matcher", "Unknown")
        avg_time = result.get("avg_time_ms", 0.0)
        avg_matches = result.get("avg_matches", 0.0)

        if has_accuracy:
            avg_inliers = result.get("avg_inliers", 0.0)
            inlier_ratio = result.get("avg_inlier_ratio", 0.0)
            print(
                f"| {matcher:10s} | {avg_time:13.2f} | {avg_matches:11.2f} | "
                f"{avg_inliers:11.2f} | {inlier_ratio:12.4f} |"
            )
        else:
            print(
                f"| {matcher:10s} | {avg_time:13.2f} | {avg_matches:11.2f} |"
            )

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if len(results) >= 2:
        fastest = min(results, key=lambda r: r.get("avg_time_ms", float("inf")))
        print(f"Fastest matcher: {fastest.get('matcher', 'Unknown')}")

        if has_accuracy:
            best_accuracy = max(results, key=lambda r: r.get("avg_inlier_ratio", 0.0))
            print(f"Best accuracy (inlier ratio): {best_accuracy.get('matcher', 'Unknown')}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate paths
    images_dir = Path(args.images)
    if not images_dir.exists():
        print(f"Error: Image directory not found: {images_dir}", file=sys.stderr)
        return 1

    benchmark_exe = Path(args.benchmark)
    if not benchmark_exe.exists():
        print(f"Error: Benchmark executable not found: {benchmark_exe}", file=sys.stderr)
        print("Please build the project first: pixi run build", file=sys.stderr)
        return 1

    camera_yaml = Path(args.camera) if args.camera else None
    if camera_yaml and not camera_yaml.exists():
        print(f"Warning: Camera YAML not found: {camera_yaml}")
        camera_yaml = None

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    results = []
    for matcher in args.matchers:
        output_file = output_dir / f"benchmark_{matcher}.json"
        result = run_benchmark(
            benchmark_exe, matcher, images_dir, output_file, camera_yaml
        )
        if result:
            results.append(result)

    # Print comparison table
    print_comparison_table(results)

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
