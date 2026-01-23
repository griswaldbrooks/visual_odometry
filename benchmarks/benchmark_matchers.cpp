#include <visual_odometry/image_loader.hpp>
#include <visual_odometry/image_matcher.hpp>
#include <visual_odometry/motion_estimator.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
    std::string image_dir;
    std::string matcher = "orb";
    std::string output_path = "benchmark_results.json";
    std::string camera_yaml;  // Optional camera intrinsics for accuracy metrics
};

void print_usage(std::string_view program) {
    std::cerr << "Usage: " << program << " --images <dir> [options]\n"
              << "\nRequired:\n"
              << "  --images <dir>     Path to image directory\n"
              << "\nOptions:\n"
              << "  --matcher <name>   Feature matcher: orb (default), matchanything, lightglue\n"
              << "  --output <file>    Output JSON file (default: benchmark_results.json)\n"
              << "  --camera <file>    Camera intrinsics YAML (for accuracy metrics)\n"
              << "  --help             Show this help message\n";
}

[[nodiscard]] auto parse_args(int argc, char* argv[]) -> std::optional<Args> {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string_view const arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return std::nullopt;
        }
        if (arg == "--images" && i + 1 < argc) {
            args.image_dir = argv[++i];
        } else if (arg == "--matcher" && i + 1 < argc) {
            args.matcher = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (arg == "--camera" && i + 1 < argc) {
            args.camera_yaml = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << '\n';
            print_usage(argv[0]);
            return std::nullopt;
        }
    }

    if (args.image_dir.empty()) {
        std::cerr << "Error: --images is required\n";
        print_usage(argv[0]);
        return std::nullopt;
    }

    return args;
}

struct BenchmarkResult {
    std::string matcher_name;
    std::size_t num_pairs{0};
    double total_time_ms{0.0};
    double avg_time_ms{0.0};

    // Accuracy metrics
    std::size_t total_matches{0};
    double avg_matches{0.0};
    std::size_t total_inliers{0};
    double avg_inliers{0.0};
    double avg_inlier_ratio{0.0};
};

[[nodiscard]] auto benchmark_matcher(
    visual_odometry::image_matcher const& matcher,
    visual_odometry::ImageLoader& loader,
    std::optional<visual_odometry::CameraIntrinsics> const& intrinsics)
    -> BenchmarkResult
{
    BenchmarkResult result;
    result.matcher_name = std::string(
        std::visit([](auto const& m) { return m.name(); }, matcher));

    loader.reset();

    std::cout << "Benchmarking " << result.matcher_name << "...\n";

    while (loader.has_next()) {
        auto const pair_result = loader.next_pair();
        if (!pair_result) {
            std::cerr << "Error loading pair: " << pair_result.error() << '\n';
            break;
        }

        auto const& [img1, img2] = *pair_result;

        // Time the matching operation
        auto const start = std::chrono::high_resolution_clock::now();
        auto const match_result = std::visit(
            [&](visual_odometry::matcher auto const& m) {
                return m.match_images(img1, img2);
            },
            matcher);
        auto const end = std::chrono::high_resolution_clock::now();

        auto const duration_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        result.total_time_ms += duration_ms;
        result.num_pairs++;

        // Accuracy metrics
        auto const num_matches = match_result.points1.size();
        result.total_matches += num_matches;

        std::size_t inliers = 0;
        if (intrinsics && num_matches >= 5) {
            visual_odometry::MotionEstimatorConfig const config{};
            auto const motion = visual_odometry::estimate_motion(
                match_result.points1, match_result.points2, *intrinsics, config);
            inliers = static_cast<std::size_t>(motion.inliers);
            result.total_inliers += inliers;
        }

        std::cout << "  Pair " << result.num_pairs
                  << ": " << std::fixed << std::setprecision(2) << duration_ms
                  << " ms (" << num_matches << " matches";
        if (intrinsics) {
            std::cout << ", " << inliers << " inliers";
        }
        std::cout << ")\n";
    }

    if (result.num_pairs > 0) {
        result.avg_time_ms = result.total_time_ms / static_cast<double>(result.num_pairs);
        result.avg_matches = static_cast<double>(result.total_matches) / static_cast<double>(result.num_pairs);
        if (intrinsics) {
            result.avg_inliers = static_cast<double>(result.total_inliers) / static_cast<double>(result.num_pairs);
            if (result.total_matches > 0) {
                result.avg_inlier_ratio = static_cast<double>(result.total_inliers) / static_cast<double>(result.total_matches);
            }
        }
    }

    return result;
}

void write_results_json(BenchmarkResult const& result, std::string_view output_path) {
    std::ofstream file(output_path.data());
    if (!file) {
        std::cerr << "Error: Failed to open output file: " << output_path << '\n';
        return;
    }

    file << "{\n"
         << "  \"matcher\": \"" << result.matcher_name << "\",\n"
         << "  \"num_pairs\": " << result.num_pairs << ",\n"
         << "  \"total_time_ms\": " << std::fixed << std::setprecision(2)
         << result.total_time_ms << ",\n"
         << "  \"avg_time_ms\": " << std::fixed << std::setprecision(2)
         << result.avg_time_ms << ",\n"
         << "  \"total_matches\": " << result.total_matches << ",\n"
         << "  \"avg_matches\": " << std::fixed << std::setprecision(2)
         << result.avg_matches << ",\n"
         << "  \"total_inliers\": " << result.total_inliers << ",\n"
         << "  \"avg_inliers\": " << std::fixed << std::setprecision(2)
         << result.avg_inliers << ",\n"
         << "  \"avg_inlier_ratio\": " << std::fixed << std::setprecision(4)
         << result.avg_inlier_ratio << "\n"
         << "}\n";

    std::cout << "\nResults written to: " << output_path << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
    auto const args = parse_args(argc, argv);
    if (!args) {
        return 1;
    }

    std::cout << "Benchmark Configuration\n"
              << "  Images:  " << args->image_dir << '\n'
              << "  Matcher: " << args->matcher << '\n'
              << "  Output:  " << args->output_path << '\n';
    if (!args->camera_yaml.empty()) {
        std::cout << "  Camera:  " << args->camera_yaml << '\n';
    }
    std::cout << '\n';

    // Load camera intrinsics (optional, for accuracy metrics)
    std::optional<visual_odometry::CameraIntrinsics> intrinsics;
    if (!args->camera_yaml.empty()) {
        auto const intrinsics_result =
            visual_odometry::CameraIntrinsics::load_from_yaml(args->camera_yaml);
        if (!intrinsics_result) {
            std::cerr << "Warning: Failed to load camera intrinsics: "
                      << intrinsics_result.error() << '\n';
            std::cerr << "Continuing without accuracy metrics.\n\n";
        } else {
            intrinsics = *intrinsics_result;
            std::cout << "Loaded camera intrinsics (fx=" << intrinsics_result->fx
                      << ", fy=" << intrinsics_result->fy << ")\n";
        }
    }

    // Initialize image loader
    auto loader_result = visual_odometry::ImageLoader::create(args->image_dir);
    if (!loader_result) {
        std::cerr << "Error loading images: " << loader_result.error() << '\n';
        return 1;
    }
    auto loader = std::move(*loader_result);

    std::cout << "Loaded " << loader.size() << " images\n\n";

    // Create matcher
    visual_odometry::image_matcher matcher;
    try {
        matcher = visual_odometry::create_image_matcher(args->matcher);
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << '\n';
        std::cerr << "Available matchers: orb, lightglue\n";
        return 1;
    }

    // Run benchmark
    auto const result = benchmark_matcher(matcher, loader, intrinsics);

    // Print summary
    std::cout << "\nBenchmark Summary\n"
              << "  Matcher:       " << result.matcher_name << '\n'
              << "  Image pairs:   " << result.num_pairs << '\n'
              << "  Total time:    " << std::fixed << std::setprecision(2)
              << result.total_time_ms << " ms\n"
              << "  Avg time/pair: " << std::fixed << std::setprecision(2)
              << result.avg_time_ms << " ms\n";
    if (intrinsics) {
        std::cout << "  Avg matches:   " << std::fixed << std::setprecision(2)
                  << result.avg_matches << '\n'
                  << "  Avg inliers:   " << std::fixed << std::setprecision(2)
                  << result.avg_inliers << '\n'
                  << "  Inlier ratio:  " << std::fixed << std::setprecision(4)
                  << result.avg_inlier_ratio << '\n';
    }
    std::cout << '\n';

    // Write results to JSON
    write_results_json(result, args->output_path);

    return 0;
}
