#include <visual_odometry/feature_detector.hpp>
#include <visual_odometry/feature_matcher.hpp>
#include <visual_odometry/image_loader.hpp>
#include <visual_odometry/motion_estimator.hpp>
#include <visual_odometry/trajectory.hpp>
#include <visual_odometry/visual_odometry.hpp>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {

struct Args {
    std::string image_dir;
    std::string camera_yaml = "data/kitti_camera.yaml";
    std::string output_path = "trajectory.json";
    int max_frames = 0;  // 0 = process all
};

void print_usage(std::string_view program) {
    std::cerr << "Usage: " << program << " --images <dir> [options]\n"
              << "\nRequired:\n"
              << "  --images <dir>     Path to image directory\n"
              << "\nOptions:\n"
              << "  --camera <file>    Camera intrinsics YAML (default: data/kitti_camera.yaml)\n"
              << "  --output <file>    Output trajectory JSON (default: trajectory.json)\n"
              << "  --max-frames <n>   Maximum frames to process (default: all)\n"
              << "  --help             Show this help message\n";
}

auto parse_args(int argc, char* argv[]) -> std::optional<Args> {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return std::nullopt;
        }
        if (arg == "--images" && i + 1 < argc) {
            args.image_dir = argv[++i];
        } else if (arg == "--camera" && i + 1 < argc) {
            args.camera_yaml = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (arg == "--max-frames" && i + 1 < argc) {
            args.max_frames = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
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

}  // namespace

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);
    if (!args) {
        return 1;
    }

    std::cout << "Visual Odometry Pipeline\n"
              << "  Images:     " << args->image_dir << "\n"
              << "  Camera:     " << args->camera_yaml << "\n"
              << "  Output:     " << args->output_path << "\n"
              << "  Max frames: " << (args->max_frames > 0 ? std::to_string(args->max_frames) : "all") << "\n\n";

    // Load camera intrinsics
    auto intrinsics_result = visual_odometry::CameraIntrinsics::load_from_yaml(args->camera_yaml);
    if (!intrinsics_result) {
        std::cerr << "Error: " << intrinsics_result.error() << "\n";
        return 1;
    }
    auto const& intrinsics = *intrinsics_result;
    std::cout << "Loaded camera intrinsics (fx=" << intrinsics.fx
              << ", fy=" << intrinsics.fy << ")\n";

    // Initialize image loader
    auto loader_result = visual_odometry::ImageLoader::create(args->image_dir);
    if (!loader_result) {
        std::cerr << "Error: " << loader_result.error() << "\n";
        return 1;
    }
    auto loader = std::move(*loader_result);
    std::cout << "Found " << loader.size() << " images\n";

    if (loader.size() < 2) {
        std::cerr << "Error: Need at least 2 images for visual odometry\n";
        return 1;
    }

    // Initialize VO components
    visual_odometry::FeatureDetector detector;
    visual_odometry::FeatureMatcher matcher;
    visual_odometry::MotionEstimator estimator(intrinsics);
    visual_odometry::Trajectory trajectory;

    // Determine number of frames to process
    auto const total_pairs = loader.size() - 1;
    auto const max_pairs = args->max_frames > 0
        ? std::min(static_cast<size_t>(args->max_frames), total_pairs)
        : total_pairs;

    std::cout << "\nProcessing " << max_pairs << " frame pairs...\n";

    // Process frame pairs
    int valid_motions = 0;
    for (size_t i = 0; i < max_pairs; ++i) {
        auto pair_result = loader.load_image_pair(i);
        if (!pair_result) {
            std::cerr << "Warning: Failed to load pair " << i << ": "
                      << pair_result.error() << "\n";
            continue;
        }
        auto const& [img1, img2] = *pair_result;

        // Detect features
        auto const det1 = detector.detect(img1);
        auto const det2 = detector.detect(img2);

        // Match features
        auto const match_result = matcher.match(
            det1.descriptors, det2.descriptors,
            det1.keypoints, det2.keypoints);

        // Estimate motion
        auto const motion = estimator.estimate(match_result.points1, match_result.points2);

        // Add to trajectory
        if (trajectory.add_motion(motion)) {
            ++valid_motions;
        }

        // Progress output
        if ((i + 1) % 100 == 0 || i == max_pairs - 1) {
            std::cout << "  Frame " << std::setw(5) << (i + 1) << "/" << max_pairs
                      << " | Features: " << det1.keypoints.size()
                      << " | Matches: " << match_result.matches.size()
                      << " | Inliers: " << motion.inliers
                      << (motion.valid ? "" : " [FAILED]") << "\n";
        }
    }

    std::cout << "\nProcessed " << max_pairs << " pairs, "
              << valid_motions << " valid motions\n";

    // Save trajectory to JSON
    auto save_result = trajectory.save_to_json(args->output_path);
    if (!save_result) {
        std::cerr << "Error: Failed to save trajectory: " << save_result.error() << "\n";
        return 1;
    }

    std::cout << "Saved trajectory with " << trajectory.size()
              << " poses to: " << args->output_path << "\n";

    return 0;
}
