#include <visual_odometry/image_matcher.hpp>
#include <visual_odometry/onnx_session.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace visual_odometry {

namespace {

// Simple JSON array parser for [[x,y], ...] format
auto parse_point_array(std::string_view json) -> std::vector<cv::Point2f> {
    std::vector<cv::Point2f> points;
    std::regex point_regex(R"(\[\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\])");

    std::string json_str(json);
    auto begin = std::sregex_iterator(json_str.begin(), json_str.end(), point_regex);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        auto x = std::stof((*it)[1].str());
        auto y = std::stof((*it)[2].str());
        points.emplace_back(x, y);
    }

    return points;
}

// Execute command and capture stdout
auto exec_command(std::string const& cmd) -> std::string {
    std::array<char, 4096> buffer{};
    std::string result;

    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed");
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}

// Create temporary file path
auto make_temp_path(std::string_view suffix) -> std::filesystem::path {
    auto path = std::filesystem::temp_directory_path() / ("vo_match_" + std::to_string(std::rand()) + std::string(suffix));
    return path;
}

// Preprocess image for LightGlue ONNX inference
// Input: grayscale or BGR cv::Mat
// Output: float vector in NCHW format (1, 3, H, W), normalized to [0,1]
auto preprocess_image_for_lightglue(cv::Mat const& img)
    -> std::pair<std::vector<float>, std::vector<int64_t>> {
    cv::Mat rgb;

    // Convert to RGB if needed
    if (img.channels() == 1) {
        cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
    } else if (img.channels() == 3) {
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, rgb, cv::COLOR_BGRA2RGB);
    } else {
        rgb = img.clone();
    }

    // Convert to float32 and normalize to [0, 1]
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // Prepare output tensor in NCHW format
    int64_t const height = float_img.rows;
    int64_t const width = float_img.cols;
    std::vector<int64_t> const shape = {1, 3, height, width};

    // Convert HWC to CHW format
    std::vector<float> tensor_data(static_cast<size_t>(3 * height * width));
    for (int64_t c = 0; c < 3; ++c) {
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                auto const src_idx = static_cast<size_t>(h * width * 3 + w * 3 + c);
                auto const dst_idx = static_cast<size_t>(c * height * width + h * width + w);
                tensor_data[dst_idx] = float_img.ptr<float>()[src_idx];
            }
        }
    }

    return {std::move(tensor_data), shape};
}

}  // namespace

OrbImageMatcher::OrbImageMatcher(int max_features, float ratio_threshold)
    : detector_(max_features), matcher_(ratio_threshold) {}

auto OrbImageMatcher::match_images(cv::Mat const& img1,
                                   cv::Mat const& img2) const -> MatchResult {
    // Detect features in both images
    auto const det1 = detector_.detect(img1);
    auto const det2 = detector_.detect(img2);

    // Match features
    return matcher_.match(det1.descriptors, det2.descriptors,
                          det1.keypoints, det2.keypoints);
}

MatchAnythingMatcher::MatchAnythingMatcher(std::filesystem::path script_path,
                                           std::string python_exe,
                                           float threshold)
    : script_path_(std::move(script_path)),
      python_exe_(std::move(python_exe)),
      threshold_(threshold) {}

auto MatchAnythingMatcher::match_images(cv::Mat const& img1,
                                        cv::Mat const& img2) const -> MatchResult {
    MatchResult result;

    // Write images to temporary files
    auto const temp1 = make_temp_path("_1.png");
    auto const temp2 = make_temp_path("_2.png");

    cv::imwrite(temp1.string(), img1);
    cv::imwrite(temp2.string(), img2);

    // Build command
    std::ostringstream cmd;
    cmd << python_exe_ << " " << script_path_.string()
        << " " << temp1.string()
        << " " << temp2.string()
        << " --threshold " << threshold_;

    // Execute and capture output
    std::string output;
    try {
        output = exec_command(cmd.str());
    } catch (std::exception const& e) {
        std::filesystem::remove(temp1);
        std::filesystem::remove(temp2);
        return result;
    }

    // Clean up temp files
    std::filesystem::remove(temp1);
    std::filesystem::remove(temp2);

    // Check for error in output
    if (output.find("\"error\"") != std::string::npos) {
        return result;
    }

    // Parse keypoints from JSON
    // Format: {"keypoints0": [[x,y],...], "keypoints1": [[x,y],...], "scores": [...]}
    auto const kp0_start = output.find("\"keypoints0\"");
    auto const kp1_start = output.find("\"keypoints1\"");

    if (kp0_start == std::string::npos || kp1_start == std::string::npos) {
        return result;
    }

    // Extract keypoints0 array
    auto const kp0_array_start = output.find('[', kp0_start);
    auto const kp0_array_end = output.find(']', output.find(']', kp0_array_start) + 1);
    auto const kp0_json = output.substr(kp0_array_start, kp0_array_end - kp0_array_start + 1);

    // Extract keypoints1 array
    auto const kp1_array_start = output.find('[', kp1_start);
    auto const kp1_array_end = output.find(']', output.find(']', kp1_array_start) + 1);
    auto const kp1_json = output.substr(kp1_array_start, kp1_array_end - kp1_array_start + 1);

    result.points1 = parse_point_array(kp0_json);
    result.points2 = parse_point_array(kp1_json);

    // Create dummy DMatch entries (MatchAnything doesn't provide traditional descriptors)
    for (size_t i = 0; i < result.points1.size(); ++i) {
        cv::DMatch match;
        match.queryIdx = static_cast<int>(i);
        match.trainIdx = static_cast<int>(i);
        match.distance = 0.0f;
        result.matches.push_back(match);
    }

    return result;
}

// LightGlueImageMatcher implementation

LightGlueImageMatcher::LightGlueImageMatcher(std::filesystem::path model_path)
    : model_path_(std::move(model_path)),
      session_(std::make_unique<OnnxSession>(model_path_)) {}

LightGlueImageMatcher::LightGlueImageMatcher(LightGlueImageMatcher&&) noexcept = default;
auto LightGlueImageMatcher::operator=(LightGlueImageMatcher&&) noexcept -> LightGlueImageMatcher& = default;
LightGlueImageMatcher::~LightGlueImageMatcher() = default;

auto LightGlueImageMatcher::match_images(cv::Mat const& img1,
                                         cv::Mat const& img2) const -> MatchResult {
    MatchResult result;

    // Preprocess images
    auto [data0, shape0] = preprocess_image_for_lightglue(img1);
    auto [data1, shape1] = preprocess_image_for_lightglue(img2);

    // Create input tensors
    auto tensor0 = create_tensor(data0, shape0);
    auto tensor1 = create_tensor(data1, shape1);

    // Prepare input/output names
    std::array<char const*, 2> input_names = {"image0", "image1"};
    std::array<char const*, 4> output_names = {"kpts0", "kpts1", "matches0", "mscores0"};

    // Run inference (const_cast needed because ORT API is not const-correct)
    auto& mutable_session = const_cast<OnnxSession&>(*session_);
    std::array<Ort::Value, 2> inputs = {std::move(tensor0), std::move(tensor1)};

    auto outputs = mutable_session.run(
        std::span{input_names},
        std::span{inputs},
        std::span{output_names});

    // Extract matched keypoints from outputs
    // outputs[0] = kpts0: [1, N, 2]
    // outputs[1] = kpts1: [1, M, 2]
    // outputs[2] = matches0: [K, 2] - indices into kpts0 and kpts1
    // outputs[3] = mscores0: [K] - match scores

    if (outputs.size() < 4) {
        return result;  // Inference failed
    }

    auto const& kpts0 = outputs[0];
    auto const& kpts1 = outputs[1];
    auto const& matches = outputs[2];

    // Get tensor info
    auto const kpts0_info = kpts0.GetTensorTypeAndShapeInfo();
    auto const kpts1_info = kpts1.GetTensorTypeAndShapeInfo();
    auto const matches_info = matches.GetTensorTypeAndShapeInfo();

    auto const kpts0_shape = kpts0_info.GetShape();
    auto const kpts1_shape = kpts1_info.GetShape();
    auto const matches_shape = matches_info.GetShape();

    // Get raw data pointers
    auto const* kpts0_data = kpts0.GetTensorData<float>();
    auto const* kpts1_data = kpts1.GetTensorData<float>();
    auto const* matches_data = matches.GetTensorData<int64_t>();

    // Number of matches
    auto const num_matches = matches_shape.empty() ? 0 : static_cast<size_t>(matches_shape[0]);
    auto const num_kpts0 = (kpts0_shape.size() >= 2) ? static_cast<size_t>(kpts0_shape[1]) : 0;
    auto const num_kpts1 = (kpts1_shape.size() >= 2) ? static_cast<size_t>(kpts1_shape[1]) : 0;

    // Extract matched points
    result.points1.reserve(num_matches);
    result.points2.reserve(num_matches);
    result.matches.reserve(num_matches);

    for (size_t i = 0; i < num_matches; ++i) {
        auto const idx0 = static_cast<size_t>(matches_data[i * 2]);
        auto const idx1 = static_cast<size_t>(matches_data[i * 2 + 1]);

        if (idx0 < num_kpts0 && idx1 < num_kpts1) {
            // Get keypoint coordinates (x, y)
            float const x0 = kpts0_data[idx0 * 2];
            float const y0 = kpts0_data[idx0 * 2 + 1];
            float const x1 = kpts1_data[idx1 * 2];
            float const y1 = kpts1_data[idx1 * 2 + 1];

            result.points1.emplace_back(x0, y0);
            result.points2.emplace_back(x1, y1);

            cv::DMatch match;
            match.queryIdx = static_cast<int>(i);
            match.trainIdx = static_cast<int>(i);
            match.distance = 0.0f;
            result.matches.push_back(match);
        }
    }

    return result;
}

auto create_matcher(std::string_view name) -> std::unique_ptr<ImageMatcher> {
    // Convert to lowercase for comparison
    std::string lower_name;
    lower_name.reserve(name.size());
    for (char c : name) {
        lower_name.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (lower_name == "orb" || lower_name.empty()) {
        return std::make_unique<OrbImageMatcher>();
    }

    if (lower_name == "matchanything") {
        return std::make_unique<MatchAnythingMatcher>();
    }

    if (lower_name == "lightglue") {
        return std::make_unique<LightGlueImageMatcher>();
    }

    return nullptr;
}

}  // namespace visual_odometry
