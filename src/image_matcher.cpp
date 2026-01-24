#include <visual_odometry/image_matcher.hpp>
#include <visual_odometry/onnx_session.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>

namespace visual_odometry {

namespace {

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

auto match_images_orb(cv::Mat const& img1,
                      cv::Mat const& img2,
                      feature_detector_config const& detector_config,
                      feature_matcher_config const& matcher_config)
    -> match_result
{
    // Detect features in both images
    auto const det1 = detect_features(img1, detector_config);
    auto const det2 = detect_features(img2, detector_config);

    // Match features
    return match_features(det1.descriptors, det2.descriptors,
                          det1.keypoints, det2.keypoints,
                          matcher_config);
}

// lightglue_matcher implementation

lightglue_matcher::lightglue_matcher(std::filesystem::path model_path)
    : model_path_(std::move(model_path)),
      session_(std::make_unique<OnnxSession>(model_path_)) {}

lightglue_matcher::lightglue_matcher(lightglue_matcher&&) noexcept = default;
auto lightglue_matcher::operator=(lightglue_matcher&&) noexcept -> lightglue_matcher& = default;
lightglue_matcher::~lightglue_matcher() = default;

auto lightglue_matcher::match_images(cv::Mat const& img1,
                                     cv::Mat const& img2) const -> match_result {
    match_result result;

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

auto create_image_matcher(std::string_view name) -> image_matcher {
    // Convert to lowercase for comparison
    std::string lower_name;
    lower_name.reserve(name.size());
    for (char c : name) {
        lower_name.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (lower_name == "orb" || lower_name.empty()) {
        return orb_matcher{};
    }

    if (lower_name == "lightglue") {
        return lightglue_matcher{};
    }

    throw std::runtime_error("Unknown matcher: " + std::string(name));
}

}  // namespace visual_odometry
