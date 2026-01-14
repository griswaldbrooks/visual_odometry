#include <visual_odometry/image_matcher.hpp>
#include <opencv2/imgcodecs.hpp>
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

    return nullptr;
}

}  // namespace visual_odometry
