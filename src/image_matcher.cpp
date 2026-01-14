#include <visual_odometry/image_matcher.hpp>
#include <algorithm>

namespace visual_odometry {

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

    // Future: add "matchanything" here
    // if (lower_name == "matchanything") {
    //     return std::make_unique<MatchAnythingMatcher>();
    // }

    return nullptr;
}

}  // namespace visual_odometry
