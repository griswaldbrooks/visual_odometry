#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace visual_odometry {

/**
 * @brief Detects ORB features in images.
 */
class FeatureDetector {
public:
    /**
     * @brief Construct a new Feature Detector.
     * @param max_features Maximum number of features to detect.
     */
    explicit FeatureDetector(int max_features = 2000);

    /**
     * @brief Detect keypoints and compute descriptors.
     * @param image Grayscale input image.
     * @param keypoints Output keypoints.
     * @param descriptors Output descriptors.
     */
    void detect(const cv::Mat& image,
                std::vector<cv::KeyPoint>& keypoints,
                cv::Mat& descriptors) const;

    /**
     * @brief Detect keypoints only (no descriptors).
     * @param image Grayscale input image.
     * @return Vector of detected keypoints.
     */
    std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat& image) const;

    /**
     * @brief Draw keypoints on an image.
     * @param image Input image.
     * @param keypoints Keypoints to draw.
     * @return Image with keypoints drawn.
     */
    static cv::Mat drawKeypoints(const cv::Mat& image,
                                  const std::vector<cv::KeyPoint>& keypoints);

private:
    cv::Ptr<cv::ORB> orb_;
};

}  // namespace visual_odometry
