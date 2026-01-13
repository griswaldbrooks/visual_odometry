#include <gtest/gtest.h>
#include <visual_odometry/feature_detector.hpp>
#include <opencv2/imgproc.hpp>

class FeatureDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test image with some texture (checkerboard pattern)
        test_image_ = cv::Mat(480, 640, CV_8UC1);
        for (int y = 0; y < test_image_.rows; ++y) {
            for (int x = 0; x < test_image_.cols; ++x) {
                test_image_.at<uchar>(y, x) =
                    ((x / 32) % 2 == (y / 32) % 2) ? 255 : 0;
            }
        }
        // Add some noise for more realistic features
        cv::Mat noise(test_image_.size(), CV_8UC1);
        cv::randn(noise, 0, 25);
        test_image_ += noise;
    }

    cv::Mat test_image_;
};

TEST_F(FeatureDetectorTest, DetectsKeypoints) {
    visual_odometry::FeatureDetector detector;
    auto const keypoints = detector.detect_keypoints(test_image_);

    EXPECT_GT(keypoints.size(), 0);
}

TEST_F(FeatureDetectorTest, DetectsKeypointsAndDescriptors) {
    visual_odometry::FeatureDetector detector;
    auto const result = detector.detect(test_image_);

    EXPECT_GT(result.keypoints.size(), 0);
    EXPECT_FALSE(result.descriptors.empty());
    EXPECT_EQ(static_cast<size_t>(result.descriptors.rows), result.keypoints.size());
    EXPECT_EQ(result.descriptors.cols, 32);  // ORB descriptor size
}

TEST_F(FeatureDetectorTest, RespectsMaxFeatures) {
    int const max_features = 100;
    visual_odometry::FeatureDetector detector(max_features);
    auto const keypoints = detector.detect_keypoints(test_image_);

    EXPECT_LE(keypoints.size(), static_cast<size_t>(max_features));
}

TEST_F(FeatureDetectorTest, DrawsKeypoints) {
    visual_odometry::FeatureDetector detector;
    auto const keypoints = detector.detect_keypoints(test_image_);
    cv::Mat const output = visual_odometry::FeatureDetector::draw_keypoints(test_image_, keypoints);

    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.channels(), 3);  // Color output
}

TEST_F(FeatureDetectorTest, HandlesEmptyImage) {
    visual_odometry::FeatureDetector detector;
    cv::Mat const empty_image;
    auto const keypoints = detector.detect_keypoints(empty_image);

    EXPECT_EQ(keypoints.size(), 0);
}

TEST_F(FeatureDetectorTest, HandlesUniformImage) {
    visual_odometry::FeatureDetector detector;
    cv::Mat const uniform_image(480, 640, CV_8UC1, cv::Scalar(128));
    auto const keypoints = detector.detect_keypoints(uniform_image);

    // Uniform image should have very few or no keypoints
    EXPECT_LT(keypoints.size(), 10);
}
