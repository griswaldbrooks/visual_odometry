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
    // GIVEN a feature detector config with default settings
    auto const config = visual_odometry::FeatureDetectorConfig{};

    // WHEN detecting keypoints in a textured image
    auto const keypoints = visual_odometry::detect_keypoints_only(test_image_, config);

    // THEN keypoints should be detected
    EXPECT_GT(keypoints.size(), 0);
}

TEST_F(FeatureDetectorTest, DetectsKeypointsAndDescriptors) {
    // GIVEN a feature detector config
    auto const config = visual_odometry::FeatureDetectorConfig{};

    // WHEN detecting features with descriptors
    auto const result = visual_odometry::detect_features(test_image_, config);

    // THEN keypoints and descriptors should be returned
    EXPECT_GT(result.keypoints.size(), 0);
    EXPECT_FALSE(result.descriptors.empty());
    // AND descriptor count should match keypoint count
    EXPECT_EQ(static_cast<size_t>(result.descriptors.rows), result.keypoints.size());
    // AND ORB descriptors should be 32 bytes
    EXPECT_EQ(result.descriptors.cols, 32);
}

TEST_F(FeatureDetectorTest, RespectsMaxFeatures) {
    // GIVEN a detector config with limited max features
    int const max_features = 100;
    auto const config = visual_odometry::FeatureDetectorConfig{.max_features = max_features};

    // WHEN detecting keypoints
    auto const keypoints = visual_odometry::detect_keypoints_only(test_image_, config);

    // THEN keypoint count should not exceed max_features
    EXPECT_LE(keypoints.size(), static_cast<size_t>(max_features));
}

TEST_F(FeatureDetectorTest, DrawsKeypoints) {
    // GIVEN detected keypoints
    auto const config = visual_odometry::FeatureDetectorConfig{};
    auto const keypoints = visual_odometry::detect_keypoints_only(test_image_, config);

    // WHEN drawing keypoints
    cv::Mat const output = visual_odometry::draw_keypoints(test_image_, keypoints);

    // THEN output should be a valid color image
    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.channels(), 3);
}

TEST_F(FeatureDetectorTest, HandlesEmptyImage) {
    // GIVEN an empty image and config
    auto const config = visual_odometry::FeatureDetectorConfig{};
    cv::Mat const empty_image;

    // WHEN detecting keypoints
    auto const keypoints = visual_odometry::detect_keypoints_only(empty_image, config);

    // THEN no keypoints should be returned
    EXPECT_EQ(keypoints.size(), 0);
}

TEST_F(FeatureDetectorTest, HandlesUniformImage) {
    // GIVEN a uniform (featureless) image and config
    auto const config = visual_odometry::FeatureDetectorConfig{};
    cv::Mat const uniform_image(480, 640, CV_8UC1, cv::Scalar(128));

    // WHEN detecting keypoints
    auto const keypoints = visual_odometry::detect_keypoints_only(uniform_image, config);

    // THEN very few or no keypoints should be found
    EXPECT_LT(keypoints.size(), 10);
}
