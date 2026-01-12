#include <gtest/gtest.h>
#include <visual_odometry/feature_detector.hpp>
#include <opencv2/imgproc.hpp>

class FeatureDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test image with some texture (checkerboard pattern)
        testImage_ = cv::Mat(480, 640, CV_8UC1);
        for (int y = 0; y < testImage_.rows; ++y) {
            for (int x = 0; x < testImage_.cols; ++x) {
                testImage_.at<uchar>(y, x) =
                    ((x / 32) % 2 == (y / 32) % 2) ? 255 : 0;
            }
        }
        // Add some noise for more realistic features
        cv::Mat noise(testImage_.size(), CV_8UC1);
        cv::randn(noise, 0, 25);
        testImage_ += noise;
    }

    cv::Mat testImage_;
};

TEST_F(FeatureDetectorTest, DetectsKeypoints) {
    visual_odometry::FeatureDetector detector;
    auto keypoints = detector.detectKeypoints(testImage_);

    EXPECT_GT(keypoints.size(), 0);
}

TEST_F(FeatureDetectorTest, DetectsKeypointsAndDescriptors) {
    visual_odometry::FeatureDetector detector;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    detector.detect(testImage_, keypoints, descriptors);

    EXPECT_GT(keypoints.size(), 0);
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(static_cast<size_t>(descriptors.rows), keypoints.size());
    EXPECT_EQ(descriptors.cols, 32);  // ORB descriptor size
}

TEST_F(FeatureDetectorTest, RespectsMaxFeatures) {
    int maxFeatures = 100;
    visual_odometry::FeatureDetector detector(maxFeatures);
    auto keypoints = detector.detectKeypoints(testImage_);

    EXPECT_LE(keypoints.size(), static_cast<size_t>(maxFeatures));
}

TEST_F(FeatureDetectorTest, DrawsKeypoints) {
    visual_odometry::FeatureDetector detector;
    auto keypoints = detector.detectKeypoints(testImage_);
    cv::Mat output = visual_odometry::FeatureDetector::drawKeypoints(testImage_, keypoints);

    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.channels(), 3);  // Color output
}

TEST_F(FeatureDetectorTest, HandlesEmptyImage) {
    visual_odometry::FeatureDetector detector;
    cv::Mat emptyImage;
    auto keypoints = detector.detectKeypoints(emptyImage);

    EXPECT_EQ(keypoints.size(), 0);
}

TEST_F(FeatureDetectorTest, HandlesUniformImage) {
    visual_odometry::FeatureDetector detector;
    cv::Mat uniformImage(480, 640, CV_8UC1, cv::Scalar(128));
    auto keypoints = detector.detectKeypoints(uniformImage);

    // Uniform image should have very few or no keypoints
    EXPECT_LT(keypoints.size(), 10);
}
