#include <gtest/gtest.h>
#include <visual_odometry/feature_detector.hpp>
#include <visual_odometry/feature_matcher.hpp>

#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>

class feature_matcher_test : public ::testing::Test {
protected:
    void SetUp() override {
        // Create two similar test images (shifted checkerboard)
        image1_ = cv::Mat(480, 640, CV_8UC1);
        image2_ = cv::Mat(480, 640, CV_8UC1);

        for (int y = 0; y < image1_.rows; ++y) {
            for (int x = 0; x < image1_.cols; ++x) {
                image1_.at<uchar>(y, x) = ((x / 32) % 2 == (y / 32) % 2) ? 255 : 0;
                // Second image shifted by 10 pixels
                int const x2 = (x + 10) % image2_.cols;
                image2_.at<uchar>(y, x) = ((x2 / 32) % 2 == (y / 32) % 2) ? 255 : 0;
            }
        }

        // Add noise
        cv::Mat noise1(image1_.size(), CV_8UC1);
        cv::Mat noise2(image2_.size(), CV_8UC1);
        cv::randn(noise1, 0, 15);
        cv::randn(noise2, 0, 15);
        image1_ += noise1;
        image2_ += noise2;

        // Detect features using pure function API
        auto const config = visual_odometry::feature_detector_config{};
        auto const result1 = visual_odometry::detect_features(image1_, config);
        auto const result2 = visual_odometry::detect_features(image2_, config);
        keypoints1_ = result1.keypoints;
        keypoints2_ = result2.keypoints;
        descriptors1_ = result1.descriptors;
        descriptors2_ = result2.descriptors;
    }

    cv::Mat image1_;
    cv::Mat image2_;
    std::vector<cv::KeyPoint> keypoints1_;
    std::vector<cv::KeyPoint> keypoints2_;
    cv::Mat descriptors1_;
    cv::Mat descriptors2_;
};

TEST_F(feature_matcher_test, MatchesSimilarImages) {
    // GIVEN a feature matcher config
    auto const config = visual_odometry::feature_matcher_config{};

    // WHEN matching descriptors from two similar images
    auto const result = visual_odometry::match_features(
        descriptors1_, descriptors2_, keypoints1_, keypoints2_, config);

    // THEN matches should be found
    EXPECT_GT(result.matches.size(), 0);
    // AND point arrays should match in size
    EXPECT_EQ(result.points1.size(), result.matches.size());
    EXPECT_EQ(result.points2.size(), result.matches.size());
}

TEST_F(feature_matcher_test, MatchedPointsAreConsistent) {
    // GIVEN a feature matcher config with matches
    auto const config = visual_odometry::feature_matcher_config{};
    auto const result = visual_odometry::match_features(
        descriptors1_, descriptors2_, keypoints1_, keypoints2_, config);

    // WHEN examining each match
    for (size_t i = 0; i < result.matches.size(); ++i) {
        auto const idx1 = static_cast<size_t>(result.matches[i].queryIdx);
        auto const idx2 = static_cast<size_t>(result.matches[i].trainIdx);

        // THEN extracted points should match keypoint coordinates
        EXPECT_FLOAT_EQ(result.points1[i].x, keypoints1_[idx1].pt.x);
        EXPECT_FLOAT_EQ(result.points1[i].y, keypoints1_[idx1].pt.y);
        EXPECT_FLOAT_EQ(result.points2[i].x, keypoints2_[idx2].pt.x);
        EXPECT_FLOAT_EQ(result.points2[i].y, keypoints2_[idx2].pt.y);
    }
}

TEST_F(feature_matcher_test, StricterRatioReducesMatches) {
    // GIVEN matcher configs with different ratio thresholds
    auto const loose_config = visual_odometry::feature_matcher_config{.ratio_threshold = 0.9f};
    auto const strict_config = visual_odometry::feature_matcher_config{.ratio_threshold = 0.5f};

    // WHEN matching with each
    auto const loose_result = visual_odometry::match_features(
        descriptors1_, descriptors2_, keypoints1_, keypoints2_, loose_config);
    auto const strict_result = visual_odometry::match_features(
        descriptors1_, descriptors2_, keypoints1_, keypoints2_, strict_config);

    // THEN stricter threshold should produce fewer matches
    EXPECT_GE(loose_result.matches.size(), strict_result.matches.size());
}

TEST_F(feature_matcher_test, HandlesEmptyDescriptors) {
    // GIVEN a feature matcher config
    auto const config = visual_odometry::feature_matcher_config{};
    cv::Mat const empty_desc;

    // WHEN matching with empty first descriptors
    auto result = visual_odometry::match_features(
        empty_desc, descriptors2_, {}, keypoints2_, config);

    // THEN no matches should be returned
    EXPECT_EQ(result.matches.size(), 0);

    // WHEN matching with empty second descriptors
    result = visual_odometry::match_features(
        descriptors1_, empty_desc, keypoints1_, {}, config);

    // THEN no matches should be returned
    EXPECT_EQ(result.matches.size(), 0);
}

TEST_F(feature_matcher_test, DrawsMatches) {
    // GIVEN matched features
    auto const config = visual_odometry::feature_matcher_config{};
    auto const result = visual_odometry::match_features(
        descriptors1_, descriptors2_, keypoints1_, keypoints2_, config);

    // WHEN drawing matches
    cv::Mat const output = visual_odometry::draw_feature_matches(
        image1_, keypoints1_, image2_, keypoints2_, result.matches);

    // THEN output should be a valid side-by-side image
    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.cols, image1_.cols + image2_.cols);
    EXPECT_EQ(output.channels(), 3);
}
