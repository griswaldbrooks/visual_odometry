#include <gtest/gtest.h>
#include <visual_odometry/feature_detector.hpp>
#include <visual_odometry/feature_matcher.hpp>
#include <opencv2/imgproc.hpp>

class FeatureMatcherTest : public ::testing::Test {
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

        // Detect features
        auto const result1 = detector_.detect(image1_);
        auto const result2 = detector_.detect(image2_);
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
    visual_odometry::FeatureDetector detector_;
};

TEST_F(FeatureMatcherTest, MatchesSimilarImages) {
    // GIVEN a feature matcher
    visual_odometry::FeatureMatcher matcher;

    // WHEN matching descriptors from two similar images
    auto const result = matcher.match(descriptors1_, descriptors2_, keypoints1_, keypoints2_);

    // THEN matches should be found
    EXPECT_GT(result.matches.size(), 0);
    // AND point arrays should match in size
    EXPECT_EQ(result.points1.size(), result.matches.size());
    EXPECT_EQ(result.points2.size(), result.matches.size());
}

TEST_F(FeatureMatcherTest, MatchedPointsAreConsistent) {
    // GIVEN a feature matcher with matches
    visual_odometry::FeatureMatcher matcher;
    auto const result = matcher.match(descriptors1_, descriptors2_, keypoints1_, keypoints2_);

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

TEST_F(FeatureMatcherTest, StricterRatioReducesMatches) {
    // GIVEN matchers with different ratio thresholds
    visual_odometry::FeatureMatcher loose_matcher(0.9f);
    visual_odometry::FeatureMatcher strict_matcher(0.5f);

    // WHEN matching with each
    auto const loose_result = loose_matcher.match(descriptors1_, descriptors2_, keypoints1_, keypoints2_);
    auto const strict_result = strict_matcher.match(descriptors1_, descriptors2_, keypoints1_, keypoints2_);

    // THEN stricter threshold should produce fewer matches
    EXPECT_GE(loose_result.matches.size(), strict_result.matches.size());
}

TEST_F(FeatureMatcherTest, HandlesEmptyDescriptors) {
    // GIVEN a feature matcher
    visual_odometry::FeatureMatcher matcher;
    cv::Mat const empty_desc;

    // WHEN matching with empty first descriptors
    auto result = matcher.match(empty_desc, descriptors2_, {}, keypoints2_);

    // THEN no matches should be returned
    EXPECT_EQ(result.matches.size(), 0);

    // WHEN matching with empty second descriptors
    result = matcher.match(descriptors1_, empty_desc, keypoints1_, {});

    // THEN no matches should be returned
    EXPECT_EQ(result.matches.size(), 0);
}

TEST_F(FeatureMatcherTest, DrawsMatches) {
    // GIVEN matched features
    visual_odometry::FeatureMatcher matcher;
    auto const result = matcher.match(descriptors1_, descriptors2_, keypoints1_, keypoints2_);

    // WHEN drawing matches
    cv::Mat const output = visual_odometry::FeatureMatcher::draw_matches(
        image1_, keypoints1_, image2_, keypoints2_, result.matches);

    // THEN output should be a valid side-by-side image
    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.cols, image1_.cols + image2_.cols);
    EXPECT_EQ(output.channels(), 3);
}
