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
                int x2 = (x + 10) % image2_.cols;
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
        detector_.detect(image1_, keypoints1_, descriptors1_);
        detector_.detect(image2_, keypoints2_, descriptors2_);
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
    visual_odometry::FeatureMatcher matcher;
    auto result = matcher.match(descriptors1_, descriptors2_, keypoints1_, keypoints2_);

    EXPECT_GT(result.matches.size(), 0);
    EXPECT_EQ(result.points1.size(), result.matches.size());
    EXPECT_EQ(result.points2.size(), result.matches.size());
}

TEST_F(FeatureMatcherTest, MatchedPointsAreConsistent) {
    visual_odometry::FeatureMatcher matcher;
    auto result = matcher.match(descriptors1_, descriptors2_, keypoints1_, keypoints2_);

    for (size_t i = 0; i < result.matches.size(); ++i) {
        auto idx1 = static_cast<size_t>(result.matches[i].queryIdx);
        auto idx2 = static_cast<size_t>(result.matches[i].trainIdx);

        // Verify points match the keypoint indices
        EXPECT_FLOAT_EQ(result.points1[i].x, keypoints1_[idx1].pt.x);
        EXPECT_FLOAT_EQ(result.points1[i].y, keypoints1_[idx1].pt.y);
        EXPECT_FLOAT_EQ(result.points2[i].x, keypoints2_[idx2].pt.x);
        EXPECT_FLOAT_EQ(result.points2[i].y, keypoints2_[idx2].pt.y);
    }
}

TEST_F(FeatureMatcherTest, StricterRatioReducesMatches) {
    visual_odometry::FeatureMatcher looseMatcher(0.9f);
    visual_odometry::FeatureMatcher strictMatcher(0.5f);

    auto looseResult = looseMatcher.match(descriptors1_, descriptors2_, keypoints1_, keypoints2_);
    auto strictResult = strictMatcher.match(descriptors1_, descriptors2_, keypoints1_, keypoints2_);

    EXPECT_GE(looseResult.matches.size(), strictResult.matches.size());
}

TEST_F(FeatureMatcherTest, HandlesEmptyDescriptors) {
    visual_odometry::FeatureMatcher matcher;
    cv::Mat emptyDesc;

    auto result = matcher.match(emptyDesc, descriptors2_, {}, keypoints2_);
    EXPECT_EQ(result.matches.size(), 0);

    result = matcher.match(descriptors1_, emptyDesc, keypoints1_, {});
    EXPECT_EQ(result.matches.size(), 0);
}

TEST_F(FeatureMatcherTest, DrawsMatches) {
    visual_odometry::FeatureMatcher matcher;
    auto result = matcher.match(descriptors1_, descriptors2_, keypoints1_, keypoints2_);

    cv::Mat output = visual_odometry::FeatureMatcher::drawMatches(
        image1_, keypoints1_, image2_, keypoints2_, result.matches);

    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.cols, image1_.cols + image2_.cols);
    EXPECT_EQ(output.channels(), 3);
}
