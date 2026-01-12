#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <visual_odometry/image_loader.hpp>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

class ImageLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test directory with sample images
        testDir_ = fs::temp_directory_path() / "vo_test_images";
        fs::create_directories(testDir_);

        // Create simple test images
        for (int i = 0; i < 5; ++i) {
            cv::Mat img(100, 100, CV_8UC1, cv::Scalar(i * 50));
            std::string filename = testDir_.string() + "/" +
                                   std::string(6 - std::to_string(i).length(), '0') +
                                   std::to_string(i) + ".png";
            cv::imwrite(filename, img);
        }
    }

    void TearDown() override {
        fs::remove_all(testDir_);
    }

    fs::path testDir_;
};

TEST_F(ImageLoaderTest, LoadsImagesFromDirectory) {
    visual_odometry::ImageLoader loader(testDir_.string());
    EXPECT_EQ(loader.size(), 5);
}

TEST_F(ImageLoaderTest, LoadsSingleImage) {
    visual_odometry::ImageLoader loader(testDir_.string());
    cv::Mat img = loader.loadImage(0);
    EXPECT_FALSE(img.empty());
    EXPECT_EQ(img.rows, 100);
    EXPECT_EQ(img.cols, 100);
}

TEST_F(ImageLoaderTest, LoadsImagePair) {
    visual_odometry::ImageLoader loader(testDir_.string());
    auto [img1, img2] = loader.loadImagePair(0);
    EXPECT_FALSE(img1.empty());
    EXPECT_FALSE(img2.empty());
}

TEST_F(ImageLoaderTest, IteratesThroughPairs) {
    visual_odometry::ImageLoader loader(testDir_.string());

    int count = 0;
    while (loader.hasNext()) {
        auto [img1, img2] = loader.nextPair();
        EXPECT_FALSE(img1.empty());
        EXPECT_FALSE(img2.empty());
        count++;
    }
    EXPECT_EQ(count, 4);  // 5 images = 4 pairs
}

TEST_F(ImageLoaderTest, ThrowsOnInvalidDirectory) {
    EXPECT_THROW(
        visual_odometry::ImageLoader("/nonexistent/path"),
        std::runtime_error
    );
}

TEST_F(ImageLoaderTest, ThrowsOnInvalidIndex) {
    visual_odometry::ImageLoader loader(testDir_.string());
    EXPECT_THROW(loader.loadImage(100), std::out_of_range);
}

TEST_F(ImageLoaderTest, ResetsToBeginning) {
    visual_odometry::ImageLoader loader(testDir_.string());

    loader.nextPair();
    loader.nextPair();
    loader.reset();

    EXPECT_TRUE(loader.hasNext());
    auto [img1, img2] = loader.nextPair();
    EXPECT_FALSE(img1.empty());
}
