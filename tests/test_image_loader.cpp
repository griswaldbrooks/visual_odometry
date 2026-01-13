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
        test_dir_ = fs::temp_directory_path() / "vo_test_images";
        fs::create_directories(test_dir_);

        // Create simple test images
        for (int i = 0; i < 5; ++i) {
            cv::Mat img(100, 100, CV_8UC1, cv::Scalar(i * 50));
            std::string filename = test_dir_.string() + "/" +
                                   std::string(6 - std::to_string(i).length(), '0') +
                                   std::to_string(i) + ".png";
            cv::imwrite(filename, img);
        }
    }

    void TearDown() override {
        fs::remove_all(test_dir_);
    }

    fs::path test_dir_;
};

TEST_F(ImageLoaderTest, LoadsImagesFromDirectory) {
    auto const loader_result = visual_odometry::ImageLoader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    EXPECT_EQ(loader_result.value().size(), 5);
}

TEST_F(ImageLoaderTest, LoadsSingleImage) {
    auto loader_result = visual_odometry::ImageLoader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    auto const img_result = loader.load_image(0);
    ASSERT_TRUE(img_result.has_value());
    auto const& img = img_result.value();

    EXPECT_FALSE(img.empty());
    EXPECT_EQ(img.rows, 100);
    EXPECT_EQ(img.cols, 100);
}

TEST_F(ImageLoaderTest, LoadsImagePair) {
    auto loader_result = visual_odometry::ImageLoader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    auto const pair_result = loader.load_image_pair(0);
    ASSERT_TRUE(pair_result.has_value());
    auto const& [img1, img2] = pair_result.value();

    EXPECT_FALSE(img1.empty());
    EXPECT_FALSE(img2.empty());
}

TEST_F(ImageLoaderTest, IteratesThroughPairs) {
    auto loader_result = visual_odometry::ImageLoader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    int count = 0;
    while (loader.has_next()) {
        auto const pair_result = loader.next_pair();
        ASSERT_TRUE(pair_result.has_value());
        auto const& [img1, img2] = pair_result.value();

        EXPECT_FALSE(img1.empty());
        EXPECT_FALSE(img2.empty());
        count++;
    }
    EXPECT_EQ(count, 4);  // 5 images = 4 pairs
}

TEST_F(ImageLoaderTest, ReturnsErrorOnInvalidDirectory) {
    auto const loader_result = visual_odometry::ImageLoader::create("/nonexistent/path");
    ASSERT_FALSE(loader_result.has_value());
    EXPECT_THAT(loader_result.error(), testing::HasSubstr("does not exist"));
}

TEST_F(ImageLoaderTest, ReturnsErrorOnInvalidIndex) {
    auto loader_result = visual_odometry::ImageLoader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    auto const img_result = loader.load_image(100);
    ASSERT_FALSE(img_result.has_value());
    EXPECT_THAT(img_result.error(), testing::HasSubstr("out of range"));
}

TEST_F(ImageLoaderTest, ResetsToBeginning) {
    auto loader_result = visual_odometry::ImageLoader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    (void)loader.next_pair();
    (void)loader.next_pair();
    loader.reset();

    EXPECT_TRUE(loader.has_next());
    auto const pair_result = loader.next_pair();
    ASSERT_TRUE(pair_result.has_value());
    EXPECT_FALSE(pair_result.value().first.empty());
}

TEST_F(ImageLoaderTest, HandlesEmptyDirectory) {
    auto const empty_dir = fs::temp_directory_path() / "vo_empty_test";
    fs::create_directories(empty_dir);

    auto const loader_result = visual_odometry::ImageLoader::create(empty_dir.string());
    ASSERT_TRUE(loader_result.has_value());
    EXPECT_EQ(loader_result.value().size(), 0);

    fs::remove_all(empty_dir);
}
