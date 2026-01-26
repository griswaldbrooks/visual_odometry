#include <filesystem>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <visual_odometry/image_loader.hpp>

namespace fs = std::filesystem;

class image_loader_test : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test directory with sample images
        test_dir_ = fs::temp_directory_path() / "vo_test_images";
        fs::create_directories(test_dir_);

        // Create simple test images
        for (int i = 0; i < 5; ++i) {
            cv::Mat const img(100, 100, CV_8UC1, cv::Scalar(i * 50));
            std::string const filename = test_dir_.string() + "/" +
                                         std::string(6 - std::to_string(i).length(), '0') +
                                         std::to_string(i) + ".png";
            cv::imwrite(filename, img);
        }
    }

    void TearDown() override { fs::remove_all(test_dir_); }

    fs::path test_dir_;
};

TEST_F(image_loader_test, LoadsImagesFromDirectory) {
    // GIVEN a directory with 5 images
    // WHEN creating an image_loader from the directory
    auto const loader_result = visual_odometry::image_loader::create(test_dir_.string());

    // THEN creation should succeed
    ASSERT_TRUE(loader_result.has_value());
    // AND loader should report 5 images
    EXPECT_EQ(loader_result.value().size(), 5);
}

TEST_F(image_loader_test, LoadsSingleImage) {
    // GIVEN a valid image_loader
    auto loader_result = visual_odometry::image_loader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    // WHEN loading a single image by index
    auto const img_result = loader.load_image(0);

    // THEN loading should succeed
    ASSERT_TRUE(img_result.has_value());
    auto const& img = img_result.value();

    // AND image should have expected dimensions
    EXPECT_FALSE(img.empty());
    EXPECT_EQ(img.rows, 100);
    EXPECT_EQ(img.cols, 100);
}

TEST_F(image_loader_test, LoadsImagePair) {
    // GIVEN a valid image_loader
    auto loader_result = visual_odometry::image_loader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    // WHEN loading an image pair
    auto const pair_result = loader.load_image_pair(0);

    // THEN loading should succeed
    ASSERT_TRUE(pair_result.has_value());
    auto const& [img1, img2] = pair_result.value();

    // AND both images should be valid
    EXPECT_FALSE(img1.empty());
    EXPECT_FALSE(img2.empty());
}

TEST_F(image_loader_test, IteratesThroughPairs) {
    // GIVEN a valid image_loader with 5 images
    auto loader_result = visual_odometry::image_loader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    // WHEN iterating through all pairs
    int count = 0;
    while (loader.has_next()) {
        auto const pair_result = loader.next_pair();
        ASSERT_TRUE(pair_result.has_value());
        auto const& [img1, img2] = pair_result.value();

        // THEN each pair should contain valid images
        EXPECT_FALSE(img1.empty());
        EXPECT_FALSE(img2.empty());
        count++;
    }

    // AND should produce 4 pairs (5 images - 1)
    EXPECT_EQ(count, 4);
}

TEST_F(image_loader_test, ReturnsErrorOnInvalidDirectory) {
    // GIVEN a nonexistent directory path
    // WHEN creating an image_loader
    auto const loader_result = visual_odometry::image_loader::create("/nonexistent/path");

    // THEN creation should fail with an error
    ASSERT_FALSE(loader_result.has_value());
    EXPECT_THAT(loader_result.error(), testing::HasSubstr("does not exist"));
}

TEST_F(image_loader_test, ReturnsErrorOnInvalidIndex) {
    // GIVEN a valid image_loader
    auto loader_result = visual_odometry::image_loader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    // WHEN loading an image with an out-of-range index
    auto const img_result = loader.load_image(100);

    // THEN loading should fail with an error
    ASSERT_FALSE(img_result.has_value());
    EXPECT_THAT(img_result.error(), testing::HasSubstr("out of range"));
}

TEST_F(image_loader_test, ResetsToBeginning) {
    // GIVEN a loader that has been partially iterated
    auto loader_result = visual_odometry::image_loader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();
    (void)loader.next_pair();
    (void)loader.next_pair();

    // WHEN resetting the loader
    loader.reset();

    // THEN loader should have pairs available again
    EXPECT_TRUE(loader.has_next());
    auto const pair_result = loader.next_pair();
    ASSERT_TRUE(pair_result.has_value());
    EXPECT_FALSE(pair_result.value().first.empty());
}

TEST_F(image_loader_test, HandlesEmptyDirectory) {
    // GIVEN an empty directory
    auto const empty_dir = fs::temp_directory_path() / "vo_empty_test";
    fs::create_directories(empty_dir);

    // WHEN creating an image_loader
    auto const loader_result = visual_odometry::image_loader::create(empty_dir.string());

    // THEN creation should succeed
    ASSERT_TRUE(loader_result.has_value());
    // AND size should be zero
    EXPECT_EQ(loader_result.value().size(), 0);

    // Cleanup
    fs::remove_all(empty_dir);
}
