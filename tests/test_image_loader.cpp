#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

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

TEST_F(image_loader_test, HasNoTimestampsWithoutRgbTxt) {
    // GIVEN a directory without rgb.txt
    auto loader_result = visual_odometry::image_loader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    // THEN has_timestamps should return false
    EXPECT_FALSE(loader.has_timestamps());

    // AND get_timestamp should return 0.0
    EXPECT_EQ(loader.get_timestamp(0), 0.0);
}

TEST_F(image_loader_test, LoadsImageWithTimestampReturnsZeroWithoutRgbTxt) {
    // GIVEN a directory without rgb.txt
    auto loader_result = visual_odometry::image_loader::create(test_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    // WHEN loading image with timestamp
    auto const result = loader.load_image_with_timestamp(0);

    // THEN loading should succeed
    ASSERT_TRUE(result.has_value());

    // AND timestamp should be 0.0
    EXPECT_EQ(result.value().timestamp, 0.0);

    // AND image should be valid
    EXPECT_FALSE(result.value().image.empty());
}

class image_loader_tum_test : public ::testing::Test {
protected:
    void SetUp() override {
        // Create TUM-like directory structure
        // parent/rgb/*.png
        // parent/rgb.txt
        parent_dir_ = fs::temp_directory_path() / "vo_tum_test";
        rgb_dir_ = parent_dir_ / "rgb";
        fs::create_directories(rgb_dir_);

        // Create test images with TUM-style filenames
        timestamps_ = {1305031102.175304, 1305031102.211214, 1305031102.243211};
        for (size_t i = 0; i < timestamps_.size(); ++i) {
            cv::Mat const img(100, 100, CV_8UC1, cv::Scalar(static_cast<int>(i) * 50));
            std::string const filename = std::to_string(timestamps_[i]) + ".png";
            cv::imwrite((rgb_dir_ / filename).string(), img);
        }

        // Create rgb.txt
        std::ofstream rgb_txt(parent_dir_ / "rgb.txt");
        rgb_txt << "# color images\n";
        rgb_txt << "# timestamp filename\n";
        for (auto ts : timestamps_) {
            rgb_txt << std::fixed << std::setprecision(6) << ts << " rgb/" << ts << ".png\n";
        }
        rgb_txt.close();
    }

    void TearDown() override { fs::remove_all(parent_dir_); }

    fs::path parent_dir_;
    fs::path rgb_dir_;
    std::vector<double> timestamps_;
};

TEST_F(image_loader_tum_test, ParsesRgbTxtTimestamps) {
    // GIVEN a TUM-format directory with rgb.txt
    auto loader_result = visual_odometry::image_loader::create(rgb_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    // THEN has_timestamps should return true
    EXPECT_TRUE(loader.has_timestamps());

    // AND timestamps should match
    ASSERT_EQ(loader.size(), timestamps_.size());
    for (size_t i = 0; i < timestamps_.size(); ++i) {
        EXPECT_NEAR(loader.get_timestamp(i), timestamps_[i], 1e-6);
    }
}

TEST_F(image_loader_tum_test, LoadsImageWithTimestamp) {
    // GIVEN a TUM-format directory
    auto loader_result = visual_odometry::image_loader::create(rgb_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    // WHEN loading image with timestamp
    auto const result = loader.load_image_with_timestamp(0);

    // THEN loading should succeed
    ASSERT_TRUE(result.has_value());

    // AND timestamp should match
    EXPECT_NEAR(result.value().timestamp, timestamps_[0], 1e-6);

    // AND image should be valid
    EXPECT_FALSE(result.value().image.empty());
}

TEST_F(image_loader_tum_test, LoadsImagePairWithTimestamps) {
    // GIVEN a TUM-format directory
    auto loader_result = visual_odometry::image_loader::create(rgb_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    // WHEN loading image pair with timestamps
    auto const result = loader.load_image_pair_with_timestamps(0);

    // THEN loading should succeed
    ASSERT_TRUE(result.has_value());

    // AND timestamps should match
    EXPECT_NEAR(result.value().first.timestamp, timestamps_[0], 1e-6);
    EXPECT_NEAR(result.value().second.timestamp, timestamps_[1], 1e-6);

    // AND images should be valid
    EXPECT_FALSE(result.value().first.image.empty());
    EXPECT_FALSE(result.value().second.image.empty());
}

TEST_F(image_loader_tum_test, NextPairWithTimestamps) {
    // GIVEN a TUM-format directory
    auto loader_result = visual_odometry::image_loader::create(rgb_dir_.string());
    ASSERT_TRUE(loader_result.has_value());
    auto& loader = loader_result.value();

    // WHEN iterating with timestamps
    auto const result = loader.next_pair_with_timestamps();

    // THEN loading should succeed
    ASSERT_TRUE(result.has_value());

    // AND timestamps should match first pair
    EXPECT_NEAR(result.value().first.timestamp, timestamps_[0], 1e-6);
    EXPECT_NEAR(result.value().second.timestamp, timestamps_[1], 1e-6);
}
