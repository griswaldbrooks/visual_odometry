#include <gtest/gtest.h>
#include <visual_odometry/onnx_session.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace {

// Try multiple paths to find the model (handles different working directories)
auto find_model_path() -> std::filesystem::path {
    std::vector<std::filesystem::path> const candidates = {
        "models/disk_lightglue_end2end.onnx",
        "../models/disk_lightglue_end2end.onnx",
        "../../models/disk_lightglue_end2end.onnx",
        std::filesystem::path{__FILE__}.parent_path().parent_path() /
            "models/disk_lightglue_end2end.onnx",
    };
    for (auto const& path : candidates) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return "models/disk_lightglue_end2end.onnx";  // Return default (will fail)
}

auto const k_test_model_path = find_model_path();

auto model_exists() -> bool {
    return std::filesystem::exists(k_test_model_path);
}

}  // namespace

class onnx_session_test : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_exists()) {
            GTEST_SKIP() << "Test model not found: " << k_test_model_path;
        }
    }
};

TEST_F(onnx_session_test, LoadsModelSuccessfully) {
    // GIVEN a valid ONNX model path
    // WHEN creating an onnx_session
    visual_odometry::onnx_session const session{k_test_model_path};

    // THEN the session should have inputs and outputs
    EXPECT_GT(session.num_inputs(), 0);
    EXPECT_GT(session.num_outputs(), 0);
}

TEST_F(onnx_session_test, ReturnsInputNames) {
    // GIVEN a loaded ONNX session
    visual_odometry::onnx_session const session{k_test_model_path};

    // WHEN getting input names
    auto const& names = session.input_names();

    // THEN names should be non-empty strings
    EXPECT_EQ(names.size(), session.num_inputs());
    for (auto const& name : names) {
        EXPECT_FALSE(name.empty());
    }
}

TEST_F(onnx_session_test, ReturnsOutputNames) {
    // GIVEN a loaded ONNX session
    visual_odometry::onnx_session const session{k_test_model_path};

    // WHEN getting output names
    auto const& names = session.output_names();

    // THEN names should be non-empty strings
    EXPECT_EQ(names.size(), session.num_outputs());
    for (auto const& name : names) {
        EXPECT_FALSE(name.empty());
    }
}

TEST_F(onnx_session_test, ReturnsInputShape) {
    // GIVEN a loaded ONNX session
    visual_odometry::onnx_session const session{k_test_model_path};

    // WHEN getting input shape for first input
    auto const shape = session.input_shape(0);

    // THEN shape should have dimensions
    EXPECT_GT(shape.size(), 0);

    // Print model info for debugging
    std::cout << "\n=== Model Info ===\n";
    std::cout << "Inputs (" << session.num_inputs() << "):\n";
    for (size_t i = 0; i < session.num_inputs(); ++i) {
        std::cout << "  " << session.input_names()[i] << ": [";
        auto const s = session.input_shape(i);
        for (size_t j = 0; j < s.size(); ++j) {
            if (j > 0) {
                std::cout << ", ";
            }
            std::cout << s[j];
        }
        std::cout << "]\n";
    }
    std::cout << "Outputs (" << session.num_outputs() << "):\n";
    for (size_t i = 0; i < session.num_outputs(); ++i) {
        std::cout << "  " << session.output_names()[i] << ": [";
        auto const s = session.output_shape(i);
        for (size_t j = 0; j < s.size(); ++j) {
            if (j > 0) {
                std::cout << ", ";
            }
            std::cout << s[j];
        }
        std::cout << "]\n";
    }
    std::cout << "==================\n\n";
}

TEST_F(onnx_session_test, ThrowsOnInvalidInputIndex) {
    // GIVEN a loaded ONNX session
    visual_odometry::onnx_session const session{k_test_model_path};

    // WHEN requesting shape for invalid index
    // THEN should throw out_of_range
    EXPECT_THROW(session.input_shape(999), std::out_of_range);
}

TEST(onnx_sessionErrorTest, ThrowsOnMissingFile) {
    // GIVEN a non-existent model path
    auto const bad_path = std::filesystem::path{"nonexistent_model.onnx"};

    // WHEN creating an onnx_session
    // THEN should throw Ort::Exception
    EXPECT_THROW(visual_odometry::onnx_session{bad_path}, Ort::Exception);
}

TEST(CreateTensorTest, CreatesTensorFromData) {
    // GIVEN float data and shape
    std::vector<float> const data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> const shape{2, 3};

    // WHEN creating a tensor
    auto tensor = visual_odometry::create_tensor(data, shape);

    // THEN tensor should be valid
    EXPECT_TRUE(tensor.IsTensor());
    auto const info = tensor.GetTensorTypeAndShapeInfo();
    EXPECT_EQ(info.GetShape(), shape);
    EXPECT_EQ(info.GetElementCount(), 6);
}
