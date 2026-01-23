#include <gtest/gtest.h>
#include <visual_odometry/onnx_session.hpp>

#include <filesystem>

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

auto const kTestModelPath = find_model_path();

auto model_exists() -> bool {
    return std::filesystem::exists(kTestModelPath);
}

}  // namespace

class OnnxSessionTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_exists()) {
            GTEST_SKIP() << "Test model not found: " << kTestModelPath;
        }
    }
};

TEST_F(OnnxSessionTest, LoadsModelSuccessfully) {
    // GIVEN a valid ONNX model path
    // WHEN creating an OnnxSession
    visual_odometry::OnnxSession session{kTestModelPath};

    // THEN the session should have inputs and outputs
    EXPECT_GT(session.num_inputs(), 0);
    EXPECT_GT(session.num_outputs(), 0);
}

TEST_F(OnnxSessionTest, ReturnsInputNames) {
    // GIVEN a loaded ONNX session
    visual_odometry::OnnxSession session{kTestModelPath};

    // WHEN getting input names
    auto const& names = session.input_names();

    // THEN names should be non-empty strings
    EXPECT_EQ(names.size(), session.num_inputs());
    for (auto const& name : names) {
        EXPECT_FALSE(name.empty());
    }
}

TEST_F(OnnxSessionTest, ReturnsOutputNames) {
    // GIVEN a loaded ONNX session
    visual_odometry::OnnxSession session{kTestModelPath};

    // WHEN getting output names
    auto const& names = session.output_names();

    // THEN names should be non-empty strings
    EXPECT_EQ(names.size(), session.num_outputs());
    for (auto const& name : names) {
        EXPECT_FALSE(name.empty());
    }
}

TEST_F(OnnxSessionTest, ReturnsInputShape) {
    // GIVEN a loaded ONNX session
    visual_odometry::OnnxSession session{kTestModelPath};

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
            if (j > 0) std::cout << ", ";
            std::cout << s[j];
        }
        std::cout << "]\n";
    }
    std::cout << "Outputs (" << session.num_outputs() << "):\n";
    for (size_t i = 0; i < session.num_outputs(); ++i) {
        std::cout << "  " << session.output_names()[i] << ": [";
        auto const s = session.output_shape(i);
        for (size_t j = 0; j < s.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << s[j];
        }
        std::cout << "]\n";
    }
    std::cout << "==================\n\n";
}

TEST_F(OnnxSessionTest, ThrowsOnInvalidInputIndex) {
    // GIVEN a loaded ONNX session
    visual_odometry::OnnxSession session{kTestModelPath};

    // WHEN requesting shape for invalid index
    // THEN should throw out_of_range
    EXPECT_THROW(session.input_shape(999), std::out_of_range);
}

TEST(OnnxSessionErrorTest, ThrowsOnMissingFile) {
    // GIVEN a non-existent model path
    auto const bad_path = std::filesystem::path{"nonexistent_model.onnx"};

    // WHEN creating an OnnxSession
    // THEN should throw Ort::Exception
    EXPECT_THROW(visual_odometry::OnnxSession{bad_path}, Ort::Exception);
}

TEST(CreateTensorTest, CreatesTensorFromData) {
    // GIVEN float data and shape
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> shape{2, 3};

    // WHEN creating a tensor
    auto tensor = visual_odometry::create_tensor(data, shape);

    // THEN tensor should be valid
    EXPECT_TRUE(tensor.IsTensor());
    auto const info = tensor.GetTensorTypeAndShapeInfo();
    EXPECT_EQ(info.GetShape(), shape);
    EXPECT_EQ(info.GetElementCount(), 6);
}
