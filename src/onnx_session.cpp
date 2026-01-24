#include <visual_odometry/onnx_session.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <vector>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

namespace visual_odometry {

auto onnx_session::get_env() -> Ort::Env& {
    static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "visual_odometry"};
    return env;
}

onnx_session::onnx_session(std::filesystem::path const& model_path)
    : onnx_session(model_path, Ort::SessionOptions{}) {}

onnx_session::onnx_session(std::filesystem::path const& model_path,
                         Ort::SessionOptions const& session_options)
    : session_{get_env(), model_path.c_str(), session_options} {
    // Cache input names
    auto const num_inputs = session_.GetInputCount();
    input_names_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator_);
        input_names_.emplace_back(name.get());
    }

    // Cache output names
    auto const num_outputs = session_.GetOutputCount();
    output_names_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_.GetOutputNameAllocated(i, allocator_);
        output_names_.emplace_back(name.get());
    }
}

auto onnx_session::run(std::span<char const* const> input_names,
                      std::span<Ort::Value> input_tensors,
                      std::span<char const* const> output_names)
    -> std::vector<Ort::Value> {
    if (input_names.size() != input_tensors.size()) {
        throw std::invalid_argument(
            "input_names and input_tensors must have the same size");
    }

    return session_.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_names.size(),
        output_names.data(),
        output_names.size());
}

auto onnx_session::input_shape(size_t index) const -> std::vector<int64_t> {
    if (index >= session_.GetInputCount()) {
        throw std::out_of_range("Input index out of range");
    }
    auto const type_info = session_.GetInputTypeInfo(index);
    auto const tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

auto onnx_session::output_shape(size_t index) const -> std::vector<int64_t> {
    if (index >= session_.GetOutputCount()) {
        throw std::out_of_range("Output index out of range");
    }
    auto const type_info = session_.GetOutputTypeInfo(index);
    auto const tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

auto create_tensor(float const* data, std::span<int64_t const> shape)
    -> Ort::Value {
    // Calculate total size
    int64_t total_size = 1;
    for (auto const dim : shape) {
        total_size *= dim;
    }

    // Use allocator to create tensor and copy data
    Ort::AllocatorWithDefaultOptions const allocator;
    auto tensor = Ort::Value::CreateTensor<float>(
        allocator,
        shape.data(),
        shape.size());

    // Copy data into the allocated tensor
    auto* tensor_data = tensor.GetTensorMutableData<float>();
    std::copy(data, data + total_size, tensor_data);

    return tensor;
}

auto create_tensor(std::vector<float> const& data,
                   std::span<int64_t const> shape) -> Ort::Value {
    return create_tensor(data.data(), shape);
}

}  // namespace visual_odometry
