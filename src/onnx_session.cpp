#include <visual_odometry/onnx_session.hpp>

#include <stdexcept>

namespace visual_odometry {

auto OnnxSession::get_env() -> Ort::Env& {
    static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "visual_odometry"};
    return env;
}

OnnxSession::OnnxSession(std::filesystem::path const& model_path)
    : OnnxSession(model_path, Ort::SessionOptions{}) {}

OnnxSession::OnnxSession(std::filesystem::path const& model_path,
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

auto OnnxSession::run(std::span<char const* const> input_names,
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

auto OnnxSession::input_shape(size_t index) const -> std::vector<int64_t> {
    if (index >= session_.GetInputCount()) {
        throw std::out_of_range("Input index out of range");
    }
    auto const type_info = session_.GetInputTypeInfo(index);
    auto const tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

auto OnnxSession::output_shape(size_t index) const -> std::vector<int64_t> {
    if (index >= session_.GetOutputCount()) {
        throw std::out_of_range("Output index out of range");
    }
    auto const type_info = session_.GetOutputTypeInfo(index);
    auto const tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

auto create_tensor(float const* data, std::span<int64_t const> shape)
    -> Ort::Value {
    auto const memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Calculate total size
    int64_t total_size = 1;
    for (auto const dim : shape) {
        total_size *= dim;
    }

    // Create tensor (const_cast is safe - ONNX Runtime doesn't modify input data)
    return Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(data),
        static_cast<size_t>(total_size),
        shape.data(),
        shape.size());
}

auto create_tensor(std::vector<float> const& data,
                   std::span<int64_t const> shape) -> Ort::Value {
    return create_tensor(data.data(), shape);
}

}  // namespace visual_odometry
