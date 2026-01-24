#pragma once

#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace visual_odometry {

/**
 * @brief RAII wrapper for ONNX Runtime inference session.
 *
 * Provides a simplified interface for loading ONNX models and running inference.
 * Thread-safe for concurrent inference calls on the same session.
 */
class onnx_session {
public:
    /**
     * @brief Construct an ONNX session from a model file.
     * @param model_path Path to the .onnx model file.
     * @throws Ort::Exception if model loading fails.
     */
    explicit onnx_session(std::filesystem::path const& model_path);

    /**
     * @brief Construct an ONNX session with custom session options.
     * @param model_path Path to the .onnx model file.
     * @param session_options Custom session options (e.g., for GPU execution).
     * @throws Ort::Exception if model loading fails.
     */
    onnx_session(std::filesystem::path const& model_path,
                Ort::SessionOptions const& session_options);

    // Move-only (Ort::Session is not copyable)
    onnx_session(onnx_session const&) = delete;
    auto operator=(onnx_session const&) -> onnx_session& = delete;
    onnx_session(onnx_session&&) noexcept = default;
    auto operator=(onnx_session&&) noexcept -> onnx_session& = default;
    ~onnx_session() = default;

    /**
     * @brief Run inference with the given inputs.
     * @param input_names Names of the input tensors.
     * @param input_tensors Input tensor values (will be consumed/moved).
     * @param output_names Names of the output tensors to retrieve.
     * @return Vector of output tensor values.
     * @throws Ort::Exception if inference fails.
     *
     * Note: This method is not const because Ort::Session::Run is not const,
     * even though it doesn't modify the session state. Thread-safe for
     * concurrent calls.
     */
    [[nodiscard]] auto run(std::span<char const* const> input_names,
                           std::span<Ort::Value> input_tensors,
                           std::span<char const* const> output_names)
        -> std::vector<Ort::Value>;

    /**
     * @brief Get the names of all input tensors.
     */
    [[nodiscard]] auto input_names() const -> std::vector<std::string> const& {
        return input_names_;
    }

    /**
     * @brief Get the names of all output tensors.
     */
    [[nodiscard]] auto output_names() const -> std::vector<std::string> const& {
        return output_names_;
    }

    /**
     * @brief Get the number of inputs.
     */
    [[nodiscard]] auto num_inputs() const noexcept -> size_t {
        return input_names_.size();
    }

    /**
     * @brief Get the number of outputs.
     */
    [[nodiscard]] auto num_outputs() const noexcept -> size_t {
        return output_names_.size();
    }

    /**
     * @brief Get the shape of an input tensor.
     * @param index Input index (0-based).
     * @return Vector of dimension sizes (-1 for dynamic dimensions).
     */
    [[nodiscard]] auto input_shape(size_t index) const -> std::vector<int64_t>;

    /**
     * @brief Get the shape of an output tensor.
     * @param index Output index (0-based).
     * @return Vector of dimension sizes (-1 for dynamic dimensions).
     */
    [[nodiscard]] auto output_shape(size_t index) const -> std::vector<int64_t>;

private:
    // Static environment shared across all sessions
    static auto get_env() -> Ort::Env&;

    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

/**
 * @brief Create a float tensor from raw data.
 * @param data Pointer to float data.
 * @param shape Tensor shape.
 * @return Ort::Value containing the tensor.
 */
[[nodiscard]] auto create_tensor(float const* data,
                                  std::span<int64_t const> shape)
    -> Ort::Value;

/**
 * @brief Create a float tensor from a vector.
 * @param data Vector of float data.
 * @param shape Tensor shape.
 * @return Ort::Value containing the tensor.
 */
[[nodiscard]] auto create_tensor(std::vector<float> const& data,
                                  std::span<int64_t const> shape)
    -> Ort::Value;

}  // namespace visual_odometry
