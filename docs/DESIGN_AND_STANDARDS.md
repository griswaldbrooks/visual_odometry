# Visual Odometry Design and Standards

**Last Updated:** 2026-01-25

This document describes the implementation details, coding patterns, and design standards for the visual odometry codebase. For high-level system architecture and algorithmic overview, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Modern C++20 Patterns](#modern-c20-patterns)
3. [Python Integration](#python-integration)
4. [Extension Points](#extension-points)
5. [Build System](#build-system)
6. [Implementation Guidelines](#implementation-guidelines)

---

## Design Principles

### 1. Functions Over Classes

Pure functions are preferred for stateless operations. Classes are used only when:

- State must persist between method calls
- Resource management requires RAII
- Iteration requires mutable state

**Example:**

```cpp
// Stateless operation → Pure function
[[nodiscard]] auto detect_features(
    cv::Mat const& image,
    feature_detector_config const& config = {})
    -> detection_result;

// Stateful resource → RAII class
class onnx_session {
    // Manages ONNX Runtime inference engine lifetime
};
```

### 2. State at System Boundaries

Stateful components belong at system edges, not in core algorithms:

- **I/O boundaries:** `image_loader` (file iteration)
- **Resource boundaries:** `onnx_session` (inference engine)
- **Accumulation boundaries:** `Trajectory` (pose accumulation)

Core algorithms (detection, matching, motion estimation) are pure functions.

**Justification Examples:**

**Trajectory (Pose Accumulation):**

```cpp
class Trajectory {
public:
    Trajectory();

    void add_motion(motion_estimate const& motion);
    [[nodiscard]] auto current_pose() const -> pose const&;
    [[nodiscard]] auto poses() const -> std::vector<pose> const&;
    [[nodiscard]] auto to_json() const -> std::string;

private:
    std::vector<pose> poses_;
};
```

- State persists across calls (accumulated poses)
- Each `add_motion()` depends on previous state
- This IS a system boundary (accumulation edge)

**Image Loader (I/O Boundary):**

```cpp
class image_loader {
public:
    static auto create(std::filesystem::path const& directory)
        -> tl::expected<image_loader, std::string>;

    [[nodiscard]] auto load_image_pair(size_t index) const
        -> tl::expected<std::pair<cv::Mat, cv::Mat>, std::string>;

    [[nodiscard]] auto size() const -> size_t;

private:
    std::vector<std::filesystem::path> image_paths_;
};
```

- Manages I/O state (directory contents)
- Iteration requires mutable position
- System boundary component

**ONNX Session (Resource Management):**

```cpp
class onnx_session {
public:
    static auto create(std::filesystem::path const& model_path)
        -> tl::expected<onnx_session, std::string>;

    // Move-only (non-copyable)
    onnx_session(onnx_session&&) noexcept;
    onnx_session& operator=(onnx_session&&) noexcept;

    auto run(std::vector<Ort::Value> inputs)
        -> tl::expected<std::vector<Ort::Value>, std::string>;

private:
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
};
```

- Manages ONNX Runtime lifetime (RAII)
- Inference engine is stateful
- Move-only ensures single ownership

### 3. Explicit Configuration

Functions accept configuration structs rather than maintaining mutable state:

```cpp
struct feature_detector_config {
    int max_features = 2000;
};

// Config passed explicitly, no hidden state
auto result = detect_features(image, config);
```

### 4. Error Handling via Expected

Use `tl::expected<T, std::string>` for operations that can fail:

```cpp
// Example: image_loader::create returns tl::expected
auto loader = image_loader::create(directory);
if (!loader) {
    std::cerr << "Error: " << loader.error() << "\n";
    return;
}

// Example: motion estimation returns struct with valid flag
auto motion = estimate_motion(points1, points2, intrinsics);
if (motion.valid) {
    trajectory.add_motion(motion);
} else {
    std::cerr << "Motion estimation failed\n";
}
```

**Benefits:**

- Functions that can fail return `tl::expected` (e.g., I/O operations)
- Pure computation functions return structs with validity flags
- Explicit error handling at call sites
- Automatic conversion to Python exceptions in bindings (for `tl::expected`)
- Monadic operations (map, and_then, or_else) for `tl::expected`

### 5. Zero-Cost Abstractions

Use modern C++ features to achieve abstraction without runtime overhead:

- Concepts for compile-time interface checking
- `std::variant` for polymorphism without vtables
- Templates for generic code
- Move semantics to avoid copies

---

## Modern C++20 Patterns

### 1. Concepts for Structural Typing

Instead of inheritance hierarchies, use concepts to define interfaces:

```cpp
template <typename T>
concept matcher_like = requires(T const& m, cv::Mat const& img) {
    { m.match_images(img, img) } -> std::same_as<match_result>;
    { m.name() } -> std::convertible_to<std::string_view>;
};

// Any type satisfying the concept can be used
static_assert(matcher_like<orb_matcher>);
static_assert(matcher_like<lightglue_matcher>);
```

**Benefits:**

- Duck typing in C++ (structural, not nominal)
- Better compiler error messages than SFINAE
- Compile-time verification
- No runtime overhead

### 2. std::variant for Type-Safe Unions

Instead of virtual inheritance, use `std::variant` for polymorphism:

```cpp
using image_matcher = std::variant<orb_matcher, lightglue_matcher>;

// Pattern matching with std::visit
auto name = std::visit(
    [](auto const& m) { return m.name(); },
    matcher
);
```

**Benefits:**

- Zero runtime overhead (no vtable lookup)
- Value semantics (no heap allocation required)
- Exhaustive matching (compiler error if case missing)
- Better cache locality

**Trade-offs:**

- Closed set of types (can't add new types at runtime)
- All types in variant must be known at compile time
- Perfect for our use case (fixed set of matchers)

**Full Implementation Example:**

```cpp
// Header: include/visual_odometry/image_matcher.hpp
struct orb_matcher {
    [[nodiscard]] auto match_images(
        cv::Mat const& img1,
        cv::Mat const& img2) const -> match_result;

    [[nodiscard]] auto name() const -> std::string_view {
        return "ORB";
    }
};

struct lightglue_matcher {
    explicit lightglue_matcher(std::filesystem::path const& model_path);

    // Move-only (contains onnx_session)
    lightglue_matcher(lightglue_matcher&&) noexcept = default;

    [[nodiscard]] auto match_images(
        cv::Mat const& img1,
        cv::Mat const& img2) const -> match_result;

    [[nodiscard]] auto name() const -> std::string_view {
        return "LightGlue";
    }

private:
    std::unique_ptr<onnx_session> session_;
};

// Zero-overhead polymorphism
using image_matcher = std::variant<orb_matcher, lightglue_matcher>;
```

**Usage:**

```cpp
// Create matcher
auto matcher = create_image_matcher("lightglue", model_path);

// Use via std::visit
auto result = std::visit(
    [&](auto const& m) { return m.match_images(img1, img2); },
    matcher
);
```

### 3. std::span for Non-Owning Views

Use `std::span` instead of raw pointers or vector references:

```cpp
auto match_features(
    cv::Mat const& desc1,
    cv::Mat const& desc2,
    std::span<cv::KeyPoint const> kp1,  // Non-owning view
    std::span<cv::KeyPoint const> kp2,
    feature_matcher_config const& config = {})
    -> match_result;
```

**Benefits:**

- Safety: size is always known
- Works with arrays, vectors, or any contiguous container
- Zero overhead (pointer + size)
- Communicates intent (non-owning)

### 4. [[nodiscard]] for Error Safety

Mark all functions returning values with `[[nodiscard]]`:

```cpp
[[nodiscard]] auto detect_features(
    cv::Mat const& image,
    feature_detector_config const& config = {})
    -> detection_result;
```

**Benefits:**

- Compiler warning if return value ignored
- Prevents accidental discard of `tl::expected` errors
- Documents that function has no side effects

### 5. Move Semantics for Resource Management

Use move-only types for unique resources:

```cpp
class onnx_session {
public:
    // Deleted copy operations
    onnx_session(onnx_session const&) = delete;
    onnx_session& operator=(onnx_session const&) = delete;

    // Defaulted move operations
    onnx_session(onnx_session&&) noexcept = default;
    onnx_session& operator=(onnx_session&&) noexcept = default;
};
```

**Benefits:**

- Enforces single ownership
- Prevents accidental copies of heavy resources
- Clear transfer of ownership semantics

---

## Python Integration

### Nanobind Bindings

**File:** `src/python_bindings.cpp` (321 lines)

The codebase provides comprehensive Python bindings via nanobind, offering:

- Zero-overhead interop (no copies for simple types)
- Automatic error conversion (`tl::expected` → Python exceptions)
- Full numpy integration via cvnp_nano
- Iterator protocol support
- Type hints in docstrings

### Exposed API

**Core Types:**

```python
# Data structures
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def load_from_yaml(path: str) -> CameraIntrinsics: ...

class MatchResult:
    points1: np.ndarray  # Nx2 float32
    points2: np.ndarray  # Nx2 float32
    matches: list[cv2.DMatch]

class MotionEstimate:
    rotation: np.ndarray     # 3x3 float64
    translation: np.ndarray  # 3x1 float64
    inliers: int
    valid: bool

class Pose:
    rotation: np.ndarray     # 3x3
    translation: np.ndarray  # 3x1
```

**Matchers:**

```python
class orb_matcher:
    def __init__(self): ...
    def match_images(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult: ...
    def name(self) -> str: ...

class lightglue_matcher:
    def __init__(self, model_path: str): ...
    def match_images(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult: ...
    def name(self) -> str: ...
```

**Processing:**

```python
class image_loader:
    @staticmethod
    def create(directory: str) -> image_loader: ...

    def load_image_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]: ...
    def size(self) -> int: ...

    # Iterator protocol
    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:  # Python typing hint ...

# Module-level function (NOT a class)
def estimate_motion(
    points1: np.ndarray,
    points2: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: MotionEstimatorConfig = MotionEstimatorConfig()
) -> MotionEstimate: ...

class Trajectory:
    def __init__(self): ...
    def add_motion(self, motion: MotionEstimate) -> None: ...
    def current_pose(self) -> Pose: ...
    def poses(self) -> list[Pose]: ...
    def to_json(self) -> str: ...
```

### Example Python Usage

**Basic VO Pipeline:**

```python
import visual_odometry as vo
import numpy as np

# Setup
intrinsics = vo.CameraIntrinsics.load_from_yaml("camera.yaml")
loader = vo.image_loader.create("images/")
matcher = vo.lightglue_matcher("models/lightglue.onnx")
trajectory = vo.Trajectory()

# Process
for img1, img2 in loader:
    matches = matcher.match_images(img1, img2)
    motion = vo.estimate_motion(matches.points1, matches.points2, intrinsics)

    if motion.valid:
        trajectory.add_motion(motion)
        print(f"Inliers: {motion.inliers}")

# Export
with open("trajectory.json", "w") as f:
    f.write(trajectory.to_json())
```

**Real-Time Visualization (Proposed):**

```python
import visual_odometry as vo
import viser

# Setup
server = viser.ViserServer()
trajectory = vo.Trajectory()
intrinsics = vo.CameraIntrinsics.load_from_yaml("camera.yaml")
# ... (setup loader, matcher)

# Process with live updates
for img1, img2 in loader:
    matches = matcher.match_images(img1, img2)
    motion = vo.estimate_motion(matches.points1, matches.points2, intrinsics)

    if motion.valid:
        trajectory.add_motion(motion)

        # Update visualization
        pose = trajectory.current_pose()
        server.scene.add_camera_frustum(
            name=f"camera_{len(trajectory.poses())}",
            fov=60,
            aspect=1.33,
            wxyz=rotation_to_quaternion(pose.rotation),
            position=pose.translation,
        )
```

### cv::Mat ↔ NumPy Conversion

**Automatic via cvnp_nano:**

- C++ `cv::Mat` automatically becomes `np.ndarray` in Python
- Python `np.ndarray` automatically becomes `cv::Mat` in C++
- Zero-copy for compatible types
- Proper memory management (reference counting)

**Supported conversions:**

- Grayscale images (CV_8UC1) ↔ uint8 array
- Color images (CV_8UC3) ↔ uint8 array (H, W, 3)
- Float images (CV_32FC1) ↔ float32 array
- Descriptors (CV_8UC1) ↔ uint8 array (N, descriptor_size)

### Error Handling

**C++ `tl::expected` → Python exceptions:**

```cpp
// C++ code - for functions returning tl::expected
auto loader = image_loader::create(directory);
if (!loader) {
    // In Python bindings, this becomes an exception
    throw std::runtime_error(loader.error());
}
```

```python
# Python code - tl::expected errors become exceptions
try:
    loader = vo.image_loader.create("nonexistent/")
except RuntimeError as e:
    print(f"Failed to create loader: {e}")

# Note: estimate_motion returns motion_estimate, not tl::expected
# Check the 'valid' field instead
motion = vo.estimate_motion(points1, points2, intrinsics)
if not motion.valid:
    print("Motion estimation failed")
```

---

## Extension Points

### Adding New Matchers

To add a new matcher (e.g., SuperGlue, LoFTR):

**1. Create matcher struct satisfying `matcher_like` concept:**

```cpp
// include/visual_odometry/image_matcher.hpp
struct superglue_matcher {
    explicit superglue_matcher(std::filesystem::path const& model_path);

    [[nodiscard]] auto match_images(
        cv::Mat const& img1,
        cv::Mat const& img2) const -> match_result;

    [[nodiscard]] auto name() const -> std::string_view {
        return "superglue";
    }

private:
    std::unique_ptr<onnx_session> session_;
};
```

**2. Add to variant:**

```cpp
using image_matcher = std::variant<
    orb_matcher,
    lightglue_matcher,
    superglue_matcher  // Added
>;
```

**3. Update factory function:**

```cpp
// src/image_matcher.cpp
auto create_image_matcher(
    std::string const& name,
    std::optional<std::filesystem::path> const& model_path)
    -> tl::expected<image_matcher, std::string>
{
    if (name == "orb") {
        return orb_matcher{};
    } else if (name == "lightglue") {
        // ...
    } else if (name == "superglue") {
        if (!model_path) {
            return tl::unexpected("SuperGlue requires model path");
        }
        return superglue_matcher{*model_path};
    }
    // ...
}
```

**4. Implement preprocessing/postprocessing:**

Most learned matchers need custom preprocessing:

- Resize to expected dimensions
- Normalize pixel values
- Convert to RGB/grayscale as needed
- Create ONNX input tensors

**Example pattern:**

```cpp
auto superglue_matcher::match_images(
    cv::Mat const& img1,
    cv::Mat const& img2) const -> match_result
{
    // 1. Preprocess
    auto tensor1 = preprocess_for_superglue(img1);
    auto tensor2 = preprocess_for_superglue(img2);

    // 2. Run inference
    auto outputs = session_->run({tensor1, tensor2});
    if (!outputs) {
        return match_result{};  // Empty result on error
    }

    // 3. Postprocess
    return parse_superglue_output(*outputs);
}
```

**5. Add Python bindings:**

```cpp
// src/python_bindings.cpp
nb::class_<superglue_matcher>(m, "SuperglueMatcher")
    .def(nb::init<std::string const&>(), nb::arg("model_path"))
    .def("match_images", &superglue_matcher::match_images)
    .def("name", &superglue_matcher::name);
```

### Adding Configuration Options

To add new configuration parameters:

**1. Update config struct:**

```cpp
struct motion_estimator_config {
    double ransac_threshold = 1.0;
    double ransac_probability = 0.999;
    int min_inliers = 10;  // New parameter
};
```

**2. Use in function:**

```cpp
auto estimate_motion(..., motion_estimator_config const& config)
    -> motion_estimate
{
    // ...
    if (result.inliers < config.min_inliers) {
        result.valid = false;
    }
    return result;
}
```

**3. Update Python bindings:**

```cpp
nb::class_<motion_estimator_config>(m, "MotionEstimatorConfig")
    .def(nb::init<>())
    .def_rw("ransac_threshold", &motion_estimator_config::ransac_threshold)
    .def_rw("ransac_probability", &motion_estimator_config::ransac_probability)
    .def_rw("min_inliers", &motion_estimator_config::min_inliers);  // Expose
```

### Adding New Output Formats

Currently supports TUM RGB-D JSON. To add other formats:

**1. Add method to Trajectory:**

```cpp
class Trajectory {
public:
    [[nodiscard]] auto to_json() const -> std::string;
    [[nodiscard]] auto to_kitti() const -> std::string;  // New
    [[nodiscard]] auto to_euroc() const -> std::string;  // New
};
```

**2. Implement serialization:**

```cpp
auto Trajectory::to_kitti() const -> std::string {
    std::ostringstream oss;
    for (auto const& p : poses_) {
        // KITTI format: 12 values per line (3x4 matrix)
        Eigen::Matrix3d const& R = p.rotation;
        Eigen::Vector3d const& t = p.translation;

        oss << R(0,0) << " " << R(0,1) << " " << R(0,2) << " " << t(0) << " "
            << R(1,0) << " " << R(1,1) << " " << R(1,2) << " " << t(1) << " "
            << R(2,0) << " " << R(2,1) << " " << R(2,2) << " " << t(2) << "\n";
    }
    return oss.str();
}
```

**3. Expose in Python:**

```cpp
nb::class_<Trajectory>(m, "Trajectory")
    // ...
    .def("to_json", &Trajectory::to_json)
    .def("to_kitti", &Trajectory::to_kitti)
    .def("to_euroc", &Trajectory::to_euroc);
```

---

## Build System

### CMake Configuration

**Minimum requirements:**

- CMake 3.25+
- C++20 compiler (GCC 11+, Clang 14+, MSVC 19.29+)

**Required dependencies:**

- OpenCV 4.x
- Eigen3
- yaml-cpp

**Optional dependencies:**

- ONNX Runtime (for LightGlue, auto-detected from `CONDA_PREFIX` environment)
- Python 3.8+ (for bindings, controlled by `ENABLE_PYTHON_BINDINGS`)
- nanobind (fetched automatically if Python enabled)
- GTest (for tests)

### Build Options

```bash
# Basic build (ORB only)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# With ONNX Runtime (LightGlue support)
# ONNX Runtime is auto-detected from CONDA_PREFIX environment variable
# Install via: conda install onnxruntime or pixi add onnxruntime
cmake -B build -DCMAKE_BUILD_TYPE=Release

# With Python bindings
cmake -B build -DENABLE_PYTHON_BINDINGS=ON

# Development build (tests, sanitizers)
cmake -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_TESTING=ON \
  -DENABLE_SANITIZERS=ON

# Full build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PYTHON_BINDINGS=ON \
  -DENABLE_TESTING=ON
```

### Conditional Compilation

Code uses preprocessor guards and CMake feature detection for optional features:

- **ONNX Runtime**: Auto-detected from `CONDA_PREFIX` environment variable. If found, `ONNXRUNTIME_FOUND` is set and LightGlue matcher is available.
- **Python bindings**: Controlled by `ENABLE_PYTHON_BINDINGS` CMake option. Requires Python 3.8+.
- **Testing**: Controlled by `ENABLE_TESTING` CMake option. Requires GTest.
- **Sanitizers**: Controlled by `ENABLE_SANITIZERS` CMake option (Debug builds only).

This allows building minimal configurations without heavy dependencies. ONNX Runtime is automatically included if available in a conda/pixi environment.

---

## Implementation Guidelines

### Memory Ownership

**Principle:** Functions return by value, classes manage owned resources.

**Examples:**

```cpp
// Functions return by value (move semantics avoid copies)
auto detection_result detect_features(...) -> detection_result;
auto match_result match_features(...) -> match_result;

// Classes own their resources
class Trajectory {
    std::vector<pose> poses_;  // Owned
};

class onnx_session {
    std::unique_ptr<Ort::Session> session_;  // Owned
};

// Spans are non-owning views
auto f(std::span<cv::Point2f const> points);  // Borrows, doesn't own
```

### Type System Conventions

**Important:** This codebase uses snake_case for ALL identifiers, including types:

```cpp
// Types use snake_case (not PascalCase)
struct motion_estimate { ... };
struct feature_detector_config { ... };
struct detection_result { ... };
class image_loader { ... };
struct orb_matcher { ... };

// Functions and variables also use snake_case
auto detect_features(...) -> detection_result;
auto const match_result = matcher.match_images(...);
```

### RAII Patterns

**ONNX Runtime Integration:**

```text
┌─────────────────────────────────────────┐
│ lightglue_matcher                       │
│  ├─ Contains: unique_ptr<onnx_session>  │
│  └─ Lifetime: Entire VO run             │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ onnx_session (RAII wrapper)             │
│  ├─ Loads model once in constructor     │
│  ├─ Provides run() for inference        │
│  └─ Destroys in destructor              │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ ONNX Runtime (Ort::Session)             │
│  ├─ Model loaded in memory              │
│  ├─ Optimized computational graph       │
│  └─ CPU/GPU execution providers         │
└─────────────────────────────────────────┘
```

**Workflow:**

1. **Initialization:** `lightglue_matcher` constructor loads ONNX model (one-time cost ~100-500ms)
2. **Inference:** `match_images()` calls `onnx_session.run()` (per-frame cost ~20-50ms)
3. **Cleanup:** Automatic when matcher goes out of scope

**Benefits:**

- Model loaded once, reused for all frames
- No subprocess overhead
- No file I/O per frame
- Direct memory access
- Deterministic performance

### Resource Management Best Practices

**Smart Pointers:**

- Use `std::unique_ptr` for exclusive ownership
- Use `std::shared_ptr` for shared ownership (rarely needed)
- Never use raw owning pointers

**Move Semantics:**

- Return by value and rely on move semantics
- Use move-only types for unique resources
- Default move operations when appropriate

**RAII Wrappers:**

- Wrap external resources (ONNX Runtime, file handles, etc.)
- Ensure cleanup in destructor
- Use factory functions returning `tl::expected` for fallible construction
