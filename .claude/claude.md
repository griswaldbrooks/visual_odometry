# Visual Odometry Project Standards

## Agent Delegation Policy

**IMPORTANT**: Delegate ALL tasks to the specialized agents in `.claude/agents/`:
- **project-manager** - High-level coordination and orchestration of complex tasks
- **coder** - Implementation and code writing tasks
- **code-reviewer** - Code quality reviews before committing
- **test-runner** - Running and validating tests

Always use the Task tool with the appropriate `subagent_type` to delegate work to these agents.

---

This document defines the C++ coding standards and best practices for this project. It follows the "Jason Turner style" of modern C++20 with emphasis on const-correctness, compile-time computation, and type safety.

## Quick Reference

```bash
# Run all checks
pre-commit run -a

# Build and test
pixi run build && pixi run test

# Format check only
pixi run format-check

# Static analysis
pixi run tidy
```

## Naming Conventions

| Element | Style | Example |
|---------|-------|---------|
| Functions | snake_case | `calculate_iou()`, `estimate_motion()` |
| Variables | snake_case | `match_result`, `num_inliers` |
| Parameters | snake_case | `input_image`, `threshold` |
| Member variables | snake_case_ | `ratio_threshold_`, `matcher_` |
| Types (struct/class) | PascalCase | `MotionEstimate`, `FeatureMatcher` |
| Enums | PascalCase | `MatchType`, `Status` |
| Enum values | PascalCase | `Success`, `InvalidInput` |
| Template params | PascalCase | `typename ImageType` |
| Namespaces | snake_case | `visual_odometry`, `detail` |
| Constants | snake_case | `default_threshold`, `max_iterations` |

## East Const (Mandatory)

**All `const` qualifiers go on the RIGHT side:**

```cpp
// CORRECT - East Const
int const value = 42;
std::string const& name = get_name();
auto const& points = result.points;
cv::Mat const* const ptr = &image;  // const pointer to const Mat

for (auto const& item : collection) { ... }

auto process(cv::Mat const& input) -> Result;

// WRONG - West Const (never use)
const int value = 42;
const std::string& name = get_name();
```

**Rationale:** East const reads left-to-right consistently and mirrors pointer const placement.

## Core C++20 Patterns

### Attributes (Always Use)

```cpp
// [[nodiscard]] on ALL value-returning functions
[[nodiscard]] auto calculate_iou(cv::Mat const& m1, cv::Mat const& m2) -> float;

// noexcept on functions that cannot throw
[[nodiscard]] auto size() const noexcept -> std::size_t;

// [[maybe_unused]] for intentionally unused parameters
void callback([[maybe_unused]] int event_id) { ... }

// [[likely]] / [[unlikely]] for branch hints (when profiling shows benefit)
if (result.has_value()) [[likely]] {
    return result.value();
}
```

### Trailing Return Types (Preferred)

```cpp
// Prefer trailing return type syntax
[[nodiscard]] auto estimate_motion(std::vector<cv::Point2f> const& pts1,
                                    std::vector<cv::Point2f> const& pts2) const
    -> MotionEstimate;

// Required for auto deduction
[[nodiscard]] auto get_matches() const -> auto const& { return matches_; }

// Essential for templates
template<typename T>
[[nodiscard]] auto process(T const& input) -> typename T::ResultType;
```

### Auto with Const

```cpp
// Always prefer auto with east const for local variables
auto const result = calculate_iou(mask1, mask2);
auto const& points = result.points;
auto const* ptr = get_pointer();

// Loop iterators
for (auto i = 0u; i < vec.size(); ++i) { ... }
for (auto const& [key, value] : map) { ... }
```

### constexpr Everything Possible

```cpp
// constexpr for compile-time computation
constexpr auto default_threshold = 0.75f;
constexpr auto max_features = 2000;

// constexpr functions
[[nodiscard]] constexpr auto square(int x) noexcept -> int {
    return x * x;
}

// consteval for compile-time ONLY (C++20)
[[nodiscard]] consteval auto compile_time_hash(std::string_view str) -> std::size_t {
    // Must be evaluated at compile time
    return str.size();  // simplified example
}

// if constexpr for compile-time branching
template<typename T>
auto process(T const& value) {
    if constexpr (std::is_integral_v<T>) {
        return value * 2;
    } else {
        return value;
    }
}
```

## Error Handling with tl::expected

**Use `tl::expected<T, E>` for all fallible operations:**

```cpp
#include <tl/expected.hpp>

// Return expected for operations that can fail
[[nodiscard]] auto load_image(std::filesystem::path const& path)
    -> tl::expected<cv::Mat, std::string>
{
    auto const image = cv::imread(path.string());
    if (image.empty()) {
        return tl::unexpected("Failed to load image: " + path.string());
    }
    return image;
}

// Chain operations with and_then / transform
auto result = load_image(path)
    .and_then([](cv::Mat const& img) { return detect_features(img); })
    .transform([](auto const& features) { return features.size(); });

// Handle errors
if (!result.has_value()) {
    log_error(result.error());
    return;
}
auto const count = result.value();

// Or use value_or for defaults
auto const count = result.value_or(0);
```

### Error Message Guidelines

```cpp
// Error messages should be:
// 1. Actionable - tell what went wrong
// 2. Contextual - include relevant values
// 3. Consistent - use similar patterns

return tl::unexpected("Cannot estimate motion: insufficient matches ("
    + std::to_string(matches.size()) + " < 5 required)");

return tl::unexpected("Failed to load camera intrinsics from: " + path.string());
```

## Non-Owning Views (Prefer Over References)

### std::string_view for Strings

```cpp
// PREFER: string_view for non-owning string parameters
[[nodiscard]] auto parse_config(std::string_view config) -> Config;

// Use string_view literals
using namespace std::string_view_literals;
constexpr auto default_name = "unknown"sv;

// AVOID: const string& when not storing
// [[nodiscard]] auto parse_config(std::string const& config) -> Config;
```

### std::span for Arrays

```cpp
// PREFER: span for non-owning array views
[[nodiscard]] auto process_points(std::span<cv::Point2f const> points) -> Result;

// Can accept vector, array, or raw array
std::vector<cv::Point2f> vec = ...;
process_points(vec);

std::array<cv::Point2f, 10> arr = ...;
process_points(arr);
```

## RAII and Resource Management

### Smart Pointers

```cpp
// unique_ptr for exclusive ownership
std::unique_ptr<Model> model_;

// shared_ptr for shared ownership
std::shared_ptr<InferenceHandle> handle_;

// Never use raw owning pointers
// BAD: Model* model_;
```

### Rule of Zero

```cpp
// Prefer rule of zero - let compiler generate special members
struct FeatureMatcher {
    // No need to declare destructor, copy/move constructors
    // if members handle their own resources

private:
    cv::Ptr<cv::BFMatcher> matcher_;  // cv::Ptr is a smart pointer
    float ratio_threshold_;
};
```

### Explicit Default/Delete

```cpp
// Be explicit about special members when needed
struct NonCopyable {
    NonCopyable() = default;
    ~NonCopyable() = default;

    NonCopyable(NonCopyable const&) = delete;
    auto operator=(NonCopyable const&) -> NonCopyable& = delete;

    NonCopyable(NonCopyable&&) = default;
    auto operator=(NonCopyable&&) -> NonCopyable& = default;
};
```

## Class Design

### Struct vs Class

```cpp
// struct for aggregate/POD types (public by default)
struct Point {
    double x{0.0};
    double y{0.0};
};

// struct with private section for encapsulated types
struct FeatureMatcher {
public:
    explicit FeatureMatcher(float ratio_threshold = 0.75f);

    [[nodiscard]] auto match(...) const -> MatchResult;

private:
    cv::Ptr<cv::BFMatcher> matcher_;
    float ratio_threshold_;
};

// Use final for classes not designed for inheritance
struct MotionEstimator final {
    // ...
};
```

### Member Initialization

```cpp
// Always initialize all members (prefer in-class initializers)
struct Config {
    int max_features{2000};
    float threshold{0.75f};
    bool verbose{false};
    std::string name{"default"};
};

// Use constructor initializer list for complex initialization
struct Processor {
    explicit Processor(Config const& config)
        : max_features_{config.max_features}
        , threshold_{config.threshold}
    {}

private:
    int max_features_;
    float threshold_;
};
```

## Modern Algorithms and Ranges

### Prefer Algorithms Over Raw Loops

```cpp
// PREFER: std::ranges algorithms
auto const filtered = points
    | std::views::filter([](auto const& p) { return p.score > 0.5f; })
    | std::views::transform([](auto const& p) { return p.position; });

// PREFER: std::transform over manual loop
std::transform(masks.begin(), masks.end(), std::back_inserter(results),
    [](auto const& mask) { return upscale_mask(mask); });

// PREFER: std::ranges::find over manual search
if (auto const it = std::ranges::find(items, target); it != items.end()) {
    process(*it);
}
```

### Parallel Algorithms (When Beneficial)

```cpp
// Use parallel execution for compute-intensive operations
#include <execution>

std::transform(std::execution::par_unseq,
    masks.begin(), masks.end(), results.begin(),
    [](auto const& mask) { return process_mask(mask); });

// Document when NOT to parallelize
// Note: cv::connectedComponents is not thread-safe
for (auto const& mask : masks) {
    results.push_back(cv::connectedComponents(mask));
}
```

## Small Functions and Clarity

### Single Responsibility

```cpp
// Break complex operations into named steps
[[nodiscard]] auto process_frame(cv::Mat const& frame) -> tl::expected<Pose, std::string> {
    return detect_features(frame)
        .and_then([this](auto const& features) { return match_features(features); })
        .and_then([this](auto const& matches) { return estimate_motion(matches); })
        .transform([this](auto const& motion) { return accumulate_pose(motion); });
}
```

### Avoid Output Parameters

```cpp
// PREFER: Return values (use structured bindings)
[[nodiscard]] auto decompose_essential(cv::Mat const& E)
    -> std::pair<Eigen::Matrix3d, Eigen::Vector3d>;

auto const [rotation, translation] = decompose_essential(E);

// AVOID: Output parameters
// void decompose_essential(cv::Mat const& E, Eigen::Matrix3d& R, Eigen::Vector3d& t);
```

## Testing Standards

See agent files for detailed testing patterns. Key points:

- BDD comments in ALL CAPS: `// GIVEN`, `// WHEN`, `// THEN`
- East const in test code
- ASSERT container size before indexing
- Parameterized tests to reduce duplication
- Helper functions over complex fixtures

## Tooling Integration

### Pre-commit Hooks

```bash
# Install hooks (one time)
pre-commit install

# Run all checks manually
pre-commit run -a

# Run specific hook
pre-commit run clang-format -a
pre-commit run clang-tidy -a
```

### Build Commands

```bash
pixi run configure  # CMake configure
pixi run build      # Build
pixi run test       # Run tests
pixi run tidy       # clang-tidy
pixi run format-check  # Check formatting
```

## Tips and Gotchas

### Common Mistakes to Avoid

1. **West const** - Always use east const
2. **camelCase** - Use snake_case for functions/variables
3. **Raw owning pointers** - Use smart pointers
4. **std::endl** - Use `'\n'` (faster, no flush)
5. **push_back with temporary** - Use `emplace_back`
6. **const string&** - Prefer `string_view` for non-owning
7. **Exceptions in expected paths** - Use `tl::expected`

### Performance Considerations

1. **Move semantics** - Return by value, compiler optimizes
2. **string_view** - Zero-copy for string parameters
3. **span** - Zero-copy for array parameters
4. **constexpr** - Compute at compile time when possible
5. **[[nodiscard]]** - Prevent accidentally ignoring results
