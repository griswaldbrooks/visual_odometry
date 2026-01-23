# Visual Odometry Architecture Analysis

## Design Principles

1. **Functions over classes** - Everything should be a function (or struct for data) unless it needs state between calls
2. **Classes only for state** - Only use classes when state must persist between method calls
3. **Push state to edges** - Stateful components should be at system boundaries, not in core logic
4. **Methods call functions** - Class methods should be thin wrappers calling pure functions
5. **I/O at edges** - File reading/writing should happen in main, with loaded data passed to functions

---

## Current Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ MAIN.CPP - SYSTEM BOUNDARY                                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. parse_args() - command line                                  │
│ 2. CameraIntrinsics::load_from_yaml() - reads YAML              │
│ 3. ImageLoader::create() - scans directory                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PROCESSING LOOP                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   loader.load_image_pair(i)  ←─── I/O hidden in method          │
│            ↓                                                    │
│   matcher.match_images(img1, img2)                              │
│       ├─ FeatureDetector::detect(img1)                          │
│       ├─ FeatureDetector::detect(img2)                          │
│       └─ FeatureMatcher::match(desc1, desc2)                    │
│            ↓                                                    │
│   estimator.estimate(points1, points2)                          │
│       ├─ cv::findEssentialMat (RANSAC)                          │
│       └─ cv::recoverPose                                        │
│            ↓                                                    │
│   trajectory.add_motion(motion)  ←─── Stateful (correct)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT                                                          │
├─────────────────────────────────────────────────────────────────┤
│ trajectory.save_to_json(path) - writes file                     │
└─────────────────────────────────────────────────────────────────┘
```

### Data Structures Through Pipeline

```
Image Files
    ↓ cv::imread
cv::Mat (grayscale)
    ↓ detect_features()
DetectionResult {keypoints[], descriptors}
    ↓ match_features()
MatchResult {points1[], points2[], matches[]}
    ↓ estimate_motion()
MotionEstimate {R(3x3), t(3x1), inliers, valid}
    ↓ Trajectory::add_motion()
Pose {R, t}  [accumulated in poses_]
    ↓ to_json()
JSON string → File
```

---

## Design Violations

### VIOLATION 1: FeatureDetector - Should Be Function

**Location:** `include/visual_odometry/feature_detector.hpp`

```cpp
// CURRENT - Unnecessary class wrapper
class FeatureDetector {
private:
    cv::Ptr<cv::ORB> orb_;  // Config, not state
};
```

**Problem:** `orb_` doesn't change between calls. Each `detect()` is independent.

**Should be:**
```cpp
struct FeatureDetectorConfig {
    int max_features = 2000;
};

[[nodiscard]] auto detect_features(cv::Mat const& image,
                                   FeatureDetectorConfig const& config)
    -> DetectionResult;
```

---

### VIOLATION 2: FeatureMatcher - Should Be Function

**Location:** `include/visual_odometry/feature_matcher.hpp`

```cpp
// CURRENT - Unnecessary class wrapper
class FeatureMatcher {
private:
    cv::Ptr<cv::BFMatcher> matcher_;
    float ratio_threshold_;  // Config, not state
};
```

**Should be:**
```cpp
struct FeatureMatcherConfig {
    float ratio_threshold = 0.75f;
};

[[nodiscard]] auto match_features(cv::Mat const& desc1, cv::Mat const& desc2,
                                  std::span<cv::KeyPoint const> kp1,
                                  std::span<cv::KeyPoint const> kp2,
                                  FeatureMatcherConfig const& config)
    -> MatchResult;
```

---

### VIOLATION 3: MatchAnythingMatcher - I/O Deep in Algorithm

**Location:** `src/image_matcher.cpp:73-149`

```cpp
// CURRENT - I/O happens inside algorithm
auto MatchAnythingMatcher::match_images(...) const -> MatchResult {
    cv::imwrite(temp1.string(), img1);     // I/O HERE
    cv::imwrite(temp2.string(), img2);     // I/O HERE
    output = exec_command(cmd.str());      // SUBPROCESS HERE
    std::filesystem::remove(temp1);        // I/O HERE
    std::filesystem::remove(temp2);        // I/O HERE
}
```

**Problems:**
- File I/O in algorithm method
- Subprocess spawned per call (model reloaded each time)
- 100x overhead vs embedded approach
- Security issues (command injection, predictable temp files)

**Should be:** Embedded Python via nanobind (see Nanobind section below)

---

### VIOLATION 4: OrbImageMatcher - Unnecessary Wrapper

**Location:** `include/visual_odometry/image_matcher.hpp:43-64`

```cpp
// CURRENT - Just composes two other classes
class OrbImageMatcher : public ImageMatcher {
    FeatureDetector detector_;
    FeatureMatcher matcher_;
};
```

**This is acceptable** if we need polymorphism for the plugin system. But the inner classes should become functions.

---

### CORRECT: Trajectory - Justified Stateful Class

**Location:** `include/visual_odometry/trajectory.hpp`

```cpp
class Trajectory {
    std::vector<Pose> poses_;  // Meaningful state - accumulates over time
};
```

**Why correct:**
- State persists across calls: each `add_motion()` depends on prior state
- `current_pose()` returns accumulated result of all prior motions
- This IS the edge of the system where state belongs

---

### CORRECT: Data Structs

These are all correct - pure data containers:
- `Pose` - camera pose (R, t)
- `MotionEstimate` - relative motion result
- `MatchResult` - feature matching output
- `DetectionResult` - feature detection output
- `CameraIntrinsics` - camera parameters

---

## Component Summary

| Component | Current | Should Be | State Justified? |
|-----------|---------|-----------|-----------------|
| **FeatureDetector** | Class | Function + Config struct | No |
| **FeatureMatcher** | Class | Function + Config struct | No |
| **OrbImageMatcher** | Class | OK (needs polymorphism) | N/A |
| **MatchAnythingMatcher** | Class + subprocess | Embedded Python | No |
| **MotionEstimator** | Class | Could be function, but OK | Marginal |
| **ImageLoader** | Class | OK (iteration state) | Yes |
| **Trajectory** | Class | Correct | Yes |

---

## Nanobind Integration

### Current State

The nanobind bindings (`src/python_bindings.cpp`) expose:
- ImageLoader, MatchResult, ImageMatcher, OrbImageMatcher
- CameraIntrinsics, MotionEstimate, MotionEstimator
- Pose, Trajectory
- Factory function `create_matcher()`

**cv::Mat ↔ numpy** conversion works via `cvnp_nano`.

### Problem: MatchAnything Subprocess

Current flow:
```
C++ match_images()
  → write temp PNG files
  → popen("python3 match_anything.py ...")
  → parse JSON output
  → delete temp files
```

**Issues:**
- Model loaded fresh for EVERY image pair (~500ms overhead)
- File I/O for every call
- JSON serialization/deserialization overhead
- ~100x slower than it should be

### Solution: Embedded Python

```cpp
// Proposed: Load model ONCE, call directly
class EmbeddedMatchAnything {
public:
    EmbeddedMatchAnything() {
        // Load model once
        auto match_anything = nb::module_::import_("match_anything_module");
        model_ = match_anything.attr("load_model")();
        processor_ = match_anything.attr("load_processor")();
    }

    auto match(cv::Mat const& img1, cv::Mat const& img2) -> MatchResult {
        // Direct call - no temp files, no subprocess
        auto np1 = cvnp::mat_to_nparray(img1);
        auto np2 = cvnp::mat_to_nparray(img2);
        auto result = model_.attr("match")(np1, np2);
        return parse_result(result);
    }

private:
    nb::object model_;
    nb::object processor_;
};
```

**Benefits:**
- Model loaded once (~5s), reused for all frames
- Direct memory access (no temp files)
- ~5ms per match vs ~500ms current
- No security issues (no shell commands)

---

## Viser Integration

### Current State

Three scripts use viser:
- `visualize_trajectory.py` - Post-hoc visualization from JSON
- `live_visualization.py` - Runs VO in Python, streams to viser
- `test_viser.py` - Basic connectivity test

**Current limitation:** No direct C++ → viser streaming.

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    C++ Core (Performance)                       │
├─────────────────────────────────────────────────────────────────┤
│ • detect_features() - pure function                             │
│ • match_features() - pure function                              │
│ • estimate_motion() - pure function                             │
│ • Trajectory - state accumulation                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ nanobind
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Python Orchestration Layer                      │
├─────────────────────────────────────────────────────────────────┤
│ • EmbeddedMatchAnything - ML inference                          │
│ • Viser server - real-time 3D visualization                     │
│ • Main loop - coordinates C++ functions + Python viz            │
└─────────────────────────────────────────────────────────────────┘
```

### Ideal Boundary

**C++ (speed-critical):**
- Feature detection (ORB)
- RANSAC motion estimation
- Pose composition
- Image loading

**Python (flexibility-critical):**
- ML model inference (MatchAnything, future models)
- Visualization (viser)
- Configuration/experimentation
- Analysis and debugging

### Main Loop Options

**Option A: Python-driven (recommended for flexibility)**
```python
import visual_odometry as vo

loader = vo.ImageLoader(image_dir)
matcher = EmbeddedMatchAnything()  # Python ML model
estimator = vo.MotionEstimator(intrinsics)
trajectory = vo.Trajectory()
server = viser.ViserServer()

for img1, img2 in loader:
    matches = matcher.match(img1, img2)  # Python ML
    motion = estimator.estimate(matches.points1, matches.points2)  # C++
    trajectory.add_motion(motion)  # C++

    # Real-time viz
    server.scene.add_camera_frustum(trajectory.current_pose())
```

**Option B: C++-driven with callbacks**
```cpp
// C++ drives, Python receives updates
auto on_pose = [&](Pose const& p) {
    py_viser.attr("add_pose")(p);  // Callback to Python
};

run_vo_pipeline(loader, matcher, estimator, on_pose);
```

---

## Recommended Refactoring Plan

### Phase 1: Core Functions (High Priority)

1. **Extract `detect_features()` function**
   - Create `FeatureDetectorConfig` struct
   - Make detection a pure function
   - Keep wrapper class only if needed for Python bindings

2. **Extract `match_features()` function**
   - Create `FeatureMatcherConfig` struct
   - Make matching a pure function

3. **Extract `estimate_motion()` function**
   - Create `MotionEstimatorConfig` struct
   - Pure function taking matches + config

### Phase 2: Embedded ML (High Priority)

4. **Replace MatchAnything subprocess with embedded Python**
   - Create Python module with `load_model()`, `match()`
   - Call from C++ via nanobind
   - Delete temp file logic entirely

### Phase 3: Visualization (Medium Priority)

5. **Create Python orchestration layer**
   - Main loop in Python using C++ functions
   - Viser integration for real-time viz
   - Config via Python (easy experimentation)

### Phase 4: Cleanup (Low Priority)

6. **Remove unnecessary class wrappers**
   - ImageMatcher hierarchy can stay (polymorphism needed)
   - Inner implementation becomes function calls

---

## Target Architecture

```
include/visual_odometry/
├── types.hpp              # All data structs (Pose, MotionEstimate, etc.)
├── detection.hpp          # detect_features() function
├── matching.hpp           # match_features() function
├── motion.hpp             # estimate_motion() function
├── trajectory.hpp         # Trajectory class (stateful, at edge)
├── image_loader.hpp       # ImageLoader class (I/O at edge)
└── image_matcher.hpp      # ImageMatcher interface (for polymorphism)

src/
├── detection.cpp
├── matching.cpp
├── motion.cpp
├── trajectory.cpp
├── image_loader.cpp
├── orb_image_matcher.cpp  # Thin wrapper calling functions
└── python_bindings.cpp    # Expose functions + classes

scripts/
├── main.py                # Python orchestration (NEW)
├── match_anything.py      # ML module for embedded use
├── visualize.py           # Viser visualization
└── analysis.py            # Post-processing tools
```

**Key insight:** The C++ becomes a library of pure functions + minimal stateful classes. Python orchestrates the pipeline, handles ML, and provides visualization.
