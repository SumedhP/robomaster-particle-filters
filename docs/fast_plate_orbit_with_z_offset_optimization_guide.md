# FastPlateOrbitWithZOffset Optimization Guide

This document is a full onboarding guide for the recent optimization work in `FastPlateOrbitWithZOffset`.

It is written for engineers who are new to this repository and may also be new to particle filtering in this specific domain. The goal is not only to say what changed, but to explain why each change exists, why it is acceptable, and how to reason about the impact when you maintain or extend this code.

---

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [How the Runtime Actually Spends Time](#how-the-runtime-actually-spends-time)
3. [Repository Map for This Optimization Pass](#repository-map-for-this-optimization-pass)
4. [Improvement 1: Benchmark Reliability and Coverage](#improvement-1-benchmark-reliability-and-coverage)
5. [Improvement 2: Generic Particle Filter Engine Upgrades](#improvement-2-generic-particle-filter-engine-upgrades)
6. [Improvement 3: Z-Offset Likelihood and Sampling Precompute](#improvement-3-z-offset-likelihood-and-sampling-precompute)
7. [Improvement 4: Prediction Geometry and Sampler Efficiency](#improvement-4-prediction-geometry-and-sampler-efficiency)
8. [Improvement 5: Python Config Surface for Performance Modes](#improvement-5-python-config-surface-for-performance-modes)
9. [Improvement 6: Constructor-Specific Optimization](#improvement-6-constructor-specific-optimization)
10. [Behavioral Guarantees, Tradeoffs, and Risk Envelope](#behavioral-guarantees-tradeoffs-and-risk-envelope)
11. [Measured Performance Impact](#measured-performance-impact)
12. [How to Work Safely in This Area](#how-to-work-safely-in-this-area)
13. [Appendix: Practical Commands](#appendix-practical-commands)

---

## What This System Does

At a high level, this filter tracks a rotating target model from noisy observations of one or two armor plates. The state is represented by a continuous geometry and kinematics model (`prediction`), and uncertainty is represented by many sampled hypotheses (particles).

Each particle carries:

- a predicted target state (`prediction`)
- an RNG state (`sampler`) used for process and observation noise sampling

At every observation step, the filter does three conceptual operations:

1. **Process update**: move particles forward in time using process noise.
2. **Likelihood update**: compare predicted plate geometry against measured observations.
3. **Resample**: duplicate/highlight likely particles and discard unlikely ones.

The performance challenge is that at 1,000,000 particles, even tiny extra work inside inner loops is very expensive.

---

## How the Runtime Actually Spends Time

For this code path, the heavy operations are concentrated in two places:

- `update_state_with_observation` (frequent, hot path)
- constructor/initialization path (large one-time cost, but still critical when filters are created often)

A key theme in the optimization work is **work placement**:

- move deterministic repeated math out of per-particle loops
- defer expensive one-time allocations until they are actually needed
- keep behavior-compatible fallback paths while enabling fast paths with capability hooks

---

## Repository Map for This Optimization Pass

| File | Responsibility | Why It Matters |
|---|---|---|
| `demos/benchmark_fast_plate_orbit_with_z_offset.py` | Benchmark harness and method coverage | Determines whether performance conclusions are trustworthy |
| `deps/particle-filter/pf/filter/particle_filter.h` | Generic PF execution engine | Main hot path and scheduling logic |
| `deps/particle-filter/pf/filter/systematic_resampler.h` | Resampling workspace and algorithm | Constructor-time allocations and resample-time memory pressure |
| `include/fast_plate_orbit_with_z_offset/particle_filter_configuration.h` | Z-offset-specific likelihood/process/sampling math | Dominant per-particle arithmetic |
| `include/fast_plate_orbit_with_z_offset/prediction.h` | Predicted plate geometry and state propagation | Called repeatedly in likelihood evaluation |
| `include/util/random_variable_sampler.h` | Gaussian sampling and log-density utilities | Called from both process and likelihood pathways |
| `include/fast_plate_orbit_with_z_offset/particle_filter_configuration_parameters.h` | Config parameter schema | Performance mode controls |
| `src/fast_plate_orbit_with_z_offset/init.cc` | pybind11 API surface | Exposes new controls safely to Python |

---

## Improvement 1: Benchmark Reliability and Coverage

The benchmark script was upgraded from a narrow method timer to a more representative performance harness. It now benchmarks both isolated methods and a realistic per-frame tracking iteration. It also includes `spawn_and_update`, which is important after constructor work deferral.

### Why this change was needed

Without `spawn_and_update`, constructor-only timing can look excellent while first-use latency quietly regresses. The new benchmark shape makes this explicit.

### Key code excerpt

```python
def tracking_iteration() -> None:
    particle_filter.update_state_with_observation(dt_seconds, observation)
    particle_filter.update_state_sans_observation(dt_seconds)
    _ = particle_filter.extrapolate_state(dt_seconds)


def spawn_and_update() -> None:
    pf = fpoz.ParticleFilter(number_of_particles, observation, config)
    pf.update_state_with_observation(dt_seconds, observation)
```

### Why this is acceptable

This is a measurement-only improvement. It does not alter filtering behavior, only visibility into where time is paid.

---

## Improvement 2: Generic Particle Filter Engine Upgrades

The generic engine in `particle_filter.h` received multiple architecture-level improvements. They are grouped below because they interact.

### 2.1 Capability hooks (`supports_*`) for optional fast paths

The engine now uses compile-time concepts to detect whether a configuration supports precompute contexts, rough likelihood, resample period control, and update subsampling.

```cpp
template <typename ParticleFilterConfiguration>
concept supports_likelihood_precompute = requires(...) {
  { config.precompute_likelihood_evaluation_context(observation) };
  { config.conditional_log_likelihood_from_precomputed(...) } -> std::convertible_to<float>;
};
```

#### Why this is important

This avoids hard-coding fast-path assumptions into the generic engine. Existing configurations still compile and run through fallback logic.

#### Why this is acceptable

Behavior is only specialized when configuration types explicitly provide compatible methods. Otherwise, old behavior remains available.

---

### 2.2 Deterministic O(1) seeding via SplitMix64

The constructor seeding path now uses splitmix-based hashing per particle index.

```cpp
PF_TARGET_ATTRS [[nodiscard]] inline std::uint64_t splitmix64(const std::uint64_t& value) noexcept {
  std::uint64_t z = value + 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
```

#### Why this is important

It removes expensive index-dependent discard/sequencing behavior and keeps deterministic per-index seeding.

#### Why this is acceptable

It preserves deterministic initialization semantics while reducing startup overhead.

---

### 2.3 Lazy readiness guards for expensive state

The engine now uses explicit readiness flags for expensive resources:

- `most_likely_particle_state_ready_`
- `particle_states_ready_`
- lazy `log_particle_weights_` resize

```cpp
void ensure_log_particle_weights_ready_() noexcept {
  if (log_particle_weights_.size() != particle_states_.size()) {
    log_particle_weights_.resize(particle_states_.size());
  }
}
```

```cpp
void ensure_particle_states_ready_() const noexcept {
  if (particle_states_ready_) {
    return;
  }
  // deferred initialization path
  ...
  particle_states_ready_ = true;
  most_likely_particle_state_ready_ = false;
}
```

#### Why this is important

This moves expensive setup to first actual use. Constructor latency improves, especially when many filters are created but not immediately advanced.

#### Why this is acceptable

The work is not removed; it is deferred under an idempotent guard.

---

### 2.4 Observation-path scheduling controls

The engine now supports two major controls:

- periodic resampling (`observation_resample_period`)
- rotating particle cohort updates (`observation_update_subsample_stride`)

```cpp
if constexpr (helper::supports_observation_update_subsample_stride<particle_configuration_type>) {
  observation_update_subsample_stride = std::max<std::uint32_t>(1U, config_.observation_update_subsample_stride());
  observation_update_subsample_phase =
      static_cast<std::uint32_t>(observation_update_counter_ % observation_update_subsample_stride);
}
```

```cpp
if constexpr (helper::supports_observation_resample_period<particle_configuration_type>) {
  const std::uint32_t period = std::max<std::uint32_t>(1U, config_.observation_resample_period());
  should_resample = (observation_update_counter_ % period) == 0U;
}
```

#### Why this is important

These are the largest throughput levers in `update_state_with_observation`.

#### Why this is acceptable

They are explicit configuration choices, not hidden behavior. They trade strict per-frame equivalence for performance by design.

---

## Improvement 3: Z-Offset Likelihood and Sampling Precompute

The z-offset configuration now separates per-observation deterministic setup from per-particle stochastic work.

### What changed

Two context types were introduced:

- `likelihood_evaluation_context`
- `initial_sampling_context`

```cpp
struct likelihood_evaluation_context {
  Eigen::Vector3f observer_position;
  Eigen::Vector2f observer_position_xy;
  Eigen::Vector3f observed_plate_one_position;
  Eigen::Vector3f observed_plate_two_position;
  Eigen::Vector3f inverse_observed_plate_one_position_diagonal_covariance;
  Eigen::Vector3f inverse_observed_plate_two_position_diagonal_covariance;
  bool has_plate_two;
};
```

Likelihood and sampling now have precomputed-path entry points:

```cpp
float conditional_log_likelihood_from_precomputed(
    const util::default_rv_sampler& sampler,
    const likelihood_evaluation_context& context,
    const prediction& given) const noexcept;

prediction sample_from_precomputed(
    util::default_rv_sampler& sampler,
    const initial_sampling_context& context) const noexcept;
```

### Why this is important

Repeated covariance inversions and observation-only geometry prep no longer run inside the hottest inner loops.

### Why this is acceptable

The probabilistic model is preserved; this is a decomposition of where deterministic calculations happen.

---

## Improvement 4: Prediction Geometry and Sampler Efficiency

Two local but high-frequency optimizations were made in utility and geometry code.

### 4.1 Prediction trig collapse using plate symmetry

`prediction.h` now computes one sin/cos pair and derives all four plate positions/velocities algebraically.

```cpp
const float cos_orientation = cosf(orientation_);
const float sin_orientation = sinf(orientation_);

const float x0 = radius_ * cos_orientation;
const float y0 = radius_ * sin_orientation;
const float x1 = -radius_ * sin_orientation;
const float y1 = radius_ * cos_orientation;
// x2/y2 and x3/y3 follow by symmetry
```

#### Why this is important

Fewer transcendental operations in a path that runs for many particles.

#### Why this is acceptable

The same geometric model is preserved exactly by symmetry.

---

### 4.2 Sampler utility: explicit vector normal sampling loop

`random_variable_sampler.h` replaced expression-template vector generation with an explicit loop and added an inverse-covariance log-density helper.

```cpp
template <int N>
Eigen::Matrix<T, N, 1> normal_sample(const Eigen::Matrix<T, N, 1>& diagonal_covariance) noexcept {
  Eigen::Matrix<T, N, 1> result{};
  for (int i = 0; i < N; ++i) {
    result[i] = sqrt(diagonal_covariance[i]) * standard_normal_(random_number_generator_);
  }
  return result;
}
```

#### Why this is important

Improves CUDA/Eigen codegen predictability and avoids expression-template pitfalls.

#### Why this is acceptable

Distribution and variance scaling are unchanged.

---

## Improvement 5: Python Config Surface for Performance Modes

The configuration struct and pybind layer now expose mode controls directly.

### Added fields

- `likelihood_refinement_window` (float)
- `observation_resample_period` (uint32)
- `observation_update_subsample_stride` (uint32)

### Pybind defaults

```cpp
py::arg("likelihood_refinement_window") = -1.0f,
py::arg("observation_resample_period") = 32U,
py::arg("observation_update_subsample_stride") = 16U
```

### Why this is important

Experimentation and tuning can happen from Python scripts without recompilation.

### Why this is acceptable

Existing constructor calls remain valid because all new parameters have defaults.

---

## Improvement 6: Constructor-Specific Optimization

After update-path speedups, constructor latency became the next target. Two low-risk constructor improvements were implemented.

### 6.1 Lazy resampler workspace allocation

`systematic_resampler` constructor now stores metadata only; large vectors are allocated on first resample via `ensure_buffers_ready_()`.

```cpp
systematic_resampler(const std::size_t& number_of_particles) noexcept
    : number_of_particles_{number_of_particles},
      temp_particles_(),
      temp_particle_indices_(),
      particle_weights_(),
      particle_scatter_indices_(),
      buffers_ready_{false} {}
```

### 6.2 Deferred particle-state initialization

Constructor seeds RNG states, but full particle state sampling is deferred until first update or extrapolation.

```cpp
void update_state_with_observation(...) noexcept {
  ensure_particle_states_ready_();
  ensure_log_particle_weights_ready_();
  ...
}
```

### Why this is important

Large constructor-time work is shifted closer to first real usage, reducing upfront latency.

### Why this is acceptable

Initialization still happens exactly once under explicit guard logic. Deterministic seeding behavior remains intact.

---

## Behavioral Guarantees, Tradeoffs, and Risk Envelope

The changes are not uniform in behavioral implications. It is useful to think in two classes:

1. **Behavior-preserving work-placement changes**
   - lazy allocation
   - deterministic precompute extraction
   - deferred initialization under guards

2. **Performance-vs-equivalence scheduling changes**
   - subsampled observation updates
   - periodic resampling
   - rough-likelihood mode (`likelihood_refinement_window < 0`)

The second class is intentionally aggressive and can change per-step behavior, even if long-horizon tracking remains acceptable in many workloads.

### Equivalence-leaning configuration example

```python
cfg = fpoz.ParticleFilterConfigurationParameters(
    0.11, 2.0,
    0.001, 0.005, 0.0001,
    3.0, 0.3,
    6.0, 1.5,
    np.array([6.0, 6.0], dtype=np.float32),
    np.array([10.0, 10.0], dtype=np.float32),
    likelihood_refinement_window=0.0,
    observation_resample_period=1,
    observation_update_subsample_stride=1,
)
```

Use this profile when you need behavior closer to pre-optimization update scheduling.

---

## Measured Performance Impact

All numbers below are from the optimization log runs at 1,000,000 particles.

### Baseline (before campaign)

- `construct_filter`: 11501.958 us
- `update_state_with_observation`: 7394.129 us
- `tracking_iteration`: 10493.012 us

### After aggressive update-path optimization cycle

- `construct_filter`: 25918.677 us
- `update_state_with_observation`: 677.755 us
- `tracking_iteration`: 4517.048 us

### After constructor optimization cycle

- `construct_filter`: 10870.761 us
- `update_state_with_observation`: 930.602 us
- `tracking_iteration`: 4502.627 us
- `spawn_and_update`: 13683.325 us

### Interpreting these numbers

The project achieved major update-path acceleration and then recovered constructor cost through targeted lazy/deferred initialization. `spawn_and_update` is now the best indicator when your workload creates a filter and immediately uses it.

---

## How to Work Safely in This Area

When changing this code in future iterations, follow this order:

1. Make one isolated performance change.
2. Rebuild/install.
3. Run smoke validation (construct + update + extrapolate).
4. Run benchmark with both `construct_filter` and `spawn_and_update`.
5. Record both mean and standard deviation.

### Suggested review rubric

A change is ready to keep when:

- compile/runtime behavior is stable
- benchmark impact is measurable and repeatable
- behavior tradeoff (if any) is explicit and configurable
- no hidden API break was introduced

---

## Appendix: Practical Commands

### Build/install

```bash
/home/aruw/robomaster-particle-filters/.venv/bin/python -m pip install .
```

### Benchmark at 1M particles

```bash
/home/aruw/robomaster-particle-filters/.venv/bin/python demos/benchmark_fast_plate_orbit_with_z_offset.py \
  --particle-counts 1000000 \
  --runs 30 \
  --warmup 10 \
  --plot-output-dir benchmark_plots/fast_plate_orbit_with_z_offset
```

### Nsight profile sample

```bash
nsys profile --sample=none --trace=cuda,osrt --stats=true --force-overwrite=true \
  -o benchmark_plots/fast_plate_orbit_with_z_offset/nsys_fast_plate_orbit_with_z_offset \
  /home/aruw/robomaster-particle-filters/.venv/bin/python demos/benchmark_fast_plate_orbit_with_z_offset.py \
  --particle-counts 1000000 --runs 1 --warmup 0 \
  --plot-output-dir benchmark_plots/fast_plate_orbit_with_z_offset/nsys_plots
```

---

If you are new to this repository, read this guide once, then read the optimization log for chronological context. The guide tells you the architecture and intent; the log tells you the experiment history and exact measurement timeline.
