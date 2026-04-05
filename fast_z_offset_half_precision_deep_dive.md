# Fast Z-Offset Particle Filter Optimization Deep Dive

## Document Purpose

This report documents the full optimization journey that led to the current high-performance fast z-offset particle filter implementation, with special focus on the half-precision state storage breakthrough.

The goals of this document are:

1. Explain the architecture and execution path in enough detail that another engineer can reason about performance changes safely.
2. Tie each major change to measured outcomes in the benchmark ledger.
3. Tie bottlenecks to concrete kernel hotspots from Nsight Systems summaries.
4. Explain why half precision for state fields (while keeping log weights as float) delivered a large gain.
5. Provide reproducible steps so results can be validated on the same workspace.

## Scope and Workload

The optimization target in this session is the fast z-offset implementation.

Target implementation:

- `src/fast_plate_orbit_with_z_offset/particle_filter.cu`

Primary benchmark method:

- `update_state_with_observation_and_extract_state`

Representative production-like loop semantics:

- Update with observation every frame.
- Extract state every frame.
- `dt = 1/60`.
- 1,000,000 particles.

This aligns with the benchmark harness definitions in:

- `demos/benchmark_fast_plate_orbit_with_z_offset.py`

## Measurement Protocol

The benchmark protocol used for decision-making in the run ledger is:

- Warmup iterations: 10.
- Timed iterations: 40.
- Primary metric: mean microseconds of `update_state_with_observation_and_extract_state`.
- Secondary metric: mean microseconds of `tracking_iteration`.
- Keep/discard rule: keep only if improvement is stable and meaningful in the primary metric without unacceptable regressions in practical loop behavior.

The benchmark harness supports broader sweeps, but optimization decisions were anchored to the 1M particle operating point.

## Baseline and End State

Baseline run:

- run id: `run-apr4-a-baseline-000`
- primary mean: 9024.601 us
- primary std: 15.981 us
- tracking mean: 13188.246 us

Best kept run from this session focus:

- run id: `run-apr5-a-large-038-rerun`
- primary mean: 2898.696 us
- primary std: 5.329 us
- tracking mean: 4502.549 us

Absolute improvement versus baseline:

- primary: `-6125.905 us`
- tracking: `-8685.697 us`

Relative improvement versus baseline:

- primary: `-67.88%`
- tracking: `-65.86%`

## Executive Narrative of Breakthroughs

The largest durable improvements came from a sequence of architecture-level and hotpath-level changes, not a single micro-optimization.

High-level chronology:

1. Early hot-path cleanup and caching in likelihood logic reduced obvious per-particle overhead.
2. SoA conversion dramatically improved memory access behavior and enabled subsequent field-specialized kernels.
3. Scalarized and specialized process + likelihood code reduced object construction and generic math overhead.
4. Resampling path simplifications removed unnecessary index/materialization work.
5. Most-likely reduction was rewritten around scalar state accumulation rather than heavier object-level transforms.
6. Half-precision state storage cut state traffic further while preserving float compute for math and log weights.

The half-state change succeeded because it was applied after the architecture was already aligned with memory-efficient SoA kernels, making bandwidth savings immediately actionable in dominant kernels.

## Architecture Walkthrough

### Core Components

The fast z-offset filter implementation consists of:

- RNG sampler state vector.
- Log weight vector (float).
- Particle index vector for resampling.
- SoA vectors for all state fields.
- Temporary SoA vectors for gather materialization during resampling.
- Reduction path to produce the current most-likely state estimate.

### Why SoA Matters Here

The filter updates 1M particles per frame.

For each particle, kernels touch many fields:

- radius
- z0
- z1
- orientation
- orientation velocity
- center x/y
- center velocity x/y

In AoS form, field access by warp lanes is strided by full struct size.

In SoA form, field access by warp lanes is contiguous.

Given repeated full-pass kernels every frame, this alone changes the memory behavior profile fundamentally.

### Half Precision Strategy

Half precision was not applied blindly.

The adopted strategy:

- Store particle state fields in half (`__half`).
- Keep log particle weights in float.
- Convert half -> float at read boundaries in kernels.
- Perform process and likelihood math in float.
- Convert float -> half only at write boundaries back to SoA state buffers.

This design preserves most numerical stability-sensitive arithmetic in float while reducing memory bandwidth for state arrays.

## Code Evidence: Half-State Integration

### Storage Type Alias and Converters

Source: `src/fast_plate_orbit_with_z_offset/particle_filter.cu`

```cpp
namespace helper {

using configuration_type = particle_filter_configuration;
using sampler_type = configuration_type::sampler_type;
using state_storage_type = __half;

PF_TARGET_ATTRS [[nodiscard]] inline float state_storage_to_float(const state_storage_type& value) noexcept {
  return __half2float(value);
}

PF_TARGET_ATTRS [[nodiscard]] inline state_storage_type float_to_state_storage(const float& value) noexcept {
  return __float2half_rn(value);
}

```

Interpretation:

- The storage type is explicitly centralized through `state_storage_type`.
- Conversion helpers isolate precision-boundary logic and keep callsites readable.

### Process + Weight Functor Uses Float Compute

Source: `src/fast_plate_orbit_with_z_offset/particle_filter.cu`

```cpp
struct process_and_weight_particle_fields_functor {
  configuration_type config_;
  configuration_type::process_noise_scales process_noise_scales_;
  configuration_type::likelihood_observation_cache observation_cache_;

  template <typename TupleType>
  PF_TARGET_ONLY_ATTRS void operator()(TupleType tuple) const noexcept {
    sampler_type& sampler_state = cuda::std::get<0>(tuple);
    float& particle_weight = cuda::std::get<1>(tuple);

    state_storage_type& radius_storage = cuda::std::get<2>(tuple);
    state_storage_type& z_coordinate_0_storage = cuda::std::get<3>(tuple);
    state_storage_type& z_coordinate_1_storage = cuda::std::get<4>(tuple);

    state_storage_type& orientation_storage = cuda::std::get<5>(tuple);
    state_storage_type& orientation_velocity_storage = cuda::std::get<6>(tuple);

    state_storage_type& center_x_storage = cuda::std::get<7>(tuple);
    state_storage_type& center_y_storage = cuda::std::get<8>(tuple);
    state_storage_type& center_velocity_x_storage = cuda::std::get<9>(tuple);
    state_storage_type& center_velocity_y_storage = cuda::std::get<10>(tuple);

    float radius = state_storage_to_float(radius_storage);
    float z_coordinate_0 = state_storage_to_float(z_coordinate_0_storage);
    float z_coordinate_1 = state_storage_to_float(z_coordinate_1_storage);

    float orientation = state_storage_to_float(orientation_storage);
    float orientation_velocity = state_storage_to_float(orientation_velocity_storage);

    float center_x = state_storage_to_float(center_x_storage);
    float center_y = state_storage_to_float(center_y_storage);
    float center_velocity_x = state_storage_to_float(center_velocity_x_storage);
    float center_velocity_y = state_storage_to_float(center_velocity_y_storage);

    config_.apply_process_to_fields(
      process_noise_scales_,
      sampler_state,
      radius,
      z_coordinate_0,
      z_coordinate_1,
      orientation,
      orientation_velocity,
      center_x,
      center_y,
      center_velocity_x,
      center_velocity_y);

    particle_weight = config_.conditional_log_likelihood_from_fields(
      sampler_state,
      observation_cache_,
      radius,
      z_coordinate_0,
      z_coordinate_1,
      orientation,
      center_x,
      center_y);

    radius_storage = float_to_state_storage(radius);
    z_coordinate_0_storage = float_to_state_storage(z_coordinate_0);
    z_coordinate_1_storage = float_to_state_storage(z_coordinate_1);

    orientation_storage = float_to_state_storage(orientation);
    orientation_velocity_storage = float_to_state_storage(orientation_velocity);

    center_x_storage = float_to_state_storage(center_x);
    center_y_storage = float_to_state_storage(center_y);
    center_velocity_x_storage = float_to_state_storage(center_velocity_x);
    center_velocity_y_storage = float_to_state_storage(center_velocity_y);
  }
```

Interpretation:

- All state fields are loaded from half to float before math.
- Process model and conditional log likelihood operate in float.
- Updated state is written back in half.
- Log weights remain float.

### SoA and Temporary Buffers

Source: `src/fast_plate_orbit_with_z_offset/particle_filter.cu`

```cpp

  prediction most_likely_particle_state_;

  std::size_t number_of_particles_;
  pf::filter::systematic_resampler<std::uint32_t, std::uint32_t> index_resampler_;

  pf::target_config::vector<sampler_type> sampler_states_;
  pf::target_config::vector<float> log_particle_weights_;
  pf::target_config::vector<std::uint32_t> particle_indices_;

  // Struct-of-arrays particle state storage.
  pf::target_config::vector<state_storage_type> radius_;
  pf::target_config::vector<state_storage_type> z_coordinate_0_;
  pf::target_config::vector<state_storage_type> z_coordinate_1_;
  pf::target_config::vector<state_storage_type> orientation_;
  pf::target_config::vector<state_storage_type> orientation_velocity_;
  pf::target_config::vector<state_storage_type> center_x_;
  pf::target_config::vector<state_storage_type> center_y_;
  pf::target_config::vector<state_storage_type> center_velocity_x_;
  pf::target_config::vector<state_storage_type> center_velocity_y_;

  // Temporary storage for index-gather materialization during resampling.
  pf::target_config::vector<state_storage_type> temp_radius_;
  pf::target_config::vector<state_storage_type> temp_z_coordinate_0_;
  pf::target_config::vector<state_storage_type> temp_z_coordinate_1_;
  pf::target_config::vector<state_storage_type> temp_orientation_;
  pf::target_config::vector<state_storage_type> temp_orientation_velocity_;
  pf::target_config::vector<state_storage_type> temp_center_x_;
  pf::target_config::vector<state_storage_type> temp_center_y_;
  pf::target_config::vector<state_storage_type> temp_center_velocity_x_;
  pf::target_config::vector<state_storage_type> temp_center_velocity_y_;

  [[nodiscard]] prediction extrapolate_state(const float& time_offset_seconds) const noexcept {
```

Interpretation:

- Persistent state fields and temp materialization fields both use `state_storage_type`.
- This ensures gather/resample traffic also benefits from half storage.

### Resample Materialization Path

Source: `src/fast_plate_orbit_with_z_offset/particle_filter.cu`

```cpp
  void resample_particle_fields_() noexcept {
    auto source_particle_fields = thrust::make_zip_iterator(
      radius_.begin(),
      z_coordinate_0_.begin(),
      z_coordinate_1_.begin(),
      orientation_.begin(),
      orientation_velocity_.begin(),
      center_x_.begin(),
      center_y_.begin(),
      center_velocity_x_.begin(),
      center_velocity_y_.begin());

    auto destination_particle_fields = thrust::make_zip_iterator(
      temp_radius_.begin(),
      temp_z_coordinate_0_.begin(),
      temp_z_coordinate_1_.begin(),
      temp_orientation_.begin(),
      temp_orientation_velocity_.begin(),
      temp_center_x_.begin(),
      temp_center_y_.begin(),
      temp_center_velocity_x_.begin(),
      temp_center_velocity_y_.begin());

    thrust::gather(
      pf::target_config::policy(caching_allocator_),
      particle_indices_.cbegin(),
      particle_indices_.cend(),
      source_particle_fields,
      destination_particle_fields);

    radius_.swap(temp_radius_);
    z_coordinate_0_.swap(temp_z_coordinate_0_);
    z_coordinate_1_.swap(temp_z_coordinate_1_);
    orientation_.swap(temp_orientation_);
    orientation_velocity_.swap(temp_orientation_velocity_);
    center_x_.swap(temp_center_x_);
    center_y_.swap(temp_center_y_);
    center_velocity_x_.swap(temp_center_velocity_x_);
    center_velocity_y_.swap(temp_center_velocity_y_);
  }
```

Interpretation:

- Gather writes full state-field tuples into temporary SoA buffers.
- Swapping buffers avoids extra copies.
- Because fields are half, gather write bandwidth is reduced relative to float-state storage.

### Most-Likely Reduction Path

Source: `src/fast_plate_orbit_with_z_offset/particle_filter.cu`

```cpp
  void update_most_likely_particle_state_() noexcept {
    const auto* radius = thrust::raw_pointer_cast(radius_.data());
    const auto* z_coordinate_0 = thrust::raw_pointer_cast(z_coordinate_0_.data());
    const auto* z_coordinate_1 = thrust::raw_pointer_cast(z_coordinate_1_.data());

    const auto* orientation = thrust::raw_pointer_cast(orientation_.data());
    const auto* orientation_velocity = thrust::raw_pointer_cast(orientation_velocity_.data());

    const auto* center_x = thrust::raw_pointer_cast(center_x_.data());
    const auto* center_y = thrust::raw_pointer_cast(center_y_.data());
    const auto* center_velocity_x = thrust::raw_pointer_cast(center_velocity_x_.data());
    const auto* center_velocity_y = thrust::raw_pointer_cast(center_velocity_y_.data());

    const auto index_begin = thrust::make_counting_iterator<std::uint32_t>(std::uint32_t{});
    const auto index_end = index_begin + static_cast<std::uint32_t>(number_of_particles_);

    const helper::reduction_state_transform_functor transform_functor{
      radius,
      z_coordinate_0,
      z_coordinate_1,
      orientation,
      orientation_velocity,
      center_x,
      center_y,
      center_velocity_x,
      center_velocity_y};

    const helper::reduction_particle_state reduced_particle_state = thrust::transform_reduce(
      pf::target_config::policy(caching_allocator_),
      index_begin,
      index_end,
      transform_functor,
      helper::reduction_particle_state::zero(),
      helper::reduction_particle_state_reduce_functor{});

    most_likely_particle_state_ = prediction(
      reduced_particle_state.radius_,
      reduced_particle_state.z_coordinate_0_,
      reduced_particle_state.z_coordinate_1_,
      reduced_particle_state.orientation_,
```

Interpretation:

- Reduction transform reads state through raw pointers and converts each half field to float.
- Reduction state remains float scalar fields.
- Final prediction reconstruction is done once per update.

## Benchmark Harness Evidence

Relevant benchmark definitions are in `demos/benchmark_fast_plate_orbit_with_z_offset.py`.

Snippet showing warmup/timed loop utility and method registration:

```python
def benchmark_callable(callable_under_test: Callable[[], None], runs: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        callable_under_test()

    samples_us: list[float] = []
    for _ in range(runs):
        start_ns = time.perf_counter_ns()
        callable_under_test()
        end_ns = time.perf_counter_ns()
        samples_us.append((end_ns - start_ns) / 1_000.0)

    mean_us = statistics.fmean(samples_us)
    std_us = statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0
    return mean_us, std_us


def run_benchmarks_for_particle_count(
    number_of_particles: int,
    runs: int,
    warmup: int,
    dt_seconds: float,
) -> list[BenchmarkResult]:
    observation = build_default_observation()
    config = build_default_config()

    def construct_filter() -> None:
        _ = fpoz.ParticleFilter(number_of_particles, observation, config)

    particle_filter = fpoz.ParticleFilter(number_of_particles, observation, config)

    def update_with_observation() -> None:
        particle_filter.update_state_with_observation(dt_seconds, observation)

    def update_with_observation_and_extract_state() -> None:
        particle_filter.update_state_with_observation(dt_seconds, observation)
        _ = particle_filter.extrapolate_state(dt_seconds)

    def update_sans_observation() -> None:
        particle_filter.update_state_sans_observation(dt_seconds)

    def extrapolate_state() -> None:
        _ = particle_filter.extrapolate_state(dt_seconds)

    def tracking_iteration() -> None:
        particle_filter.update_state_with_observation(dt_seconds, observation)
        particle_filter.update_state_sans_observation(dt_seconds)
        _ = particle_filter.extrapolate_state(dt_seconds)
    
    def spawn_and_update() -> None:
        pf = fpoz.ParticleFilter(number_of_particles, observation, config)
        pf.update_state_with_observation(dt_seconds, observation)

    methods: Sequence[tuple[str, Callable[[], None]]] = (
        ("construct_filter", construct_filter),
        ("update_state_with_observation", update_with_observation),
        ("update_state_with_observation_and_extract_state", update_with_observation_and_extract_state),
        ("update_state_sans_observation", update_sans_observation),
        ("extrapolate_state", extrapolate_state),
        ("tracking_iteration", tracking_iteration),
        ("spawn_and_update", spawn_and_update),
    )

    results: list[BenchmarkResult] = []
    for name, benchmark_target in methods:
        mean_us, std_us = benchmark_callable(benchmark_target, runs=runs, warmup=warmup)
        results.append(BenchmarkResult(name, mean_us, std_us))

    return results
```

Snippet showing default CLI knobs:

```python
        )
    )
    parser.add_argument("--runs", type=int, default=100, help="Timed runs per method")
    parser.add_argument("--warmup", type=int, default=20, help="Warm-up runs per method")
    parser.add_argument("--dt-seconds", type=float, default=1.0 / 60.0, help="Time delta passed to filter update/extrapolation")
    parser.add_argument("--min-particles", type=int, default=1, help="Starting particle count")
    parser.add_argument("--max-particles", type=int, default=1_000_000, help="Maximum particle count")
    parser.add_argument("--multiplier", type=int, default=10, help="Particle count multiplier between measurements")
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default="benchmark_plots/fast_plate_orbit_with_z_offset",
        help="Directory where per-method benchmark plots are written.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plot windows in addition to saving PNG files.",
    )
    parser.add_argument(
        "--particle-counts",
        type=str,
        default="",
        help="Optional comma-separated particle counts. Overrides min/max/multiplier.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be greater than 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.min_particles <= 0 or args.max_particles <= 0:
        raise ValueError("Particle counts must be greater than 0")
    if args.min_particles > args.max_particles and not args.particle_counts:
        raise ValueError("--min-particles must be <= --max-particles when --particle-counts is not provided")

    particle_counts = parse_particle_counts(args)

    print("Benchmarking fast_plate_orbit_with_z_offset.ParticleFilter")
    print(f"Runs per method: {args.runs}")
    print(f"Warm-up runs per method: {args.warmup}")
    print("Units: microseconds")

    gc_state = gc.isenabled()
    benchmark_results_by_particle_count: dict[int, list[BenchmarkResult]] = {}
```

## Profiler Hotspot Snapshots

This section summarizes key kernels from three captures:

1. Early profile (`/tmp/zoffset_profile_update_extract_v1_cuda_gpu_kern_sum.csv`).
2. Pre-half later profile (`/tmp/zoffset_profile_update_extract_v6_cuda_gpu_kern_sum.csv`).
3. Post-half focus profile (`/tmp/fpoz_run038_focus_cuda_gpu_kern_sum.csv`).

### v1 Snapshot Highlights

- process+weight kernel: 46.0%
- gather resampled fields kernel: 20.6%
- reduction transform-reduce kernel: 10.8%
- scan kernel (truncated representation): 4.9%

### v6 Snapshot Highlights

- process+weight kernel: 48.1%
- resample gather transform kernel: 22.0%
- reduction transform-reduce kernel: 11.5%
- scan kernel (truncated representation): 5.3%

### run038 Focus Snapshot Highlights

- process+weight kernel: 49.1%
- resample gather transform kernel: 22.8%
- reduction transform-reduce kernel: 11.6%
- scan kernel (truncated representation): 5.3%

Interpretation caveat:

- Percent share of total GPU kernel time can remain similar while absolute frame time drops.
- Half-state optimization mostly reduces memory traffic in hot kernels, so relative percentages can stay structurally similar while latency decreases.

## Why Half-State Helped

Half-state helped because this workload repeatedly streams state fields through large kernels.

The dominant path repeatedly performs:

1. Read many state fields per particle.
2. Compute process + likelihood.
3. Write updated fields.
4. Gather field tuples after resampling.
5. Read fields again for reduction.

When state arrays are half-sized:

- Read traffic drops for state fields.
- Write traffic drops for state fields.
- Gather traffic drops for state fields.
- Reduction input traffic drops for state fields.

Because arithmetic remained float for critical computations:

- The performance win did not require moving numerically sensitive calculations into half.
- Practical stability remained acceptable in smoke checks.

## Risks and Validation Notes

Potential risk classes introduced by half-state storage:

1. Quantization effects in slow-drift state components.
2. Potential phase/orientation wrap sensitivity under long trajectories.
3. Slight bias accumulation if process noise scales are very low.
4. Edge-case behavior under sparse/noisy observations.

Recommended validation beyond microbenchmarks:

1. Long-horizon trajectory replay against float-state baseline.
2. Drift statistics over minute-scale sequences.
3. Outlier and recovery scenarios (occlusions, abrupt maneuvers).
4. Agreement checks on orientation/z ordering behavior under wraps.

## Reproduction Steps

### Build and Install

```bash
python -m pip install --force-reinstall .
```

### 1M-Particle Benchmark (decision protocol)

```bash
python demos/benchmark_fast_plate_orbit_with_z_offset.py --particle-counts 1000000 --runs 40 --warmup 10
```

### Focused Profiling Example

```bash
nsys profile --stats=true -o /tmp/fpoz_run038_focus python demos/benchmark_fast_plate_orbit_with_z_offset.py --particle-counts 1000000 --runs 40 --warmup 10
```

## Annotated Full Run Timeline

The next section provides a line-by-line annotated timeline for every run recorded in `results.tsv`.

For each run, the report includes:

- Raw metrics.
- Delta versus baseline.
- Delta versus immediately previous run.
- Running-best context.
- Stability and subsystem attribution tags.
- Practical implication against a 60 FPS frame budget.

#### Timeline 001: `run-apr4-a-baseline-000`
- Timestamp: 2026-04-04T23:43:33-07:00
- Phase: baseline
- Status: keep
- Mean update+extract latency (us): 9024.601
- Std dev (us): 15.981
- Tracking iteration latency (us): 13188.246
- Change description: baseline zero-change run at 1M particles (runs=40 warmup=10)
- Delta vs baseline update+extract (us): +0.000
- Delta vs baseline update+extract (%): +0.000%
- Delta vs previous run update+extract (us): +0.000
- Delta vs previous run update+extract (%): +0.000%
- Running best before this run (us): 9024.601
- Delta vs running best before this run (us): +0.000
- Delta vs running best before this run (%): +0.000%
- New running best?: no
- Running best after this run (us): 9024.601
- 60 FPS budget share (update+extract): 54.148%
- 60 FPS headroom after update+extract (us): 7642.066
- Coefficient of variation (%): 0.177%
- Delta vs baseline tracking (us): +0.000
- Delta vs baseline tracking (%): +0.000%
- Delta vs previous run tracking (us): +0.000
- Delta vs previous run tracking (%): +0.000%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 002: `run-apr4-a-large-001`
- Timestamp: 2026-04-04T23:54:47-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 9018.958
- Std dev (us): 13.014
- Tracking iteration latency (us): 13009.189
- Change description: likelihood hot-path caching plus powf-to-mul in z-offset prediction
- Delta vs baseline update+extract (us): -5.643
- Delta vs baseline update+extract (%): -0.063%
- Delta vs previous run update+extract (us): -5.643
- Delta vs previous run update+extract (%): -0.063%
- Running best before this run (us): 9024.601
- Delta vs running best before this run (us): -5.643
- Delta vs running best before this run (%): -0.063%
- New running best?: yes
- Running best after this run (us): 9018.958
- 60 FPS budget share (update+extract): 54.114%
- 60 FPS headroom after update+extract (us): 7647.709
- Coefficient of variation (%): 0.144%
- Delta vs baseline tracking (us): -179.057
- Delta vs baseline tracking (%): -1.358%
- Delta vs previous run tracking (us): -179.057
- Delta vs previous run tracking (%): -1.358%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: likelihood hotpath

#### Timeline 003: `run-apr4-a-large-002`
- Timestamp: 2026-04-05T00:00:25-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 9017.437
- Std dev (us): 41.027
- Tracking iteration latency (us): 13034.218
- Change description: trig-reduced predicted_plates initial run had unacceptable std regression
- Delta vs baseline update+extract (us): -7.164
- Delta vs baseline update+extract (%): -0.079%
- Delta vs previous run update+extract (us): -1.521
- Delta vs previous run update+extract (%): -0.017%
- Running best before this run (us): 9018.958
- Delta vs running best before this run (us): -1.521
- Delta vs running best before this run (%): -0.017%
- New running best?: yes
- Running best after this run (us): 9017.437
- 60 FPS budget share (update+extract): 54.105%
- 60 FPS headroom after update+extract (us): 7649.230
- Coefficient of variation (%): 0.455%
- Delta vs baseline tracking (us): -154.028
- Delta vs baseline tracking (%): -1.168%
- Delta vs previous run tracking (us): +25.029
- Delta vs previous run tracking (%): +0.192%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: likelihood hotpath

#### Timeline 004: `run-apr4-a-large-002-rerun`
- Timestamp: 2026-04-05T00:00:25-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 9003.638
- Std dev (us): 8.513
- Tracking iteration latency (us): 13011.245
- Change description: trig-reduced predicted_plates rerun validated improvement and stability
- Delta vs baseline update+extract (us): -20.963
- Delta vs baseline update+extract (%): -0.232%
- Delta vs previous run update+extract (us): -13.799
- Delta vs previous run update+extract (%): -0.153%
- Running best before this run (us): 9017.437
- Delta vs running best before this run (us): -13.799
- Delta vs running best before this run (%): -0.153%
- New running best?: yes
- Running best after this run (us): 9003.638
- 60 FPS budget share (update+extract): 54.022%
- 60 FPS headroom after update+extract (us): 7663.029
- Coefficient of variation (%): 0.095%
- Delta vs baseline tracking (us): -177.001
- Delta vs baseline tracking (%): -1.342%
- Delta vs previous run tracking (us): -22.973
- Delta vs previous run tracking (%): -0.176%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: likelihood hotpath

#### Timeline 005: `run-apr4-a-large-003`
- Timestamp: 2026-04-05T00:04:43-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 9004.182
- Std dev (us): 19.302
- Tracking iteration latency (us): 13013.579
- Change description: manual reduction candidate selection regressed mean and std vs best kept state
- Delta vs baseline update+extract (us): -20.419
- Delta vs baseline update+extract (%): -0.226%
- Delta vs previous run update+extract (us): +0.544
- Delta vs previous run update+extract (%): +0.006%
- Running best before this run (us): 9003.638
- Delta vs running best before this run (us): +0.544
- Delta vs running best before this run (%): +0.006%
- New running best?: no
- Running best after this run (us): 9003.638
- 60 FPS budget share (update+extract): 54.025%
- 60 FPS headroom after update+extract (us): 7662.485
- Coefficient of variation (%): 0.214%
- Delta vs baseline tracking (us): -174.667
- Delta vs baseline tracking (%): -1.324%
- Delta vs previous run tracking (us): +2.334
- Delta vs previous run tracking (%): +0.018%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; state reduction

#### Timeline 006: `run-apr4-a-large-002-confirm`
- Timestamp: 2026-04-05T00:08:16-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 9004.233
- Std dev (us): 9.587
- Tracking iteration latency (us): 13040.349
- Change description: post-revert confirmation of kept large-001 plus large-002 state
- Delta vs baseline update+extract (us): -20.368
- Delta vs baseline update+extract (%): -0.226%
- Delta vs previous run update+extract (us): +0.051
- Delta vs previous run update+extract (%): +0.001%
- Running best before this run (us): 9003.638
- Delta vs running best before this run (us): +0.595
- Delta vs running best before this run (%): +0.007%
- New running best?: no
- Running best after this run (us): 9003.638
- 60 FPS budget share (update+extract): 54.025%
- 60 FPS headroom after update+extract (us): 7662.434
- Coefficient of variation (%): 0.106%
- Delta vs baseline tracking (us): -147.897
- Delta vs baseline tracking (%): -1.121%
- Delta vs previous run tracking (us): +26.770
- Delta vs previous run tracking (%): +0.206%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 007: `run-apr5-a-large-004`
- Timestamp: 2026-04-05T08:29:09-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7961.264
- Std dev (us): 15.950
- Tracking iteration latency (us): 11915.307
- Change description: position-only predicted plate path in likelihood; skip unused velocity math
- Delta vs baseline update+extract (us): -1063.337
- Delta vs baseline update+extract (%): -11.783%
- Delta vs previous run update+extract (us): -1042.969
- Delta vs previous run update+extract (%): -11.583%
- Running best before this run (us): 9003.638
- Delta vs running best before this run (us): -1042.374
- Delta vs running best before this run (%): -11.577%
- New running best?: yes
- Running best after this run (us): 7961.264
- 60 FPS budget share (update+extract): 47.768%
- 60 FPS headroom after update+extract (us): 8705.403
- Coefficient of variation (%): 0.200%
- Delta vs baseline tracking (us): -1272.939
- Delta vs baseline tracking (%): -9.652%
- Delta vs previous run tracking (us): -1125.042
- Delta vs previous run tracking (%): -8.627%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: likelihood hotpath; process model

#### Timeline 008: `run-apr5-a-large-004-rerun`
- Timestamp: 2026-04-05T08:29:09-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7979.017
- Std dev (us): 18.909
- Tracking iteration latency (us): 11932.073
- Change description: stability rerun for position-only predicted plate path
- Delta vs baseline update+extract (us): -1045.584
- Delta vs baseline update+extract (%): -11.586%
- Delta vs previous run update+extract (us): +17.753
- Delta vs previous run update+extract (%): +0.223%
- Running best before this run (us): 7961.264
- Delta vs running best before this run (us): +17.753
- Delta vs running best before this run (%): +0.223%
- New running best?: no
- Running best after this run (us): 7961.264
- 60 FPS budget share (update+extract): 47.874%
- 60 FPS headroom after update+extract (us): 8687.650
- Coefficient of variation (%): 0.237%
- Delta vs baseline tracking (us): -1256.173
- Delta vs baseline tracking (%): -9.525%
- Delta vs previous run tracking (us): +16.766
- Delta vs previous run tracking (%): +0.141%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: likelihood hotpath

#### Timeline 009: `run-apr5-a-large-005`
- Timestamp: 2026-04-05T08:34:17-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7960.478
- Std dev (us): 10.795
- Tracking iteration latency (us): 12070.884
- Change description: precomputed sigma values and sigma-based sampling to avoid repeated sqrt in process/sample paths
- Delta vs baseline update+extract (us): -1064.123
- Delta vs baseline update+extract (%): -11.791%
- Delta vs previous run update+extract (us): -18.539
- Delta vs previous run update+extract (%): -0.232%
- Running best before this run (us): 7961.264
- Delta vs running best before this run (us): -0.786
- Delta vs running best before this run (%): -0.010%
- New running best?: yes
- Running best after this run (us): 7960.478
- 60 FPS budget share (update+extract): 47.763%
- 60 FPS headroom after update+extract (us): 8706.189
- Coefficient of variation (%): 0.136%
- Delta vs baseline tracking (us): -1117.362
- Delta vs baseline tracking (%): -8.472%
- Delta vs previous run tracking (us): +138.811
- Delta vs previous run tracking (%): +1.163%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; process model

#### Timeline 010: `run-apr5-a-large-005-rerun`
- Timestamp: 2026-04-05T08:34:17-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7948.616
- Std dev (us): 19.532
- Tracking iteration latency (us): 11922.715
- Change description: stability rerun for sigma-precompute sampling change
- Delta vs baseline update+extract (us): -1075.985
- Delta vs baseline update+extract (%): -11.923%
- Delta vs previous run update+extract (us): -11.862
- Delta vs previous run update+extract (%): -0.149%
- Running best before this run (us): 7960.478
- Delta vs running best before this run (us): -11.862
- Delta vs running best before this run (%): -0.149%
- New running best?: yes
- Running best after this run (us): 7948.616
- 60 FPS budget share (update+extract): 47.692%
- 60 FPS headroom after update+extract (us): 8718.051
- Coefficient of variation (%): 0.246%
- Delta vs baseline tracking (us): -1265.531
- Delta vs baseline tracking (%): -9.596%
- Delta vs previous run tracking (us): -148.169
- Delta vs previous run tracking (%): -1.227%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; process model

#### Timeline 011: `run-apr5-a-large-006`
- Timestamp: 2026-04-05T08:39:42-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7945.282
- Std dev (us): 13.732
- Tracking iteration latency (us): 11910.530
- Change description: use precomputed diagonal precision in likelihood density evaluation
- Delta vs baseline update+extract (us): -1079.319
- Delta vs baseline update+extract (%): -11.960%
- Delta vs previous run update+extract (us): -3.334
- Delta vs previous run update+extract (%): -0.042%
- Running best before this run (us): 7948.616
- Delta vs running best before this run (us): -3.334
- Delta vs running best before this run (%): -0.042%
- New running best?: yes
- Running best after this run (us): 7945.282
- 60 FPS budget share (update+extract): 47.672%
- 60 FPS headroom after update+extract (us): 8721.385
- Coefficient of variation (%): 0.173%
- Delta vs baseline tracking (us): -1277.716
- Delta vs baseline tracking (%): -9.688%
- Delta vs previous run tracking (us): -12.185
- Delta vs previous run tracking (%): -0.102%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: likelihood hotpath

#### Timeline 012: `run-apr5-a-large-006-rerun`
- Timestamp: 2026-04-05T08:39:42-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7914.040
- Std dev (us): 10.746
- Tracking iteration latency (us): 11883.999
- Change description: stability rerun for precomputed precision likelihood optimization
- Delta vs baseline update+extract (us): -1110.561
- Delta vs baseline update+extract (%): -12.306%
- Delta vs previous run update+extract (us): -31.242
- Delta vs previous run update+extract (%): -0.393%
- Running best before this run (us): 7945.282
- Delta vs running best before this run (us): -31.242
- Delta vs running best before this run (%): -0.393%
- New running best?: yes
- Running best after this run (us): 7914.040
- 60 FPS budget share (update+extract): 47.484%
- 60 FPS headroom after update+extract (us): 8752.627
- Coefficient of variation (%): 0.136%
- Delta vs baseline tracking (us): -1304.247
- Delta vs baseline tracking (%): -9.889%
- Delta vs previous run tracking (us): -26.531
- Delta vs previous run tracking (%): -0.223%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: likelihood hotpath

#### Timeline 013: `run-apr5-a-large-007`
- Timestamp: 2026-04-05T08:49:43-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 8193.576
- Std dev (us): 9.697
- Tracking iteration latency (us): 12160.057
- Change description: cached observed-plate precision plus radius-based visibility denominator regressed update mean substantially
- Delta vs baseline update+extract (us): -831.025
- Delta vs baseline update+extract (%): -9.208%
- Delta vs previous run update+extract (us): +279.536
- Delta vs previous run update+extract (%): +3.532%
- Running best before this run (us): 7914.040
- Delta vs running best before this run (us): +279.536
- Delta vs running best before this run (%): +3.532%
- New running best?: no
- Running best after this run (us): 7914.040
- 60 FPS budget share (update+extract): 49.161%
- 60 FPS headroom after update+extract (us): 8473.091
- Coefficient of variation (%): 0.118%
- Delta vs baseline tracking (us): -1028.189
- Delta vs baseline tracking (%): -7.796%
- Delta vs previous run tracking (us): +276.058
- Delta vs previous run tracking (%): +2.323%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: likelihood hotpath

#### Timeline 014: `run-apr5-a-large-007-revert-confirm`
- Timestamp: 2026-04-05T08:49:43-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7930.893
- Std dev (us): 18.352
- Tracking iteration latency (us): 11890.238
- Change description: post-discard rollback confirmation near best kept large-006 state
- Delta vs baseline update+extract (us): -1093.708
- Delta vs baseline update+extract (%): -12.119%
- Delta vs previous run update+extract (us): -262.683
- Delta vs previous run update+extract (%): -3.206%
- Running best before this run (us): 7914.040
- Delta vs running best before this run (us): +16.853
- Delta vs running best before this run (%): +0.213%
- New running best?: no
- Running best after this run (us): 7914.040
- 60 FPS budget share (update+extract): 47.585%
- 60 FPS headroom after update+extract (us): 8735.774
- Coefficient of variation (%): 0.231%
- Delta vs baseline tracking (us): -1298.008
- Delta vs baseline tracking (%): -9.842%
- Delta vs previous run tracking (us): -269.819
- Delta vs previous run tracking (%): -2.219%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 015: `run-apr5-a-large-008`
- Timestamp: 2026-04-05T08:57:28-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7744.470
- Std dev (us): 10.112
- Tracking iteration latency (us): 14406.974
- Change description: deferred resample gather via index-carry improved update-only benchmark but caused major tracking-iteration regression
- Delta vs baseline update+extract (us): -1280.131
- Delta vs baseline update+extract (%): -14.185%
- Delta vs previous run update+extract (us): -186.423
- Delta vs previous run update+extract (%): -2.351%
- Running best before this run (us): 7914.040
- Delta vs running best before this run (us): -169.570
- Delta vs running best before this run (%): -2.143%
- New running best?: yes
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 46.467%
- 60 FPS headroom after update+extract (us): 8922.197
- Coefficient of variation (%): 0.131%
- Delta vs baseline tracking (us): +1218.728
- Delta vs baseline tracking (%): +9.241%
- Delta vs previous run tracking (us): +2516.736
- Delta vs previous run tracking (%): +21.166%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: memory layout + resampling

#### Timeline 016: `run-apr5-a-large-008-revert-confirm`
- Timestamp: 2026-04-05T09:01:45-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7935.735
- Std dev (us): 9.205
- Tracking iteration latency (us): 11891.010
- Change description: post-discard rollback confirmation restored balanced update and tracking metrics
- Delta vs baseline update+extract (us): -1088.866
- Delta vs baseline update+extract (%): -12.066%
- Delta vs previous run update+extract (us): +191.265
- Delta vs previous run update+extract (%): +2.470%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +191.265
- Delta vs running best before this run (%): +2.470%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.614%
- 60 FPS headroom after update+extract (us): 8730.932
- Coefficient of variation (%): 0.116%
- Delta vs baseline tracking (us): -1297.236
- Delta vs baseline tracking (%): -9.836%
- Delta vs previous run tracking (us): -2515.964
- Delta vs previous run tracking (%): -17.464%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 017: `run-apr5-a-large-009`
- Timestamp: 2026-04-05T09:06:10-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7917.926
- Std dev (us): 14.004
- Tracking iteration latency (us): 11862.867
- Change description: reduced configuration capture footprint was close but did not improve update mean versus best kept state
- Delta vs baseline update+extract (us): -1106.675
- Delta vs baseline update+extract (%): -12.263%
- Delta vs previous run update+extract (us): -17.809
- Delta vs previous run update+extract (%): -0.224%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +173.456
- Delta vs running best before this run (%): +2.240%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.508%
- 60 FPS headroom after update+extract (us): 8748.741
- Coefficient of variation (%): 0.177%
- Delta vs baseline tracking (us): -1325.379
- Delta vs baseline tracking (%): -10.050%
- Delta vs previous run tracking (us): -28.143
- Delta vs previous run tracking (%): -0.237%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 018: `run-apr5-a-large-009-rerun`
- Timestamp: 2026-04-05T09:06:10-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7922.998
- Std dev (us): 20.552
- Tracking iteration latency (us): 11872.892
- Change description: stability rerun confirmed no primary-metric win over best kept configuration
- Delta vs baseline update+extract (us): -1101.603
- Delta vs baseline update+extract (%): -12.207%
- Delta vs previous run update+extract (us): +5.072
- Delta vs previous run update+extract (%): +0.064%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +178.528
- Delta vs running best before this run (%): +2.305%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.538%
- 60 FPS headroom after update+extract (us): 8743.669
- Coefficient of variation (%): 0.259%
- Delta vs baseline tracking (us): -1315.354
- Delta vs baseline tracking (%): -9.974%
- Delta vs previous run tracking (us): +10.025
- Delta vs previous run tracking (%): +0.085%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 019: `run-apr5-a-large-009-revert-confirm`
- Timestamp: 2026-04-05T09:08:54-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7918.187
- Std dev (us): 8.597
- Tracking iteration latency (us): 11860.723
- Change description: post-discard rollback confirmation remained near best update mean with strong stability
- Delta vs baseline update+extract (us): -1106.414
- Delta vs baseline update+extract (%): -12.260%
- Delta vs previous run update+extract (us): -4.811
- Delta vs previous run update+extract (%): -0.061%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +173.717
- Delta vs running best before this run (%): +2.243%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.509%
- 60 FPS headroom after update+extract (us): 8748.480
- Coefficient of variation (%): 0.109%
- Delta vs baseline tracking (us): -1327.523
- Delta vs baseline tracking (%): -10.066%
- Delta vs previous run tracking (us): -12.169
- Delta vs previous run tracking (%): -0.102%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 020: `run-apr5-a-large-010`
- Timestamp: 2026-04-05T09:12:49-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7902.611
- Std dev (us): 14.521
- Tracking iteration latency (us): 11843.571
- Change description: scalarized orientation-candidate reduction looked promising initially but did not hold across reruns
- Delta vs baseline update+extract (us): -1121.990
- Delta vs baseline update+extract (%): -12.433%
- Delta vs previous run update+extract (us): -15.576
- Delta vs previous run update+extract (%): -0.197%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +158.141
- Delta vs running best before this run (%): +2.042%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.416%
- 60 FPS headroom after update+extract (us): 8764.056
- Coefficient of variation (%): 0.184%
- Delta vs baseline tracking (us): -1344.675
- Delta vs baseline tracking (%): -10.196%
- Delta vs previous run tracking (us): -17.152
- Delta vs previous run tracking (%): -0.145%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; state reduction

#### Timeline 021: `run-apr5-a-large-010-rerun`
- Timestamp: 2026-04-05T09:12:49-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7940.276
- Std dev (us): 83.700
- Tracking iteration latency (us): 11873.196
- Change description: first stability rerun showed large variance and mean regression
- Delta vs baseline update+extract (us): -1084.325
- Delta vs baseline update+extract (%): -12.015%
- Delta vs previous run update+extract (us): +37.665
- Delta vs previous run update+extract (%): +0.477%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +195.806
- Delta vs running best before this run (%): +2.528%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.642%
- 60 FPS headroom after update+extract (us): 8726.391
- Coefficient of variation (%): 1.054%
- Delta vs baseline tracking (us): -1315.050
- Delta vs baseline tracking (%): -9.971%
- Delta vs previous run tracking (us): +29.625
- Delta vs previous run tracking (%): +0.250%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Moderately noisy (0.70% <= CV < 1.50%).
- Likely subsystem attribution: mixed

#### Timeline 022: `run-apr5-a-large-010-rerun2`
- Timestamp: 2026-04-05T09:12:49-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7913.665
- Std dev (us): 30.540
- Tracking iteration latency (us): 11847.382
- Change description: second rerun returned near-best mean but still elevated variance
- Delta vs baseline update+extract (us): -1110.936
- Delta vs baseline update+extract (%): -12.310%
- Delta vs previous run update+extract (us): -26.611
- Delta vs previous run update+extract (%): -0.335%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +169.195
- Delta vs running best before this run (%): +2.185%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.482%
- 60 FPS headroom after update+extract (us): 8753.002
- Coefficient of variation (%): 0.386%
- Delta vs baseline tracking (us): -1340.864
- Delta vs baseline tracking (%): -10.167%
- Delta vs previous run tracking (us): -25.814
- Delta vs previous run tracking (%): -0.217%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: mixed

#### Timeline 023: `run-apr5-a-large-010-rerun3`
- Timestamp: 2026-04-05T09:12:49-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7920.029
- Std dev (us): 23.129
- Tracking iteration latency (us): 11883.920
- Change description: fourth sample did not confirm consistent primary-metric gain
- Delta vs baseline update+extract (us): -1104.572
- Delta vs baseline update+extract (%): -12.240%
- Delta vs previous run update+extract (us): +6.364
- Delta vs previous run update+extract (%): +0.080%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +175.559
- Delta vs running best before this run (%): +2.267%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.520%
- 60 FPS headroom after update+extract (us): 8746.638
- Coefficient of variation (%): 0.292%
- Delta vs baseline tracking (us): -1304.326
- Delta vs baseline tracking (%): -9.890%
- Delta vs previous run tracking (us): +36.538
- Delta vs previous run tracking (%): +0.308%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 024: `run-apr5-a-large-010-revert-confirm`
- Timestamp: 2026-04-05T09:15:42-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7915.645
- Std dev (us): 8.086
- Tracking iteration latency (us): 11860.152
- Change description: post-discard rollback confirmation returned to stable near-best performance
- Delta vs baseline update+extract (us): -1108.956
- Delta vs baseline update+extract (%): -12.288%
- Delta vs previous run update+extract (us): -4.384
- Delta vs previous run update+extract (%): -0.055%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +171.175
- Delta vs running best before this run (%): +2.210%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.494%
- 60 FPS headroom after update+extract (us): 8751.022
- Coefficient of variation (%): 0.102%
- Delta vs baseline tracking (us): -1328.094
- Delta vs baseline tracking (%): -10.070%
- Delta vs previous run tracking (us): -23.768
- Delta vs previous run tracking (%): -0.200%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 025: `run-apr5-a-large-011`
- Timestamp: 2026-04-05T09:19:06-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7939.820
- Std dev (us): 11.529
- Tracking iteration latency (us): 11860.199
- Change description: fast intrinsic exp/log substitutions regressed update mean without tracking gains
- Delta vs baseline update+extract (us): -1084.781
- Delta vs baseline update+extract (%): -12.020%
- Delta vs previous run update+extract (us): +24.175
- Delta vs previous run update+extract (%): +0.305%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +195.350
- Delta vs running best before this run (%): +2.522%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.639%
- 60 FPS headroom after update+extract (us): 8726.847
- Coefficient of variation (%): 0.145%
- Delta vs baseline tracking (us): -1328.047
- Delta vs baseline tracking (%): -10.070%
- Delta vs previous run tracking (us): +0.047
- Delta vs previous run tracking (%): +0.000%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 026: `run-apr5-a-large-011-revert-confirm`
- Timestamp: 2026-04-05T09:22:05-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7927.302
- Std dev (us): 17.538
- Tracking iteration latency (us): 11867.655
- Change description: post-discard rollback confirmation restored prior behavior envelope
- Delta vs baseline update+extract (us): -1097.299
- Delta vs baseline update+extract (%): -12.159%
- Delta vs previous run update+extract (us): -12.518
- Delta vs previous run update+extract (%): -0.158%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +182.832
- Delta vs running best before this run (%): +2.361%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.564%
- 60 FPS headroom after update+extract (us): 8739.365
- Coefficient of variation (%): 0.221%
- Delta vs baseline tracking (us): -1320.591
- Delta vs baseline tracking (%): -10.013%
- Delta vs previous run tracking (us): +7.456
- Delta vs previous run tracking (%): +0.063%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 027: `run-apr5-a-large-012`
- Timestamp: 2026-04-05T09:25:26-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7920.587
- Std dev (us): 12.685
- Tracking iteration latency (us): 11853.235
- Change description: scalarized prediction center math improved tracking but regressed primary update mean
- Delta vs baseline update+extract (us): -1104.014
- Delta vs baseline update+extract (%): -12.233%
- Delta vs previous run update+extract (us): -6.715
- Delta vs previous run update+extract (%): -0.085%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +176.117
- Delta vs running best before this run (%): +2.274%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.524%
- 60 FPS headroom after update+extract (us): 8746.080
- Coefficient of variation (%): 0.160%
- Delta vs baseline tracking (us): -1335.011
- Delta vs baseline tracking (%): -10.123%
- Delta vs previous run tracking (us): -14.420
- Delta vs previous run tracking (%): -0.122%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 028: `run-apr5-a-large-012-rerun`
- Timestamp: 2026-04-05T09:25:26-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7920.060
- Std dev (us): 19.214
- Tracking iteration latency (us): 11843.383
- Change description: rerun confirmed persistent primary-metric regression despite secondary improvement
- Delta vs baseline update+extract (us): -1104.541
- Delta vs baseline update+extract (%): -12.239%
- Delta vs previous run update+extract (us): -0.527
- Delta vs previous run update+extract (%): -0.007%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +175.590
- Delta vs running best before this run (%): +2.267%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.520%
- 60 FPS headroom after update+extract (us): 8746.607
- Coefficient of variation (%): 0.243%
- Delta vs baseline tracking (us): -1344.863
- Delta vs baseline tracking (%): -10.197%
- Delta vs previous run tracking (us): -9.852
- Delta vs previous run tracking (%): -0.083%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 029: `run-apr5-a-large-012-revert-confirm`
- Timestamp: 2026-04-05T09:28:21-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7936.869
- Std dev (us): 30.235
- Tracking iteration latency (us): 11861.385
- Change description: post-discard rollback confirmation re-established prior implementation state
- Delta vs baseline update+extract (us): -1087.732
- Delta vs baseline update+extract (%): -12.053%
- Delta vs previous run update+extract (us): +16.809
- Delta vs previous run update+extract (%): +0.212%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +192.399
- Delta vs running best before this run (%): +2.484%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.621%
- 60 FPS headroom after update+extract (us): 8729.798
- Coefficient of variation (%): 0.381%
- Delta vs baseline tracking (us): -1326.861
- Delta vs baseline tracking (%): -10.061%
- Delta vs previous run tracking (us): +18.002
- Delta vs previous run tracking (%): +0.152%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: mixed

#### Timeline 030: `run-apr5-a-large-013`
- Timestamp: 2026-04-05T09:32:15-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7915.076
- Std dev (us): 9.589
- Tracking iteration latency (us): 11852.706
- Change description: 2D sampler overloads showed mixed results and no stable primary improvement
- Delta vs baseline update+extract (us): -1109.525
- Delta vs baseline update+extract (%): -12.294%
- Delta vs previous run update+extract (us): -21.793
- Delta vs previous run update+extract (%): -0.275%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +170.606
- Delta vs running best before this run (%): +2.203%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.490%
- 60 FPS headroom after update+extract (us): 8751.591
- Coefficient of variation (%): 0.121%
- Delta vs baseline tracking (us): -1335.540
- Delta vs baseline tracking (%): -10.127%
- Delta vs previous run tracking (us): -8.679
- Delta vs previous run tracking (%): -0.073%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; random generation

#### Timeline 031: `run-apr5-a-large-013-rerun`
- Timestamp: 2026-04-05T09:32:15-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7939.811
- Std dev (us): 10.958
- Tracking iteration latency (us): 11894.723
- Change description: first rerun regressed both primary and secondary metrics
- Delta vs baseline update+extract (us): -1084.790
- Delta vs baseline update+extract (%): -12.020%
- Delta vs previous run update+extract (us): +24.735
- Delta vs previous run update+extract (%): +0.313%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +195.341
- Delta vs running best before this run (%): +2.522%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.639%
- 60 FPS headroom after update+extract (us): 8726.856
- Coefficient of variation (%): 0.138%
- Delta vs baseline tracking (us): -1293.523
- Delta vs baseline tracking (%): -9.808%
- Delta vs previous run tracking (us): +42.017
- Delta vs previous run tracking (%): +0.354%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 032: `run-apr5-a-large-013-rerun2`
- Timestamp: 2026-04-05T09:32:15-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7909.380
- Std dev (us): 7.758
- Tracking iteration latency (us): 11848.029
- Change description: second rerun was strong but contradicted prior sample; stability unresolved
- Delta vs baseline update+extract (us): -1115.221
- Delta vs baseline update+extract (%): -12.358%
- Delta vs previous run update+extract (us): -30.431
- Delta vs previous run update+extract (%): -0.383%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +164.910
- Delta vs running best before this run (%): +2.129%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.456%
- 60 FPS headroom after update+extract (us): 8757.287
- Coefficient of variation (%): 0.098%
- Delta vs baseline tracking (us): -1340.217
- Delta vs baseline tracking (%): -10.162%
- Delta vs previous run tracking (us): -46.694
- Delta vs previous run tracking (%): -0.393%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 033: `run-apr5-a-large-013-rerun3`
- Timestamp: 2026-04-05T09:32:15-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7915.301
- Std dev (us): 18.595
- Tracking iteration latency (us): 11889.972
- Change description: fourth sample failed to establish consistent gain trend
- Delta vs baseline update+extract (us): -1109.300
- Delta vs baseline update+extract (%): -12.292%
- Delta vs previous run update+extract (us): +5.921
- Delta vs previous run update+extract (%): +0.075%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +170.831
- Delta vs running best before this run (%): +2.206%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.492%
- 60 FPS headroom after update+extract (us): 8751.366
- Coefficient of variation (%): 0.235%
- Delta vs baseline tracking (us): -1298.274
- Delta vs baseline tracking (%): -9.844%
- Delta vs previous run tracking (us): +41.943
- Delta vs previous run tracking (%): +0.354%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 034: `run-apr5-a-large-013-revert-confirm`
- Timestamp: 2026-04-05T09:35:25-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7982.756
- Std dev (us): 41.034
- Tracking iteration latency (us): 11973.389
- Change description: initial rollback confirmation produced transient high-variance outlier
- Delta vs baseline update+extract (us): -1041.845
- Delta vs baseline update+extract (%): -11.544%
- Delta vs previous run update+extract (us): +67.455
- Delta vs previous run update+extract (%): +0.852%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +238.286
- Delta vs running best before this run (%): +3.077%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.897%
- 60 FPS headroom after update+extract (us): 8683.911
- Coefficient of variation (%): 0.514%
- Delta vs baseline tracking (us): -1214.857
- Delta vs baseline tracking (%): -9.212%
- Delta vs previous run tracking (us): +83.417
- Delta vs previous run tracking (%): +0.702%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: mixed

#### Timeline 035: `run-apr5-a-large-013-revert-confirm-rerun`
- Timestamp: 2026-04-05T09:35:25-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7917.055
- Std dev (us): 10.793
- Tracking iteration latency (us): 11887.755
- Change description: follow-up rollback confirmation returned to expected near-best range
- Delta vs baseline update+extract (us): -1107.546
- Delta vs baseline update+extract (%): -12.273%
- Delta vs previous run update+extract (us): -65.701
- Delta vs previous run update+extract (%): -0.823%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +172.585
- Delta vs running best before this run (%): +2.228%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.502%
- 60 FPS headroom after update+extract (us): 8749.612
- Coefficient of variation (%): 0.136%
- Delta vs baseline tracking (us): -1300.491
- Delta vs baseline tracking (%): -9.861%
- Delta vs previous run tracking (us): -85.634
- Delta vs previous run tracking (%): -0.715%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 036: `run-apr5-a-large-014`
- Timestamp: 2026-04-05T09:38:52-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 9381.641
- Std dev (us): 13.107
- Tracking iteration latency (us): 15134.637
- Change description: switching default sampler RNG engine to taus88 caused major regressions in update and tracking paths
- Delta vs baseline update+extract (us): +357.040
- Delta vs baseline update+extract (%): +3.956%
- Delta vs previous run update+extract (us): +1464.586
- Delta vs previous run update+extract (%): +18.499%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +1637.171
- Delta vs running best before this run (%): +21.140%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 56.290%
- 60 FPS headroom after update+extract (us): 7285.026
- Coefficient of variation (%): 0.140%
- Delta vs baseline tracking (us): +1946.391
- Delta vs baseline tracking (%): +14.759%
- Delta vs previous run tracking (us): +3246.882
- Delta vs previous run tracking (%): +27.313%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; random generation

#### Timeline 037: `run-apr5-a-large-014-revert-confirm`
- Timestamp: 2026-04-05T09:41:45-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7920.722
- Std dev (us): 9.873
- Tracking iteration latency (us): 11861.507
- Change description: rollback to default RNG restored expected performance envelope
- Delta vs baseline update+extract (us): -1103.879
- Delta vs baseline update+extract (%): -12.232%
- Delta vs previous run update+extract (us): -1460.919
- Delta vs previous run update+extract (%): -15.572%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +176.252
- Delta vs running best before this run (%): +2.276%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.524%
- 60 FPS headroom after update+extract (us): 8745.945
- Coefficient of variation (%): 0.125%
- Delta vs baseline tracking (us): -1326.739
- Delta vs baseline tracking (%): -10.060%
- Delta vs previous run tracking (us): -3273.130
- Delta vs previous run tracking (%): -21.627%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; random generation

#### Timeline 038: `run-apr5-a-large-015`
- Timestamp: 2026-04-05T09:53:57-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7920.874
- Std dev (us): 15.130
- Tracking iteration latency (us): 11872.503
- Change description: precompute initial orbit for sample_from via config precompute path regressed update mean
- Delta vs baseline update+extract (us): -1103.727
- Delta vs baseline update+extract (%): -12.230%
- Delta vs previous run update+extract (us): +0.152
- Delta vs previous run update+extract (%): +0.002%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +176.404
- Delta vs running best before this run (%): +2.278%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.525%
- 60 FPS headroom after update+extract (us): 8745.793
- Coefficient of variation (%): 0.191%
- Delta vs baseline tracking (us): -1315.743
- Delta vs baseline tracking (%): -9.977%
- Delta vs previous run tracking (us): +10.996
- Delta vs previous run tracking (%): +0.093%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 039: `run-apr5-a-large-015-revert-confirm`
- Timestamp: 2026-04-05T09:57:04-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7915.191
- Std dev (us): 14.040
- Tracking iteration latency (us): 11882.862
- Change description: rollback confirmation after discarded precomputed-initial-orbit change returned near best state
- Delta vs baseline update+extract (us): -1109.410
- Delta vs baseline update+extract (%): -12.293%
- Delta vs previous run update+extract (us): -5.683
- Delta vs previous run update+extract (%): -0.072%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +170.721
- Delta vs running best before this run (%): +2.204%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.491%
- 60 FPS headroom after update+extract (us): 8751.476
- Coefficient of variation (%): 0.177%
- Delta vs baseline tracking (us): -1305.384
- Delta vs baseline tracking (%): -9.898%
- Delta vs previous run tracking (us): +10.359
- Delta vs previous run tracking (%): +0.087%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 040: `run-apr5-a-large-016`
- Timestamp: 2026-04-05T10:00:06-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 21074.779
- Std dev (us): 73.901
- Tracking iteration latency (us): 25234.146
- Change description: metropolis-style resampler replacement massively regressed update and tracking metrics
- Delta vs baseline update+extract (us): +12050.178
- Delta vs baseline update+extract (%): +133.526%
- Delta vs previous run update+extract (us): +13159.588
- Delta vs previous run update+extract (%): +166.257%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +13330.309
- Delta vs running best before this run (%): +172.127%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 126.449%
- 60 FPS headroom after update+extract (us): -4408.112
- Coefficient of variation (%): 0.351%
- Delta vs baseline tracking (us): +12045.900
- Delta vs baseline tracking (%): +91.338%
- Delta vs previous run tracking (us): +13351.284
- Delta vs previous run tracking (%): +112.357%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: memory layout + resampling; random generation

#### Timeline 041: `run-apr5-a-large-016-revert-confirm`
- Timestamp: 2026-04-05T10:03:12-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 8184.371
- Std dev (us): 350.099
- Tracking iteration latency (us): 12162.989
- Change description: initial rollback confirmation after discarded metropolis resampler showed transient high-variance outlier
- Delta vs baseline update+extract (us): -840.230
- Delta vs baseline update+extract (%): -9.310%
- Delta vs previous run update+extract (us): -12890.408
- Delta vs previous run update+extract (%): -61.165%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +439.901
- Delta vs running best before this run (%): +5.680%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 49.106%
- 60 FPS headroom after update+extract (us): 8482.296
- Coefficient of variation (%): 4.278%
- Delta vs baseline tracking (us): -1025.257
- Delta vs baseline tracking (%): -7.774%
- Delta vs previous run tracking (us): -13071.157
- Delta vs previous run tracking (%): -51.799%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: High variance (CV >= 1.50%); requires caution.
- Likely subsystem attribution: memory layout + resampling; random generation

#### Timeline 042: `run-apr5-a-large-016-revert-confirm-rerun`
- Timestamp: 2026-04-05T10:03:12-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7917.954
- Std dev (us): 9.072
- Tracking iteration latency (us): 11872.802
- Change description: follow-up rollback confirmation restored expected performance envelope after reverting metropolis resampler
- Delta vs baseline update+extract (us): -1106.647
- Delta vs baseline update+extract (%): -12.263%
- Delta vs previous run update+extract (us): -266.417
- Delta vs previous run update+extract (%): -3.255%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +173.484
- Delta vs running best before this run (%): +2.240%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.508%
- 60 FPS headroom after update+extract (us): 8748.713
- Coefficient of variation (%): 0.115%
- Delta vs baseline tracking (us): -1315.444
- Delta vs baseline tracking (%): -9.974%
- Delta vs previous run tracking (us): -290.187
- Delta vs previous run tracking (%): -2.386%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: memory layout + resampling; random generation

#### Timeline 043: `run-apr5-a-large-017`
- Timestamp: 2026-04-05T10:06:26-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7957.376
- Std dev (us): 28.855
- Tracking iteration latency (us): 11885.471
- Change description: per-seed initializer seeding (no discard sequence walk) regressed update mean and variance
- Delta vs baseline update+extract (us): -1067.225
- Delta vs baseline update+extract (%): -11.826%
- Delta vs previous run update+extract (us): +39.422
- Delta vs previous run update+extract (%): +0.498%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +212.906
- Delta vs running best before this run (%): +2.749%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.744%
- 60 FPS headroom after update+extract (us): 8709.291
- Coefficient of variation (%): 0.363%
- Delta vs baseline tracking (us): -1302.775
- Delta vs baseline tracking (%): -9.878%
- Delta vs previous run tracking (us): +12.669
- Delta vs previous run tracking (%): +0.107%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: mixed; random generation

#### Timeline 044: `run-apr5-a-large-017-revert-confirm`
- Timestamp: 2026-04-05T10:09:19-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 8124.399
- Std dev (us): 91.005
- Tracking iteration latency (us): 12045.017
- Change description: initial rollback confirmation after discarded per-seed initializer produced transient high-variance outlier
- Delta vs baseline update+extract (us): -900.202
- Delta vs baseline update+extract (%): -9.975%
- Delta vs previous run update+extract (us): +167.023
- Delta vs previous run update+extract (%): +2.099%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +379.929
- Delta vs running best before this run (%): +4.906%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 48.746%
- 60 FPS headroom after update+extract (us): 8542.268
- Coefficient of variation (%): 1.120%
- Delta vs baseline tracking (us): -1143.229
- Delta vs baseline tracking (%): -8.669%
- Delta vs previous run tracking (us): +159.546
- Delta vs previous run tracking (%): +1.342%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Moderately noisy (0.70% <= CV < 1.50%).
- Likely subsystem attribution: mixed; random generation

#### Timeline 045: `run-apr5-a-large-017-revert-confirm-rerun`
- Timestamp: 2026-04-05T10:09:19-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7918.515
- Std dev (us): 11.766
- Tracking iteration latency (us): 11869.010
- Change description: follow-up rollback confirmation restored expected near-best behavior after reverting seeding change
- Delta vs baseline update+extract (us): -1106.086
- Delta vs baseline update+extract (%): -12.256%
- Delta vs previous run update+extract (us): -205.884
- Delta vs previous run update+extract (%): -2.534%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +174.045
- Delta vs running best before this run (%): +2.247%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.511%
- 60 FPS headroom after update+extract (us): 8748.152
- Coefficient of variation (%): 0.149%
- Delta vs baseline tracking (us): -1319.236
- Delta vs baseline tracking (%): -10.003%
- Delta vs previous run tracking (us): -176.007
- Delta vs previous run tracking (%): -1.461%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; random generation

#### Timeline 046: `run-apr5-a-large-018`
- Timestamp: 2026-04-05T10:13:35-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 14457.872
- Std dev (us): 598.505
- Tracking iteration latency (us): 25600.623
- Change description: philox-backed sampler engine with thrust normal distribution caused severe update and tracking regressions
- Delta vs baseline update+extract (us): +5433.271
- Delta vs baseline update+extract (%): +60.205%
- Delta vs previous run update+extract (us): +6539.357
- Delta vs previous run update+extract (%): +82.583%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +6713.402
- Delta vs running best before this run (%): +86.686%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 86.747%
- 60 FPS headroom after update+extract (us): 2208.795
- Coefficient of variation (%): 4.140%
- Delta vs baseline tracking (us): +12412.377
- Delta vs baseline tracking (%): +94.117%
- Delta vs previous run tracking (us): +13731.613
- Delta vs previous run tracking (%): +115.693%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: High variance (CV >= 1.50%); requires caution.
- Likely subsystem attribution: mixed; random generation

#### Timeline 047: `run-apr5-a-large-018-revert-confirm`
- Timestamp: 2026-04-05T10:16:21-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7926.409
- Std dev (us): 9.782
- Tracking iteration latency (us): 11885.570
- Change description: rollback confirmation after discarded philox sampler experiment restored expected performance envelope
- Delta vs baseline update+extract (us): -1098.192
- Delta vs baseline update+extract (%): -12.169%
- Delta vs previous run update+extract (us): -6531.463
- Delta vs previous run update+extract (%): -45.176%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +181.939
- Delta vs running best before this run (%): +2.349%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.558%
- 60 FPS headroom after update+extract (us): 8740.258
- Coefficient of variation (%): 0.123%
- Delta vs baseline tracking (us): -1302.676
- Delta vs baseline tracking (%): -9.878%
- Delta vs previous run tracking (us): -13715.053
- Delta vs previous run tracking (%): -53.573%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; random generation

#### Timeline 048: `run-apr5-a-large-019`
- Timestamp: 2026-04-05T10:23:05-07:00
- Phase: large
- Status: crash
- Mean update+extract latency (us): 0.000
- Std dev (us): 0.000
- Tracking iteration latency (us): 0.000
- Change description: constant-memory params experiment compiled but crashed during particle filter construction
- Delta vs baseline update+extract (us): n/a
- Delta vs baseline update+extract (%): n/a
- Delta vs previous run update+extract (us): n/a
- Delta vs previous run update+extract (%): n/a
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): n/a
- Delta vs running best before this run (%): n/a
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): n/a
- 60 FPS headroom after update+extract (us): n/a
- Coefficient of variation (%): n/a
- Delta vs baseline tracking (us): n/a
- Delta vs baseline tracking (%): n/a
- Delta vs previous run tracking (us): n/a
- Delta vs previous run tracking (%): n/a
- Keep/discard/crash interpretation: Run crashed before valid timing; treat as functional failure, not a performance datapoint.
- Stability classification: No stability classification (crash).
- Likely subsystem attribution: mixed

#### Timeline 049: `run-apr5-a-large-019-revert-confirm`
- Timestamp: 2026-04-05T10:25:56-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7939.844
- Std dev (us): 16.242
- Tracking iteration latency (us): 11864.987
- Change description: rollback confirmation after crashing constant-memory params experiment restored stable execution
- Delta vs baseline update+extract (us): -1084.757
- Delta vs baseline update+extract (%): -12.020%
- Delta vs previous run update+extract (us): +13.435
- Delta vs previous run update+extract (%): +0.169%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +195.374
- Delta vs running best before this run (%): +2.523%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.639%
- 60 FPS headroom after update+extract (us): 8726.823
- Coefficient of variation (%): 0.205%
- Delta vs baseline tracking (us): -1323.259
- Delta vs baseline tracking (%): -10.034%
- Delta vs previous run tracking (us): -20.583
- Delta vs previous run tracking (%): -0.173%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 050: `run-apr5-a-large-020`
- Timestamp: 2026-04-05T10:31:12-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7917.406
- Std dev (us): 6.984
- Tracking iteration latency (us): 11861.983
- Change description: cub DeviceReduce with preallocated temp storage improved variance but did not beat best update mean
- Delta vs baseline update+extract (us): -1107.195
- Delta vs baseline update+extract (%): -12.269%
- Delta vs previous run update+extract (us): -22.438
- Delta vs previous run update+extract (%): -0.283%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +172.936
- Delta vs running best before this run (%): +2.233%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.504%
- 60 FPS headroom after update+extract (us): 8749.261
- Coefficient of variation (%): 0.088%
- Delta vs baseline tracking (us): -1326.263
- Delta vs baseline tracking (%): -10.056%
- Delta vs previous run tracking (us): -3.004
- Delta vs previous run tracking (%): -0.025%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; state reduction

#### Timeline 051: `run-apr5-a-large-020-rerun`
- Timestamp: 2026-04-05T10:31:12-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7931.157
- Std dev (us): 26.068
- Tracking iteration latency (us): 11874.324
- Change description: rerun confirmed no stable primary-metric improvement for cub-based most-likely reduction
- Delta vs baseline update+extract (us): -1093.444
- Delta vs baseline update+extract (%): -12.116%
- Delta vs previous run update+extract (us): +13.751
- Delta vs previous run update+extract (%): +0.174%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +186.687
- Delta vs running best before this run (%): +2.411%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.587%
- 60 FPS headroom after update+extract (us): 8735.510
- Coefficient of variation (%): 0.329%
- Delta vs baseline tracking (us): -1313.922
- Delta vs baseline tracking (%): -9.963%
- Delta vs previous run tracking (us): +12.341
- Delta vs previous run tracking (%): +0.104%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: mixed; state reduction

#### Timeline 052: `run-apr5-a-large-020-revert-confirm`
- Timestamp: 2026-04-05T10:34:31-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7926.489
- Std dev (us): 19.589
- Tracking iteration latency (us): 11902.792
- Change description: rollback confirmation after discarded cub reduction experiment restored expected range
- Delta vs baseline update+extract (us): -1098.112
- Delta vs baseline update+extract (%): -12.168%
- Delta vs previous run update+extract (us): -4.668
- Delta vs previous run update+extract (%): -0.059%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): +182.019
- Delta vs running best before this run (%): +2.350%
- New running best?: no
- Running best after this run (us): 7744.470
- 60 FPS budget share (update+extract): 47.559%
- 60 FPS headroom after update+extract (us): 8740.178
- Coefficient of variation (%): 0.247%
- Delta vs baseline tracking (us): -1285.454
- Delta vs baseline tracking (%): -9.747%
- Delta vs previous run tracking (us): +28.468
- Delta vs previous run tracking (%): +0.240%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; state reduction

#### Timeline 053: `run-apr5-a-large-021`
- Timestamp: 2026-04-05T10:38:26-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7730.745
- Std dev (us): 8.660
- Tracking iteration latency (us): 11715.057
- Change description: fused log-sum-exp normalization removed intermediate particle_weights pass and reduced resampler passes
- Delta vs baseline update+extract (us): -1293.856
- Delta vs baseline update+extract (%): -14.337%
- Delta vs previous run update+extract (us): -195.744
- Delta vs previous run update+extract (%): -2.469%
- Running best before this run (us): 7744.470
- Delta vs running best before this run (us): -13.725
- Delta vs running best before this run (%): -0.177%
- New running best?: yes
- Running best after this run (us): 7730.745
- 60 FPS budget share (update+extract): 46.384%
- 60 FPS headroom after update+extract (us): 8935.922
- Coefficient of variation (%): 0.112%
- Delta vs baseline tracking (us): -1473.189
- Delta vs baseline tracking (%): -11.170%
- Delta vs previous run tracking (us): -187.735
- Delta vs previous run tracking (%): -1.577%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: memory layout + resampling; random generation

#### Timeline 054: `run-apr5-a-large-021-rerun`
- Timestamp: 2026-04-05T10:38:40-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7770.221
- Std dev (us): 118.155
- Tracking iteration latency (us): 11716.554
- Change description: first stability rerun retained major mean gain but showed elevated variance
- Delta vs baseline update+extract (us): -1254.380
- Delta vs baseline update+extract (%): -13.900%
- Delta vs previous run update+extract (us): +39.476
- Delta vs previous run update+extract (%): +0.511%
- Running best before this run (us): 7730.745
- Delta vs running best before this run (us): +39.476
- Delta vs running best before this run (%): +0.511%
- New running best?: no
- Running best after this run (us): 7730.745
- 60 FPS budget share (update+extract): 46.621%
- 60 FPS headroom after update+extract (us): 8896.446
- Coefficient of variation (%): 1.521%
- Delta vs baseline tracking (us): -1471.692
- Delta vs baseline tracking (%): -11.159%
- Delta vs previous run tracking (us): +1.497
- Delta vs previous run tracking (%): +0.013%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: High variance (CV >= 1.50%); requires caution.
- Likely subsystem attribution: mixed

#### Timeline 055: `run-apr5-a-large-021-rerun2`
- Timestamp: 2026-04-05T10:38:53-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7735.991
- Std dev (us): 19.153
- Tracking iteration latency (us): 11693.775
- Change description: second stability rerun confirmed strong low-variance gain for fused normalization
- Delta vs baseline update+extract (us): -1288.610
- Delta vs baseline update+extract (%): -14.279%
- Delta vs previous run update+extract (us): -34.230
- Delta vs previous run update+extract (%): -0.441%
- Running best before this run (us): 7730.745
- Delta vs running best before this run (us): +5.246
- Delta vs running best before this run (%): +0.068%
- New running best?: no
- Running best after this run (us): 7730.745
- 60 FPS budget share (update+extract): 46.416%
- 60 FPS headroom after update+extract (us): 8930.676
- Coefficient of variation (%): 0.248%
- Delta vs baseline tracking (us): -1494.471
- Delta vs baseline tracking (%): -11.332%
- Delta vs previous run tracking (us): -22.779
- Delta vs previous run tracking (%): -0.194%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 056: `run-apr5-a-large-022`
- Timestamp: 2026-04-05T10:43:21-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7734.991
- Std dev (us): 24.834
- Tracking iteration latency (us): 11687.221
- Change description: fused init seed+sample pass improved tracking but did not beat best update mean
- Delta vs baseline update+extract (us): -1289.610
- Delta vs baseline update+extract (%): -14.290%
- Delta vs previous run update+extract (us): -1.000
- Delta vs previous run update+extract (%): -0.013%
- Running best before this run (us): 7730.745
- Delta vs running best before this run (us): +4.246
- Delta vs running best before this run (%): +0.055%
- New running best?: no
- Running best after this run (us): 7730.745
- 60 FPS budget share (update+extract): 46.410%
- 60 FPS headroom after update+extract (us): 8931.676
- Coefficient of variation (%): 0.321%
- Delta vs baseline tracking (us): -1501.025
- Delta vs baseline tracking (%): -11.382%
- Delta vs previous run tracking (us): -6.554
- Delta vs previous run tracking (%): -0.056%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: mixed; random generation

#### Timeline 057: `run-apr5-a-large-022-rerun`
- Timestamp: 2026-04-05T10:43:21-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7732.466
- Std dev (us): 19.638
- Tracking iteration latency (us): 11685.378
- Change description: first rerun remained close but still above best update mean
- Delta vs baseline update+extract (us): -1292.135
- Delta vs baseline update+extract (%): -14.318%
- Delta vs previous run update+extract (us): -2.525
- Delta vs previous run update+extract (%): -0.033%
- Running best before this run (us): 7730.745
- Delta vs running best before this run (us): +1.721
- Delta vs running best before this run (%): +0.022%
- New running best?: no
- Running best after this run (us): 7730.745
- 60 FPS budget share (update+extract): 46.395%
- 60 FPS headroom after update+extract (us): 8934.201
- Coefficient of variation (%): 0.254%
- Delta vs baseline tracking (us): -1502.868
- Delta vs baseline tracking (%): -11.396%
- Delta vs previous run tracking (us): -1.843
- Delta vs previous run tracking (%): -0.016%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 058: `run-apr5-a-large-022-rerun2`
- Timestamp: 2026-04-05T10:43:21-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7740.213
- Std dev (us): 18.160
- Tracking iteration latency (us): 11690.043
- Change description: second rerun confirmed no primary-metric gain from fused init pass
- Delta vs baseline update+extract (us): -1284.388
- Delta vs baseline update+extract (%): -14.232%
- Delta vs previous run update+extract (us): +7.747
- Delta vs previous run update+extract (%): +0.100%
- Running best before this run (us): 7730.745
- Delta vs running best before this run (us): +9.468
- Delta vs running best before this run (%): +0.122%
- New running best?: no
- Running best after this run (us): 7730.745
- 60 FPS budget share (update+extract): 46.441%
- 60 FPS headroom after update+extract (us): 8926.454
- Coefficient of variation (%): 0.235%
- Delta vs baseline tracking (us): -1498.203
- Delta vs baseline tracking (%): -11.360%
- Delta vs previous run tracking (us): +4.665
- Delta vs previous run tracking (%): +0.040%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 059: `run-apr5-a-large-022-revert-confirm`
- Timestamp: 2026-04-05T10:46:15-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7724.082
- Std dev (us): 10.030
- Tracking iteration latency (us): 11687.135
- Change description: rollback confirmation after discarding fused init yielded new best on fused-normalization baseline
- Delta vs baseline update+extract (us): -1300.519
- Delta vs baseline update+extract (%): -14.411%
- Delta vs previous run update+extract (us): -16.131
- Delta vs previous run update+extract (%): -0.208%
- Running best before this run (us): 7730.745
- Delta vs running best before this run (us): -6.663
- Delta vs running best before this run (%): -0.086%
- New running best?: yes
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.344%
- 60 FPS headroom after update+extract (us): 8942.585
- Coefficient of variation (%): 0.130%
- Delta vs baseline tracking (us): -1501.111
- Delta vs baseline tracking (%): -11.382%
- Delta vs previous run tracking (us): -2.908
- Delta vs previous run tracking (%): -0.025%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 060: `run-apr5-a-large-023`
- Timestamp: 2026-04-05T10:49:24-07:00
- Phase: fine
- Status: discard
- Mean update+extract latency (us): 7741.491
- Std dev (us): 12.478
- Tracking iteration latency (us): 11700.629
- Change description: fast exp intrinsic in fused normalization regressed update mean versus current best
- Delta vs baseline update+extract (us): -1283.110
- Delta vs baseline update+extract (%): -14.218%
- Delta vs previous run update+extract (us): +17.409
- Delta vs previous run update+extract (%): +0.225%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +17.409
- Delta vs running best before this run (%): +0.225%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.449%
- 60 FPS headroom after update+extract (us): 8925.176
- Coefficient of variation (%): 0.161%
- Delta vs baseline tracking (us): -1487.617
- Delta vs baseline tracking (%): -11.280%
- Delta vs previous run tracking (us): +13.494
- Delta vs previous run tracking (%): +0.115%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 061: `run-apr5-a-large-023-revert-confirm`
- Timestamp: 2026-04-05T10:52:08-07:00
- Phase: fine
- Status: keep
- Mean update+extract latency (us): 7726.738
- Std dev (us): 10.398
- Tracking iteration latency (us): 11697.555
- Change description: rollback confirmation after discarded fast-exp tweak preserved near-best fused-normalization performance
- Delta vs baseline update+extract (us): -1297.863
- Delta vs baseline update+extract (%): -14.381%
- Delta vs previous run update+extract (us): -14.753
- Delta vs previous run update+extract (%): -0.191%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +2.656
- Delta vs running best before this run (%): +0.034%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.360%
- 60 FPS headroom after update+extract (us): 8939.929
- Coefficient of variation (%): 0.135%
- Delta vs baseline tracking (us): -1490.691
- Delta vs baseline tracking (%): -11.303%
- Delta vs previous run tracking (us): -3.074
- Delta vs previous run tracking (%): -0.026%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 062: `run-apr5-a-large-024`
- Timestamp: 2026-04-05T10:55:20-07:00
- Phase: fine
- Status: discard
- Mean update+extract latency (us): 7732.225
- Std dev (us): 11.312
- Tracking iteration latency (us): 11702.862
- Change description: gather-based resample materialization was close but slower than best for update mean
- Delta vs baseline update+extract (us): -1292.376
- Delta vs baseline update+extract (%): -14.321%
- Delta vs previous run update+extract (us): +5.487
- Delta vs previous run update+extract (%): +0.071%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +8.143
- Delta vs running best before this run (%): +0.105%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.393%
- 60 FPS headroom after update+extract (us): 8934.442
- Coefficient of variation (%): 0.146%
- Delta vs baseline tracking (us): -1485.384
- Delta vs baseline tracking (%): -11.263%
- Delta vs previous run tracking (us): +5.307
- Delta vs previous run tracking (%): +0.045%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: memory layout + resampling

#### Timeline 063: `run-apr5-a-large-024-revert-confirm`
- Timestamp: 2026-04-05T10:58:20-07:00
- Phase: fine
- Status: keep
- Mean update+extract latency (us): 7739.978
- Std dev (us): 10.641
- Tracking iteration latency (us): 11719.825
- Change description: initial rollback confirmation after run024 discard landed above best with elevated tracking variance
- Delta vs baseline update+extract (us): -1284.623
- Delta vs baseline update+extract (%): -14.235%
- Delta vs previous run update+extract (us): +7.753
- Delta vs previous run update+extract (%): +0.100%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +15.896
- Delta vs running best before this run (%): +0.206%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.440%
- 60 FPS headroom after update+extract (us): 8926.689
- Coefficient of variation (%): 0.137%
- Delta vs baseline tracking (us): -1468.421
- Delta vs baseline tracking (%): -11.134%
- Delta vs previous run tracking (us): +16.963
- Delta vs previous run tracking (%): +0.145%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 064: `run-apr5-a-large-024-revert-confirm-rerun`
- Timestamp: 2026-04-05T10:58:20-07:00
- Phase: fine
- Status: keep
- Mean update+extract latency (us): 7734.315
- Std dev (us): 22.104
- Tracking iteration latency (us): 11684.371
- Change description: follow-up rollback confirmation returned to expected near-best fused-normalization range
- Delta vs baseline update+extract (us): -1290.286
- Delta vs baseline update+extract (%): -14.297%
- Delta vs previous run update+extract (us): -5.663
- Delta vs previous run update+extract (%): -0.073%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +10.233
- Delta vs running best before this run (%): +0.132%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.406%
- 60 FPS headroom after update+extract (us): 8932.352
- Coefficient of variation (%): 0.286%
- Delta vs baseline tracking (us): -1503.875
- Delta vs baseline tracking (%): -11.403%
- Delta vs previous run tracking (us): -35.454
- Delta vs previous run tracking (%): -0.303%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 065: `run-apr5-a-large-025`
- Timestamp: 2026-04-05T11:09:11-07:00
- Phase: large
- Status: crash
- Mean update+extract latency (us): 0.000
- Std dev (us): 0.000
- Tracking iteration latency (us): 0.000
- Change description: direct curand Philox normal4 sampler crashed under full benchmark harness at 1M particles
- Delta vs baseline update+extract (us): n/a
- Delta vs baseline update+extract (%): n/a
- Delta vs previous run update+extract (us): n/a
- Delta vs previous run update+extract (%): n/a
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): n/a
- Delta vs running best before this run (%): n/a
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): n/a
- 60 FPS headroom after update+extract (us): n/a
- Coefficient of variation (%): n/a
- Delta vs baseline tracking (us): n/a
- Delta vs baseline tracking (%): n/a
- Delta vs previous run tracking (us): n/a
- Delta vs previous run tracking (%): n/a
- Keep/discard/crash interpretation: Run crashed before valid timing; treat as functional failure, not a performance datapoint.
- Stability classification: No stability classification (crash).
- Likely subsystem attribution: mixed; random generation

#### Timeline 066: `run-apr5-a-large-025-focus`
- Timestamp: 2026-04-05T11:09:11-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 14060.610
- Std dev (us): 744.169
- Tracking iteration latency (us): 24889.641
- Change description: focused update-tracking measurement for direct Philox showed severe regression even without constructor-loop pressure
- Delta vs baseline update+extract (us): +5036.009
- Delta vs baseline update+extract (%): +55.803%
- Delta vs previous run update+extract (us): +6326.295
- Delta vs previous run update+extract (%): +81.795%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +6336.528
- Delta vs running best before this run (%): +82.036%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 84.364%
- 60 FPS headroom after update+extract (us): 2606.057
- Coefficient of variation (%): 5.293%
- Delta vs baseline tracking (us): +11701.395
- Delta vs baseline tracking (%): +88.726%
- Delta vs previous run tracking (us): +13205.270
- Delta vs previous run tracking (%): +113.017%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: High variance (CV >= 1.50%); requires caution.
- Likely subsystem attribution: mixed; random generation

#### Timeline 067: `run-apr5-a-large-025-revert-confirm`
- Timestamp: 2026-04-05T11:12:44-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7753.095
- Std dev (us): 22.772
- Tracking iteration latency (us): 11715.318
- Change description: initial rollback confirmation after direct Philox failure was above best envelope
- Delta vs baseline update+extract (us): -1271.506
- Delta vs baseline update+extract (%): -14.089%
- Delta vs previous run update+extract (us): -6307.515
- Delta vs previous run update+extract (%): -44.859%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +29.013
- Delta vs running best before this run (%): +0.376%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.519%
- 60 FPS headroom after update+extract (us): 8913.572
- Coefficient of variation (%): 0.294%
- Delta vs baseline tracking (us): -1472.928
- Delta vs baseline tracking (%): -11.168%
- Delta vs previous run tracking (us): -13174.323
- Delta vs previous run tracking (%): -52.931%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; random generation

#### Timeline 068: `run-apr5-a-large-025-revert-confirm-rerun`
- Timestamp: 2026-04-05T11:12:44-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7729.575
- Std dev (us): 9.231
- Tracking iteration latency (us): 11698.472
- Change description: follow-up rollback confirmation restored expected fused-normalization baseline after Philox revert
- Delta vs baseline update+extract (us): -1295.026
- Delta vs baseline update+extract (%): -14.350%
- Delta vs previous run update+extract (us): -23.520
- Delta vs previous run update+extract (%): -0.303%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +5.493
- Delta vs running best before this run (%): +0.071%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.377%
- 60 FPS headroom after update+extract (us): 8937.092
- Coefficient of variation (%): 0.119%
- Delta vs baseline tracking (us): -1489.774
- Delta vs baseline tracking (%): -11.296%
- Delta vs previous run tracking (us): -16.846
- Delta vs previous run tracking (%): -0.144%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; random generation

#### Timeline 069: `run-apr5-a-large-026`
- Timestamp: 2026-04-05T11:18:50-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7997.192
- Std dev (us): 21.351
- Tracking iteration latency (us): 11932.378
- Change description: adaptive ESS-gated resampling with weight carry-over added overhead and regressed both update and tracking
- Delta vs baseline update+extract (us): -1027.409
- Delta vs baseline update+extract (%): -11.385%
- Delta vs previous run update+extract (us): +267.617
- Delta vs previous run update+extract (%): +3.462%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +273.110
- Delta vs running best before this run (%): +3.536%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 47.983%
- 60 FPS headroom after update+extract (us): 8669.475
- Coefficient of variation (%): 0.267%
- Delta vs baseline tracking (us): -1255.868
- Delta vs baseline tracking (%): -9.523%
- Delta vs previous run tracking (us): +233.906
- Delta vs previous run tracking (%): +1.999%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 070: `run-apr5-a-large-026-rerun`
- Timestamp: 2026-04-05T11:18:50-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 7969.808
- Std dev (us): 14.780
- Tracking iteration latency (us): 11920.144
- Change description: rerun confirmed ESS-gated path did not provide win on this workload
- Delta vs baseline update+extract (us): -1054.793
- Delta vs baseline update+extract (%): -11.688%
- Delta vs previous run update+extract (us): -27.384
- Delta vs previous run update+extract (%): -0.342%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +245.726
- Delta vs running best before this run (%): +3.181%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 47.819%
- 60 FPS headroom after update+extract (us): 8696.859
- Coefficient of variation (%): 0.185%
- Delta vs baseline tracking (us): -1268.102
- Delta vs baseline tracking (%): -9.615%
- Delta vs previous run tracking (us): -12.234
- Delta vs previous run tracking (%): -0.103%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 071: `run-apr5-a-large-026-revert-confirm`
- Timestamp: 2026-04-05T11:22:22-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 7726.937
- Std dev (us): 9.499
- Tracking iteration latency (us): 11683.803
- Change description: rollback confirmation after ESS-gated discard restored near-best fused-normalization behavior
- Delta vs baseline update+extract (us): -1297.664
- Delta vs baseline update+extract (%): -14.379%
- Delta vs previous run update+extract (us): -242.871
- Delta vs previous run update+extract (%): -3.047%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +2.855
- Delta vs running best before this run (%): +0.037%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.362%
- 60 FPS headroom after update+extract (us): 8939.730
- Coefficient of variation (%): 0.123%
- Delta vs baseline tracking (us): -1504.443
- Delta vs baseline tracking (%): -11.407%
- Delta vs previous run tracking (us): -236.341
- Delta vs previous run tracking (%): -1.983%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 072: `run-apr5-a-large-027`
- Timestamp: 2026-04-05T11:26:35-07:00
- Phase: fine
- Status: discard
- Mean update+extract latency (us): 7792.865
- Std dev (us): 13.988
- Tracking iteration latency (us): 11789.194
- Change description: reciprocal-sqrt visibility logit math regressed update and tracking means
- Delta vs baseline update+extract (us): -1231.736
- Delta vs baseline update+extract (%): -13.649%
- Delta vs previous run update+extract (us): +65.928
- Delta vs previous run update+extract (%): +0.853%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +68.783
- Delta vs running best before this run (%): +0.891%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.757%
- 60 FPS headroom after update+extract (us): 8873.802
- Coefficient of variation (%): 0.179%
- Delta vs baseline tracking (us): -1399.052
- Delta vs baseline tracking (%): -10.608%
- Delta vs previous run tracking (us): +105.391
- Delta vs previous run tracking (%): +0.902%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: likelihood hotpath; process model

#### Timeline 073: `run-apr5-a-large-027-revert-confirm`
- Timestamp: 2026-04-05T11:29:37-07:00
- Phase: fine
- Status: keep
- Mean update+extract latency (us): 7738.968
- Std dev (us): 19.019
- Tracking iteration latency (us): 11712.308
- Change description: initial rollback confirmation after run027 discard remained above best baseline range
- Delta vs baseline update+extract (us): -1285.633
- Delta vs baseline update+extract (%): -14.246%
- Delta vs previous run update+extract (us): -53.897
- Delta vs previous run update+extract (%): -0.692%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +14.886
- Delta vs running best before this run (%): +0.193%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.434%
- 60 FPS headroom after update+extract (us): 8927.699
- Coefficient of variation (%): 0.246%
- Delta vs baseline tracking (us): -1475.938
- Delta vs baseline tracking (%): -11.191%
- Delta vs previous run tracking (us): -76.886
- Delta vs previous run tracking (%): -0.652%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 074: `run-apr5-a-large-027-revert-confirm-rerun`
- Timestamp: 2026-04-05T11:29:37-07:00
- Phase: fine
- Status: keep
- Mean update+extract latency (us): 7750.145
- Std dev (us): 37.037
- Tracking iteration latency (us): 11703.281
- Change description: follow-up rollback confirmation stayed elevated likely due transient thermal/load conditions
- Delta vs baseline update+extract (us): -1274.456
- Delta vs baseline update+extract (%): -14.122%
- Delta vs previous run update+extract (us): +11.177
- Delta vs previous run update+extract (%): +0.144%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): +26.063
- Delta vs running best before this run (%): +0.337%
- New running best?: no
- Running best after this run (us): 7724.082
- 60 FPS budget share (update+extract): 46.501%
- 60 FPS headroom after update+extract (us): 8916.522
- Coefficient of variation (%): 0.478%
- Delta vs baseline tracking (us): -1484.965
- Delta vs baseline tracking (%): -11.260%
- Delta vs previous run tracking (us): -9.027
- Delta vs previous run tracking (%): -0.077%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: mixed

#### Timeline 075: `run-apr5-a-large-028`
- Timestamp: 2026-04-05T11:42:31-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 5719.159
- Std dev (us): 7.740
- Tracking iteration latency (us): 8476.656
- Change description: SoA particle-state storage for fast z-offset filter with index-based resample gather (major update and tracking reduction)
- Delta vs baseline update+extract (us): -3305.442
- Delta vs baseline update+extract (%): -36.627%
- Delta vs previous run update+extract (us): -2030.986
- Delta vs previous run update+extract (%): -26.206%
- Running best before this run (us): 7724.082
- Delta vs running best before this run (us): -2004.923
- Delta vs running best before this run (%): -25.957%
- New running best?: yes
- Running best after this run (us): 5719.159
- 60 FPS budget share (update+extract): 34.315%
- 60 FPS headroom after update+extract (us): 10947.508
- Coefficient of variation (%): 0.135%
- Delta vs baseline tracking (us): -4711.590
- Delta vs baseline tracking (%): -35.726%
- Delta vs previous run tracking (us): -3226.625
- Delta vs previous run tracking (%): -27.570%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: memory layout + resampling; state reduction

#### Timeline 076: `run-apr5-a-large-028-rerun`
- Timestamp: 2026-04-05T11:42:31-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 5719.964
- Std dev (us): 6.929
- Tracking iteration latency (us): 8468.259
- Change description: stability rerun confirmed SoA particle-state storage gain at 1M particles
- Delta vs baseline update+extract (us): -3304.637
- Delta vs baseline update+extract (%): -36.618%
- Delta vs previous run update+extract (us): +0.805
- Delta vs previous run update+extract (%): +0.014%
- Running best before this run (us): 5719.159
- Delta vs running best before this run (us): +0.805
- Delta vs running best before this run (%): +0.014%
- New running best?: no
- Running best after this run (us): 5719.159
- 60 FPS budget share (update+extract): 34.320%
- 60 FPS headroom after update+extract (us): 10946.703
- Coefficient of variation (%): 0.121%
- Delta vs baseline tracking (us): -4719.987
- Delta vs baseline tracking (%): -35.789%
- Delta vs previous run tracking (us): -8.397
- Delta vs previous run tracking (%): -0.099%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: memory layout + resampling

#### Timeline 077: `run-apr5-a-large-029`
- Timestamp: 2026-04-05T12:01:53-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 11762.609
- Std dev (us): 746.170
- Tracking iteration latency (us): 21770.413
- Change description: retry direct curand Philox sampler on SoA baseline regressed update and tracking severely
- Delta vs baseline update+extract (us): +2738.008
- Delta vs baseline update+extract (%): +30.339%
- Delta vs previous run update+extract (us): +6042.645
- Delta vs previous run update+extract (%): +105.641%
- Running best before this run (us): 5719.159
- Delta vs running best before this run (us): +6043.450
- Delta vs running best before this run (%): +105.670%
- New running best?: no
- Running best after this run (us): 5719.159
- 60 FPS budget share (update+extract): 70.576%
- 60 FPS headroom after update+extract (us): 4904.058
- Coefficient of variation (%): 6.344%
- Delta vs baseline tracking (us): +8582.167
- Delta vs baseline tracking (%): +65.074%
- Delta vs previous run tracking (us): +13302.154
- Delta vs previous run tracking (%): +157.083%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: High variance (CV >= 1.50%); requires caution.
- Likely subsystem attribution: memory layout + resampling; random generation

#### Timeline 078: `run-apr5-a-large-029-revert-confirm`
- Timestamp: 2026-04-05T12:04:50-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 5739.935
- Std dev (us): 24.932
- Tracking iteration latency (us): 8500.477
- Change description: rollback confirmation after Philox discard restored SoA performance envelope
- Delta vs baseline update+extract (us): -3284.666
- Delta vs baseline update+extract (%): -36.397%
- Delta vs previous run update+extract (us): -6022.674
- Delta vs previous run update+extract (%): -51.202%
- Running best before this run (us): 5719.159
- Delta vs running best before this run (us): +20.776
- Delta vs running best before this run (%): +0.363%
- New running best?: no
- Running best after this run (us): 5719.159
- 60 FPS budget share (update+extract): 34.440%
- 60 FPS headroom after update+extract (us): 10926.732
- Coefficient of variation (%): 0.434%
- Delta vs baseline tracking (us): -4687.769
- Delta vs baseline tracking (%): -35.545%
- Delta vs previous run tracking (us): -13269.936
- Delta vs previous run tracking (%): -60.954%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: memory layout + resampling; random generation

#### Timeline 079: `run-apr5-a-large-030`
- Timestamp: 2026-04-05T12:17:16-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 5753.555
- Std dev (us): 45.387
- Tracking iteration latency (us): 8462.724
- Change description: field-level process+likelihood path initial sample regressed update with elevated variance
- Delta vs baseline update+extract (us): -3271.046
- Delta vs baseline update+extract (%): -36.246%
- Delta vs previous run update+extract (us): +13.620
- Delta vs previous run update+extract (%): +0.237%
- Running best before this run (us): 5719.159
- Delta vs running best before this run (us): +34.396
- Delta vs running best before this run (%): +0.601%
- New running best?: no
- Running best after this run (us): 5719.159
- 60 FPS budget share (update+extract): 34.521%
- 60 FPS headroom after update+extract (us): 10913.112
- Coefficient of variation (%): 0.789%
- Delta vs baseline tracking (us): -4725.522
- Delta vs baseline tracking (%): -35.831%
- Delta vs previous run tracking (us): -37.753
- Delta vs previous run tracking (%): -0.444%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Moderately noisy (0.70% <= CV < 1.50%).
- Likely subsystem attribution: likelihood hotpath; process model

#### Timeline 080: `run-apr5-a-large-030-rerun`
- Timestamp: 2026-04-05T12:17:16-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 5672.139
- Std dev (us): 32.479
- Tracking iteration latency (us): 8397.489
- Change description: stability rerun for field-level process+likelihood path showed clear update and tracking gains
- Delta vs baseline update+extract (us): -3352.462
- Delta vs baseline update+extract (%): -37.148%
- Delta vs previous run update+extract (us): -81.416
- Delta vs previous run update+extract (%): -1.415%
- Running best before this run (us): 5719.159
- Delta vs running best before this run (us): -47.020
- Delta vs running best before this run (%): -0.822%
- New running best?: yes
- Running best after this run (us): 5672.139
- 60 FPS budget share (update+extract): 34.033%
- 60 FPS headroom after update+extract (us): 10994.528
- Coefficient of variation (%): 0.573%
- Delta vs baseline tracking (us): -4790.757
- Delta vs baseline tracking (%): -36.326%
- Delta vs previous run tracking (us): -65.235
- Delta vs previous run tracking (%): -0.771%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: likelihood hotpath; process model

#### Timeline 081: `run-apr5-a-large-030-rerun2`
- Timestamp: 2026-04-05T12:17:16-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 5653.460
- Std dev (us): 17.560
- Tracking iteration latency (us): 8398.432
- Change description: second rerun confirmed improved update mean for field-level process+likelihood path
- Delta vs baseline update+extract (us): -3371.141
- Delta vs baseline update+extract (%): -37.355%
- Delta vs previous run update+extract (us): -18.679
- Delta vs previous run update+extract (%): -0.329%
- Running best before this run (us): 5672.139
- Delta vs running best before this run (us): -18.679
- Delta vs running best before this run (%): -0.329%
- New running best?: yes
- Running best after this run (us): 5653.460
- 60 FPS budget share (update+extract): 33.921%
- 60 FPS headroom after update+extract (us): 11013.207
- Coefficient of variation (%): 0.311%
- Delta vs baseline tracking (us): -4789.814
- Delta vs baseline tracking (%): -36.319%
- Delta vs previous run tracking (us): +0.943
- Delta vs previous run tracking (%): +0.011%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: likelihood hotpath; process model

#### Timeline 082: `run-apr5-a-large-031`
- Timestamp: 2026-04-05T12:17:16-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 4903.257
- Std dev (us): 8.527
- Tracking iteration latency (us): 6898.009
- Change description: scalar SoA most-likely reduction replaced prediction-object reduction in hot path
- Delta vs baseline update+extract (us): -4121.344
- Delta vs baseline update+extract (%): -45.668%
- Delta vs previous run update+extract (us): -750.203
- Delta vs previous run update+extract (%): -13.270%
- Running best before this run (us): 5653.460
- Delta vs running best before this run (us): -750.203
- Delta vs running best before this run (%): -13.270%
- New running best?: yes
- Running best after this run (us): 4903.257
- 60 FPS budget share (update+extract): 29.420%
- 60 FPS headroom after update+extract (us): 11763.410
- Coefficient of variation (%): 0.174%
- Delta vs baseline tracking (us): -6290.237
- Delta vs baseline tracking (%): -47.696%
- Delta vs previous run tracking (us): -1500.423
- Delta vs previous run tracking (%): -17.866%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: memory layout + resampling; state reduction

#### Timeline 083: `run-apr5-a-large-031-rerun`
- Timestamp: 2026-04-05T12:17:16-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 4896.349
- Std dev (us): 8.807
- Tracking iteration latency (us): 6866.890
- Change description: stability rerun confirmed major gain from scalar SoA reduction
- Delta vs baseline update+extract (us): -4128.252
- Delta vs baseline update+extract (%): -45.744%
- Delta vs previous run update+extract (us): -6.908
- Delta vs previous run update+extract (%): -0.141%
- Running best before this run (us): 4903.257
- Delta vs running best before this run (us): -6.908
- Delta vs running best before this run (%): -0.141%
- New running best?: yes
- Running best after this run (us): 4896.349
- 60 FPS budget share (update+extract): 29.378%
- 60 FPS headroom after update+extract (us): 11770.318
- Coefficient of variation (%): 0.180%
- Delta vs baseline tracking (us): -6321.356
- Delta vs baseline tracking (%): -47.932%
- Delta vs previous run tracking (us): -31.119
- Delta vs previous run tracking (%): -0.451%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: memory layout + resampling; state reduction

#### Timeline 084: `run-apr5-a-large-031-rerun2`
- Timestamp: 2026-04-05T12:17:16-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 4903.591
- Std dev (us): 7.260
- Tracking iteration latency (us): 6875.655
- Change description: second rerun validated low-variance performance envelope for scalar SoA reduction
- Delta vs baseline update+extract (us): -4121.010
- Delta vs baseline update+extract (%): -45.664%
- Delta vs previous run update+extract (us): +7.242
- Delta vs previous run update+extract (%): +0.148%
- Running best before this run (us): 4896.349
- Delta vs running best before this run (us): +7.242
- Delta vs running best before this run (%): +0.148%
- New running best?: no
- Running best after this run (us): 4896.349
- 60 FPS budget share (update+extract): 29.422%
- 60 FPS headroom after update+extract (us): 11763.076
- Coefficient of variation (%): 0.148%
- Delta vs baseline tracking (us): -6312.591
- Delta vs baseline tracking (%): -47.865%
- Delta vs previous run tracking (us): +8.765
- Delta vs previous run tracking (%): +0.128%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: memory layout + resampling; state reduction

#### Timeline 085: `run-apr5-a-large-032`
- Timestamp: 2026-04-05T12:22:48-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3617.100
- Std dev (us): 6.880
- Tracking iteration latency (us): 5565.081
- Change description: scalar fixed-size sort and scalar precision likelihood path replaced generic position sorting in hotspot
- Delta vs baseline update+extract (us): -5407.501
- Delta vs baseline update+extract (%): -59.920%
- Delta vs previous run update+extract (us): -1286.491
- Delta vs previous run update+extract (%): -26.236%
- Running best before this run (us): 4896.349
- Delta vs running best before this run (us): -1279.249
- Delta vs running best before this run (%): -26.127%
- New running best?: yes
- Running best after this run (us): 3617.100
- 60 FPS budget share (update+extract): 21.703%
- 60 FPS headroom after update+extract (us): 13049.567
- Coefficient of variation (%): 0.190%
- Delta vs baseline tracking (us): -7623.165
- Delta vs baseline tracking (%): -57.803%
- Delta vs previous run tracking (us): -1310.574
- Delta vs previous run tracking (%): -19.061%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: likelihood hotpath

#### Timeline 086: `run-apr5-a-large-032-rerun`
- Timestamp: 2026-04-05T12:22:48-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3597.708
- Std dev (us): 7.060
- Tracking iteration latency (us): 5529.237
- Change description: stability rerun confirmed major update and tracking gains for scalar likelihood path
- Delta vs baseline update+extract (us): -5426.893
- Delta vs baseline update+extract (%): -60.134%
- Delta vs previous run update+extract (us): -19.392
- Delta vs previous run update+extract (%): -0.536%
- Running best before this run (us): 3617.100
- Delta vs running best before this run (us): -19.392
- Delta vs running best before this run (%): -0.536%
- New running best?: yes
- Running best after this run (us): 3597.708
- 60 FPS budget share (update+extract): 21.586%
- 60 FPS headroom after update+extract (us): 13068.959
- Coefficient of variation (%): 0.196%
- Delta vs baseline tracking (us): -7659.009
- Delta vs baseline tracking (%): -58.075%
- Delta vs previous run tracking (us): -35.844
- Delta vs previous run tracking (%): -0.644%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: likelihood hotpath

#### Timeline 087: `run-apr5-a-large-032-rerun2`
- Timestamp: 2026-04-05T12:22:48-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3608.877
- Std dev (us): 11.173
- Tracking iteration latency (us): 5538.674
- Change description: second rerun validated low-variance envelope for scalar likelihood optimization
- Delta vs baseline update+extract (us): -5415.724
- Delta vs baseline update+extract (%): -60.011%
- Delta vs previous run update+extract (us): +11.169
- Delta vs previous run update+extract (%): +0.310%
- Running best before this run (us): 3597.708
- Delta vs running best before this run (us): +11.169
- Delta vs running best before this run (%): +0.310%
- New running best?: no
- Running best after this run (us): 3597.708
- 60 FPS budget share (update+extract): 21.653%
- 60 FPS headroom after update+extract (us): 13057.790
- Coefficient of variation (%): 0.310%
- Delta vs baseline tracking (us): -7649.572
- Delta vs baseline tracking (%): -58.003%
- Delta vs previous run tracking (us): +9.437
- Delta vs previous run tracking (%): +0.171%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: likelihood hotpath

#### Timeline 088: `run-apr5-a-large-033`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3525.863
- Std dev (us): 9.801
- Tracking iteration latency (us): 5461.190
- Change description: index-only resampling path removed identity-index materialization; mean_us reflects update+extract workload
- Delta vs baseline update+extract (us): -5498.738
- Delta vs baseline update+extract (%): -60.931%
- Delta vs previous run update+extract (us): -83.014
- Delta vs previous run update+extract (%): -2.300%
- Running best before this run (us): 3597.708
- Delta vs running best before this run (us): -71.845
- Delta vs running best before this run (%): -1.997%
- New running best?: yes
- Running best after this run (us): 3525.863
- 60 FPS budget share (update+extract): 21.155%
- 60 FPS headroom after update+extract (us): 13140.804
- Coefficient of variation (%): 0.278%
- Delta vs baseline tracking (us): -7727.056
- Delta vs baseline tracking (%): -58.590%
- Delta vs previous run tracking (us): -77.484
- Delta vs previous run tracking (%): -1.399%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: memory layout + resampling

#### Timeline 089: `run-apr5-a-large-033-rerun`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3535.592
- Std dev (us): 29.559
- Tracking iteration latency (us): 5461.136
- Change description: stability rerun for index-only resampling path on update+extract workload
- Delta vs baseline update+extract (us): -5489.009
- Delta vs baseline update+extract (%): -60.823%
- Delta vs previous run update+extract (us): +9.729
- Delta vs previous run update+extract (%): +0.276%
- Running best before this run (us): 3525.863
- Delta vs running best before this run (us): +9.729
- Delta vs running best before this run (%): +0.276%
- New running best?: no
- Running best after this run (us): 3525.863
- 60 FPS budget share (update+extract): 21.214%
- 60 FPS headroom after update+extract (us): 13131.075
- Coefficient of variation (%): 0.836%
- Delta vs baseline tracking (us): -7727.110
- Delta vs baseline tracking (%): -58.591%
- Delta vs previous run tracking (us): -0.054
- Delta vs previous run tracking (%): -0.001%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Moderately noisy (0.70% <= CV < 1.50%).
- Likely subsystem attribution: memory layout + resampling

#### Timeline 090: `run-apr5-a-large-033-rerun2`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3531.364
- Std dev (us): 8.165
- Tracking iteration latency (us): 5499.659
- Change description: second rerun confirmed consistent frame-time reduction from index-only resampling
- Delta vs baseline update+extract (us): -5493.237
- Delta vs baseline update+extract (%): -60.870%
- Delta vs previous run update+extract (us): -4.228
- Delta vs previous run update+extract (%): -0.120%
- Running best before this run (us): 3525.863
- Delta vs running best before this run (us): +5.501
- Delta vs running best before this run (%): +0.156%
- New running best?: no
- Running best after this run (us): 3525.863
- 60 FPS budget share (update+extract): 21.188%
- 60 FPS headroom after update+extract (us): 13135.303
- Coefficient of variation (%): 0.231%
- Delta vs baseline tracking (us): -7688.587
- Delta vs baseline tracking (%): -58.299%
- Delta vs previous run tracking (us): +38.523
- Delta vs previous run tracking (%): +0.705%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: memory layout + resampling; state reduction

#### Timeline 091: `run-apr5-a-large-034`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3506.002
- Std dev (us): 10.404
- Tracking iteration latency (us): 5471.282
- Change description: precomputed process-noise scales moved dt-dependent math out of per-particle process kernel
- Delta vs baseline update+extract (us): -5518.599
- Delta vs baseline update+extract (%): -61.151%
- Delta vs previous run update+extract (us): -25.362
- Delta vs previous run update+extract (%): -0.718%
- Running best before this run (us): 3525.863
- Delta vs running best before this run (us): -19.861
- Delta vs running best before this run (%): -0.563%
- New running best?: yes
- Running best after this run (us): 3506.002
- 60 FPS budget share (update+extract): 21.036%
- 60 FPS headroom after update+extract (us): 13160.665
- Coefficient of variation (%): 0.297%
- Delta vs baseline tracking (us): -7716.964
- Delta vs baseline tracking (%): -58.514%
- Delta vs previous run tracking (us): -28.377
- Delta vs previous run tracking (%): -0.516%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; process model

#### Timeline 092: `run-apr5-a-large-034-rerun`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3535.498
- Std dev (us): 7.733
- Tracking iteration latency (us): 5481.527
- Change description: stability rerun for process-noise-scale hoisting path
- Delta vs baseline update+extract (us): -5489.103
- Delta vs baseline update+extract (%): -60.824%
- Delta vs previous run update+extract (us): +29.496
- Delta vs previous run update+extract (%): +0.841%
- Running best before this run (us): 3506.002
- Delta vs running best before this run (us): +29.496
- Delta vs running best before this run (%): +0.841%
- New running best?: no
- Running best after this run (us): 3506.002
- 60 FPS budget share (update+extract): 21.213%
- 60 FPS headroom after update+extract (us): 13131.169
- Coefficient of variation (%): 0.219%
- Delta vs baseline tracking (us): -7706.719
- Delta vs baseline tracking (%): -58.436%
- Delta vs previous run tracking (us): +10.245
- Delta vs previous run tracking (%): +0.187%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; process model

#### Timeline 093: `run-apr5-a-large-034-rerun2`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3504.664
- Std dev (us): 14.296
- Tracking iteration latency (us): 5436.863
- Change description: second rerun confirmed retained gain from process-noise-scale hoisting
- Delta vs baseline update+extract (us): -5519.937
- Delta vs baseline update+extract (%): -61.165%
- Delta vs previous run update+extract (us): -30.834
- Delta vs previous run update+extract (%): -0.872%
- Running best before this run (us): 3506.002
- Delta vs running best before this run (us): -1.338
- Delta vs running best before this run (%): -0.038%
- New running best?: yes
- Running best after this run (us): 3504.664
- 60 FPS budget share (update+extract): 21.028%
- 60 FPS headroom after update+extract (us): 13162.003
- Coefficient of variation (%): 0.408%
- Delta vs baseline tracking (us): -7751.383
- Delta vs baseline tracking (%): -58.775%
- Delta vs previous run tracking (us): -44.664
- Delta vs previous run tracking (%): -0.815%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: mixed; process model

#### Timeline 094: `run-apr5-a-large-035`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 3503.752
- Std dev (us): 9.517
- Tracking iteration latency (us): 5443.424
- Change description: analytic quarter-turn reducer was not stable across reruns and provided no consistent gain
- Delta vs baseline update+extract (us): -5520.849
- Delta vs baseline update+extract (%): -61.176%
- Delta vs previous run update+extract (us): -0.912
- Delta vs previous run update+extract (%): -0.026%
- Running best before this run (us): 3504.664
- Delta vs running best before this run (us): -0.912
- Delta vs running best before this run (%): -0.026%
- New running best?: yes
- Running best after this run (us): 3503.752
- 60 FPS budget share (update+extract): 21.023%
- 60 FPS headroom after update+extract (us): 13162.915
- Coefficient of variation (%): 0.272%
- Delta vs baseline tracking (us): -7744.822
- Delta vs baseline tracking (%): -58.725%
- Delta vs previous run tracking (us): +6.561
- Delta vs previous run tracking (%): +0.121%
- Keep/discard/crash interpretation: Despite a local latency gain, the run was discarded due to instability, tradeoffs, or inconsistent rerun behavior.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 095: `run-apr5-a-large-035-rerun`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 3535.363
- Std dev (us): 7.415
- Tracking iteration latency (us): 5474.278
- Change description: first rerun regressed frame time versus kept run-034 envelope
- Delta vs baseline update+extract (us): -5489.238
- Delta vs baseline update+extract (%): -60.825%
- Delta vs previous run update+extract (us): +31.611
- Delta vs previous run update+extract (%): +0.902%
- Running best before this run (us): 3503.752
- Delta vs running best before this run (us): +31.611
- Delta vs running best before this run (%): +0.902%
- New running best?: no
- Running best after this run (us): 3503.752
- 60 FPS budget share (update+extract): 21.212%
- 60 FPS headroom after update+extract (us): 13131.304
- Coefficient of variation (%): 0.210%
- Delta vs baseline tracking (us): -7713.968
- Delta vs baseline tracking (%): -58.491%
- Delta vs previous run tracking (us): +30.854
- Delta vs previous run tracking (%): +0.567%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 096: `run-apr5-a-large-035-rerun2`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: discard
- Mean update+extract latency (us): 3539.026
- Std dev (us): 20.263
- Tracking iteration latency (us): 5467.259
- Change description: second rerun confirmed no stable win for analytic quarter-turn reducer
- Delta vs baseline update+extract (us): -5485.575
- Delta vs baseline update+extract (%): -60.785%
- Delta vs previous run update+extract (us): +3.663
- Delta vs previous run update+extract (%): +0.104%
- Running best before this run (us): 3503.752
- Delta vs running best before this run (us): +35.274
- Delta vs running best before this run (%): +1.007%
- New running best?: no
- Running best after this run (us): 3503.752
- 60 FPS budget share (update+extract): 21.234%
- 60 FPS headroom after update+extract (us): 13127.641
- Coefficient of variation (%): 0.573%
- Delta vs baseline tracking (us): -7720.987
- Delta vs baseline tracking (%): -58.544%
- Delta vs previous run tracking (us): -7.019
- Delta vs previous run tracking (%): -0.128%
- Keep/discard/crash interpretation: Run was discarded because it did not produce a robust, decision-quality improvement.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: mixed

#### Timeline 097: `run-apr5-a-large-035-revert-confirm`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3509.919
- Std dev (us): 8.365
- Tracking iteration latency (us): 5447.881
- Change description: revert confirmation after discarding analytic quarter-turn reducer
- Delta vs baseline update+extract (us): -5514.682
- Delta vs baseline update+extract (%): -61.107%
- Delta vs previous run update+extract (us): -29.107
- Delta vs previous run update+extract (%): -0.822%
- Running best before this run (us): 3503.752
- Delta vs running best before this run (us): +6.167
- Delta vs running best before this run (%): +0.176%
- New running best?: no
- Running best after this run (us): 3503.752
- 60 FPS budget share (update+extract): 21.060%
- 60 FPS headroom after update+extract (us): 13156.748
- Coefficient of variation (%): 0.238%
- Delta vs baseline tracking (us): -7740.365
- Delta vs baseline tracking (%): -58.691%
- Delta vs previous run tracking (us): -19.378
- Delta vs previous run tracking (%): -0.354%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed

#### Timeline 098: `run-apr5-a-large-036`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3385.340
- Std dev (us): 11.622
- Tracking iteration latency (us): 5324.785
- Change description: zero-copy index handoff in resampler via vector swap removed per-frame index copy overhead
- Delta vs baseline update+extract (us): -5639.261
- Delta vs baseline update+extract (%): -62.488%
- Delta vs previous run update+extract (us): -124.579
- Delta vs previous run update+extract (%): -3.549%
- Running best before this run (us): 3503.752
- Delta vs running best before this run (us): -118.412
- Delta vs running best before this run (%): -3.380%
- New running best?: yes
- Running best after this run (us): 3385.340
- 60 FPS budget share (update+extract): 20.312%
- 60 FPS headroom after update+extract (us): 13281.327
- Coefficient of variation (%): 0.343%
- Delta vs baseline tracking (us): -7863.461
- Delta vs baseline tracking (%): -59.625%
- Delta vs previous run tracking (us): -123.096
- Delta vs previous run tracking (%): -2.260%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: memory layout + resampling; random generation

#### Timeline 099: `run-apr5-a-large-036-rerun`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3399.534
- Std dev (us): 45.024
- Tracking iteration latency (us): 5320.471
- Change description: stability rerun for zero-copy resampler index handoff
- Delta vs baseline update+extract (us): -5625.067
- Delta vs baseline update+extract (%): -62.330%
- Delta vs previous run update+extract (us): +14.194
- Delta vs previous run update+extract (%): +0.419%
- Running best before this run (us): 3385.340
- Delta vs running best before this run (us): +14.194
- Delta vs running best before this run (%): +0.419%
- New running best?: no
- Running best after this run (us): 3385.340
- 60 FPS budget share (update+extract): 20.397%
- 60 FPS headroom after update+extract (us): 13267.133
- Coefficient of variation (%): 1.324%
- Delta vs baseline tracking (us): -7867.775
- Delta vs baseline tracking (%): -59.657%
- Delta vs previous run tracking (us): -4.314
- Delta vs previous run tracking (%): -0.081%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Moderately noisy (0.70% <= CV < 1.50%).
- Likely subsystem attribution: memory layout + resampling; random generation

#### Timeline 100: `run-apr5-a-large-036-rerun2`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3385.205
- Std dev (us): 21.712
- Tracking iteration latency (us): 5339.445
- Change description: second rerun confirmed sustained gain from zero-copy resampler index handoff
- Delta vs baseline update+extract (us): -5639.396
- Delta vs baseline update+extract (%): -62.489%
- Delta vs previous run update+extract (us): -14.329
- Delta vs previous run update+extract (%): -0.421%
- Running best before this run (us): 3385.340
- Delta vs running best before this run (us): -0.135
- Delta vs running best before this run (%): -0.004%
- New running best?: yes
- Running best after this run (us): 3385.205
- 60 FPS budget share (update+extract): 20.311%
- 60 FPS headroom after update+extract (us): 13281.462
- Coefficient of variation (%): 0.641%
- Delta vs baseline tracking (us): -7848.801
- Delta vs baseline tracking (%): -59.514%
- Delta vs previous run tracking (us): +18.974
- Delta vs previous run tracking (%): +0.357%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Stable (0.30% <= CV < 0.70%).
- Likely subsystem attribution: memory layout + resampling; random generation

#### Timeline 101: `run-apr5-a-large-037`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3377.283
- Std dev (us): 5.510
- Tracking iteration latency (us): 5333.590
- Change description: scalar center-velocity noise draws replaced Eigen vector noise sampling in process path
- Delta vs baseline update+extract (us): -5647.318
- Delta vs baseline update+extract (%): -62.577%
- Delta vs previous run update+extract (us): -7.922
- Delta vs previous run update+extract (%): -0.234%
- Running best before this run (us): 3385.205
- Delta vs running best before this run (us): -7.922
- Delta vs running best before this run (%): -0.234%
- New running best?: yes
- Running best after this run (us): 3377.283
- 60 FPS budget share (update+extract): 20.264%
- 60 FPS headroom after update+extract (us): 13289.384
- Coefficient of variation (%): 0.163%
- Delta vs baseline tracking (us): -7854.656
- Delta vs baseline tracking (%): -59.558%
- Delta vs previous run tracking (us): -5.855
- Delta vs previous run tracking (%): -0.110%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; process model

#### Timeline 102: `run-apr5-a-large-037-rerun`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3382.793
- Std dev (us): 8.333
- Tracking iteration latency (us): 5324.249
- Change description: stability rerun for scalar center-velocity noise sampling
- Delta vs baseline update+extract (us): -5641.808
- Delta vs baseline update+extract (%): -62.516%
- Delta vs previous run update+extract (us): +5.510
- Delta vs previous run update+extract (%): +0.163%
- Running best before this run (us): 3377.283
- Delta vs running best before this run (us): +5.510
- Delta vs running best before this run (%): +0.163%
- New running best?: no
- Running best after this run (us): 3377.283
- 60 FPS budget share (update+extract): 20.297%
- 60 FPS headroom after update+extract (us): 13283.874
- Coefficient of variation (%): 0.246%
- Delta vs baseline tracking (us): -7863.997
- Delta vs baseline tracking (%): -59.629%
- Delta vs previous run tracking (us): -9.341
- Delta vs previous run tracking (%): -0.175%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; process model

#### Timeline 103: `run-apr5-a-large-037-rerun2`
- Timestamp: 2026-04-05T13:00:51-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 3386.236
- Std dev (us): 8.910
- Tracking iteration latency (us): 5339.116
- Change description: second rerun validated scalar center-velocity noise sampling performance envelope
- Delta vs baseline update+extract (us): -5638.365
- Delta vs baseline update+extract (%): -62.478%
- Delta vs previous run update+extract (us): +3.443
- Delta vs previous run update+extract (%): +0.102%
- Running best before this run (us): 3377.283
- Delta vs running best before this run (us): +8.953
- Delta vs running best before this run (%): +0.265%
- New running best?: no
- Running best after this run (us): 3377.283
- 60 FPS budget share (update+extract): 20.317%
- 60 FPS headroom after update+extract (us): 13280.431
- Coefficient of variation (%): 0.263%
- Delta vs baseline tracking (us): -7849.130
- Delta vs baseline tracking (%): -59.516%
- Delta vs previous run tracking (us): +14.867
- Delta vs previous run tracking (%): +0.279%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: mixed; process model

#### Timeline 104: `run-apr5-a-large-038`
- Timestamp: 2026-04-05T13:28:41-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 2934.654
- Std dev (us): 67.899
- Tracking iteration latency (us): 4506.232
- Change description: half-precision SoA state storage (all state fields except float log weights) with float compute in kernels
- Delta vs baseline update+extract (us): -6089.947
- Delta vs baseline update+extract (%): -67.482%
- Delta vs previous run update+extract (us): -451.582
- Delta vs previous run update+extract (%): -13.336%
- Running best before this run (us): 3377.283
- Delta vs running best before this run (us): -442.629
- Delta vs running best before this run (%): -13.106%
- New running best?: yes
- Running best after this run (us): 2934.654
- 60 FPS budget share (update+extract): 17.608%
- 60 FPS headroom after update+extract (us): 13732.013
- Coefficient of variation (%): 2.314%
- Delta vs baseline tracking (us): -8682.014
- Delta vs baseline tracking (%): -65.831%
- Delta vs previous run tracking (us): -832.884
- Delta vs previous run tracking (%): -15.600%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: High variance (CV >= 1.50%); requires caution.
- Likely subsystem attribution: likelihood hotpath

#### Timeline 105: `run-apr5-a-large-038-rerun`
- Timestamp: 2026-04-05T13:28:41-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 2898.696
- Std dev (us): 5.329
- Tracking iteration latency (us): 4502.549
- Change description: stability rerun for half-precision state storage path
- Delta vs baseline update+extract (us): -6125.905
- Delta vs baseline update+extract (%): -67.880%
- Delta vs previous run update+extract (us): -35.958
- Delta vs previous run update+extract (%): -1.225%
- Running best before this run (us): 2934.654
- Delta vs running best before this run (us): -35.958
- Delta vs running best before this run (%): -1.225%
- New running best?: yes
- Running best after this run (us): 2898.696
- 60 FPS budget share (update+extract): 17.392%
- 60 FPS headroom after update+extract (us): 13767.971
- Coefficient of variation (%): 0.184%
- Delta vs baseline tracking (us): -8685.697
- Delta vs baseline tracking (%): -65.859%
- Delta vs previous run tracking (us): -3.683
- Delta vs previous run tracking (%): -0.082%
- Keep/discard/crash interpretation: Kept as a positive step in the optimization frontier.
- Stability classification: Very stable (CV < 0.30%).
- Likely subsystem attribution: likelihood hotpath

#### Timeline 106: `run-apr5-a-large-038-rerun2`
- Timestamp: 2026-04-05T13:28:41-07:00
- Phase: large
- Status: keep
- Mean update+extract latency (us): 2907.441
- Std dev (us): 23.515
- Tracking iteration latency (us): 4497.022
- Change description: second rerun confirmed substantial frame-time and tracking gains for half-state storage
- Delta vs baseline update+extract (us): -6117.160
- Delta vs baseline update+extract (%): -67.783%
- Delta vs previous run update+extract (us): +8.745
- Delta vs previous run update+extract (%): +0.302%
- Running best before this run (us): 2898.696
- Delta vs running best before this run (us): +8.745
- Delta vs running best before this run (%): +0.302%
- New running best?: no
- Running best after this run (us): 2898.696
- 60 FPS budget share (update+extract): 17.445%
- 60 FPS headroom after update+extract (us): 13759.226
- Coefficient of variation (%): 0.809%
- Delta vs baseline tracking (us): -8691.224
- Delta vs baseline tracking (%): -65.901%
- Delta vs previous run tracking (us): -5.527
- Delta vs previous run tracking (%): -0.123%
- Keep/discard/crash interpretation: Kept as baseline/confirmation or as part of a stable envelope check.
- Stability classification: Moderately noisy (0.70% <= CV < 1.50%).
- Likely subsystem attribution: memory layout + resampling


## Consolidated Bottleneck-to-Fix Mapping

This section maps dominant bottlenecks to concrete code interventions and expected effect categories.

1. Bottleneck: process+weight kernel dominates GPU time.
   Fix category: remove unnecessary work and reduce state traffic.
   Representative interventions: scalarized likelihood path, field-level process path, half-state storage.

2. Bottleneck: resample materialization/gather path remains second largest.
   Fix category: reduce index and state materialization overhead.
   Representative interventions: index-only resampling, swap-based index handoff, half-state gather traffic reduction.

3. Bottleneck: most-likely reduction path significant in every frame.
   Fix category: simplify transform and state representation.
   Representative interventions: scalar SoA reduction state and direct reconstruction, half-state read path in transform functor.

4. Bottleneck: repeated dt-dependent math in per-particle process code.
   Fix category: hoist invariant math.
   Representative interventions: precomputed process-noise scales.

5. Bottleneck: generic object-heavy math in likelihood internals.
   Fix category: fixed-size scalar math with reduced temporary object construction.
   Representative interventions: scalar fixed-size sort and scalar precision-density evaluation.

## Practical Guidance for Future Experiments

1. Keep changes isolated when possible.
2. Run warmup+timed loops with fixed particle count when making keep/discard decisions.
3. Require at least one rerun for large wins before final keep decisions.
4. Separate constructor profiling from steady-state frame profiling.
5. Avoid mixing algorithmic and micro-optimization changes in one trial if attribution matters.
6. Preserve the update-with-observation-plus-extract workload semantics when optimizing for frame-time behavior.
7. Track both mean and variance; reject unstable improvements unless variance causes are understood and controlled.
8. Validate numerical behavior after precision/storage changes with finite-value checks and long-horizon replay.

## Minimal Numerical Smoke Test Ideas

1. Run repeated updates and ensure all state components remain finite.
2. Compare extracted state continuity across frames against float-state baseline.
3. Compute mean absolute state difference versus float baseline over long runs.
4. Validate orientation wrap behavior for quarter-turn symmetries.
5. Validate z-ordering behavior under one-plate and two-plate observations.

## Closing Summary

The optimization path succeeded because it progressed from broad architecture alignment (SoA and workflow simplification) into hotspot-specialized math and finally bandwidth compression via half-state storage.

The half-state change was effective not as an isolated trick but as the final step on a code path already shaped to expose memory bandwidth costs clearly.

At 1M particles, the kept run envelope shows a large reduction from baseline to the current best while maintaining practical loop semantics and acceptable stability in benchmark reruns.

