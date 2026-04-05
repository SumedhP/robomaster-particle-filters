### TIER 1: Algorithmic — eliminate work entirely
*These don't optimize the code, they eliminate entire categories of computation. Largest impact per line changed.*

**1. Delete predicted plate velocity computation (zero-effort, ~25-35% of `conditional_log_likelihood`)**
`predicted_plates()` computes `predicted_plate_velocity` for all 4 plates using sin/cos. `conditional_log_likelihood` never reads velocity — it uses only `.position()`. At 1M particles × 4 plates = 4M wasted trig pairs per update call. Create a `predicted_plate_positions_only()` path that skips velocity entirely. This is the highest ROI change in the entire codebase.

**2. Precompute orbit in `initialize_internal_state_` (one-liner before kernel, ~40-60% of init)**
Every particle's `sample_from()` call builds an identical `observed_plate_orbit_builder` and calls `from_one_plate()` or `from_two_plates()` with the same `state` and `params_`. All 1-2M particles compute the exact same center/radius/orientation and then add noise around it. Compute the orbit once on the host, pass as a `__constant__` or kernel parameter. The remaining per-particle work is just sampling noise — which is unavoidable.

---

### TIER 2: Algorithmic — replace with better parallel algorithm
*Changes the fundamental algorithm to something designed for GPU hardware.*

**5. Replace resampler with Megopolis (significant at >500K particles)**
Your systematic resampler requires 8 sequential Thrust passes (2× reduce, transform, transform_exclusive_scan, fill, scatter_if, inclusive_scan, gather). Standard multinomial, stratified, and systematic resamplers require a collective prefix-sum of weights — while this can be performed efficiently on GPUs, it requires thread communication and multiple kernel launches. The Metropolis resampler requires only pairwise ratios between weights, so threads operate independently in a single kernel launch.

The catch: with a large number of particles, Metropolis resampling becomes slow due to the non-coalesced access problem on GPU global memory. The solution is **Metropolis-C2** (coalesced variant) or **Megopolis** (2022): Megopolis achieves the same convergence rate as Metropolis but with significant performance improvements from coalesced memory access patterns. For your use case (1-2M particles, well-converged tracker), weight variance is typically modest → Metropolis-family is a good fit.

**6. Replace `thrust::default_random_engine` with `curandStatePhilox4_32_10_t` (~30-40% of `apply_process`)**
Your `apply_process` draws 11 normal samples per particle using `thrust::normal_distribution` (Box-Muller: 2 uniform draws + `logf` + `sqrtf` + trig per 2 samples = 5.5 `logf`+`sqrtf`+`sincosf` per particle at 1M = 11M expensive transcendental ops). The Philox generator produces four single-precision or two double-precision results per call and is generally more efficient than generating one result at a time due to its underlying counter-based implementation. 11 samples → 3 `curand_normal4()` calls, no trig. Additionally, Philox generators do not require transferring a state from and to global memory as long as the generator keys are deduced from intrinsic variables such as particle numbers — meaning the RNG state doesn't consume global memory bandwidth at all. Side benefit: fixing the seeding strategy.

**7. Fix seeding to use per-seed initialization, not per-sequence-id**
Current `initialize_internal_state_` uses `generator.discard(index)` — the costly approach. Changing from a fixed seed with different sequence IDs to different seeds with the same sequence and offset yields a 2× speedup for `curand_init`, because different seeds only generate one sequence for all threads with less overhead than per-thread sequence setup. Use `curand_init(base_seed + particle_index, 0, 0, &state[i])` instead.

---

### TIER 3: Memory architecture — restructure data layout
*No algorithmic change; same math, better memory access patterns.*

**8. AoS → SoA layout for `particle_states_` and `sampler_states_` (15-25% bandwidth, more on Ampere+)**
Current: `vector<prediction>` packs {radius, z0, z1, orientation, orientation_velocity, center.x, center.y, center_velocity.x, center_velocity.y} = 9 floats = 36 bytes per particle. Thread 0 reads `radius_` from offset 0, thread 1 reads from offset 36, thread 32 reads from offset 36×32 = 1152 — not a coalesced 128-byte cache line. SoA = separate `vector<float>` per field: all threads in a warp read consecutive `radius_` values = one transaction.

Running a simulation of 2 million particles using AoS layout was found to be three times slower than SoA for discrete element methods and SoA is considered best practice for GPUs due to improved memory coalescing. This is the highest-effort change but affects every kernel simultaneously.

**9. Place `particle_filter_configuration_parameters` in `__constant__` memory**
`params_` (11 floats ≈ 44 bytes) is captured by value into every lambda. At 1-2M threads, every thread broadcasts from the same address → constant memory is ideal. If every thread in a half-warp requests data from the same address in constant memory, the GPU generates only a single read request and broadcasts to every thread, reducing memory traffic to ~6% of the equivalent global memory read. Do not use `__constant__` for anything threads access at different indices — it serializes.

---

### TIER 4: Kernel-level — reduce kernel launch count and fuse operations
*Each Thrust call is a separate GPU kernel with overhead. Fusing eliminates round-trips to global memory.*

**10. Fuse the weight normalization passes in the resampler (3 → 1 kernel)**
`resample()` does: (a) `reduce` for max log weight, (b) `transform` to exponentiate, (c) `reduce` for cumulative sum. These are 3 separate global memory passes over 1-2M weights × 4 bytes = 4-8MB each. A single fused kernel can find max and partial sums in one pass using shared memory per block, with a second-pass global reduction. Fusing mapping and block-level reduction into one kernel by eliminating per-element intermediate arrays yields 1.53–3.13× speedups.

**11. Switch `transform_reduce` to `cub::DeviceReduce` for `most_likely_particle` (~4× reduction)**
For 4M elements on a Tesla K40: Thrust reduce runs at 0.490ms, CUB reduce at 0.125ms. CUB-based codes significantly outperform `transform_reduce` Thrust routines on all problem sizes. CUB requires pre-allocated temp storage (amortize this at construction) and two calls (size query + execution), but the speedup compounds since `most_likely_particle` runs after every `update_*` call.
---

### TIER 5: Instruction-level — cheaper arithmetic per particle
*Each change is small but multiplied by 1-2M particles times 4 plates per update.*

**14. Precompute `sqrtf(variance)` sigma values before kernel launch (~8-12% of `apply_process`)**
`normal_sample(variance)` calls `sqrtf(variance)` inline. All 11 variance values come from `params_` — identical for every particle. At 1M particles, that's 11M `sqrtf` calls on constant inputs. Precompute 7 scalar sigma values (and the 2D diagonal covariance `cwiseSqrt()`) on the host and pass them in. Note: `sqrtf` throughput on modern GPUs is high (32 per clock cycle on Ampere) so this is a smaller win than it appears — but it's a 1-minute change.

**16. Replace `expf` in resampler with `__expf` fast intrinsic**
The resampler's weight normalization calls `expf(log_weight - max_log_weight)` for every particle. `__expf` uses the hardware SFU with ~2ULP error — acceptable for log-weight normalization where exact precision is unnecessary. On most NVIDIA architectures this maps to a native instruction vs. the multi-instruction `expf`.


### TIER 6: Initialization-specific improvements

**19. One-pass `fill` + `transform` fused init kernel**
`initialize_internal_state_` runs: (a) `for_each` to seed all RNGs using `generator.discard(index)`, then (b) `transform` to sample from each. These are two global memory passes over 1-2M sampler states. A single fused `for_each` can seed the Philox state (counter-based, no discard needed) and immediately sample, writing only the final `prediction` — never materializing the intermediate sampler state to global memory.

**20. Preallocate CUB temp storage at construction time**
CUB reduction requires temp storage. The current Thrust `transform_reduce` allocates internally via the caching allocator. With CUB `DeviceReduce::Reduce`, you call it once with `nullptr` to query the size, allocate once in the constructor, and reuse across all subsequent calls. Eliminates hidden allocations on every update.


# Rand performance improvement guide:
Achieving High Performance
Below we present general and advanced advices that may help in achieving high performance using cuRANDDx.

General Advices
Best performance from the cuRANDDx library is achieved by generating blocks of random numbers that are as large as possible to fill the GPU for peak performance.

The pseudorandom generator, PCG, is fast in setting up initial states and generating random bits.

In cases of fused operations, use directly the generated numbers in register. Avoid reading/writing data from/to global memory unnecessarily.

For curanddx::philox4_32 generator, each thread computes 4 random numbers at a time thus the most efficient use of Philox generator is to generate a multiple of 4 times number of threads. Setting round count to be 7 for better performance than the default value, 10, while still having full Crush-resistance (see reference).

Experiment with different block size for optimal performance.

Initialization of the generator state generally requires more registers than random number generation, thus for some generators it may be beneficial to experiment separating calls to initialize the states and generate random numbers into separate kernels for better performance.

State setup can be an expensive operation for some generators such as curanddx::xorwow and curanddx::mrg32k3a. One way to speed up the setup is to use different seeds for each thread and a constant sequence number of 0. But be aware that while faster to set up, this method provides less guarantees about the mathematical properties of the generated sequences.

Advanced Advices
Use CUDA Occupancy Calculator [5] and/or cudaOccupancyMaxActiveBlocksPerMultiprocessor [6] function to determine the optimum launch parameters.

For compute loads not filling the GPU entirely, consider running parallel kernels in a separate stream.

Use the Nsight Compute CUDA Occupancy Calculator [5] to determine what extra resources are available without losing occupancy.