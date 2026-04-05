# fast_plate_orbit_with_z_offset_autoresearch

This prompt defines an autonomous experimentation loop for runtime optimization in this repository.

Primary objective:
- Minimize `update_state_with_observation` runtime at 1,000,000 particles.

Secondary objective:
- Minimize standard deviation while preserving correctness and API behavior.

Hard constraints:
- Do not defer initialization.
- Do not switch to "resample every Nth call" behavior.
- Do not fundamentally change the public library API.
- Do not add new dependencies.
- Do not perform version-control operations as part of this workflow.

## Setup

Before experimentation, complete all setup steps.

1. Choose a run label
- Use a simple run label for organization, for example: `run-apr4-a`.
- This is only for local note-keeping.

2. Read in-scope files for context
- `README.md`
- `CMakeLists.txt`
- `setup.py`
- `pyproject.toml`
- `demos/benchmark_fast_plate_orbit_with_z_offset.py`
- `src/fast_plate_orbit/particle_filter.cu`
- `include/fast_plate_orbit/particle_filter_configuration.h`
- `include/fast_plate_orbit/prediction.h`
- `deps/particle-filter/pf/filter/particle_filter.h`
- `deps/particle-filter/pf/filter/systematic_resampler.h`

3. Resolve the actual implementation path before edits
- The benchmark imports `fast_plate_orbit_with_z_offset`.
- The source tree may expose `fast_plate_orbit` names in files.
- Always trace symbol paths from benchmark import to C++ implementation before editing.
- If mapping is ambiguous, pause and verify via symbol/search commands first.

4. Verify environment and tooling
- Ensure submodules are initialized.
- Ensure venv is active.
- Ensure CUDA and profiler tools are available: `nvidia-smi`, `nsys`, `ncu`.

5. Build/install sanity
- Confirm that benchmark execution uses binaries built from this workspace state.
- Rebuild/reinstall native extension after source changes before benchmarking.

6. Initialize tracker files
- Create `results.tsv` with header only.
- Create `large_changes.md` with an empty results table.
- Create `fine_grained_changes.md` with an empty results table.
- Keep these tracker files local for note-taking.

7. Baseline run
- First run must be baseline with zero code changes.
- Record baseline in `results.tsv`.

## Experimentation policy

What you CAN do:
- Modify C++/CUDA implementation and internal algorithmic structure.
- Modify benchmark harness only if needed for more accurate measurement.
- Use Nsight Systems/Compute and CUDA profiling.
- Use external research sources.
- Use Explore subagents for parallel idea generation and literature scans.

What you CANNOT do:
- API-breaking changes.
- Deferred initialization tricks.
- "Resample every N" shortcuts.
- Dependency additions.

## Use the 13-researcher team

You have 13 chief researchers.

At each major cycle:
- Launch Explore subagents in parallel to investigate different angles.
- Assign each researcher a distinct hypothesis area.
- Request concrete proposed edits, expected impact, and risk.
- Deduplicate and prioritize ideas by estimated impact on `update_state_with_observation`.

## Measurement protocol

Target workload:
- 1,000,000 particles.
- Report method-level mean and std in microseconds.

Primary command pattern:
- `python demos/benchmark_fast_plate_orbit_with_z_offset.py --particle-counts 1000000 --runs <R> --warmup <W> --plot-output-dir <dir> > run.log 2>&1`

Metric extraction:
- Parse line beginning with `update_state_with_observation,` from `run.log`.
- Also collect `tracking_iteration` as a secondary guardrail metric.

Profiler checkpoints:
- Run Nsight regularly to verify where time moved.
- Keep profiler outputs in `benchmark_plots/.../nsight/` style directories.

Stability rule:
- If mean improves but std regresses badly, treat as suspect and re-run.

## Results logging

Log every experiment to `results.tsv` (tab-separated).

Header:

`timestamp\trun_id\tphase\tmean_us\tstd_us\ttracking_mean_us\tstatus\tdescription`

Field rules:
- `phase`: `baseline`, `large`, or `fine`.
- `status`: `keep`, `discard`, or `crash`.
- For crashes, use `0.000` for numeric fields.
- `description` should be concise but specific.

## Phase 1: large changes first

Goal:
- Pursue coarse, high-leverage architectural or algorithmic improvements first.
- Seek cumulative major gains before micro-tuning.

Examples of large changes:
- Eliminate unnecessary allocations or host synchronizations on the hot path.
- Rework data flow to reduce global memory pressure.
- Fuse logically adjacent GPU passes where feasible.
- Move repeated per-call work to reusable precomputed state when legal.

When Phase 1 produces meaningful wins:
- Pause and document in `large_changes.md`.
- For each kept large change, include:
	- What changed
	- Why it should help
	- Measured impact (mean/std deltas)
	- Tradeoffs/complexity cost

Then continue to Phase 2.

## Phase 2: fine-grained optimizations

Goal:
- Squeeze additional latency and variance reductions after large wins.

Examples:
- Kernel launch configuration tuning.
- Branch divergence reduction.
- Memory access/coalescing refinements.
- Math simplifications and instruction-level cleanup.

Document kept fine-grained wins in `fine_grained_changes.md` with impact and tradeoffs.

## Keep/discard decision rule

For each experiment:
- If primary metric improves with acceptable stability, keep and advance.
- If equal or worse, discard and revert local code changes for that experiment.
- If crash, log crash, revert, and continue.

Simplicity preference:
- If two variants are similar in performance, keep the simpler one.

## Experiment loop

Loop forever after setup:

1. Define one experiment hypothesis and assign a `run_id`.
2. Implement changes.
3. Build/install updated extension.
4. Run benchmark with output redirected to `run.log`.
5. Extract metrics.
6. If run failed, inspect logs, attempt quick fix, otherwise mark crash.
7. Append row to `results.tsv`.
8. Keep/discard and move to next hypothesis.

Do not ask for human confirmation mid-loop.
Continue autonomously until manually interrupted.

## Operational notes

- Avoid flooding context: always redirect long outputs to log files.
- If a run hangs or exceeds practical bounds, terminate and log as failure.
- Re-read in-scope files periodically for new optimization angles.
- If stuck, synthesize ideas from profiler data plus external research, then continue.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!