from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from robomaster_particle_filters import fast_plate_orbit_with_z_offset as fpoz


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    mean_us: float
    std_us: float
    min_us: float
    max_us: float


def _load_cuda_synchronize() -> Callable[[], int] | None:
    candidates: list[str] = []
    libcudart = ctypes.util.find_library("cudart")
    if libcudart:
        candidates.append(libcudart)
    candidates.extend([
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ])

    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
        except OSError:
            continue

        try:
            sync = lib.cudaDeviceSynchronize
            sync.restype = ctypes.c_int
            sync.argtypes = []
            return sync
        except AttributeError:
            continue

    return None


CUDA_SYNCHRONIZE = _load_cuda_synchronize()


def synchronize_cuda() -> None:
    if CUDA_SYNCHRONIZE is None:
        return
    result = CUDA_SYNCHRONIZE()
    if result != 0:
        raise RuntimeError(f"cudaDeviceSynchronize failed with error code {result}")


def build_default_config() -> fpoz.ParticleFilterConfigurationParameters:
    return fpoz.ParticleFilterConfigurationParameters(
        0.11,
        2.0,
        0.001,
        0.005,
        0.0001,
        3.0,
        0.3,
        6.0,
        1.5,
        np.array([6.0, 6.0], dtype=np.float32),
        np.array([10.0, 10.0], dtype=np.float32),
    )


def build_default_observation() -> fpoz.Observation:
    observer_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    plate_one_position = np.array([0.12, 0.0, 0.0], dtype=np.float32)
    plate_two_position = np.array([0.0, 0.12, 0.0], dtype=np.float32)
    plate_position_diagonal_covariance = np.array([0.0001, 0.0001, 0.0001], dtype=np.float32)

    plate_one = fpoz.ObservedPlate(plate_one_position, plate_position_diagonal_covariance)
    plate_two = fpoz.ObservedPlate(plate_two_position, plate_position_diagonal_covariance)
    return fpoz.Observation.from_two_plates(observer_position, plate_one, plate_two)


def benchmark_callable(callable_under_test: Callable[[], None], runs: int, warmup: int) -> BenchmarkResult:
    for _ in range(warmup):
        callable_under_test()
        synchronize_cuda()

    samples_us: list[float] = []
    for _ in range(runs):
        start_ns = time.perf_counter_ns()
        callable_under_test()
        synchronize_cuda()
        end_ns = time.perf_counter_ns()
        samples_us.append((end_ns - start_ns) / 1000.0)

    return BenchmarkResult(
        method="",
        mean_us=statistics.fmean(samples_us),
        std_us=statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
        min_us=min(samples_us),
        max_us=max(samples_us),
    )


def run_runtime_benchmark(
    number_of_particles: int,
    runs: int,
    warmup: int,
    dt_seconds: float,
) -> list[BenchmarkResult]:
    observation = build_default_observation()
    config = build_default_config()
    particle_filter = fpoz.ParticleFilter(number_of_particles, observation, config)

    def update_with_observation() -> None:
        particle_filter.update_state_with_observation(dt_seconds, observation)

    def update_sans_observation() -> None:
        particle_filter.update_state_sans_observation(dt_seconds)

    def extrapolate_state() -> None:
        _ = particle_filter.extrapolate_state(dt_seconds)

    methods: Sequence[tuple[str, Callable[[], None]]] = (
        ("update_state_with_observation", update_with_observation),
        ("update_state_sans_observation", update_sans_observation),
        ("extrapolate_state", extrapolate_state),
    )

    results: list[BenchmarkResult] = []
    for method_name, method_callable in methods:
        result = benchmark_callable(method_callable, runs=runs, warmup=warmup)
        results.append(
            BenchmarkResult(
                method=method_name,
                mean_us=result.mean_us,
                std_us=result.std_us,
                min_us=result.min_us,
                max_us=result.max_us,
            )
        )

    return results


def run_profiler_workload(
    method: str,
    number_of_particles: int,
    iterations: int,
    dt_seconds: float,
) -> None:
    observation = build_default_observation()
    config = build_default_config()
    particle_filter = fpoz.ParticleFilter(number_of_particles, observation, config)

    methods: dict[str, Callable[[], None]] = {
        "update_state_with_observation": lambda: particle_filter.update_state_with_observation(dt_seconds, observation),
        "update_state_sans_observation": lambda: particle_filter.update_state_sans_observation(dt_seconds),
        "extrapolate_state": lambda: particle_filter.extrapolate_state(dt_seconds),
    }

    benchmark_target = methods[method]

    # Warmup before profiling to avoid one-time startup costs in the trace.
    for _ in range(8):
        benchmark_target()
        synchronize_cuda()

    for _ in range(iterations):
        benchmark_target()

    synchronize_cuda()
    print(f"Completed profiler workload for {method}: {iterations} iterations")


def print_results(number_of_particles: int, results: Sequence[BenchmarkResult]) -> None:
    print(f"\nParticles: {number_of_particles}")
    print("method,mean_us,std_us,min_us,max_us")
    for result in results:
        print(
            f"{result.method}, \t"
            f"{result.mean_us:.3f}, \t"
            f"{result.std_us:.3f}, \t"
            f"{result.min_us:.3f}, \t"
            f"{result.max_us:.3f}"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FastPlateOrbitWithZOffset ParticleFilter methods and "
            "provide dedicated profiler workloads."
        )
    )
    parser.add_argument("--particle-count", type=int, default=(1 << 20), help="Number of particles to use.")
    parser.add_argument("--runs", type=int, default=1000, help="Timed runs per method.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per method.")
    parser.add_argument("--dt-seconds", type=float, default=0.016, help="Delta time passed to each method call.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to save runtime benchmark results as JSON.",
    )
    parser.add_argument(
        "--profile-method",
        type=str,
        default=None,
        choices=["update_state_with_observation", "update_state_sans_observation", "extrapolate_state"],
        help=(
            "Run only the selected method for many iterations, intended for CUDA profiler runs "
            "(nsys/ncu)."
        ),
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=200,
        help="Number of iterations to execute for profiler workload mode.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    gc.disable()
    np.random.seed(0)

    if args.profile_method is not None:
        run_profiler_workload(
            method=args.profile_method,
            number_of_particles=args.particle_count,
            iterations=args.profile_iters,
            dt_seconds=args.dt_seconds,
        )
        return 0

    results = run_runtime_benchmark(
        number_of_particles=args.particle_count,
        runs=args.runs,
        warmup=args.warmup,
        dt_seconds=args.dt_seconds,
    )
    print_results(args.particle_count, results)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "particle_count": args.particle_count,
            "runs": args.runs,
            "warmup": args.warmup,
            "dt_seconds": args.dt_seconds,
            "results": [asdict(result) for result in results],
        }
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved benchmark JSON: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
