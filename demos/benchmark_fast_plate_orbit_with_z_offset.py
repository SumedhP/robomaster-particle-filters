#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from robomaster_particle_filters import fast_plate_orbit_with_z_offset as fpoz


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    mean_us: float
    std_us: float


def build_default_config() -> fpoz.ParticleFilterConfigurationParameters:
    return fpoz.ParticleFilterConfigurationParameters(
        0.11,  # radius_prior
        2.0,  # visibility_logit_coefficient
        0.001,  # radius_prior_variance_one_plate
        0.005,  # radius_prior_variance_two_plates
        0.0001,  # radius_process_variance
        3.0,  # z_coordinate_common_process_variance
        0.3,  # z_coordinate_offset_process_variance
        6.0,  # orientation_velocity_prior_variance
        1.5,  # orientation_velocity_process_variance
        np.array([6.0, 6.0], dtype=np.float32),
        np.array([10.0, 10.0], dtype=np.float32),
    )


def build_default_observation() -> fpoz.Observation:
    observer_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    plate_one_position = np.array([0.12, 0.00, 0.00], dtype=np.float32)
    plate_two_position = np.array([0.00, 0.12, 0.00], dtype=np.float32)
    plate_position_diagonal_covariance = np.array([0.0001, 0.0001, 0.0001], dtype=np.float32)

    plate_one = fpoz.ObservedPlate(plate_one_position, plate_position_diagonal_covariance)
    plate_two = fpoz.ObservedPlate(plate_two_position, plate_position_diagonal_covariance)
    return fpoz.Observation.from_two_plates(observer_position, plate_one, plate_two)


def parse_particle_counts(args: argparse.Namespace) -> list[int]:
    if args.particle_counts:
        values = [int(item.strip()) for item in args.particle_counts.split(",") if item.strip()]
        if not values:
            raise ValueError("--particle-counts was provided but no valid integer values were parsed")
        return values

    counts: list[int] = []
    current = args.min_particles
    while current <= args.max_particles:
        counts.append(current)
        if args.multiplier <= 1:
            break
        current *= args.multiplier

    if not counts:
        counts = [args.min_particles]

    if counts[-1] != args.max_particles:
        counts.append(args.max_particles)

    return sorted(set(counts))


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


def print_results(number_of_particles: int, results: Iterable[BenchmarkResult]) -> None:
    print(f"\nParticles: {number_of_particles:,}")
    print("method,mean_us,std_us")
    for result in results:
        print(f"{result.method},{result.mean_us:.3f},{result.std_us:.3f}")


def save_method_plots(
    benchmark_results_by_particle_count: dict[int, list[BenchmarkResult]],
    output_directory: Path,
    show_plots: bool,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    method_names = sorted(
        {
            result.method
            for results in benchmark_results_by_particle_count.values()
            for result in results
        }
    )
    particle_counts = sorted(benchmark_results_by_particle_count.keys())

    for method_name in method_names:
        x_values: list[int] = []
        mean_values: list[float] = []
        std_values: list[float] = []

        for particle_count in particle_counts:
            method_result = next(
                (result for result in benchmark_results_by_particle_count[particle_count] if result.method == method_name),
                None,
            )
            if method_result is None:
                continue
            x_values.append(particle_count)
            mean_values.append(method_result.mean_us)
            std_values.append(method_result.std_us)

        if not x_values:
            continue

        figure, axis = plt.subplots(figsize=(9, 5))
        axis.errorbar(
            x_values,
            mean_values,
            yerr=std_values,
            fmt="o-",
            capsize=4,
            linewidth=1.5,
        )
        axis.set_xscale("log")
        axis.set_xlabel("Particle count")
        axis.set_ylabel("Latency (microseconds)")
        axis.set_title(f"{method_name}: mean latency ± std dev")
        axis.grid(True, which="both", linestyle="--", alpha=0.35)

        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.ticklabel_format(style="plain", axis="x")

        figure.tight_layout()
        output_path = output_directory / f"{method_name}.png"
        figure.savefig(output_path, dpi=200)
        print(f"Saved plot: {output_path}")

        if show_plots:
            plt.show(block=False)
        plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fast_plate_orbit_with_z_offset particle filter methods and report mean/std in microseconds."
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
    gc.disable()
    try:
        for number_of_particles in particle_counts:
            try:
                results = run_benchmarks_for_particle_count(
                    number_of_particles=number_of_particles,
                    runs=args.runs,
                    warmup=args.warmup,
                    dt_seconds=args.dt_seconds,
                )
                benchmark_results_by_particle_count[number_of_particles] = results
                print_results(number_of_particles, results)
            except Exception as exc:
                print(f"\nParticles: {number_of_particles:,}")
                print(f"FAILED: {type(exc).__name__}: {exc}")

        if benchmark_results_by_particle_count:
            save_method_plots(
                benchmark_results_by_particle_count=benchmark_results_by_particle_count,
                output_directory=Path(args.plot_output_dir),
                show_plots=args.show_plots,
            )
        else:
            print("No successful benchmark results were produced; skipping plot generation.")
    finally:
        if gc_state:
            gc.enable()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
