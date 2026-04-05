#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
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


def build_observation(x_offset: float, y_offset: float, yawish_offset: float) -> fpoz.Observation:
    observer_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    plate_one_position = np.array([0.12 + x_offset, 0.00 + y_offset, 0.00], dtype=np.float32)
    plate_two_position = np.array([0.00 - y_offset, 0.12 + x_offset, yawish_offset], dtype=np.float32)
    plate_position_diagonal_covariance = np.array([0.0001, 0.0001, 0.0001], dtype=np.float32)

    plate_one = fpoz.ObservedPlate(plate_one_position, plate_position_diagonal_covariance)
    plate_two = fpoz.ObservedPlate(plate_two_position, plate_position_diagonal_covariance)
    return fpoz.Observation.from_two_plates(observer_position, plate_one, plate_two)


def build_observation_ring(count: int) -> list[fpoz.Observation]:
    observations: list[fpoz.Observation] = []
    for i in range(count):
        phase = float(i) / float(max(1, count))
        x_offset = 0.004 * np.sin(2.0 * np.pi * phase)
        y_offset = 0.004 * np.cos(2.0 * np.pi * phase)
        yawish_offset = 0.002 * np.sin(4.0 * np.pi * phase)
        observations.append(build_observation(x_offset, y_offset, yawish_offset))
    return observations


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


def run_benchmarks(
    number_of_particles: int,
    runs: int,
    warmup: int,
    dt_seconds: float,
    observation_count: int,
) -> list[BenchmarkResult]:
    observations = build_observation_ring(observation_count)
    config = build_default_config()

    observation_index = 0

    def next_observation() -> fpoz.Observation:
        nonlocal observation_index
        obs = observations[observation_index]
        observation_index = (observation_index + 1) % len(observations)
        return obs

    initial_observation = observations[0]

    def construct_filter() -> None:
        _ = fpoz.ParticleFilter(number_of_particles, next_observation(), config)

    particle_filter = fpoz.ParticleFilter(number_of_particles, initial_observation, config)

    def reinitialize() -> None:
        particle_filter.reinitialize(next_observation())

    def reinitialize_exact() -> None:
        particle_filter.reinitialize(next_observation(), True)

    def reinitialize_and_extract_state() -> None:
        particle_filter.reinitialize(next_observation(), True)
        _ = particle_filter.extrapolate_state(dt_seconds)

    def update_with_observation_and_extract_state() -> None:
        obs = next_observation()
        particle_filter.update_state_with_observation(dt_seconds, obs)
        _ = particle_filter.extrapolate_state(dt_seconds)

    def reinitialize_then_update_and_extract_state() -> None:
        obs = next_observation()
        particle_filter.reinitialize(obs)
        particle_filter.update_state_with_observation(dt_seconds, obs)
        _ = particle_filter.extrapolate_state(dt_seconds)

    methods: Sequence[tuple[str, Callable[[], None]]] = (
        ("construct_filter", construct_filter),
        ("reinitialize_dispatch_only", reinitialize),
        ("reinitialize_exact", reinitialize_exact),
        ("reinitialize_and_extract_state", reinitialize_and_extract_state),
        ("update_state_with_observation_and_extract_state", update_with_observation_and_extract_state),
        ("reinitialize_then_update_and_extract_state", reinitialize_then_update_and_extract_state),
    )

    results: list[BenchmarkResult] = []
    for name, benchmark_target in methods:
        mean_us, std_us = benchmark_callable(benchmark_target, runs=runs, warmup=warmup)
        results.append(BenchmarkResult(name, mean_us, std_us))

    return results


def print_results(number_of_particles: int, results: Sequence[BenchmarkResult]) -> None:
    print(f"Particles: {number_of_particles:,}")
    print("method,mean_us,std_us")
    for result in results:
        print(f"{result.method},{result.mean_us:.3f},{result.std_us:.3f}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark constructor-vs-reinitialize latency for fast_plate_orbit_with_z_offset.ParticleFilter"
        )
    )
    parser.add_argument("--runs", type=int, default=100, help="Timed runs per method")
    parser.add_argument("--warmup", type=int, default=20, help="Warm-up runs per method")
    parser.add_argument("--dt-seconds", type=float, default=1.0 / 60.0, help="Time delta passed to filter update/extrapolation")
    parser.add_argument("--particle-count", type=int, default=1_000_000, help="Particle count")
    parser.add_argument(
        "--observation-count",
        type=int,
        default=64,
        help="Number of prebuilt observations cycled through during benchmarks.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be greater than 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.particle_count <= 0:
        raise ValueError("--particle-count must be greater than 0")
    if args.observation_count <= 0:
        raise ValueError("--observation-count must be greater than 0")

    print("Benchmarking fast_plate_orbit_with_z_offset.ParticleFilter reinitialization")
    print(f"Runs per method: {args.runs}")
    print(f"Warm-up runs per method: {args.warmup}")
    print(f"Observation ring size: {args.observation_count}")
    print("Units: microseconds")

    results = run_benchmarks(
        number_of_particles=args.particle_count,
        runs=args.runs,
        warmup=args.warmup,
        dt_seconds=args.dt_seconds,
        observation_count=args.observation_count,
    )

    print_results(args.particle_count, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
