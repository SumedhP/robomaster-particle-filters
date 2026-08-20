"""End to end smoke test for the plate_orbit_v3 Python bindings.

Simulates the same spinning, translating robot as test/v3_test.cc and asserts
the filter recovers the geometry. Run it from anywhere:

    python3 test/test_v3_bindings.py
"""
import math
import os
import sys

# The extension module lives in build/ until the package is installed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "build"))

import numpy as np
from robomaster_particle_filters import plate_orbit_v3 as v3

DT, STEPS, SIGMA = 0.0166, 600, 0.01
R0, R1, Z0, Z1, OMEGA = 0.245, 0.26, 0.08, 0.16, 3.0
OBSERVER = np.array([0.0, 0.0, 0.0], dtype=np.float32)
NOISE_VAR = np.full(3, SIGMA * SIGMA, dtype=np.float32)


def params():
    return v3.ParticleFilterConfigurationParameters(
        0.30,      # radius_prior
        1.0,       # visibility_logit_coefficient
        0.0025,    # radius_prior_variance_one_plate
        0.000225,  # radius_prior_variance_two_plates
        5e-6,      # radius_common_process_variance
        20.0,      # radius_common_reversion_time_constant
        0.0025,    # radius_offset_stationary_variance
        5.0,       # radius_offset_reversion_time_constant
        4e-7,      # z_common_process_variance
        0.0016,    # z_offset_stationary_variance
        5.0,       # z_offset_reversion_time_constant
        0.25,      # orientation_prior_variance
        100.0,     # orientation_velocity_prior_variance
        5.0,       # orientation_velocity_process_variance
        np.array([0.5, 0.5], dtype=np.float32),
        np.array([0.25, 0.25], dtype=np.float32),
        0.5,       # resample_effective_sample_size_fraction
    )


def true_plates(cx, cy, theta):
    radius, height = (R0, R1, R0, R1), (Z0, Z1, Z0, Z1)
    return [
        np.array([cx + radius[k] * math.cos(theta + k * math.pi / 2),
                  cy + radius[k] * math.sin(theta + k * math.pi / 2),
                  height[k]], dtype=np.float32)
        for k in range(4)
    ]


def observe(rng, cx, cy, theta):
    plates = true_plates(cx, cy, theta)
    to_observer = OBSERVER[:2] - np.array([cx, cy])
    to_observer /= np.linalg.norm(to_observer)

    visible = [k for k, p in enumerate(plates)
               if float(to_observer @ ((p[:2] - [cx, cy]) / np.linalg.norm(p[:2] - [cx, cy]))) > 0.25]
    if not visible:
        visible = [0]

    def noisy(k):
        pos = (plates[k] + rng.normal(0.0, SIGMA, 3)).astype(np.float32)
        return v3.ObservedPlate(pos, NOISE_VAR)

    if len(visible) >= 2:
        return v3.Observation.from_two_plates(OBSERVER, noisy(visible[0]), noisy(visible[1]))
    return v3.Observation.from_one_plate(OBSERVER, noisy(visible[0]))


def main():
    rng = np.random.default_rng(12345)
    cx, cy, vx, vy, theta = 2.5, 0.4, -0.35, 0.15, 0.3

    pf = v3.ParticleFilter(512, observe(rng, cx, cy, theta), params())

    for _ in range(STEPS):
        theta = (theta + DT * OMEGA + math.pi) % (2 * math.pi) - math.pi
        cx, cy = cx + DT * vx, cy + DT * vy
        pf.update_state_with_observation(DT, observe(rng, cx, cy, theta))

    s = pf.extrapolate_state(0.0)
    r_lo, r_hi = sorted((s.radius_0(), s.radius_1()))
    z_lo, z_hi = sorted((s.z_coordinate_0(), s.z_coordinate_1()))
    center_err = float(np.linalg.norm(np.asarray(s.center()) - [cx, cy]))
    cov = np.asarray(s.covariance())

    print(f"radii   {r_lo:.4f} / {r_hi:.4f}   truth {R0} / {R1}")
    print(f"heights {z_lo:.4f} / {z_hi:.4f}   truth {Z0} / {Z1}")
    print(f"omega   {s.orientation_velocity():.3f}   truth {OMEGA}")
    print(f"center  err {center_err:.4f}   ESS {pf.effective_sample_size():.0f}/512")
    print(f"covariance {cov.shape}, sigma(r_off) {math.sqrt(cov[6][6]):.4f}")

    # Labeling is gauge, so compare sorted. Bounds are loose: 1 cm on a filter
    # that measures ~1.5 mm, so seed changes do not turn this red.
    assert abs(r_lo - R0) < 0.01 and abs(r_hi - R1) < 0.01, "radii did not converge"
    assert abs(z_lo - Z0) < 0.01 and abs(z_hi - Z1) < 0.01, "plate heights did not converge"
    assert abs(s.orientation_velocity() - OMEGA) < 0.5, "angular velocity did not converge"
    assert center_err < 0.05, "center did not converge"
    assert cov.shape == (9, 9) and np.all(np.diag(cov) > 0.0), "covariance not PSD-diagonal"
    assert len(s.predicted_plates()) == 4
    print("\nOK")


if __name__ == "__main__":
    main()
