#include <plate_orbit_v3/particle_filter_configuration.h>
#include <util/rb_particle_filter.h>

#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct truth {
  float cx, cy, vx, vy;
  float theta, omega;
  float r0, r1;
  float z0, z1;
};

plate_orbit_v3::particle_filter_configuration_parameters default_params() {
  plate_orbit_v3::particle_filter_configuration_parameters p{};
  p.radius_prior = 0.30f;
  p.visibility_logit_coefficient = 1.0f;

  p.radius_prior_variance_one_plate = 0.0025f;
  p.radius_prior_variance_two_plates = 0.000225f;

  p.radius_common_process_variance = 5e-6f;
  p.radius_common_reversion_time_constant = 20.0f;

  p.radius_offset_stationary_variance = 0.0025f;  // 5 cm stationary sigma
  p.radius_offset_reversion_time_constant = 5.0f;

  p.z_common_process_variance = 4e-7f;

  p.z_offset_stationary_variance = 0.0016f;  // 4 cm stationary sigma
  p.z_offset_reversion_time_constant = 5.0f;

  p.orientation_prior_variance = 0.25f;  // 0.5 rad sigma
  p.orientation_velocity_prior_variance = 100.0f;
  p.orientation_velocity_process_variance = 5.0f;

  p.center_velocity_prior_diagonal_covariance = Eigen::Vector2f(0.5f, 0.5f);
  p.center_velocity_process_diagonal_covariance = Eigen::Vector2f(0.25f, 0.25f);

  p.resample_effective_sample_size_fraction = 0.5f;
  return p;
}

// Ground truth plate positions, using the same geometry the filter assumes.
void true_plates(const truth& t, Eigen::Vector3f out[4]) {
  const float radius[4] = {t.r0, t.r1, t.r0, t.r1};
  const float height[4] = {t.z0, t.z1, t.z0, t.z1};
  for (int k = 0; k < 4; ++k) {
    const float angle = t.theta + static_cast<float>(k) * static_cast<float>(M_PI_2);
    out[k] = Eigen::Vector3f(t.cx + radius[k] * cosf(angle), t.cy + radius[k] * sinf(angle), height[k]);
  }
}

float wrap_pi(float a) {
  while (a > static_cast<float>(M_PI)) a -= 2.0f * static_cast<float>(M_PI);
  while (a < -static_cast<float>(M_PI)) a += 2.0f * static_cast<float>(M_PI);
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t particle_count = (argc > 1) ? std::stoul(argv[1]) : 4096;
  const float omega_true = (argc > 2) ? std::stof(argv[2]) : 3.0f;

  const float dt = 1.0f / 60.0f;  // 60 FPS camera
  const int steps = 600;  // 10 seconds at 60 FPS
  const float noise_sigma = 0.01f;  // 1 cm per axis
  const Eigen::Vector3f noise_variance(noise_sigma * noise_sigma, noise_sigma * noise_sigma, noise_sigma * noise_sigma);
  const Eigen::Vector3f observer(0.0f, 0.0f, 0.0f);

  truth t{};
  t.cx = 2.5f;
  t.cy = 0.4f;
  t.vx = -0.35f;
  t.vy = 0.15f;
  t.theta = 0.3f;
  t.omega = omega_true;
  t.r0 = 0.245f;
  t.r1 = 0.26f;   // mean 0.2525 against a 0.30 prior, 7.5 mm offset
  t.z0 = 0.08f;
  t.z1 = 0.16f;   // 8 cm plate height split

  std::mt19937 rng(12345);
  std::normal_distribution<float> gauss(0.0f, noise_sigma);

  auto make_observation = [&](const truth& state) {
    Eigen::Vector3f plates[4];
    true_plates(state, plates);

    // Visible plates: those whose outward normal faces the observer.
    const Eigen::Vector2f observer_delta = observer.head<2>() - Eigen::Vector2f(state.cx, state.cy);
    int visible[4];
    int visible_count = 0;
    for (int k = 0; k < 4; ++k) {
      const Eigen::Vector2f plate_delta = plates[k].head<2>() - Eigen::Vector2f(state.cx, state.cy);
      const float similarity = observer_delta.normalized().dot(plate_delta.normalized());
      if (similarity > 0.25f) { visible[visible_count++] = k; }
    }
    if (visible_count == 0) {
      visible[0] = 0;
      visible_count = 1;
    }

    auto noisy = [&](int k) {
      return plate_orbit_v3::observed_plate(
          plates[k] + Eigen::Vector3f(gauss(rng), gauss(rng), gauss(rng)), noise_variance);
    };

    if (visible_count >= 2) {
      return plate_orbit_v3::observation::from_two_plates(observer, noisy(visible[0]), noisy(visible[1]));
    }
    return plate_orbit_v3::observation::from_one_plate(observer, noisy(visible[0]));
  };

  const auto params = default_params();
  auto observation = make_observation(t);

  util::rb_particle_filter<plate_orbit_v3::particle_filter_configuration> filter(
      particle_count, params.resample_effective_sample_size_fraction, observation, params);

  printf("plate_orbit_v3 (RBPF)  particles=%zu  omega_true=%.2f rad/s  dt=%.3f\n", particle_count, omega_true, dt);
  printf("truth: center=(%.3f,%.3f) vel=(%.2f,%.2f) r0=%.3f r1=%.3f z0=%.3f z1=%.3f\n\n",
         t.cx, t.cy, t.vx, t.vy, t.r0, t.r1, t.z0, t.z1);
  printf("%6s %8s %9s %9s %8s %8s %8s %8s %8s %7s %4s\n",
         "step", "t(s)", "cen_err", "omega", "r0", "r1", "z0", "z1", "plate_er", "ESS%", "rs");

  int resample_count = 0;
  const auto wall_start = std::chrono::steady_clock::now();

  for (int step = 1; step <= steps; ++step) {
    t.theta = wrap_pi(t.theta + dt * t.omega);
    t.cx += dt * t.vx;
    t.cy += dt * t.vy;

    observation = make_observation(t);
    filter.update_state_with_observation(dt, observation);
    if (filter.resampled()) { ++resample_count; }

    if (step % 60 == 0 || step == 1) {
      const auto s = filter.extrapolate_state(0.0f);
      const float center_error = (s.center() - Eigen::Vector2f(t.cx, t.cy)).norm();

      // Plate set error: compare the predicted plate cloud against truth,
      // matching each true plate to its nearest prediction. This is the
      // labeling invariant way to score, since the plate indices are gauge.
      Eigen::Vector3f truth_plates[4];
      true_plates(t, truth_plates);
      const auto predicted = s.predicted_plates_for_host();

      float plate_error = 0.0f;
      for (int k = 0; k < 4; ++k) {
        float best = 1e9f;
        for (int j = 0; j < 4; ++j) {
          best = std::min(best, (truth_plates[k] - predicted[j].position()).norm());
        }
        plate_error += best;
      }
      plate_error *= 0.25f;

      printf("%6d %8.3f %9.4f %9.3f %8.3f %8.3f %8.3f %8.3f %8.4f %7.1f %4s\n",
             step, step * dt, center_error, s.orientation_velocity(),
             s.radius_0(), s.radius_1(), s.z_coordinate_0(), s.z_coordinate_1(),
             plate_error,
             100.0f * filter.effective_sample_size() / static_cast<float>(particle_count),
             filter.resampled() ? "yes" : "");
    }
  }

  const auto wall_end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

  const auto s = filter.extrapolate_state(0.0f);
  const float radius_small = std::min(s.radius_0(), s.radius_1());
  const float radius_large = std::max(s.radius_0(), s.radius_1());
  const float z_low = std::min(s.z_coordinate_0(), s.z_coordinate_1());
  const float z_high = std::max(s.z_coordinate_0(), s.z_coordinate_1());

  printf("\nfinal (sorted, labeling is gauge):\n");
  printf("  radii   est %.4f / %.4f   truth %.4f / %.4f   err %.4f / %.4f\n",
         radius_small, radius_large, std::min(t.r0, t.r1), std::max(t.r0, t.r1),
         std::fabs(radius_small - std::min(t.r0, t.r1)), std::fabs(radius_large - std::max(t.r0, t.r1)));
  printf("  heights est %.4f / %.4f   truth %.4f / %.4f   err %.4f / %.4f\n",
         z_low, z_high, std::min(t.z0, t.z1), std::max(t.z0, t.z1),
         std::fabs(z_low - std::min(t.z0, t.z1)), std::fabs(z_high - std::max(t.z0, t.z1)));
  printf("  omega   est %.3f   truth %.3f   err %.3f\n", s.orientation_velocity(), t.omega,
         std::fabs(s.orientation_velocity() - t.omega));
  printf("  center  est (%.3f,%.3f)  truth (%.3f,%.3f)  err %.4f\n",
         s.center()[0], s.center()[1], t.cx, t.cy, (s.center() - Eigen::Vector2f(t.cx, t.cy)).norm());
  printf("  cen vel est (%.3f,%.3f)  truth (%.3f,%.3f)\n",
         s.center_velocity()[0], s.center_velocity()[1], t.vx, t.vy);
  const auto cov = s.covariance_for_host();
  printf("  posterior sigma:  omega %.3f  cx %.4f  cy %.4f  r_c %.4f  r_off %.4f  z_c %.4f  z_off %.4f\n",
         std::sqrt(std::max(0.0f, cov(0,0))), std::sqrt(std::max(0.0f, cov(1,1))), std::sqrt(std::max(0.0f, cov(2,2))),
         std::sqrt(std::max(0.0f, cov(5,5))), std::sqrt(std::max(0.0f, cov(6,6))),
         std::sqrt(std::max(0.0f, cov(7,7))), std::sqrt(std::max(0.0f, cov(8,8))));

  printf("\n%d steps in %.1f ms  (%.4f ms/step on %d threads, %.1f%% of a %.1f ms frame)   resampled %d/%d steps\n",
         steps, elapsed_ms, elapsed_ms / steps, omp_get_max_threads(),
         100.0 * (elapsed_ms / steps) / (1000.0 * dt), 1000.0 * dt, resample_count, steps);
  return 0;
}
