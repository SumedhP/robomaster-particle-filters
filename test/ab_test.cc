// Head to head: the stock upstream pf::filter::particle_filter (which
// resamples unconditionally every observation) against util::rb_particle_filter
// (accumulating weights, ESS gated resampling), driving the identical
// plate_orbit_v3 configuration.
//
// Reports steady state error and, more importantly, the spread of the
// estimator across independent seeds. That spread is the Monte Carlo variance
// of the filter itself, which is exactly what resampling policy affects.

#include <pf/filter/particle_filter.h>
#include <plate_orbit_v3/particle_filter_configuration.h>
#include <util/rb_particle_filter.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

struct truth {
  float cx, cy, vx, vy, theta, omega, r0, r1, z0, z1;
};

plate_orbit_v3::particle_filter_configuration_parameters default_params() {
  plate_orbit_v3::particle_filter_configuration_parameters p{};
  p.radius_prior = 0.30f;
  p.visibility_logit_coefficient = 1.0f;
  p.radius_prior_variance_one_plate = 0.0025f;
  p.radius_prior_variance_two_plates = 0.000225f;
  p.radius_common_process_variance = 5e-6f;
  p.radius_common_reversion_time_constant = 20.0f;
  p.radius_offset_stationary_variance = 0.0025f;
  p.radius_offset_reversion_time_constant = 5.0f;
  p.z_common_process_variance = 4e-7f;
  p.z_offset_stationary_variance = 0.0016f;
  p.z_offset_reversion_time_constant = 5.0f;
  p.orientation_prior_variance = 0.25f;
  p.orientation_velocity_prior_variance = 100.0f;
  p.orientation_velocity_process_variance = 5.0f;
  p.center_velocity_prior_diagonal_covariance = Eigen::Vector2f(0.5f, 0.5f);
  p.center_velocity_process_diagonal_covariance = Eigen::Vector2f(0.25f, 0.25f);
  p.resample_effective_sample_size_fraction = 0.5f;
  return p;
}

void true_plates(const truth& t, Eigen::Vector3f out[4]) {
  const float radius[4] = {t.r0, t.r1, t.r0, t.r1};
  const float height[4] = {t.z0, t.z1, t.z0, t.z1};
  for (int k = 0; k < 4; ++k) {
    const float a = t.theta + static_cast<float>(k) * static_cast<float>(M_PI_2);
    out[k] = Eigen::Vector3f(t.cx + radius[k] * cosf(a), t.cy + radius[k] * sinf(a), height[k]);
  }
}

float wrap_pi(float a) {
  while (a > static_cast<float>(M_PI)) a -= 2.0f * static_cast<float>(M_PI);
  while (a < -static_cast<float>(M_PI)) a += 2.0f * static_cast<float>(M_PI);
  return a;
}

struct run_result {
  float z_offset_error;
  float r_offset_error;
  float center_error;
  float omega_error;
  double milliseconds;
  int resample_count;
};

// One trajectory, one seed. Templated over the filter driver so both share
// every other line of code.
template <typename Filter, typename Construct>
run_result run_once(std::size_t particles, float omega_true, unsigned seed, Construct construct) {
  const float dt = 1.0f / 60.0f;  // 60 FPS camera
  const int steps = STEPS;
  const int score_from = 2 * STEPS / 3;
  const float sigma = 0.01f;
  const Eigen::Vector3f noise_variance(sigma * sigma, sigma * sigma, sigma * sigma);
  const Eigen::Vector3f observer(0.0f, 0.0f, 0.0f);

  truth t{2.5f, 0.4f, -0.35f, 0.15f, 0.3f, omega_true, 0.245f, 0.26f, 0.08f, 0.16f};
  const float true_r_offset = 0.5f * (t.r0 - t.r1);
  const float true_z_offset = 0.5f * (t.z0 - t.z1);

  std::mt19937 rng(seed);
  std::normal_distribution<float> gauss(0.0f, sigma);

  auto make_observation = [&](const truth& s) {
    Eigen::Vector3f plates[4];
    true_plates(s, plates);
    const Eigen::Vector2f observer_delta = observer.head<2>() - Eigen::Vector2f(s.cx, s.cy);
    int visible[4];
    int count = 0;
    for (int k = 0; k < 4; ++k) {
      const Eigen::Vector2f plate_delta = plates[k].head<2>() - Eigen::Vector2f(s.cx, s.cy);
      if (observer_delta.normalized().dot(plate_delta.normalized()) > 0.25f) { visible[count++] = k; }
    }
    if (count == 0) { visible[count++] = 0; }
    auto noisy = [&](int k) {
      return plate_orbit_v3::observed_plate(
          plates[k] + Eigen::Vector3f(gauss(rng), gauss(rng), gauss(rng)), noise_variance);
    };
    if (count >= 2) { return plate_orbit_v3::observation::from_two_plates(observer, noisy(visible[0]), noisy(visible[1])); }
    return plate_orbit_v3::observation::from_one_plate(observer, noisy(visible[0]));
  };

  auto observation = make_observation(t);
  Filter filter = construct(observation);

  double z_sum = 0.0, r_sum = 0.0, c_sum = 0.0, w_sum = 0.0;
  int scored = 0;
  int resample_count = 0;

  const auto start = std::chrono::steady_clock::now();
  for (int step = 1; step <= steps; ++step) {
    t.theta = wrap_pi(t.theta + dt * t.omega);
    t.cx += dt * t.vx;
    t.cy += dt * t.vy;

    observation = make_observation(t);
    filter.update_state_with_observation(dt, observation);

    if constexpr (requires { filter.resampled(); }) {
      if (filter.resampled()) { ++resample_count; }
    } else {
      ++resample_count;
    }

    if (step >= score_from) {
      const auto s = filter.extrapolate_state(0.0f);
      // Offsets are sign ambiguous under the plate relabeling gauge, so score
      // on magnitude.
      const float r_offset = 0.5f * (s.radius_0() - s.radius_1());
      const float z_offset = 0.5f * (s.z_coordinate_0() - s.z_coordinate_1());
      r_sum += std::fabs(std::fabs(r_offset) - std::fabs(true_r_offset));
      z_sum += std::fabs(std::fabs(z_offset) - std::fabs(true_z_offset));
      c_sum += (s.center() - Eigen::Vector2f(t.cx, t.cy)).norm();
      w_sum += std::fabs(std::fabs(s.orientation_velocity()) - std::fabs(t.omega));
      ++scored;
    }
  }
  const auto end = std::chrono::steady_clock::now();

  return run_result{
      static_cast<float>(z_sum / scored),
      static_cast<float>(r_sum / scored),
      static_cast<float>(c_sum / scored),
      static_cast<float>(w_sum / scored),
      std::chrono::duration<double, std::milli>(end - start).count() / steps,
      resample_count,
  };
}

struct summary {
  float mean;
  float spread;
};

summary summarize(const std::vector<float>& values) {
  double sum = 0.0;
  for (float v : values) sum += v;
  const double mean = sum / values.size();
  double variance = 0.0;
  for (float v : values) variance += (v - mean) * (v - mean);
  variance /= std::max<std::size_t>(1, values.size() - 1);
  return summary{static_cast<float>(mean), static_cast<float>(std::sqrt(variance))};
}

}  // namespace

int main(int argc, char** argv) {
  const int seeds = (argc > 1) ? std::stoi(argv[1]) : 12;
  const float omega_true = (argc > 2) ? std::stof(argv[2]) : 3.0f;
  const auto params = default_params();

  using stock_filter = pf::filter::particle_filter<plate_orbit_v3::particle_filter_configuration>;
  using gated_filter = util::rb_particle_filter<plate_orbit_v3::particle_filter_configuration>;

  const std::size_t counts[] = {COUNTS};

  printf("plate_orbit_v3 config, omega=%.1f, %d seeds each, scored over final third of %d steps\n\n",
         omega_true, seeds, STEPS);
  printf("%-7s %-6s | %-17s | %-17s | %-17s | %-15s | %8s %7s\n",
         "policy", "N", "z_offset err (m)", "r_offset err (m)", "center err (m)", "omega err", "ms/step", "resamp");
  printf("%-7s %-6s | %-17s | %-17s | %-17s | %-15s | %8s %7s\n",
         "", "", "mean +- seed sd", "mean +- seed sd", "mean +- seed sd", "mean +- sd", "", "%");

  for (const std::size_t n : counts) {
    for (int policy = 0; policy < 2; ++policy) {
      std::vector<float> z, r, c, w;
      double ms = 0.0;
      long resamples = 0;

      for (int s = 0; s < seeds; ++s) {
        run_result result{};
        if (policy == 0) {
          result = run_once<stock_filter>(n, omega_true, 1000u + s, [&](const auto& obs) {
            return stock_filter(n, obs, params);
          });
        } else {
          result = run_once<gated_filter>(n, omega_true, 1000u + s, [&](const auto& obs) {
            return gated_filter(n, params.resample_effective_sample_size_fraction, obs, params);
          });
        }
        z.push_back(result.z_offset_error);
        r.push_back(result.r_offset_error);
        c.push_back(result.center_error);
        w.push_back(result.omega_error);
        ms += result.milliseconds;
        resamples += result.resample_count;
      }

      const auto zs = summarize(z);
      const auto rs = summarize(r);
      const auto cs = summarize(c);
      const auto ws = summarize(w);

      printf("%-7s %-6zu | %.4f +- %.4f | %.4f +- %.4f | %.4f +- %.4f | %.3f +- %.3f | %8.3f %6.0f%%\n",
             (policy == 0) ? "stock" : "gated", n,
             zs.mean, zs.spread, rs.mean, rs.spread, cs.mean, cs.spread, ws.mean, ws.spread,
             ms / seeds, 100.0 * static_cast<double>(resamples) / (seeds * static_cast<double>(STEPS)));
    }
    printf("\n");
  }
  return 0;
}
