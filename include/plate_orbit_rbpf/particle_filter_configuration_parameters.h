#pragma once

#include <Eigen/Dense>

namespace plate_orbit_rbpf {

struct particle_filter_configuration_parameters {
  // Geometry prior.
  float radius_prior;
  float visibility_logit_coefficient;

  float radius_prior_variance_one_plate;
  float radius_prior_variance_two_plates;

  // Radius: slow random walk with a weak pull back toward radius_prior. The
  // pull replaces v2's hard [0.2, 1.0] clamp in to_radius, which was the last
  // remaining nonlinearity in the linear substate.
  float radius_process_variance;
  float radius_reversion_time_constant;

  float z_common_process_variance;

  // The height offset is Ornstein-Uhlenbeck rather than a clamped random
  // walk. stationary_variance replaces v2's offset_limit: set it so that two
  // or three standard deviations covers the height split you expect.
  float z_offset_stationary_variance;
  float z_offset_reversion_time_constant;

  // Orientation. The prior variance is new here: the orientation is now the
  // only sampled continuous quantity, so it needs explicit initial spread
  // rather than inheriting diversity from the center and radius draws.
  float orientation_prior_variance;
  float orientation_velocity_prior_variance;
  float orientation_velocity_process_variance;

  Eigen::Vector2f center_velocity_prior_diagonal_covariance;
  Eigen::Vector2f center_velocity_process_diagonal_covariance;
};

}  // namespace plate_orbit_rbpf
