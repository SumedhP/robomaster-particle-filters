#pragma once

#include <Eigen/Dense>

namespace plate_orbit_v3 {

struct particle_filter_configuration_parameters {
  // Geometry prior.
  float radius_prior;
  float visibility_logit_coefficient;

  float radius_prior_variance_one_plate;
  float radius_prior_variance_two_plates;

  // Common mode radius: slow random walk with a weak pull back toward
  // radius_prior. The pull replaces v2's hard [0.2, 1.0] clamp in to_radius,
  // which was the last remaining nonlinearity in the linear substate.
  float radius_common_process_variance;
  float radius_common_reversion_time_constant;

  // Offset modes are Ornstein-Uhlenbeck rather than clamped random walks.
  // stationary_variance replaces v2's offset_limit: set it so that two or
  // three standard deviations covers the physical asymmetry you expect.
  float radius_offset_stationary_variance;
  float radius_offset_reversion_time_constant;

  float z_common_process_variance;

  float z_offset_stationary_variance;
  float z_offset_reversion_time_constant;

  // Orientation. The prior variance is new in v3: the orientation is now the
  // only sampled continuous quantity, so it needs explicit initial spread
  // rather than inheriting diversity from the center and radius draws.
  float orientation_prior_variance;
  float orientation_velocity_prior_variance;
  float orientation_velocity_process_variance;

  Eigen::Vector2f center_velocity_prior_diagonal_covariance;
  Eigen::Vector2f center_velocity_process_diagonal_covariance;

  // Resample only when the effective sample size falls below this fraction of
  // the population. Rao-Blackwellized particles carry a full Gaussian each, so
  // unconditional resampling destroys diversity in the one dimension that is
  // actually being sampled while gaining nothing.
  float resample_effective_sample_size_fraction;
};

}  // namespace plate_orbit_v3
