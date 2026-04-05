#pragma once

#include <Eigen/Dense>

#include <cstdint>

namespace fast_plate_orbit_with_z_offset {

struct particle_filter_configuration_parameters {
  float radius_prior;
  float visibility_logit_coefficient;

  float radius_prior_variance_one_plate;
  float radius_prior_variance_two_plates;
  float radius_process_variance;

  float z_coordinate_common_process_variance;
  float z_coordinate_offset_process_variance;

  float orientation_velocity_prior_variance;
  float orientation_velocity_process_variance;

  Eigen::Vector2f center_velocity_prior_diagonal_covariance;
  Eigen::Vector2f center_velocity_process_diagonal_covariance;

  float likelihood_refinement_window{-1.0f};
  std::uint32_t observation_resample_period{32U};
  std::uint32_t observation_update_subsample_stride{16U};
};

}  // namespace fast_plate_orbit_with_z_offset
