#pragma once

#include <fast_plate_orbit_with_z_offset/initialization_prior.h>

#include <Eigen/Dense>

namespace fast_plate_orbit_with_z_offset {

struct particle_filter_configuration_parameters {
  initialization_prior default_initialization_prior;
  float visibility_logit_coefficient;

  float mirrored_yaw_penalty;

  float radius_prior_variance_one_plate;
  float radius_prior_variance_two_plates;
  float radius_process_variance;

  float z_coordinate_common_process_variance;
  float z_coordinate_offset_process_variance;
  float z_height_prior_variance;

  float orientation_velocity_prior_variance;
  float orientation_velocity_process_variance;

  Eigen::Vector2f center_velocity_prior_diagonal_covariance;
  Eigen::Vector2f center_velocity_process_diagonal_covariance;
};

}  // namespace fast_plate_orbit_with_z_offset
