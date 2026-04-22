#pragma once

#include <fast_plate_orbit_with_z_offset/observation.h>

#include <Eigen/Dense>

namespace fast_plate_orbit_with_z_offset {

struct initialization_prior {
  Eigen::Vector3f observer_position_estimate;
  float observer_position_confidence;
  float observer_position_variance;

  float z_coordinate_0_estimate;
  float z_coordinate_1_estimate;
  float z_coordinate_confidence;
  float z_coordinate_variance;

  PF_TARGET_ATTRS [[nodiscard]] static initialization_prior from_observation(const observation& state) noexcept {
    const float observed_z = state.plate_one().position().z();
    return initialization_prior{
        state.observer_position(),
        0.0f,
        1.0e-4f,
        observed_z,
        observed_z,
        0.0f,
        state.plate_one().position_diagonal_covariance().z()};
  }
};

}  // namespace fast_plate_orbit_with_z_offset
