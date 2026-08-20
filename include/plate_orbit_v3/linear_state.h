#pragma once

#include <pf/config/target_config.h>

#include <Eigen/Dense>

namespace plate_orbit_v3 {

// The Rao-Blackwellized linear substate. Conditional on the orientation
// (which is carried by the particle) both the process model and the plate
// position measurement model are exactly linear in these nine quantities,
// so their posterior is an exact Gaussian maintained by a Kalman filter.
//
//   index  quantity
//   -----  -------------------------------------------------------------
//     0    orientation_velocity   (rad / s)
//     1    center x               (m)
//     2    center y               (m)
//     3    center velocity x      (m / s)
//     4    center velocity y      (m / s)
//     5    radius common mode     (m)      r_k = r_c + s_k * r_o
//     6    radius offset mode     (m)      s_k = +1 (k even), -1 (k odd)
//     7    z common mode          (m)      z_k = z_c + s_k * z_o
//     8    z offset mode          (m)
//
// The (common, offset) basis replaces v2's (value_0, value_1) pair. It makes
// the process noise diagonal, turns the pi/2 plate relabeling symmetry into a
// signed permutation, and lets the bounded-offset clamp become linear
// Ornstein-Uhlenbeck mean reversion.

struct linear_state {
  static constexpr int dimension = 9;

  static constexpr int index_orientation_velocity = 0;
  static constexpr int index_center_x = 1;
  static constexpr int index_center_y = 2;
  static constexpr int index_center_velocity_x = 3;
  static constexpr int index_center_velocity_y = 4;
  static constexpr int index_radius_common = 5;
  static constexpr int index_radius_offset = 6;
  static constexpr int index_z_common = 7;
  static constexpr int index_z_offset = 8;

  using vector_type = Eigen::Matrix<float, dimension, 1>;
  using matrix_type = Eigen::Matrix<float, dimension, dimension>;

  vector_type mean_;
  matrix_type covariance_;

  PF_TARGET_ATTRS [[nodiscard]] const vector_type& mean() const noexcept { return mean_; }
  PF_TARGET_ATTRS [[nodiscard]] const matrix_type& covariance() const noexcept { return covariance_; }

  PF_TARGET_ATTRS [[nodiscard]] const float& orientation_velocity() const noexcept {
    return mean_[index_orientation_velocity];
  }

  PF_TARGET_ATTRS [[nodiscard]] Eigen::Vector2f center() const noexcept {
    return mean_.segment<2>(index_center_x);
  }

  PF_TARGET_ATTRS [[nodiscard]] Eigen::Vector2f center_velocity() const noexcept {
    return mean_.segment<2>(index_center_velocity_x);
  }

  PF_TARGET_ATTRS [[nodiscard]] float radius_0() const noexcept {
    return mean_[index_radius_common] + mean_[index_radius_offset];
  }

  PF_TARGET_ATTRS [[nodiscard]] float radius_1() const noexcept {
    return mean_[index_radius_common] - mean_[index_radius_offset];
  }

  PF_TARGET_ATTRS [[nodiscard]] float z_coordinate_0() const noexcept {
    return mean_[index_z_common] + mean_[index_z_offset];
  }

  PF_TARGET_ATTRS [[nodiscard]] float z_coordinate_1() const noexcept {
    return mean_[index_z_common] - mean_[index_z_offset];
  }

  // theta -> theta + pi/2 maps plate k onto plate k+1, which negates both
  // offset modes and leaves every other coordinate alone. As a similarity
  // transform this is a signed diagonal, so it applies to the covariance as
  // T * P * T^T with no matrix multiply required.
  PF_TARGET_ATTRS void apply_quarter_turn_relabeling() noexcept {
    mean_[index_radius_offset] = -mean_[index_radius_offset];
    mean_[index_z_offset] = -mean_[index_z_offset];

    for (int i = 0; i < dimension; ++i) {
      if (i == index_radius_offset || i == index_z_offset) { continue; }
      covariance_(i, index_radius_offset) = -covariance_(i, index_radius_offset);
      covariance_(index_radius_offset, i) = -covariance_(index_radius_offset, i);
      covariance_(i, index_z_offset) = -covariance_(i, index_z_offset);
      covariance_(index_z_offset, i) = -covariance_(index_z_offset, i);
    }
  }

  PF_TARGET_ATTRS void symmetrize() noexcept {
    covariance_ = (0.5f * (covariance_ + covariance_.transpose())).eval();
  }

  PF_TARGET_ATTRS linear_state() noexcept : mean_{vector_type::Zero()}, covariance_{matrix_type::Zero()} {}

  PF_TARGET_ATTRS linear_state(const vector_type& mean, const matrix_type& covariance) noexcept
      : mean_{mean}, covariance_{covariance} {}
};

}  // namespace plate_orbit_v3
