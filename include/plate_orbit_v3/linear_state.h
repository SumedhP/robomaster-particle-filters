#pragma once

#include <pf/config/target_config.h>

#include <Eigen/Dense>

namespace plate_orbit_v3 {

// The Rao-Blackwellized linear substate. Conditional on the orientation
// (which is carried by the particle) both the process model and the plate
// position measurement model are exactly linear in these eight quantities,
// so their posterior is an exact Gaussian maintained by a Kalman filter.
//
//   index  quantity
//   -----  -------------------------------------------------------------
//     0    orientation_velocity   (rad / s)
//     1    center x               (m)
//     2    center y               (m)
//     3    center velocity x      (m / s)
//     4    center velocity y      (m / s)
//     5    radius                 (m)      shared by all four plates
//     6    z common mode          (m)      z_k = z_c + s_k * z_o
//     7    z offset mode          (m)      s_k = +1 (k even), -1 (k odd)
//
// The chassis is modeled as circular: one radius for all four plates. The
// plate heights keep the (common, offset) split, because opposing pairs
// genuinely do sit at two different heights.
//
// This asymmetry between radius and height is not arbitrary. Perturbing the
// center by delta and every radius by -delta * e_k leaves all four plate
// positions unchanged when the yaw is fixed, so per-plate radii are only
// weakly identifiable against center position. Heights have no such
// degeneracy: the body z axis stays aligned with world z whatever the robot
// does, so the height split is observable whenever a plate is seen.

struct linear_state {
  static constexpr int dimension = 8;

  static constexpr int index_orientation_velocity = 0;
  static constexpr int index_center_x = 1;
  static constexpr int index_center_y = 2;
  static constexpr int index_center_velocity_x = 3;
  static constexpr int index_center_velocity_y = 4;
  static constexpr int index_radius = 5;
  static constexpr int index_z_common = 6;
  static constexpr int index_z_offset = 7;

  // DontAlign is required, not cosmetic. At dimension 8 these types are 32 and
  // 256 bytes, which makes them "fixed-size vectorizable" in Eigen's sense, so
  // Eigen would demand 32 byte alignment for every one of them. These live
  // inside prediction, which lives inside a thrust vector whose allocator makes
  // no such guarantee, and the misaligned AVX load segfaults. At dimension 9
  // the types were 36 bytes and the question never arose.
  using vector_type = Eigen::Matrix<float, dimension, 1, Eigen::DontAlign>;
  using matrix_type = Eigen::Matrix<float, dimension, dimension, Eigen::DontAlign>;

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

  PF_TARGET_ATTRS [[nodiscard]] const float& radius() const noexcept { return mean_[index_radius]; }

  PF_TARGET_ATTRS [[nodiscard]] float z_coordinate_0() const noexcept {
    return mean_[index_z_common] + mean_[index_z_offset];
  }

  PF_TARGET_ATTRS [[nodiscard]] float z_coordinate_1() const noexcept {
    return mean_[index_z_common] - mean_[index_z_offset];
  }

  // theta -> theta + pi/2 maps plate k onto plate k+1, which negates the
  // height offset mode and leaves every other coordinate alone. As a
  // similarity transform this is a signed diagonal, so it applies to the
  // covariance as T * P * T^T with no matrix multiply required.
  PF_TARGET_ATTRS void apply_quarter_turn_relabeling() noexcept {
    mean_[index_z_offset] = -mean_[index_z_offset];

    for (int i = 0; i < dimension; ++i) {
      if (i == index_z_offset) { continue; }
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
