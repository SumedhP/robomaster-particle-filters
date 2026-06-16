#pragma once

#include <pf/config/target_config.h>
#include <thrust/execution_policy.h>

#include <Eigen/Dense>

namespace fast_plate_orbit_with_z_offset {

class observed_plate {
 private:
  Eigen::Vector3f position_;
  Eigen::Vector3f position_diagonal_covariance_;
  float yaw_;
  float yaw_variance_;

 public:
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& position() const noexcept { return position_; }

  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& position_diagonal_covariance() const noexcept {
    return position_diagonal_covariance_;
  }

  PF_TARGET_ATTRS [[nodiscard]] const float& yaw() const noexcept { return yaw_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& yaw_variance() const noexcept { return yaw_variance_; }

  PF_TARGET_ATTRS
  observed_plate(
      const Eigen::Vector3f& position,
      const Eigen::Vector3f& position_diagonal_covariance,
      const float& yaw,
      const float& yaw_variance) noexcept
      : position_{position},
        position_diagonal_covariance_{position_diagonal_covariance},
        yaw_{yaw},
        yaw_variance_{yaw_variance} {}
};

}  // namespace fast_plate_orbit_with_z_offset
