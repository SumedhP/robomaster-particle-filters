#pragma once

#include <thrust/execution_policy.h>

#include <Eigen/Dense>

namespace fast_plate_orbit_with_z_offset {

class predicted_plate {
 private:
  Eigen::Vector3f position_;

 public:
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& position() const noexcept { return position_; }

  PF_TARGET_ATTRS predicted_plate() noexcept : position_{Eigen::Vector3f::Zero()} {}

  PF_TARGET_ATTRS predicted_plate(const Eigen::Vector3f& position) noexcept
      : position_{position} {}
};

}  // namespace fast_plate_orbit_with_z_offset
