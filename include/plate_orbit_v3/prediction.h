#pragma once

#include <pf/util/device_array.h>
#include <plate_orbit_v3/linear_state.h>
#include <plate_orbit_v3/predicted_plate.h>

#include <Eigen/Dense>
#include <array>
#include <cmath>

namespace plate_orbit_v3 {

namespace helper {

PF_TARGET_ATTRS [[nodiscard]] inline Eigen::Vector3f rpad_zero(const Eigen::Vector2f& vector) noexcept {
  return (Eigen::Vector3f{} << vector, Eigen::Matrix<float, 1, 1>::Zero()).finished();
}

// As in v2: the plate set {theta, theta + pi/2, theta + pi, theta + 3pi/2}
// is invariant under theta -> theta + pi, so the orientation is only ever
// meaningful modulo pi.
PF_TARGET_ATTRS [[nodiscard]] inline float to_orientation(const float& angle_radians) noexcept {
  const float value = fmodf(angle_radians, static_cast<float>(M_PI));
  return (value < 0.0f) ? value + static_cast<float>(M_PI) : value;
}

}  // namespace helper

// A single particle: one hypothesis about the orientation, together with the
// exact Gaussian posterior over everything else conditional on that hypothesis.
class prediction {
 private:
  static constexpr std::size_t number_of_plates = 4;

  float orientation_;
  linear_state linear_;

  struct plate_parameters {
    int index;
    float angle;
    float sign;
    float radius;
    float z_coordinate;
  };

 public:
  PF_TARGET_ATTRS [[nodiscard]] const float& orientation() const noexcept { return orientation_; }
  PF_TARGET_ATTRS [[nodiscard]] const linear_state& linear() const noexcept { return linear_; }
  PF_TARGET_ATTRS [[nodiscard]] linear_state& linear() noexcept { return linear_; }

  PF_TARGET_ATTRS void set_orientation(const float& value) noexcept { orientation_ = helper::to_orientation(value); }

  // Mirrors the v2 accessor surface so downstream consumers do not change.
  PF_TARGET_ATTRS [[nodiscard]] float radius_0() const noexcept { return linear_.radius_0(); }
  PF_TARGET_ATTRS [[nodiscard]] float radius_1() const noexcept { return linear_.radius_1(); }
  PF_TARGET_ATTRS [[nodiscard]] float z_coordinate_0() const noexcept { return linear_.z_coordinate_0(); }
  PF_TARGET_ATTRS [[nodiscard]] float z_coordinate_1() const noexcept { return linear_.z_coordinate_1(); }
  PF_TARGET_ATTRS [[nodiscard]] float orientation_velocity() const noexcept { return linear_.orientation_velocity(); }
  PF_TARGET_ATTRS [[nodiscard]] Eigen::Vector2f center() const noexcept { return linear_.center(); }
  PF_TARGET_ATTRS [[nodiscard]] Eigen::Vector2f center_velocity() const noexcept { return linear_.center_velocity(); }

  // New in v3: the filter now reports calibrated uncertainty rather than only
  // a point estimate, which is what makes downstream shot gating possible.
  [[nodiscard]] Eigen::Matrix<float, 9, 9> covariance_for_host() const noexcept { return linear_.covariance(); }

  PF_TARGET_ATTRS [[nodiscard]] pf::util::device_array<plate_parameters, number_of_plates> plate_parameters_array()
      const noexcept {
    const float radius_common = linear_.mean()[linear_state::index_radius_common];
    const float radius_offset = linear_.mean()[linear_state::index_radius_offset];
    const float z_common = linear_.mean()[linear_state::index_z_common];
    const float z_offset = linear_.mean()[linear_state::index_z_offset];

    pf::util::device_array<plate_parameters, number_of_plates> result{};
    for (int k = 0; k < static_cast<int>(number_of_plates); ++k) {
      const float sign = (k % 2 == 0) ? 1.0f : -1.0f;
      result[k] = plate_parameters{
          .index = k,
          .angle = orientation_ + static_cast<float>(k) * static_cast<float>(M_PI_2),
          .sign = sign,
          .radius = radius_common + sign * radius_offset,
          .z_coordinate = z_common + sign * z_offset,
      };
    }
    return result;
  }

  PF_TARGET_ATTRS [[nodiscard]] pf::util::device_array<predicted_plate, number_of_plates> predicted_plates()
      const noexcept {
    const Eigen::Vector2f center = linear_.center();
    const Eigen::Vector2f center_velocity = linear_.center_velocity();
    const float orientation_velocity = linear_.orientation_velocity();

    return plate_parameters_array().transformed([&](const plate_parameters& value) {
      const float cos_angle = cosf(value.angle);
      const float sin_angle = sinf(value.angle);

      const Eigen::Vector3f position =
          helper::rpad_zero(center) +
          (Eigen::Vector3f{} << value.radius * cos_angle, value.radius * sin_angle, value.z_coordinate).finished();

      const Eigen::Vector3f velocity =
          helper::rpad_zero(center_velocity) +
          orientation_velocity *
              (Eigen::Vector3f{} << -value.radius * sin_angle, value.radius * cos_angle, 0.0f).finished();

      return predicted_plate(position, velocity);
    });
  }

  [[nodiscard]] std::array<predicted_plate, number_of_plates> predicted_plates_for_host() const noexcept {
    return predicted_plates().to_host_array();
  }

  // Deterministic extrapolation for bullet time-of-flight lead. The covariance
  // is propagated too, so the caller can gate on predicted uncertainty at the
  // moment of impact rather than at the moment of observation.
  PF_TARGET_ATTRS [[nodiscard]] prediction extrapolate_state(const float& time_offset_seconds) const noexcept {
    linear_state::matrix_type transition = linear_state::matrix_type::Identity();
    transition(linear_state::index_center_x, linear_state::index_center_velocity_x) = time_offset_seconds;
    transition(linear_state::index_center_y, linear_state::index_center_velocity_y) = time_offset_seconds;

    const linear_state::vector_type mean = transition * linear_.mean();
    const linear_state::matrix_type covariance = transition * linear_.covariance() * transition.transpose();

    const float orientation = orientation_ + time_offset_seconds * linear_.orientation_velocity();
    return prediction(orientation, linear_state(mean, covariance));
  }

  PF_TARGET_ATTRS prediction() noexcept : orientation_{0.0f}, linear_{} {}

  PF_TARGET_ATTRS prediction(const float& orientation, const linear_state& linear) noexcept
      : orientation_{helper::to_orientation(orientation)}, linear_{linear} {}
};

}  // namespace plate_orbit_v3
