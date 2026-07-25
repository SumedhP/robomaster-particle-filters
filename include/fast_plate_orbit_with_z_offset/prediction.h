#pragma once

#include <fast_plate_orbit_with_z_offset/observation.h>
#include <fast_plate_orbit_with_z_offset/predicted_plate.h>
#include <pf/util/device_array.h>

#include <Eigen/Dense>
#include <array>

namespace fast_plate_orbit_with_z_offset {

namespace helper {

PF_TARGET_ATTRS [[nodiscard]] inline float to_radius(const float& radius) noexcept {
  constexpr float min_radius = 0.05f;
  return thrust::max(min_radius, radius);
}

PF_TARGET_ATTRS [[nodiscard]] inline float to_orientation(const float& angle_radians) noexcept {
  const float value = fmod(angle_radians, M_PI);
  return (value < 0.0f) ? value + M_PI : value;
}

PF_TARGET_ATTRS [[nodiscard]] inline Eigen::Vector3f rpad_zero(const Eigen::Vector2f& vector) noexcept {
  return (Eigen::Vector3f{} << vector, Eigen::Matrix<float, 1, 1>::Zero()).finished();
}

struct z_coordinate_update_configuration {
  static constexpr float offset_limit = 0.08f;
  PF_TARGET_ATTRS [[nodiscard]] static inline float post_process(const float& value) noexcept { return value; }
};

template <typename U>
PF_TARGET_ATTRS inline void update_value_offsets(
    const float& d_value_common,
    const float& d_value_0,
    const float& d_value_1,
    float& value_0,
    float& value_1) noexcept {
  constexpr float offset_limit = U::offset_limit;

  const float value_common = 0.5f * (value_0 + value_1) + d_value_common;
  const float value_offset_0 = thrust::min(offset_limit, thrust::max(-offset_limit, value_0 + d_value_0 - value_common));
  const float value_offset_1 = thrust::min(offset_limit, thrust::max(-offset_limit, value_1 + d_value_1 - value_common));

  value_0 = U::post_process(value_common + value_offset_0);
  value_1 = U::post_process(value_common + value_offset_1);
}

}  // namespace helper

class prediction {
 private:
  static constexpr size_t number_of_plates = 4;

  float radius_;

  float z_coordinate_0_;
  float z_coordinate_1_;

  float orientation_;
  float orientation_velocity_;

  Eigen::Vector2f center_;
  Eigen::Vector2f center_velocity_;

  struct angle_offset_and_z_coordinate {
    float angle_offset;
    float z_coordinate;
  };

 public:
  [[nodiscard]] std::array<predicted_plate, number_of_plates> predicted_plates_for_host() const noexcept {
    return predicted_plates().to_host_array();
  }

  PF_TARGET_ATTRS [[nodiscard]] pf::util::device_array<predicted_plate, number_of_plates> predicted_plates() const noexcept {
    const pf::util::device_array<angle_offset_and_z_coordinate, number_of_plates> angle_offsets_and_z_coordinates = {
        angle_offset_and_z_coordinate{0.0f, z_coordinate_0_},
        angle_offset_and_z_coordinate{M_PI_2, z_coordinate_1_},
        angle_offset_and_z_coordinate{M_PI, z_coordinate_0_},
        angle_offset_and_z_coordinate{M_PI + M_PI_2, z_coordinate_1_},
    };

    return angle_offsets_and_z_coordinates.transformed([this](const angle_offset_and_z_coordinate& value) {
      const float angle = value.angle_offset + orientation_;

      const Eigen::Vector3f predicted_plate_position =
          helper::rpad_zero(center_) +
          (Eigen::Vector3f{} << radius_ * cosf(angle), radius_ * sinf(angle), value.z_coordinate).finished();

      const Eigen::Vector3f predicted_plate_velocity =
          helper::rpad_zero(center_velocity_) +
          orientation_velocity_ * radius_ * (Eigen::Vector3f{} << -sinf(angle), cosf(angle), 0.0f).finished();

      return predicted_plate(predicted_plate_position, predicted_plate_velocity);
    });
  }

  PF_TARGET_ATTRS [[nodiscard]] prediction extrapolate_state(const float& time_offset_seconds) const noexcept {
    return prediction(
        radius_,
        z_coordinate_0_,
        z_coordinate_1_,
        orientation_ + time_offset_seconds * orientation_velocity_,
        orientation_velocity_,
        center_ + time_offset_seconds * center_velocity_,
        center_velocity_);
  }

  PF_TARGET_ATTRS void update_state(
      const float& time_offset_seconds,
      const float& radius_noise,
      const float& orientation_velocity_noise_0,
      const float& orientation_velocity_noise_1,
      const float& z_coordinate_noise_common,
      const float& z_coordinate_noise_0,
      const float& z_coordinate_noise_1,
      const Eigen::Vector2f& center_xy_velocity_noise_0,
      const Eigen::Vector2f& center_xy_velocity_noise_1) noexcept {
    static constexpr float one_half = 1.0 / 2.0;
    static constexpr float one_twelfth = 1.0 / 12.0;

    const float radius_noise_scale = sqrtf(time_offset_seconds);
    const float z_coordinate_noise_scale = radius_noise_scale;
    const float velocity_noise_scale = radius_noise_scale;
    const float position_noise_scale = sqrtf(one_twelfth) * powf(velocity_noise_scale, 3);

    const float d_radius = radius_noise_scale * radius_noise;

    const float d_z_coordinate_common = z_coordinate_noise_scale * z_coordinate_noise_common;
    const float d_z_coordinate_0 = z_coordinate_noise_scale * z_coordinate_noise_0;
    const float d_z_coordinate_1 = z_coordinate_noise_scale * z_coordinate_noise_1;

    const float d_orientation_velocity = velocity_noise_scale * orientation_velocity_noise_1;
    const float d_orientation = time_offset_seconds * orientation_velocity_ +
                                one_half * time_offset_seconds * d_orientation_velocity +
                                position_noise_scale * orientation_velocity_noise_0;

    const Eigen::Vector2f d_center_velocity = velocity_noise_scale * center_xy_velocity_noise_1;
    const Eigen::Vector2f d_center = time_offset_seconds * center_velocity_ + one_half * time_offset_seconds * d_center_velocity +
                                     position_noise_scale * center_xy_velocity_noise_0;

    radius_ = helper::to_radius(radius_ + d_radius);

    helper::update_value_offsets<helper::z_coordinate_update_configuration>(
        d_z_coordinate_common, d_z_coordinate_0, d_z_coordinate_1, z_coordinate_0_, z_coordinate_1_);

    orientation_ = helper::to_orientation(orientation_ + d_orientation);
    orientation_velocity_ = orientation_velocity_ + d_orientation_velocity;

    center_ = center_ + d_center;
    center_velocity_ = center_velocity_ + d_center_velocity;
  }

  PF_TARGET_ATTRS [[nodiscard]] const float& radius() const noexcept { return radius_; }

  PF_TARGET_ATTRS [[nodiscard]] const float& z_coordinate_0() const noexcept { return z_coordinate_0_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& z_coordinate_1() const noexcept { return z_coordinate_1_; }

  PF_TARGET_ATTRS [[nodiscard]] const float& orientation() const noexcept { return orientation_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& orientation_velocity() const noexcept { return orientation_velocity_; }
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector2f& center() const noexcept { return center_; }
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector2f& center_velocity() const noexcept { return center_velocity_; }

  PF_TARGET_ATTRS prediction() noexcept
      : radius_{0.0f},
        z_coordinate_0_{0.0f},
        z_coordinate_1_{0.0f},
        orientation_{0.0f},
        orientation_velocity_{0.0f},
        center_{Eigen::Vector2f::Zero()},
        center_velocity_{Eigen::Vector2f::Zero()} {}

  PF_TARGET_ATTRS prediction(
      const float& radius,
      const float& z_coordinate_0,
      const float& z_coordinate_1,
      const float& orientation,
      const float& orientation_velocity,
      const Eigen::Vector2f& center,
      const Eigen::Vector2f& center_velocity) noexcept
      : radius_{helper::to_radius(radius)},
        z_coordinate_0_{z_coordinate_0},
        z_coordinate_1_{z_coordinate_1},
        orientation_{helper::to_orientation(orientation)},
        orientation_velocity_{orientation_velocity},
        center_{center},
        center_velocity_{center_velocity} {}
};

}  // namespace fast_plate_orbit_with_z_offset
