#pragma once

#include <pf/util/device_array.h>
#include <fast_plate_orbit_with_z_offset/observation.h>
#include <fast_plate_orbit_with_z_offset/predicted_plate.h>

#include <Eigen/Dense>
#include <array>

namespace fast_plate_orbit_with_z_offset {

namespace helper {

PF_TARGET_ATTRS [[nodiscard]] inline Eigen::Vector3f rpad_zero(const Eigen::Vector2f& vector) noexcept {
  return (Eigen::Vector3f{} << vector, Eigen::Matrix<float, 1, 1>::Zero()).finished();
}

PF_TARGET_ATTRS [[nodiscard]] inline float to_orientation(const float& angle_radians) noexcept {
  const float value = fmod(angle_radians, M_PI);
  return (value < 0.0f) ? value + M_PI : value;
}

PF_TARGET_ATTRS [[nodiscard]] inline float to_radius(const float& radius) noexcept {
  constexpr float min_radius = 0.2f;
  constexpr float max_radius = 0.6f;
  return thrust::max(min_radius, thrust::min(max_radius, radius));
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

 public:
  [[nodiscard]] std::array<predicted_plate, number_of_plates> predicted_plates_for_host() const noexcept {
    return predicted_plates().to_host_array();
  }

  PF_TARGET_ATTRS [[nodiscard]] pf::util::device_array<Eigen::Vector3f, number_of_plates> predicted_plate_positions()
      const noexcept {
    const Eigen::Vector3f center_with_zero_z = helper::rpad_zero(center_);
    const float cos_orientation = cosf(orientation_);
    const float sin_orientation = sinf(orientation_);

    const float x0 = radius_ * cos_orientation;
    const float y0 = radius_ * sin_orientation;

    const float x1 = -radius_ * sin_orientation;
    const float y1 = radius_ * cos_orientation;

    const float x2 = -x0;
    const float y2 = -y0;

    const float x3 = -x1;
    const float y3 = -y1;

    return {
      center_with_zero_z + (Eigen::Vector3f{} << x0, y0, z_coordinate_0_).finished(),
      center_with_zero_z + (Eigen::Vector3f{} << x1, y1, z_coordinate_1_).finished(),
      center_with_zero_z + (Eigen::Vector3f{} << x2, y2, z_coordinate_0_).finished(),
      center_with_zero_z + (Eigen::Vector3f{} << x3, y3, z_coordinate_1_).finished(),
    };
  }

  PF_TARGET_ATTRS [[nodiscard]] pf::util::device_array<predicted_plate, number_of_plates> predicted_plates() const noexcept {
    const Eigen::Vector3f center_with_zero_z = helper::rpad_zero(center_);
    const Eigen::Vector3f center_velocity_with_zero_z = helper::rpad_zero(center_velocity_);

    const float cos_orientation = cosf(orientation_);
    const float sin_orientation = sinf(orientation_);

    const float x0 = radius_ * cos_orientation;
    const float y0 = radius_ * sin_orientation;
    const float vx0 = orientation_velocity_ * (-radius_ * sin_orientation);
    const float vy0 = orientation_velocity_ * (radius_ * cos_orientation);

    const float x1 = -radius_ * sin_orientation;
    const float y1 = radius_ * cos_orientation;
    const float vx1 = orientation_velocity_ * (-radius_ * cos_orientation);
    const float vy1 = orientation_velocity_ * (-radius_ * sin_orientation);

    const float x2 = -x0;
    const float y2 = -y0;
    const float vx2 = -vx0;
    const float vy2 = -vy0;

    const float x3 = -x1;
    const float y3 = -y1;
    const float vx3 = -vx1;
    const float vy3 = -vy1;

    return {
      predicted_plate(
        center_with_zero_z + (Eigen::Vector3f{} << x0, y0, z_coordinate_0_).finished(),
        center_velocity_with_zero_z + (Eigen::Vector3f{} << vx0, vy0, 0.0f).finished()),
      predicted_plate(
        center_with_zero_z + (Eigen::Vector3f{} << x1, y1, z_coordinate_1_).finished(),
        center_velocity_with_zero_z + (Eigen::Vector3f{} << vx1, vy1, 0.0f).finished()),
      predicted_plate(
        center_with_zero_z + (Eigen::Vector3f{} << x2, y2, z_coordinate_0_).finished(),
        center_velocity_with_zero_z + (Eigen::Vector3f{} << vx2, vy2, 0.0f).finished()),
      predicted_plate(
        center_with_zero_z + (Eigen::Vector3f{} << x3, y3, z_coordinate_1_).finished(),
        center_velocity_with_zero_z + (Eigen::Vector3f{} << vx3, vy3, 0.0f).finished()),
    };
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
      const float& z_coordinate_noise_common,
      const float& z_coordinate_noise_0,
      const float& z_coordinate_noise_1,
      const float& orientation_velocity_noise_0,
      const float& orientation_velocity_noise_1,
      const Eigen::Vector2f& center_velocity_noise_0,
      const Eigen::Vector2f& center_velocity_noise_1) noexcept {
    static constexpr float one_half = 1.0 / 2.0;
    static constexpr float one_twelfth = 1.0 / 12.0;

    const float radius_noise_scale = sqrtf(time_offset_seconds);
    const float z_coordinate_noise_scale = radius_noise_scale;
    const float velocity_noise_scale = radius_noise_scale;
    const float velocity_noise_scale_squared = velocity_noise_scale * velocity_noise_scale;
    const float position_noise_scale = sqrtf(one_twelfth) * velocity_noise_scale_squared * velocity_noise_scale;
    const float half_dt = one_half * time_offset_seconds;

    const float d_radius = radius_noise_scale * radius_noise;

    const float d_z_coordinate_common = z_coordinate_noise_scale * z_coordinate_noise_common;
    const float d_z_coordinate_0 = z_coordinate_noise_scale * z_coordinate_noise_0;
    const float d_z_coordinate_1 = z_coordinate_noise_scale * z_coordinate_noise_1;

    const float d_orientation_velocity = velocity_noise_scale * orientation_velocity_noise_1;
    const float d_orientation = time_offset_seconds * orientation_velocity_ + half_dt * d_orientation_velocity +
                                position_noise_scale * orientation_velocity_noise_0;

    const Eigen::Vector2f d_center_velocity = velocity_noise_scale * center_velocity_noise_1;
    const Eigen::Vector2f d_center =
      time_offset_seconds * center_velocity_ + half_dt * d_center_velocity + position_noise_scale * center_velocity_noise_0;

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
