#pragma once

#include <pf/config/target_config.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <Eigen/Dense>

#include <cstddef>
#include <utility>

namespace fast_plate_orbit {

namespace prediction_soa {

namespace target_config = pf::target_config;

enum class field_index : int {
  radius = 0,
  orientation = 1,
  orientation_velocity = 2,
  center_x = 3,
  center_y = 4,
  center_z = 5,
  center_vel_x = 6,
  center_vel_y = 7,
};

template <field_index Index, typename Tuple>
PF_TARGET_ATTRS [[nodiscard]] inline decltype(auto) slot(Tuple&& tuple) noexcept {
  return thrust::get<static_cast<int>(Index)>(std::forward<Tuple>(tuple));
}

template <typename Tuple>
PF_TARGET_ATTRS [[nodiscard]] inline decltype(auto) radius(Tuple&& tuple) noexcept {
  return slot<field_index::radius>(std::forward<Tuple>(tuple));
}

template <typename Tuple>
PF_TARGET_ATTRS [[nodiscard]] inline decltype(auto) orientation(Tuple&& tuple) noexcept {
  return slot<field_index::orientation>(std::forward<Tuple>(tuple));
}

template <typename Tuple>
PF_TARGET_ATTRS [[nodiscard]] inline decltype(auto) orientation_velocity(Tuple&& tuple) noexcept {
  return slot<field_index::orientation_velocity>(std::forward<Tuple>(tuple));
}

template <typename Tuple>
PF_TARGET_ATTRS [[nodiscard]] inline decltype(auto) center_x(Tuple&& tuple) noexcept {
  return slot<field_index::center_x>(std::forward<Tuple>(tuple));
}

template <typename Tuple>
PF_TARGET_ATTRS [[nodiscard]] inline decltype(auto) center_y(Tuple&& tuple) noexcept {
  return slot<field_index::center_y>(std::forward<Tuple>(tuple));
}

template <typename Tuple>
PF_TARGET_ATTRS [[nodiscard]] inline decltype(auto) center_z(Tuple&& tuple) noexcept {
  return slot<field_index::center_z>(std::forward<Tuple>(tuple));
}

template <typename Tuple>
PF_TARGET_ATTRS [[nodiscard]] inline decltype(auto) center_vel_x(Tuple&& tuple) noexcept {
  return slot<field_index::center_vel_x>(std::forward<Tuple>(tuple));
}

template <typename Tuple>
PF_TARGET_ATTRS [[nodiscard]] inline decltype(auto) center_vel_y(Tuple&& tuple) noexcept {
  return slot<field_index::center_vel_y>(std::forward<Tuple>(tuple));
}

struct storage {
  target_config::vector<float> radius_field;
  target_config::vector<float> orientation_field;
  target_config::vector<float> orientation_velocity_field;
  target_config::vector<float> center_x_field;
  target_config::vector<float> center_y_field;
  target_config::vector<float> center_z_field;
  target_config::vector<float> center_vel_x_field;
  target_config::vector<float> center_vel_y_field;

  explicit storage(std::size_t n)
      : radius_field(n),
        orientation_field(n),
        orientation_velocity_field(n),
        center_x_field(n),
        center_y_field(n),
        center_z_field(n),
        center_vel_x_field(n),
        center_vel_y_field(n) {}

  [[nodiscard]] auto zip_begin() noexcept {
    return thrust::make_zip_iterator(
        radius_field.begin(),
        orientation_field.begin(),
        orientation_velocity_field.begin(),
        center_x_field.begin(),
        center_y_field.begin(),
        center_z_field.begin(),
        center_vel_x_field.begin(),
        center_vel_y_field.begin());
  }

  [[nodiscard]] auto zip_end() noexcept {
    return thrust::make_zip_iterator(
        radius_field.end(),
        orientation_field.end(),
        orientation_velocity_field.end(),
        center_x_field.end(),
        center_y_field.end(),
        center_z_field.end(),
        center_vel_x_field.end(),
        center_vel_y_field.end());
  }

  [[nodiscard]] auto zip_cbegin() const noexcept {
    return thrust::make_zip_iterator(
        radius_field.cbegin(),
        orientation_field.cbegin(),
        orientation_velocity_field.cbegin(),
        center_x_field.cbegin(),
        center_y_field.cbegin(),
        center_z_field.cbegin(),
        center_vel_x_field.cbegin(),
        center_vel_y_field.cbegin());
  }

  [[nodiscard]] auto zip_cend() const noexcept {
    return thrust::make_zip_iterator(
        radius_field.cend(),
        orientation_field.cend(),
        orientation_velocity_field.cend(),
        center_x_field.cend(),
        center_y_field.cend(),
        center_z_field.cend(),
        center_vel_x_field.cend(),
        center_vel_y_field.cend());
  }

  template <typename ExecutionPolicy, typename IndexIterator>
  void gather_into(
      const ExecutionPolicy& policy,
      IndexIterator idx_begin,
      IndexIterator idx_end,
      storage& dst) const noexcept {
    thrust::gather(policy, idx_begin, idx_end, radius_field.cbegin(), dst.radius_field.begin());
    thrust::gather(policy, idx_begin, idx_end, orientation_field.cbegin(), dst.orientation_field.begin());
    thrust::gather(
        policy,
        idx_begin,
        idx_end,
        orientation_velocity_field.cbegin(),
        dst.orientation_velocity_field.begin());
    thrust::gather(policy, idx_begin, idx_end, center_x_field.cbegin(), dst.center_x_field.begin());
    thrust::gather(policy, idx_begin, idx_end, center_y_field.cbegin(), dst.center_y_field.begin());
    thrust::gather(policy, idx_begin, idx_end, center_z_field.cbegin(), dst.center_z_field.begin());
    thrust::gather(policy, idx_begin, idx_end, center_vel_x_field.cbegin(), dst.center_vel_x_field.begin());
    thrust::gather(policy, idx_begin, idx_end, center_vel_y_field.cbegin(), dst.center_vel_y_field.begin());
  }

  void swap(storage& other) noexcept {
    radius_field.swap(other.radius_field);
    orientation_field.swap(other.orientation_field);
    orientation_velocity_field.swap(other.orientation_velocity_field);
    center_x_field.swap(other.center_x_field);
    center_y_field.swap(other.center_y_field);
    center_z_field.swap(other.center_z_field);
    center_vel_x_field.swap(other.center_vel_x_field);
    center_vel_y_field.swap(other.center_vel_y_field);
  }
};

template <typename Prediction, typename Tuple>
PF_TARGET_ATTRS [[nodiscard]] inline Prediction from_tuple(const Tuple& tuple) noexcept {
  return Prediction(
      radius(tuple),
      orientation(tuple),
      orientation_velocity(tuple),
      Eigen::Vector3f{center_x(tuple), center_y(tuple), center_z(tuple)},
      Eigen::Vector2f{center_vel_x(tuple), center_vel_y(tuple)});
}

template <typename Prediction, typename Tuple>
PF_TARGET_ATTRS inline void to_tuple(Tuple& tuple, const Prediction& value) noexcept {
  radius(tuple) = value.radius();
  orientation(tuple) = value.orientation();
  orientation_velocity(tuple) = value.orientation_velocity();
  center_x(tuple) = value.center().x();
  center_y(tuple) = value.center().y();
  center_z(tuple) = value.center().z();
  center_vel_x(tuple) = value.center_velocity().x();
  center_vel_y(tuple) = value.center_velocity().y();
}

}  // namespace prediction_soa

}  // namespace fast_plate_orbit
