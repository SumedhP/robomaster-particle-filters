#pragma once

#include <Eigen/Dense>

namespace plate_orbit_v3 {

struct observed_plate_orbit {
  float radius;
  float orientation;
  Eigen::Vector3f center;
};

}  // namespace plate_orbit_v3
