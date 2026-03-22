#pragma once

#include <Eigen/Dense>

namespace fast_plate_orbit_with_z_offset {

struct observed_plate_orbit {
  float radius;
  float orientation;
  Eigen::Vector3f center;
};

}
