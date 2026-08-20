#include <fast_plate_orbit/init.h>
#include <fast_plate_orbit_with_z_offset/init.h>
#include <plate_orbit/init.h>
#include <plate_orbit_v2/init.h>
#include <plate_orbit_rbpf/init.h>

// pybind11
#include <pybind11/pybind11.h>

PYBIND11_MODULE(robomaster_particle_filters, m) {
  fast_plate_orbit::init(m);
  fast_plate_orbit_with_z_offset::init(m);
  plate_orbit::init(m);
  plate_orbit_v2::init(m);
  plate_orbit_rbpf::init(m);
}
