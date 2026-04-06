// fast plate orbit with z offset
#include <fast_plate_orbit_with_z_offset/observation.h>
#include <fast_plate_orbit_with_z_offset/observed_plate.h>
#include <fast_plate_orbit_with_z_offset/particle_filter.h>
#include <fast_plate_orbit_with_z_offset/particle_filter_configuration_parameters.h>
#include <fast_plate_orbit_with_z_offset/predicted_plate.h>
#include <fast_plate_orbit_with_z_offset/prediction.h>

// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace fast_plate_orbit_with_z_offset {

namespace py = pybind11;

void init(py::module_& m) noexcept {
  auto fast_plate_orbit_with_z_offset = m.def_submodule("fast_plate_orbit_with_z_offset");

  py::class_<observed_plate>(fast_plate_orbit_with_z_offset, "ObservedPlate")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>())
      .def("position", &observed_plate::position)
      .def("position_diagonal_covariance", &observed_plate::position_diagonal_covariance);

  py::class_<predicted_plate>(fast_plate_orbit_with_z_offset, "PredictedPlate")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>())
      .def(py::init<Eigen::Vector3f>())
      .def("position", &predicted_plate::position)
      .def("velocity", &predicted_plate::velocity);

  py::class_<observation>(fast_plate_orbit_with_z_offset, "Observation")
      .def_static("from_one_plate", &observation::from_one_plate)
      .def_static("from_two_plates", &observation::from_two_plates);

  py::class_<prediction>(fast_plate_orbit_with_z_offset, "Prediction")
      .def("predicted_plates", &prediction::predicted_plates_for_host)
      .def("radius", &prediction::radius)
      .def("z_coordinate_0", &prediction::z_coordinate_0)
      .def("z_coordinate_1", &prediction::z_coordinate_1)
      .def("orientation", &prediction::orientation)
      .def("orientation_velocity", &prediction::orientation_velocity)
      .def("center", &prediction::center)
      .def("center_velocity", &prediction::center_velocity)
      .def("extrapolate_state", &prediction::extrapolate_state);

  py::class_<particle_filter_configuration_parameters>(
      fast_plate_orbit_with_z_offset,
      "ParticleFilterConfigurationParameters")
      .def(py::init<float, float, float, float, float, float, float, float, float, Eigen::Vector2f, Eigen::Vector2f>());

  py::class_<particle_filter>(fast_plate_orbit_with_z_offset, "ParticleFilter")
      .def(py::init<size_t, observation, particle_filter_configuration_parameters>())
      .def("extrapolate_state", &particle_filter::extrapolate_state, py::call_guard<py::gil_scoped_release>())

      .def(
          "reinitialize",
          &particle_filter::reinitialize,
          py::call_guard<py::gil_scoped_release>())

      .def(
          "update_state_sans_observation",
          &particle_filter::update_state_sans_observation,
          py::call_guard<py::gil_scoped_release>())

      .def(
          "update_state_with_observation",
          &particle_filter::update_state_with_observation,
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace fast_plate_orbit_with_z_offset
