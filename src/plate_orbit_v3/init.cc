// plate orbit v3
#include <plate_orbit_v3/observation.h>
#include <plate_orbit_v3/observed_plate.h>
#include <plate_orbit_v3/particle_filter.h>
#include <plate_orbit_v3/particle_filter_configuration_parameters.h>
#include <plate_orbit_v3/predicted_plate.h>
#include <plate_orbit_v3/prediction.h>

// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace plate_orbit_v3 {

namespace py = pybind11;

void init(py::module_& m) noexcept {
  auto plate_orbit_v3 = m.def_submodule("plate_orbit_v3");

  py::class_<observed_plate>(plate_orbit_v3, "ObservedPlate")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>())
      .def("position", &observed_plate::position)
      .def("position_diagonal_covariance", &observed_plate::position_diagonal_covariance);

  py::class_<predicted_plate>(plate_orbit_v3, "PredictedPlate")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>())
      .def("position", &predicted_plate::position)
      .def("velocity", &predicted_plate::velocity);

  py::class_<observation>(plate_orbit_v3, "Observation")
      .def_static("from_one_plate", &observation::from_one_plate)
      .def_static("from_two_plates", &observation::from_two_plates);

  py::class_<prediction>(plate_orbit_v3, "Prediction")
      .def("predicted_plates", &prediction::predicted_plates_for_host)
      .def("radius_0", &prediction::radius_0)
      .def("radius_1", &prediction::radius_1)
      .def("z_coordinate_0", &prediction::z_coordinate_0)
      .def("z_coordinate_1", &prediction::z_coordinate_1)
      .def("orientation", &prediction::orientation)
      .def("orientation_velocity", &prediction::orientation_velocity)
      .def("center", &prediction::center)
      .def("center_velocity", &prediction::center_velocity)
      // New in v3: the full 9x9 posterior covariance over the linear substate,
      // ordered as in plate_orbit_v3::linear_state.
      .def("covariance", &prediction::covariance_for_host)
      .def("extrapolate_state", &prediction::extrapolate_state);

  py::class_<particle_filter_configuration_parameters>(plate_orbit_v3, "ParticleFilterConfigurationParameters")
      .def(py::init<
           float,   // radius_prior
           float,   // visibility_logit_coefficient
           float,   // radius_prior_variance_one_plate
           float,   // radius_prior_variance_two_plates
           float,   // radius_common_process_variance
           float,   // radius_common_reversion_time_constant
           float,   // radius_offset_stationary_variance
           float,   // radius_offset_reversion_time_constant
           float,   // z_common_process_variance
           float,   // z_offset_stationary_variance
           float,   // z_offset_reversion_time_constant
           float,   // orientation_prior_variance
           float,   // orientation_velocity_prior_variance
           float,   // orientation_velocity_process_variance
           Eigen::Vector2f,   // center_velocity_prior_diagonal_covariance
           Eigen::Vector2f,   // center_velocity_process_diagonal_covariance
           float>());         // resample_effective_sample_size_fraction

  py::class_<particle_filter>(plate_orbit_v3, "ParticleFilter")
      .def(py::init<size_t, observation, particle_filter_configuration_parameters>())
      .def("extrapolate_state", &particle_filter::extrapolate_state, py::call_guard<py::gil_scoped_release>())

      // New in v3: diagnostics for the resampling schedule. A healthy track
      // holds a high effective sample size and resamples only occasionally.
      .def("effective_sample_size", &particle_filter::effective_sample_size, py::call_guard<py::gil_scoped_release>())
      .def("resampled", &particle_filter::resampled, py::call_guard<py::gil_scoped_release>())

      .def(
          "update_state_sans_observation",
          &particle_filter::update_state_sans_observation,
          py::call_guard<py::gil_scoped_release>())

      .def(
          "update_state_with_observation",
          &particle_filter::update_state_with_observation,
          py::call_guard<py::gil_scoped_release>())

      .def(
          "reinitialize",
          static_cast<void (particle_filter::*)(const observation&) noexcept>(&particle_filter::reinitialize),
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace plate_orbit_v3
