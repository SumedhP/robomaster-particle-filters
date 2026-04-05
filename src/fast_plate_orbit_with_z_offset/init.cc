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
      .def(
          py::init([](
                       const float radius_prior,
                       const float visibility_logit_coefficient,
                       const float radius_prior_variance_one_plate,
                       const float radius_prior_variance_two_plates,
                       const float radius_process_variance,
                       const float z_coordinate_common_process_variance,
                       const float z_coordinate_offset_process_variance,
                       const float orientation_velocity_prior_variance,
                       const float orientation_velocity_process_variance,
                       const Eigen::Vector2f& center_velocity_prior_diagonal_covariance,
                       const Eigen::Vector2f& center_velocity_process_diagonal_covariance,
                       const float likelihood_refinement_window,
                       const std::uint32_t observation_resample_period,
                       const std::uint32_t observation_update_subsample_stride) {
            return particle_filter_configuration_parameters{
                .radius_prior = radius_prior,
                .visibility_logit_coefficient = visibility_logit_coefficient,
                .radius_prior_variance_one_plate = radius_prior_variance_one_plate,
                .radius_prior_variance_two_plates = radius_prior_variance_two_plates,
                .radius_process_variance = radius_process_variance,
                .z_coordinate_common_process_variance = z_coordinate_common_process_variance,
                .z_coordinate_offset_process_variance = z_coordinate_offset_process_variance,
                .orientation_velocity_prior_variance = orientation_velocity_prior_variance,
                .orientation_velocity_process_variance = orientation_velocity_process_variance,
                .center_velocity_prior_diagonal_covariance = center_velocity_prior_diagonal_covariance,
                .center_velocity_process_diagonal_covariance = center_velocity_process_diagonal_covariance,
                .likelihood_refinement_window = likelihood_refinement_window,
                .observation_resample_period = observation_resample_period,
                                .observation_update_subsample_stride = observation_update_subsample_stride,
            };
          }),
          py::arg("radius_prior"),
          py::arg("visibility_logit_coefficient"),
          py::arg("radius_prior_variance_one_plate"),
          py::arg("radius_prior_variance_two_plates"),
          py::arg("radius_process_variance"),
          py::arg("z_coordinate_common_process_variance"),
          py::arg("z_coordinate_offset_process_variance"),
          py::arg("orientation_velocity_prior_variance"),
          py::arg("orientation_velocity_process_variance"),
          py::arg("center_velocity_prior_diagonal_covariance"),
          py::arg("center_velocity_process_diagonal_covariance"),
          py::arg("likelihood_refinement_window") = -1.0f,
          py::arg("observation_resample_period") = 32U,
          py::arg("observation_update_subsample_stride") = 16U);

  py::class_<particle_filter>(fast_plate_orbit_with_z_offset, "ParticleFilter")
      .def(py::init<size_t, observation, particle_filter_configuration_parameters>())
      .def("extrapolate_state", &particle_filter::extrapolate_state, py::call_guard<py::gil_scoped_release>())

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
