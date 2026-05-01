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

#include <algorithm>
#include <array>
#include <cmath>

namespace fast_plate_orbit_with_z_offset {

namespace py = pybind11;

namespace {

[[nodiscard]] inline float wrap_half_turn_host(const float angle_radians) noexcept {
    float value = fmodf(angle_radians + M_PI_2, M_PI);
    if (value < 0.0f) {
        value += M_PI;
    }
    return value - M_PI_2;
}

[[nodiscard]] inline float clamped_positive_variance(const float variance) noexcept {
    return std::max(1.0e-6f, variance);
}

[[nodiscard]] inline py::dict error_vs_observed_plate(
        const observed_plate& observed,
        const Eigen::Vector3f& predicted_position,
        const float predicted_yaw) {
    const Eigen::Vector3f position_error = observed.position() - predicted_position;
    const float yaw_error_raw = observed.yaw() - predicted_yaw;
    const float yaw_error_wrapped = wrap_half_turn_host(yaw_error_raw);

    py::dict error;
    error["dx"] = position_error.x();
    error["dy"] = position_error.y();
    error["dz"] = position_error.z();
    error["position_error_norm"] = position_error.norm();
    error["yaw_error_raw"] = yaw_error_raw;
    error["yaw_error_wrapped"] = yaw_error_wrapped;
    error["yaw_error_abs"] = fabsf(yaw_error_wrapped);
    error["observed_yaw"] = observed.yaw();
    error["observed_yaw_variance"] = observed.yaw_variance();
    return error;
}

[[nodiscard]] inline float score_assignment(const observed_plate& observed, const Eigen::Vector3f& predicted_position, const float predicted_yaw) noexcept {
    const Eigen::Vector3f e = observed.position() - predicted_position;
    const Eigen::Vector3f v = observed.position_diagonal_covariance();
    const float yaw_error = wrap_half_turn_host(observed.yaw() - predicted_yaw);
    const float yaw_variance = clamped_positive_variance(observed.yaw_variance());

    const float position_term =
            -0.5f *
            (e.x() * e.x() / clamped_positive_variance(v.x()) +
             e.y() * e.y() / clamped_positive_variance(v.y()) +
             e.z() * e.z() / clamped_positive_variance(v.z()));

    const float yaw_term = -0.5f * yaw_error * yaw_error / yaw_variance;
    return position_term + yaw_term;
}

}  // namespace

void init(py::module_& m) noexcept {
  auto fast_plate_orbit_with_z_offset = m.def_submodule("fast_plate_orbit_with_z_offset");

  py::class_<observed_plate>(fast_plate_orbit_with_z_offset, "ObservedPlate")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f, float, float>())
      .def("position", &observed_plate::position)
      .def("position_diagonal_covariance", &observed_plate::position_diagonal_covariance)
      .def("yaw", &observed_plate::yaw)
      .def("yaw_variance", &observed_plate::yaw_variance);

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
      .def(py::init<float, float, float, float, float, float, float, float, float, float, float, Eigen::Vector2f, Eigen::Vector2f>());

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
                    py::call_guard<py::gil_scoped_release>())

            .def("telemetry", [](const particle_filter& filter, const observation& state) {
                const prediction optimal_particle = filter.extrapolate_state(0.0f);
                const auto predicted_plates = optimal_particle.predicted_plates_for_host();

                const Eigen::Vector2f center_xy = optimal_particle.center();
                const float cx = center_xy.x();
                const float cy = center_xy.y();

                const float ox = state.observer_position().x();
                const float oy = state.observer_position().y();
                const float oz = state.observer_position().z();

                std::array<float, 4> squared_distances{};
                std::array<int, 4> sort_indices{0, 1, 2, 3};
                std::array<float, 4> predicted_yaws{};

                for (int i = 0; i < 4; ++i) {
                    const Eigen::Vector3f& p = predicted_plates[i].position();
                    const float dx = ox - p.x();
                    const float dy = oy - p.y();
                    const float dz = oz - p.z();

                    squared_distances[i] = dx * dx + dy * dy + dz * dz;
                    predicted_yaws[i] = atan2f(p.y() - cy, p.x() - cx);
                }

                std::sort(sort_indices.begin(), sort_indices.end(), [&](const int a, const int b) {
                    return squared_distances[a] < squared_distances[b];
                });

                const observed_plate& observed_one = state.plate_one();
                const bool has_plate_two = state.plate_two().has_value();

                std::array<int, 4> matched_observed_index{-1, -1, -1, -1};
                py::dict assignment;

                if (!has_plate_two) {
                    matched_observed_index[sort_indices[0]] = 0;
                    assignment["has_two_observed_plates"] = false;
                    assignment["selected_assignment"] = "single_plate";
                    assignment["predicted_indices_considered"] = py::make_tuple(sort_indices[0], sort_indices[1]);
                } else {
                    const observed_plate& observed_two = *state.plate_two();
                    const int i0 = sort_indices[0];
                    const int i1 = sort_indices[1];

                    const float score_one =
                            score_assignment(observed_one, predicted_plates[i0].position(), predicted_yaws[i0]) +
                            score_assignment(observed_two, predicted_plates[i1].position(), predicted_yaws[i1]);

                    const float score_two =
                            score_assignment(observed_one, predicted_plates[i1].position(), predicted_yaws[i1]) +
                            score_assignment(observed_two, predicted_plates[i0].position(), predicted_yaws[i0]);

                    if (score_one >= score_two) {
                        matched_observed_index[i0] = 0;
                        matched_observed_index[i1] = 1;
                        assignment["selected_assignment"] = "plate1->closest,plate2->second_closest";
                    } else {
                        matched_observed_index[i1] = 0;
                        matched_observed_index[i0] = 1;
                        assignment["selected_assignment"] = "plate1->second_closest,plate2->closest";
                    }

                    assignment["has_two_observed_plates"] = true;
                    assignment["predicted_indices_considered"] = py::make_tuple(i0, i1);
                    assignment["assignment_one_score"] = score_one;
                    assignment["assignment_two_score"] = score_two;
                }

                py::list per_plate;
                for (int i = 0; i < 4; ++i) {
                    const Eigen::Vector3f& p = predicted_plates[i].position();
                    const float distance_to_observer = sqrtf(squared_distances[i]);

                    py::dict plate;
                    plate["predicted_index"] = i;
                    plate["predicted_position"] = p;
                    plate["predicted_yaw"] = predicted_yaws[i];
                    plate["distance_to_observer"] = distance_to_observer;
                    plate["distance_rank"] = (i == sort_indices[0]) ? 0 : (i == sort_indices[1]) ? 1 : (i == sort_indices[2]) ? 2 : 3;
                    plate["matched_observed_plate_index"] = matched_observed_index[i];

                    plate["error_vs_observed_plate_1"] = error_vs_observed_plate(observed_one, p, predicted_yaws[i]);
                    if (has_plate_two) {
                        plate["error_vs_observed_plate_2"] = error_vs_observed_plate(*state.plate_two(), p, predicted_yaws[i]);
                    } else {
                        plate["error_vs_observed_plate_2"] = py::none();
                    }

                    per_plate.append(plate);
                }

                py::dict result;
                result["optimal_particle"] = optimal_particle;
                result["observer_position"] = state.observer_position();
                result["assignment"] = assignment;
                result["plates"] = per_plate;
                return result;
            });

    py::module_ sys = py::module_::import("sys");
    sys.attr("modules")["robomaster_particle_filters.fast_plate_orbit_with_z_offset"] = fast_plate_orbit_with_z_offset;
}

}  // namespace fast_plate_orbit_with_z_offset
