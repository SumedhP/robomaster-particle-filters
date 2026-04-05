#pragma once

#include <pf/config/target_config.h>
#include <pf/filter/particle_reduction_state.h>
#include <pf/util/device_array.h>
#include <fast_plate_orbit_with_z_offset/observation.h>
#include <fast_plate_orbit_with_z_offset/observed_plate.h>
#include <fast_plate_orbit_with_z_offset/observed_plate_orbit.h>
#include <fast_plate_orbit_with_z_offset/observed_plate_orbit_builder.h>
#include <fast_plate_orbit_with_z_offset/particle_filter_configuration_parameters.h>
#include <fast_plate_orbit_with_z_offset/predicted_plate.h>
#include <fast_plate_orbit_with_z_offset/prediction.h>
#include <thrust/random.h>
#include <util/random_variable_sampler.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace fast_plate_orbit_with_z_offset {

namespace helper {

PF_TARGET_ONLY_ATTRS [[nodiscard]] inline float log_sigmoid(const float& x) noexcept { return -logf(1.0f + expf(-x)); }

}  // namespace helper

struct most_likely_particle_reduction_impl {
  using state_type = pf::filter::particle_reduction_state<prediction>;
  static constexpr float half_pi = M_PI_2;

  struct orientation_and_z_coordinates_and_radius {
    float orientation;

    float radius_;

    float z_coordinate_0;
    float z_coordinate_1;
  };

  PF_TARGET_ONLY_ATTRS [[nodiscard]] inline state_type operator()(const state_type& a, const state_type& b) const noexcept {
    // this does not necessarily provide the correct mean in the MLE sense,
    // though it gives a reasonable approximation for most inputs.

    const prediction a_particle = a.most_likely_particle();
    const prediction b_particle = b.most_likely_particle();

    const pf::util::device_array<orientation_and_z_coordinates_and_radius, 5> candidates = {
      orientation_and_z_coordinates_and_radius{
            .orientation = a_particle.orientation() + 0.0f * half_pi,
            .radius_ = a_particle.radius(),
            .z_coordinate_0 = a_particle.z_coordinate_0(),
            .z_coordinate_1 = a_particle.z_coordinate_1(),
        },

      orientation_and_z_coordinates_and_radius{
            .orientation = a_particle.orientation() + 1.0f * half_pi,
            .radius_ = a_particle.radius(),
            .z_coordinate_0 = a_particle.z_coordinate_1(),
            .z_coordinate_1 = a_particle.z_coordinate_0(),
        },

      orientation_and_z_coordinates_and_radius{
            .orientation = a_particle.orientation() - 1.0f * half_pi,
            .radius_ = a_particle.radius(),
            .z_coordinate_0 = a_particle.z_coordinate_1(),
            .z_coordinate_1 = a_particle.z_coordinate_0(),
        },

      orientation_and_z_coordinates_and_radius{
            .orientation = a_particle.orientation() + 2.0f * half_pi,
            .radius_ = a_particle.radius(),
            .z_coordinate_0 = a_particle.z_coordinate_0(),
            .z_coordinate_1 = a_particle.z_coordinate_1(),
        },

      orientation_and_z_coordinates_and_radius{
            .orientation = a_particle.orientation() - 2.0f * half_pi,
            .radius_ = a_particle.radius(),
            .z_coordinate_0 = a_particle.z_coordinate_0(),
            .z_coordinate_1 = a_particle.z_coordinate_1(),
        },
    };

    const auto target = b_particle.orientation();
    const auto [a_orientation, a_radius, a_z_coordinate_0, a_z_coordinate_1] =
        *candidates.minimum_by([target](const auto& value) { return abs(target - value.orientation); });

    const float alpha = static_cast<float>(b.count()) / static_cast<float>(a.count() + b.count());
    const float c_alpha = 1.0f - alpha;

    const float radius = c_alpha * a_radius + alpha * b_particle.radius();

    const float z_coordinate_0 = c_alpha * a_z_coordinate_0 + alpha * b_particle.z_coordinate_0();
    const float z_coordinate_1 = c_alpha * a_z_coordinate_1 + alpha * b_particle.z_coordinate_1();

    const float orientation = c_alpha * a_orientation + alpha * b_particle.orientation();
    const float orientation_velocity = c_alpha * a_particle.orientation_velocity() + alpha * b_particle.orientation_velocity();

    const Eigen::Vector2f center = c_alpha * a_particle.center() + alpha * b_particle.center();
    const Eigen::Vector2f center_velocity = c_alpha * a_particle.center_velocity() + alpha * b_particle.center_velocity();

    const auto state = prediction(radius, z_coordinate_0, z_coordinate_1, orientation, orientation_velocity, center, center_velocity);

    return state_type{state, a.count() + b.count()};
  }
};

class particle_filter_configuration {
 public:
  struct likelihood_evaluation_context {
    Eigen::Vector3f observer_position;
    Eigen::Vector2f observer_position_xy;

    Eigen::Vector3f observed_plate_one_position;
    Eigen::Vector3f observed_plate_two_position;

    Eigen::Vector3f inverse_observed_plate_one_position_diagonal_covariance;
    Eigen::Vector3f inverse_observed_plate_two_position_diagonal_covariance;

    bool has_plate_two;
  };

  struct initial_sampling_context {
    observed_plate_orbit orbit;
    float radius_variance;
    Eigen::Vector3f plate_one_position_diagonal_covariance;
  };

 private:

  PF_TARGET_ATTRS [[nodiscard]] static Eigen::Vector3f inverse_diagonal_covariance(
      const Eigen::Vector3f& diagonal_covariance) noexcept {
    constexpr float minimum_variance = 1.0e-12f;

    return Eigen::Vector3f{
        1.0f / thrust::max(minimum_variance, diagonal_covariance[0]),
        1.0f / thrust::max(minimum_variance, diagonal_covariance[1]),
        1.0f / thrust::max(minimum_variance, diagonal_covariance[2]),
    };
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] static float unnormalized_normal_log_density_from_inverse_covariance(
      const Eigen::Vector3f& inverse_diagonal_covariance,
      const Eigen::Vector3f& x) noexcept {
    return -0.5f * x.cwiseProduct(inverse_diagonal_covariance).dot(x);
  }

  particle_filter_configuration_parameters params_;

 public:
  using observation_type = observation;
  using prediction_type = prediction;
  using sampler_type = util::default_rv_sampler;

  [[nodiscard]] most_likely_particle_reduction_impl most_likely_particle_reduction() const noexcept {
    return most_likely_particle_reduction_impl{};
  }

  PF_TARGET_ATTRS [[nodiscard]] likelihood_evaluation_context precompute_likelihood_evaluation_context(
      const observation& state) const noexcept {
    const observed_plate& plate_one = state.plate_one();
    const observed_plate plate_two = state.plate_two().has_value() ? *state.plate_two() : plate_one;

    return likelihood_evaluation_context{
        .observer_position = state.observer_position(),
        .observer_position_xy = state.observer_position().head<2>(),
        .observed_plate_one_position = plate_one.position(),
        .observed_plate_two_position = plate_two.position(),
        .inverse_observed_plate_one_position_diagonal_covariance =
            inverse_diagonal_covariance(plate_one.position_diagonal_covariance()),
        .inverse_observed_plate_two_position_diagonal_covariance =
            inverse_diagonal_covariance(plate_two.position_diagonal_covariance()),
        .has_plate_two = state.plate_two().has_value(),
    };
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] float conditional_log_likelihood_from_precomputed(
      const util::default_rv_sampler& sampler,
      const likelihood_evaluation_context& context,
      const prediction& given) const noexcept {
    (void)sampler;

    const auto predicted_positions = given.predicted_plate_positions().selection_sort_by(
        [observer_position = context.observer_position](const Eigen::Vector3f& predicted_position) {
          return (observer_position - predicted_position).squaredNorm();
        });

    const auto [pred_0, pred_1, pred_2, pred_3] = predicted_positions;

    constexpr float minimum_norm_squared = 1.0e-12f;
    const Eigen::Vector2f center = given.center();
    const Eigen::Vector2f observer_delta = context.observer_position_xy - center;
    const float observer_delta_norm_inverse = rsqrtf(thrust::max(minimum_norm_squared, observer_delta.squaredNorm()));

    auto visibility_logit = [&, this](const Eigen::Vector3f& predicted_position) {
      const Eigen::Vector2f plate_delta = predicted_position.head<2>() - center;
      const float plate_delta_norm_inverse = rsqrtf(thrust::max(minimum_norm_squared, plate_delta.squaredNorm()));
      const float similarity = observer_delta.dot(plate_delta) * observer_delta_norm_inverse * plate_delta_norm_inverse;
      return params_.visibility_logit_coefficient * similarity;
    };

    const float logit_visibility_0 = visibility_logit(pred_0);
    const float logit_visibility_1 = visibility_logit(pred_1);
    const float logit_visibility_2 = visibility_logit(pred_2);
    const float logit_visibility_3 = visibility_logit(pred_3);

    auto log_p_of = [](const Eigen::Vector3f& observed_position,
                       const Eigen::Vector3f& inverse_observed_position_diagonal_covariance,
                       const Eigen::Vector3f& predicted_position) {
      const Eigen::Vector3f error = observed_position - predicted_position;
      return unnormalized_normal_log_density_from_inverse_covariance(inverse_observed_position_diagonal_covariance, error);
    };

    if (!context.has_plate_two) {
      const float pr_visibility = helper::log_sigmoid(logit_visibility_0) + helper::log_sigmoid(-logit_visibility_1) +
                                  helper::log_sigmoid(-logit_visibility_2) + helper::log_sigmoid(-logit_visibility_3);

      return pr_visibility + log_p_of(
                                 context.observed_plate_one_position,
                                 context.inverse_observed_plate_one_position_diagonal_covariance,
                                 pred_0);
    }

    const float log_pr_visibility = helper::log_sigmoid(logit_visibility_0) + helper::log_sigmoid(logit_visibility_1) +
                                    helper::log_sigmoid(-logit_visibility_2) + helper::log_sigmoid(-logit_visibility_3);

    const float assignment_one =
        log_p_of(
        context.observed_plate_one_position,
        context.inverse_observed_plate_one_position_diagonal_covariance,
        pred_0) +
        log_p_of(
        context.observed_plate_two_position,
        context.inverse_observed_plate_two_position_diagonal_covariance,
        pred_1);

    const float assignment_two =
        log_p_of(
        context.observed_plate_one_position,
        context.inverse_observed_plate_one_position_diagonal_covariance,
        pred_1) +
        log_p_of(
        context.observed_plate_two_position,
        context.inverse_observed_plate_two_position_diagonal_covariance,
        pred_0);

    return log_pr_visibility + std::max(assignment_one, assignment_two);
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] float rough_conditional_log_likelihood_from_precomputed(
      const util::default_rv_sampler& sampler,
      const likelihood_evaluation_context& context,
      const prediction& given) const noexcept {
    (void)sampler;

    const Eigen::Vector2f center = given.center();
    const float radius = given.radius();
    const float radius_squared = radius * radius;

    auto rough_ring_log_p = [center, radius_squared](
                  const Eigen::Vector3f& observed_position,
                  const Eigen::Vector3f& inverse_observed_position_diagonal_covariance) {
      const float distance_squared = (observed_position.head<2>() - center).squaredNorm();
      const float radial_error = distance_squared - radius_squared;

      const float inverse_xy_variance =
        0.5f * (inverse_observed_position_diagonal_covariance[0] + inverse_observed_position_diagonal_covariance[1]);

      return -0.5f * inverse_xy_variance * radial_error * radial_error;
    };

    auto rough_z_log_p = [](const float observed_z,
                const float predicted_z,
                const Eigen::Vector3f& inverse_observed_position_diagonal_covariance) {
      const float z_error = observed_z - predicted_z;
      return -0.5f * inverse_observed_position_diagonal_covariance[2] * z_error * z_error;
    };

    if (!context.has_plate_two) {
      const float z0 = rough_z_log_p(
        context.observed_plate_one_position[2],
        given.z_coordinate_0(),
        context.inverse_observed_plate_one_position_diagonal_covariance);
      const float z1 = rough_z_log_p(
        context.observed_plate_one_position[2],
        given.z_coordinate_1(),
        context.inverse_observed_plate_one_position_diagonal_covariance);

      return rough_ring_log_p(
           context.observed_plate_one_position,
           context.inverse_observed_plate_one_position_diagonal_covariance) +
         thrust::max(z0, z1);
    }

    const float assignment_one =
      rough_ring_log_p(
        context.observed_plate_one_position,
        context.inverse_observed_plate_one_position_diagonal_covariance) +
      rough_ring_log_p(
        context.observed_plate_two_position,
        context.inverse_observed_plate_two_position_diagonal_covariance) +
      rough_z_log_p(
        context.observed_plate_one_position[2],
        given.z_coordinate_0(),
        context.inverse_observed_plate_one_position_diagonal_covariance) +
      rough_z_log_p(
        context.observed_plate_two_position[2],
        given.z_coordinate_1(),
        context.inverse_observed_plate_two_position_diagonal_covariance);

    const float assignment_two =
      rough_ring_log_p(
        context.observed_plate_one_position,
        context.inverse_observed_plate_one_position_diagonal_covariance) +
      rough_ring_log_p(
        context.observed_plate_two_position,
        context.inverse_observed_plate_two_position_diagonal_covariance) +
      rough_z_log_p(
        context.observed_plate_one_position[2],
        given.z_coordinate_1(),
        context.inverse_observed_plate_one_position_diagonal_covariance) +
      rough_z_log_p(
        context.observed_plate_two_position[2],
        given.z_coordinate_0(),
        context.inverse_observed_plate_two_position_diagonal_covariance);

    return thrust::max(assignment_one, assignment_two);
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] float conditional_log_likelihood(
      const util::default_rv_sampler& sampler,
      const observation& state,
      const prediction& given) const noexcept {
    const auto context = precompute_likelihood_evaluation_context(state);
    return conditional_log_likelihood_from_precomputed(sampler, context, given);
  }

  PF_TARGET_ATTRS [[nodiscard]] float refinement_log_likelihood_window() const noexcept {
    return params_.likelihood_refinement_window;
  }

  PF_TARGET_ATTRS [[nodiscard]] std::uint32_t observation_resample_period() const noexcept {
    return params_.observation_resample_period;
  }

  PF_TARGET_ATTRS [[nodiscard]] std::uint32_t observation_update_subsample_stride() const noexcept {
    return params_.observation_update_subsample_stride;
  }

  PF_TARGET_ATTRS [[nodiscard]] initial_sampling_context precompute_initial_sampling_context(
      const observation& state) const noexcept {
    const observed_plate_orbit_builder builder(params_.radius_prior, state.observer_position());

    const observed_plate_orbit orbit = state.plate_two().has_value() ?
                                           builder.from_two_plates(state.plate_one(), *state.plate_two()) :
                                           builder.from_one_plate(state.plate_one());

    const float radius_variance =
        state.plate_two().has_value() ? params_.radius_prior_variance_two_plates : params_.radius_prior_variance_one_plate;

    return initial_sampling_context{
        .orbit = orbit,
        .radius_variance = radius_variance,
        .plate_one_position_diagonal_covariance = state.plate_one().position_diagonal_covariance(),
    };
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] prediction sample_from_precomputed(
      util::default_rv_sampler& sampler,
      const initial_sampling_context& context) const noexcept {
    const float radius = context.orbit.radius + sampler.normal_sample(context.radius_variance);

    const float orientation = context.orbit.orientation;
    const float orientation_velocity = sampler.normal_sample(params_.orientation_velocity_prior_variance);

    const Eigen::Vector3f center = context.orbit.center + sampler.normal_sample(context.plate_one_position_diagonal_covariance);
    const Eigen::Vector2f center_xy = center.head<2>();

    const float z_coordinate_0 = center.tail<1>().value();
    const float z_coordinate_1 = center.tail<1>().value();

    const Eigen::Vector2f center_xy_velocity = sampler.normal_sample(params_.center_velocity_prior_diagonal_covariance);

    return prediction(radius, z_coordinate_0, z_coordinate_1, orientation, orientation_velocity, center_xy, center_xy_velocity);
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] prediction sample_from(util::default_rv_sampler& sampler, const observation& state)
      const noexcept {
    const auto context = precompute_initial_sampling_context(state);
    return sample_from_precomputed(sampler, context);
  }

  PF_TARGET_ONLY_ATTRS void apply_process(const float& time_offset_seconds, util::default_rv_sampler& sampler, prediction& state)
      const noexcept {
    const float radius_noise = sampler.normal_sample(params_.radius_process_variance);

    const float z_coordinate_noise_common = sampler.normal_sample(params_.z_coordinate_common_process_variance);
    const float z_coordinate_noise_0 = sampler.normal_sample(params_.z_coordinate_offset_process_variance);
    const float z_coordinate_noise_1 = sampler.normal_sample(params_.z_coordinate_offset_process_variance);

    const float orientation_velocity_noise_0 = sampler.normal_sample(params_.orientation_velocity_process_variance);
    const float orientation_velocity_noise_1 = sampler.normal_sample(params_.orientation_velocity_process_variance);

    const Eigen::Vector2f center_velocity_noise_0 = sampler.normal_sample(params_.center_velocity_process_diagonal_covariance);
    const Eigen::Vector2f center_velocity_noise_1 = sampler.normal_sample(params_.center_velocity_process_diagonal_covariance);

    state.update_state(
        time_offset_seconds,
        radius_noise,
        z_coordinate_noise_common,
        z_coordinate_noise_0,
        z_coordinate_noise_1,
        orientation_velocity_noise_0,
        orientation_velocity_noise_1,
        center_velocity_noise_0,
        center_velocity_noise_1);
  }

  particle_filter_configuration(const particle_filter_configuration_parameters& params) noexcept : params_{params} {}
};

}  // namespace fast_plate_orbit_with_z_offset
