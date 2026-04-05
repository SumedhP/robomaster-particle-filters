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
 private:
  particle_filter_configuration_parameters params_;

  float radius_prior_standard_deviation_one_plate_;
  float radius_prior_standard_deviation_two_plates_;
  float radius_process_standard_deviation_;

  float z_coordinate_common_process_standard_deviation_;
  float z_coordinate_offset_process_standard_deviation_;

  float orientation_velocity_prior_standard_deviation_;
  float orientation_velocity_process_standard_deviation_;

  Eigen::Vector2f center_velocity_prior_diagonal_standard_deviation_;
  Eigen::Vector2f center_velocity_process_diagonal_standard_deviation_;

 public:
  using observation_type = observation;
  using prediction_type = prediction;
  using sampler_type = util::default_rv_sampler;

  struct process_noise_scales {
    float time_offset_seconds;
    float radius_noise_scale;
    float z_coordinate_noise_scale;
    float velocity_noise_scale;
    float position_noise_scale;
  };

  struct likelihood_observation_cache {
    float observer_x;
    float observer_y;
    float observer_z;

    float plate_one_x;
    float plate_one_y;
    float plate_one_z;

    float plate_one_precision_x;
    float plate_one_precision_y;
    float plate_one_precision_z;

    bool has_plate_two;

    float plate_two_x;
    float plate_two_y;
    float plate_two_z;

    float plate_two_precision_x;
    float plate_two_precision_y;
    float plate_two_precision_z;
  };

  struct initialization_observation_cache {
    float orbit_radius;
    float orbit_orientation;

    float orbit_center_x;
    float orbit_center_y;
    float orbit_center_z;

    float radius_standard_deviation;

    float plate_one_position_standard_deviation_x;
    float plate_one_position_standard_deviation_y;
    float plate_one_position_standard_deviation_z;
  };

    PF_TARGET_ATTRS [[nodiscard]] static process_noise_scales
    compute_process_noise_scales(const float& time_offset_seconds) noexcept {
    static constexpr float sqrt_one_twelfth = 0.28867513459481287f;

    const float velocity_noise_scale = sqrtf(time_offset_seconds);
    const float velocity_noise_scale_squared = velocity_noise_scale * velocity_noise_scale;

    return process_noise_scales{
        time_offset_seconds,
      velocity_noise_scale,
      velocity_noise_scale,
      velocity_noise_scale,
      sqrt_one_twelfth * velocity_noise_scale_squared * velocity_noise_scale};
  }

  PF_TARGET_ATTRS [[nodiscard]] static likelihood_observation_cache
  build_likelihood_observation_cache(const observation& state) noexcept {
    const Eigen::Vector3f observer_position = state.observer_position();

    const observed_plate& plate_one = state.plate_one();
    const Eigen::Vector3f plate_one_position = plate_one.position();
    const Eigen::Vector3f plate_one_covariance = plate_one.position_diagonal_covariance();

    likelihood_observation_cache cache{
      observer_position(0),
      observer_position(1),
      observer_position(2),
      plate_one_position(0),
      plate_one_position(1),
      plate_one_position(2),
      1.0f / plate_one_covariance(0),
      1.0f / plate_one_covariance(1),
      1.0f / plate_one_covariance(2),
      false,
      0.0f,
      0.0f,
      0.0f,
      0.0f,
      0.0f,
      0.0f,
    };

    const auto& plate_two = state.plate_two();
    if (plate_two.has_value()) {
      const observed_plate& plate_two_value = *plate_two;
      const Eigen::Vector3f plate_two_position = plate_two_value.position();
      const Eigen::Vector3f plate_two_covariance = plate_two_value.position_diagonal_covariance();

      cache.has_plate_two = true;

      cache.plate_two_x = plate_two_position(0);
      cache.plate_two_y = plate_two_position(1);
      cache.plate_two_z = plate_two_position(2);

      cache.plate_two_precision_x = 1.0f / plate_two_covariance(0);
      cache.plate_two_precision_y = 1.0f / plate_two_covariance(1);
      cache.plate_two_precision_z = 1.0f / plate_two_covariance(2);
    }

    return cache;
  }

  PF_TARGET_ATTRS [[nodiscard]] initialization_observation_cache
  build_initialization_observation_cache(const observation& state) const noexcept {
    const observed_plate_orbit_builder builder(params_.radius_prior, state.observer_position());

    const observed_plate_orbit orbit = state.plate_two().has_value() ?
                                           builder.from_two_plates(state.plate_one(), *state.plate_two()) :
                                           builder.from_one_plate(state.plate_one());

    const float radius_standard_deviation =
      state.plate_two().has_value() ? radius_prior_standard_deviation_two_plates_ : radius_prior_standard_deviation_one_plate_;

    const Eigen::Vector3f plate_one_covariance = state.plate_one().position_diagonal_covariance();

    return initialization_observation_cache{
      orbit.radius,
      orbit.orientation,
      orbit.center(0),
      orbit.center(1),
      orbit.center(2),
      radius_standard_deviation,
      sqrtf(plate_one_covariance(0)),
      sqrtf(plate_one_covariance(1)),
      sqrtf(plate_one_covariance(2)),
    };
  }

  [[nodiscard]] most_likely_particle_reduction_impl most_likely_particle_reduction() const noexcept {
    return most_likely_particle_reduction_impl{};
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] float conditional_log_likelihood_from_fields(
      const sampler_type& sampler,
      const likelihood_observation_cache& observation_cache,
      const float& radius,
      const float& z_coordinate_0,
      const float& z_coordinate_1,
      const float& orientation,
      const float& center_x,
      const float& center_y) const noexcept {
    const float observer_x = observation_cache.observer_x;
    const float observer_y = observation_cache.observer_y;
    const float observer_z = observation_cache.observer_z;

#ifdef __CUDA_ARCH__
    float sin_orientation;
    float cos_orientation;
    __sincosf(orientation, &sin_orientation, &cos_orientation);
#else
    const float sin_orientation = sinf(orientation);
    const float cos_orientation = cosf(orientation);
#endif

    const float radius_cos_0 = radius * cos_orientation;
    const float radius_sin_0 = radius * sin_orientation;

    float predicted_x[4] = {
      center_x + radius_cos_0,
      center_x - radius_sin_0,
      center_x - radius_cos_0,
      center_x + radius_sin_0,
    };

    float predicted_y[4] = {
      center_y + radius_sin_0,
      center_y + radius_cos_0,
      center_y - radius_sin_0,
      center_y - radius_cos_0,
    };

    float predicted_z[4] = {z_coordinate_0, z_coordinate_1, z_coordinate_0, z_coordinate_1};

    float squared_distance[4] = {
      (observer_x - predicted_x[0]) * (observer_x - predicted_x[0]) + (observer_y - predicted_y[0]) * (observer_y - predicted_y[0]) +
          (observer_z - predicted_z[0]) * (observer_z - predicted_z[0]),
      (observer_x - predicted_x[1]) * (observer_x - predicted_x[1]) + (observer_y - predicted_y[1]) * (observer_y - predicted_y[1]) +
          (observer_z - predicted_z[1]) * (observer_z - predicted_z[1]),
      (observer_x - predicted_x[2]) * (observer_x - predicted_x[2]) + (observer_y - predicted_y[2]) * (observer_y - predicted_y[2]) +
          (observer_z - predicted_z[2]) * (observer_z - predicted_z[2]),
      (observer_x - predicted_x[3]) * (observer_x - predicted_x[3]) + (observer_y - predicted_y[3]) * (observer_y - predicted_y[3]) +
          (observer_z - predicted_z[3]) * (observer_z - predicted_z[3]),
    };

    auto swap_entries = [&](const int lhs, const int rhs) {
      if (squared_distance[lhs] <= squared_distance[rhs]) {
        return;
      }

      std::swap(squared_distance[lhs], squared_distance[rhs]);
      std::swap(predicted_x[lhs], predicted_x[rhs]);
      std::swap(predicted_y[lhs], predicted_y[rhs]);
      std::swap(predicted_z[lhs], predicted_z[rhs]);
    };

    // Sorting network for 4 elements.
    swap_entries(0, 1);
    swap_entries(2, 3);
    swap_entries(0, 2);
    swap_entries(1, 3);
    swap_entries(1, 2);

    const float observer_delta_x = observer_x - center_x;
    const float observer_delta_y = observer_y - center_y;
    const float observer_delta_norm_squared = observer_delta_x * observer_delta_x + observer_delta_y * observer_delta_y;
  #ifdef __CUDA_ARCH__
    const float observer_inverse_norm = rsqrtf(fmaxf(observer_delta_norm_squared, 1.0e-12f));
  #else
    const float observer_inverse_norm = 1.0f / sqrtf(fmaxf(observer_delta_norm_squared, 1.0e-12f));
  #endif

    const float inverse_radius = 1.0f / fmaxf(fabsf(radius), 1.0e-6f);
    const float visibility_scale = params_.visibility_logit_coefficient * observer_inverse_norm * inverse_radius;

    auto visibility_logit_for = [&](const int index) {
      const float plate_delta_x = predicted_x[index] - center_x;
      const float plate_delta_y = predicted_y[index] - center_y;

      const float similarity_numerator = observer_delta_x * plate_delta_x + observer_delta_y * plate_delta_y;
      return visibility_scale * similarity_numerator;
    };

    const float logit_visibility_0 = visibility_logit_for(0);
    const float logit_visibility_1 = visibility_logit_for(1);
    const float logit_visibility_2 = visibility_logit_for(2);
    const float logit_visibility_3 = visibility_logit_for(3);

    auto log_p_of = [&sampler, &predicted_x, &predicted_y, &predicted_z](
                      const float observed_x,
                      const float observed_y,
                      const float observed_z,
                      const float precision_x,
                      const float precision_y,
                      const float precision_z,
                      const int index) {
      const float x_error = observed_x - predicted_x[index];
      const float y_error = observed_y - predicted_y[index];
      const float z_error = observed_z - predicted_z[index];

            return sampler.unnormalized_normal_log_density_from_precision(precision_x, x_error) +
              sampler.unnormalized_normal_log_density_from_precision(precision_y, y_error) +
              sampler.unnormalized_normal_log_density_from_precision(precision_z, z_error);
    };

    if (!observation_cache.has_plate_two) {
      const float pr_visibility = helper::log_sigmoid(logit_visibility_0) + helper::log_sigmoid(-logit_visibility_1) +
                                  helper::log_sigmoid(-logit_visibility_2) + helper::log_sigmoid(-logit_visibility_3);

      return pr_visibility + log_p_of(
                               observation_cache.plate_one_x,
                               observation_cache.plate_one_y,
                               observation_cache.plate_one_z,
                               observation_cache.plate_one_precision_x,
                               observation_cache.plate_one_precision_y,
                               observation_cache.plate_one_precision_z,
                               0);
    }

    const float log_pr_visibility = helper::log_sigmoid(logit_visibility_0) + helper::log_sigmoid(logit_visibility_1) +
                                    helper::log_sigmoid(-logit_visibility_2) + helper::log_sigmoid(-logit_visibility_3);

    const float plate_one_given_pred_0 = log_p_of(
      observation_cache.plate_one_x,
      observation_cache.plate_one_y,
      observation_cache.plate_one_z,
      observation_cache.plate_one_precision_x,
      observation_cache.plate_one_precision_y,
      observation_cache.plate_one_precision_z,
      0);
    const float plate_one_given_pred_1 = log_p_of(
      observation_cache.plate_one_x,
      observation_cache.plate_one_y,
      observation_cache.plate_one_z,
      observation_cache.plate_one_precision_x,
      observation_cache.plate_one_precision_y,
      observation_cache.plate_one_precision_z,
      1);

    const float plate_two_given_pred_0 = log_p_of(
      observation_cache.plate_two_x,
      observation_cache.plate_two_y,
      observation_cache.plate_two_z,
      observation_cache.plate_two_precision_x,
      observation_cache.plate_two_precision_y,
      observation_cache.plate_two_precision_z,
      0);
    const float plate_two_given_pred_1 = log_p_of(
      observation_cache.plate_two_x,
      observation_cache.plate_two_y,
      observation_cache.plate_two_z,
      observation_cache.plate_two_precision_x,
      observation_cache.plate_two_precision_y,
      observation_cache.plate_two_precision_z,
      1);

    const float assignment_one = plate_one_given_pred_0 + plate_two_given_pred_1;
    const float assignment_two = plate_one_given_pred_1 + plate_two_given_pred_0;
    return log_pr_visibility + std::max(assignment_one, assignment_two);
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] float conditional_log_likelihood_from_fields(
      const sampler_type& sampler,
      const observation& state,
      const float& radius,
      const float& z_coordinate_0,
      const float& z_coordinate_1,
      const float& orientation,
      const float& center_x,
      const float& center_y) const noexcept {
    const likelihood_observation_cache observation_cache = build_likelihood_observation_cache(state);

    return conditional_log_likelihood_from_fields(
      sampler,
        observation_cache,
        radius,
        z_coordinate_0,
        z_coordinate_1,
        orientation,
        center_x,
        center_y);
  }

  PF_TARGET_ONLY_ATTRS void apply_process_to_fields(
      const process_noise_scales& process_scales,
      sampler_type& sampler,
      float& radius,
      float& z_coordinate_0,
      float& z_coordinate_1,
      float& orientation,
      float& orientation_velocity,
      float& center_x,
      float& center_y,
      float& center_velocity_x,
      float& center_velocity_y) const noexcept {
    const float center_velocity_process_standard_deviation_x = center_velocity_process_diagonal_standard_deviation_(0);
    const float center_velocity_process_standard_deviation_y = center_velocity_process_diagonal_standard_deviation_(1);

    const float radius_noise = sampler.normal_sample_from_standard_deviation(radius_process_standard_deviation_);

    const float z_coordinate_noise_common =
      sampler.normal_sample_from_standard_deviation(z_coordinate_common_process_standard_deviation_);
    const float z_coordinate_noise_0 =
      sampler.normal_sample_from_standard_deviation(z_coordinate_offset_process_standard_deviation_);
    const float z_coordinate_noise_1 =
      sampler.normal_sample_from_standard_deviation(z_coordinate_offset_process_standard_deviation_);

    const float orientation_velocity_noise_0 =
      sampler.normal_sample_from_standard_deviation(orientation_velocity_process_standard_deviation_);
    const float orientation_velocity_noise_1 =
      sampler.normal_sample_from_standard_deviation(orientation_velocity_process_standard_deviation_);

    const float center_velocity_noise_0_x =
      sampler.normal_sample_from_standard_deviation(center_velocity_process_standard_deviation_x);
    const float center_velocity_noise_0_y =
      sampler.normal_sample_from_standard_deviation(center_velocity_process_standard_deviation_y);

    const float center_velocity_noise_1_x =
      sampler.normal_sample_from_standard_deviation(center_velocity_process_standard_deviation_x);
    const float center_velocity_noise_1_y =
      sampler.normal_sample_from_standard_deviation(center_velocity_process_standard_deviation_y);

    static constexpr float one_half = 0.5f;

    const float d_radius = process_scales.radius_noise_scale * radius_noise;

    const float d_z_coordinate_common = process_scales.z_coordinate_noise_scale * z_coordinate_noise_common;
    const float d_z_coordinate_0 = process_scales.z_coordinate_noise_scale * z_coordinate_noise_0;
    const float d_z_coordinate_1 = process_scales.z_coordinate_noise_scale * z_coordinate_noise_1;

    const float d_orientation_velocity = process_scales.velocity_noise_scale * orientation_velocity_noise_1;
    const float d_orientation = process_scales.time_offset_seconds * orientation_velocity +
                                one_half * process_scales.time_offset_seconds * d_orientation_velocity +
                                process_scales.position_noise_scale * orientation_velocity_noise_0;

    const float d_center_velocity_x = process_scales.velocity_noise_scale * center_velocity_noise_1_x;
    const float d_center_velocity_y = process_scales.velocity_noise_scale * center_velocity_noise_1_y;
    const float d_center_x = process_scales.time_offset_seconds * center_velocity_x +
                             one_half * process_scales.time_offset_seconds * d_center_velocity_x +
                 process_scales.position_noise_scale * center_velocity_noise_0_x;
    const float d_center_y = process_scales.time_offset_seconds * center_velocity_y +
                             one_half * process_scales.time_offset_seconds * d_center_velocity_y +
                 process_scales.position_noise_scale * center_velocity_noise_0_y;

    radius = helper::to_radius(radius + d_radius);

    helper::update_value_offsets<helper::z_coordinate_update_configuration>(
        d_z_coordinate_common,
        d_z_coordinate_0,
        d_z_coordinate_1,
        z_coordinate_0,
        z_coordinate_1);

    float next_orientation = orientation + d_orientation;
    if (next_orientation < 0.0f) {
      next_orientation += M_PI;
      if (next_orientation < 0.0f) {
        next_orientation = helper::to_orientation(next_orientation);
      }
    } else if (next_orientation >= M_PI) {
      next_orientation -= M_PI;
      if (next_orientation >= M_PI) {
        next_orientation = helper::to_orientation(next_orientation);
      }
    }

    orientation = next_orientation;
    orientation_velocity += d_orientation_velocity;

    center_x += d_center_x;
    center_y += d_center_y;
    center_velocity_x += d_center_velocity_x;
    center_velocity_y += d_center_velocity_y;
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] float conditional_log_likelihood(
      const sampler_type& sampler,
      const observation& state,
      const prediction& given) const noexcept {
    const Eigen::Vector2f center = given.center();
    return conditional_log_likelihood_from_fields(
      sampler,
        state,
        given.radius(),
        given.z_coordinate_0(),
        given.z_coordinate_1(),
        given.orientation(),
        center(0),
        center(1));
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] prediction sample_from(sampler_type& sampler, const observation& state)
      const noexcept {
    const initialization_observation_cache cache = build_initialization_observation_cache(state);
    return sample_from_cache(sampler, cache);
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] prediction sample_from_cache(
      sampler_type& sampler,
      const initialization_observation_cache& cache) const noexcept {
    const float radius = cache.orbit_radius + sampler.normal_sample_from_standard_deviation(cache.radius_standard_deviation);

    const float orientation = cache.orbit_orientation;
    const float orientation_velocity = sampler.normal_sample_from_standard_deviation(orientation_velocity_prior_standard_deviation_);

    const float center_x = cache.orbit_center_x +
                           sampler.normal_sample_from_standard_deviation(cache.plate_one_position_standard_deviation_x);
    const float center_y = cache.orbit_center_y +
                           sampler.normal_sample_from_standard_deviation(cache.plate_one_position_standard_deviation_y);
    const float center_z = cache.orbit_center_z +
                           sampler.normal_sample_from_standard_deviation(cache.plate_one_position_standard_deviation_z);

    const float center_velocity_x =
      sampler.normal_sample_from_standard_deviation(center_velocity_prior_diagonal_standard_deviation_(0));
    const float center_velocity_y =
      sampler.normal_sample_from_standard_deviation(center_velocity_prior_diagonal_standard_deviation_(1));

    return prediction(
      radius,
      center_z,
      center_z,
      orientation,
      orientation_velocity,
      Eigen::Vector2f(center_x, center_y),
      Eigen::Vector2f(center_velocity_x, center_velocity_y));
  }

  PF_TARGET_ONLY_ATTRS void apply_process(const float& time_offset_seconds, sampler_type& sampler, prediction& state)
      const noexcept {
    const float radius_noise = sampler.normal_sample_from_standard_deviation(radius_process_standard_deviation_);

    const float z_coordinate_noise_common =
      sampler.normal_sample_from_standard_deviation(z_coordinate_common_process_standard_deviation_);
    const float z_coordinate_noise_0 =
      sampler.normal_sample_from_standard_deviation(z_coordinate_offset_process_standard_deviation_);
    const float z_coordinate_noise_1 =
      sampler.normal_sample_from_standard_deviation(z_coordinate_offset_process_standard_deviation_);

    const float orientation_velocity_noise_0 =
      sampler.normal_sample_from_standard_deviation(orientation_velocity_process_standard_deviation_);
    const float orientation_velocity_noise_1 =
      sampler.normal_sample_from_standard_deviation(orientation_velocity_process_standard_deviation_);

    const Eigen::Vector2f center_velocity_noise_0 =
      sampler.normal_sample_from_standard_deviation(center_velocity_process_diagonal_standard_deviation_);
    const Eigen::Vector2f center_velocity_noise_1 =
      sampler.normal_sample_from_standard_deviation(center_velocity_process_diagonal_standard_deviation_);

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

  particle_filter_configuration(const particle_filter_configuration_parameters& params) noexcept
      : params_{params},
        radius_prior_standard_deviation_one_plate_{sqrtf(params_.radius_prior_variance_one_plate)},
        radius_prior_standard_deviation_two_plates_{sqrtf(params_.radius_prior_variance_two_plates)},
        radius_process_standard_deviation_{sqrtf(params_.radius_process_variance)},
        z_coordinate_common_process_standard_deviation_{sqrtf(params_.z_coordinate_common_process_variance)},
        z_coordinate_offset_process_standard_deviation_{sqrtf(params_.z_coordinate_offset_process_variance)},
        orientation_velocity_prior_standard_deviation_{sqrtf(params_.orientation_velocity_prior_variance)},
        orientation_velocity_process_standard_deviation_{sqrtf(params_.orientation_velocity_process_variance)},
        center_velocity_prior_diagonal_standard_deviation_{params_.center_velocity_prior_diagonal_covariance.cwiseSqrt()},
        center_velocity_process_diagonal_standard_deviation_{params_.center_velocity_process_diagonal_covariance.cwiseSqrt()} {}
};

}  // namespace fast_plate_orbit_with_z_offset
