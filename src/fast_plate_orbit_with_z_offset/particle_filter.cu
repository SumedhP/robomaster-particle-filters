#include <fast_plate_orbit_with_z_offset/particle_filter.h>
#include <fast_plate_orbit_with_z_offset/particle_filter_configuration.h>
#include <pf/filter/particle_filter.h>
#include <pf/filter/systematic_resampler.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda/std/tuple>
#include <utility>

namespace fast_plate_orbit_with_z_offset {

namespace helper {

using configuration_type = particle_filter_configuration;
using sampler_type = configuration_type::sampler_type;
using state_storage_type = __half;

PF_TARGET_ATTRS [[nodiscard]] inline float state_storage_to_float(const state_storage_type& value) noexcept {
  return __half2float(value);
}

PF_TARGET_ATTRS [[nodiscard]] inline state_storage_type float_to_state_storage(const float& value) noexcept {
  return __float2half_rn(value);
}

struct reduction_particle_state {
  float radius_;
  float z_coordinate_0_;
  float z_coordinate_1_;
  float orientation_;
  float orientation_velocity_;
  float center_x_;
  float center_y_;
  float center_velocity_x_;
  float center_velocity_y_;
  std::uint32_t count_;

  PF_TARGET_ATTRS [[nodiscard]] static reduction_particle_state zero() noexcept {
    return reduction_particle_state{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0U};
  }

  PF_TARGET_ATTRS [[nodiscard]] static reduction_particle_state from_fields(
      const float radius,
      const float z_coordinate_0,
      const float z_coordinate_1,
      const float orientation,
      const float orientation_velocity,
      const float center_x,
      const float center_y,
      const float center_velocity_x,
      const float center_velocity_y) noexcept {
    return reduction_particle_state{
        radius,
        z_coordinate_0,
        z_coordinate_1,
        orientation,
        orientation_velocity,
        center_x,
        center_y,
        center_velocity_x,
        center_velocity_y,
        1U};
  }
};

template <typename SamplerType>
struct seed_sampler_state_functor {
  template <typename TupleType>
  PF_TARGET_ATTRS void operator()(TupleType tuple) const noexcept {
    const std::uint64_t particle_index = static_cast<std::uint64_t>(cuda::std::get<0>(tuple));
    cuda::std::get<1>(tuple).seed_from_index(particle_index);
  }
};

struct process_particle_fields_functor {
  configuration_type config_;
  configuration_type::process_noise_scales process_noise_scales_;

  template <typename TupleType>
  PF_TARGET_ONLY_ATTRS void operator()(TupleType tuple) const noexcept {
    sampler_type& sampler_state = cuda::std::get<0>(tuple);

    state_storage_type& radius_storage = cuda::std::get<1>(tuple);
    state_storage_type& z_coordinate_0_storage = cuda::std::get<2>(tuple);
    state_storage_type& z_coordinate_1_storage = cuda::std::get<3>(tuple);

    state_storage_type& orientation_storage = cuda::std::get<4>(tuple);
    state_storage_type& orientation_velocity_storage = cuda::std::get<5>(tuple);

    state_storage_type& center_x_storage = cuda::std::get<6>(tuple);
    state_storage_type& center_y_storage = cuda::std::get<7>(tuple);
    state_storage_type& center_velocity_x_storage = cuda::std::get<8>(tuple);
    state_storage_type& center_velocity_y_storage = cuda::std::get<9>(tuple);

    float radius = state_storage_to_float(radius_storage);
    float z_coordinate_0 = state_storage_to_float(z_coordinate_0_storage);
    float z_coordinate_1 = state_storage_to_float(z_coordinate_1_storage);

    float orientation = state_storage_to_float(orientation_storage);
    float orientation_velocity = state_storage_to_float(orientation_velocity_storage);

    float center_x = state_storage_to_float(center_x_storage);
    float center_y = state_storage_to_float(center_y_storage);
    float center_velocity_x = state_storage_to_float(center_velocity_x_storage);
    float center_velocity_y = state_storage_to_float(center_velocity_y_storage);

    config_.apply_process_to_fields(
      process_noise_scales_,
      sampler_state,
      radius,
      z_coordinate_0,
      z_coordinate_1,
      orientation,
      orientation_velocity,
      center_x,
      center_y,
      center_velocity_x,
      center_velocity_y);

    radius_storage = float_to_state_storage(radius);
    z_coordinate_0_storage = float_to_state_storage(z_coordinate_0);
    z_coordinate_1_storage = float_to_state_storage(z_coordinate_1);

    orientation_storage = float_to_state_storage(orientation);
    orientation_velocity_storage = float_to_state_storage(orientation_velocity);

    center_x_storage = float_to_state_storage(center_x);
    center_y_storage = float_to_state_storage(center_y);
    center_velocity_x_storage = float_to_state_storage(center_velocity_x);
    center_velocity_y_storage = float_to_state_storage(center_velocity_y);
  }
};

struct process_and_weight_particle_fields_functor {
  configuration_type config_;
  configuration_type::process_noise_scales process_noise_scales_;
  configuration_type::likelihood_observation_cache observation_cache_;

  template <typename TupleType>
  PF_TARGET_ONLY_ATTRS void operator()(TupleType tuple) const noexcept {
    sampler_type& sampler_state = cuda::std::get<0>(tuple);
    float& particle_weight = cuda::std::get<1>(tuple);

    state_storage_type& radius_storage = cuda::std::get<2>(tuple);
    state_storage_type& z_coordinate_0_storage = cuda::std::get<3>(tuple);
    state_storage_type& z_coordinate_1_storage = cuda::std::get<4>(tuple);

    state_storage_type& orientation_storage = cuda::std::get<5>(tuple);
    state_storage_type& orientation_velocity_storage = cuda::std::get<6>(tuple);

    state_storage_type& center_x_storage = cuda::std::get<7>(tuple);
    state_storage_type& center_y_storage = cuda::std::get<8>(tuple);
    state_storage_type& center_velocity_x_storage = cuda::std::get<9>(tuple);
    state_storage_type& center_velocity_y_storage = cuda::std::get<10>(tuple);

    float radius = state_storage_to_float(radius_storage);
    float z_coordinate_0 = state_storage_to_float(z_coordinate_0_storage);
    float z_coordinate_1 = state_storage_to_float(z_coordinate_1_storage);

    float orientation = state_storage_to_float(orientation_storage);
    float orientation_velocity = state_storage_to_float(orientation_velocity_storage);

    float center_x = state_storage_to_float(center_x_storage);
    float center_y = state_storage_to_float(center_y_storage);
    float center_velocity_x = state_storage_to_float(center_velocity_x_storage);
    float center_velocity_y = state_storage_to_float(center_velocity_y_storage);

    config_.apply_process_to_fields(
      process_noise_scales_,
      sampler_state,
      radius,
      z_coordinate_0,
      z_coordinate_1,
      orientation,
      orientation_velocity,
      center_x,
      center_y,
      center_velocity_x,
      center_velocity_y);

    particle_weight = config_.conditional_log_likelihood_from_fields(
      sampler_state,
      observation_cache_,
      radius,
      z_coordinate_0,
      z_coordinate_1,
      orientation,
      center_x,
      center_y);

    radius_storage = float_to_state_storage(radius);
    z_coordinate_0_storage = float_to_state_storage(z_coordinate_0);
    z_coordinate_1_storage = float_to_state_storage(z_coordinate_1);

    orientation_storage = float_to_state_storage(orientation);
    orientation_velocity_storage = float_to_state_storage(orientation_velocity);

    center_x_storage = float_to_state_storage(center_x);
    center_y_storage = float_to_state_storage(center_y);
    center_velocity_x_storage = float_to_state_storage(center_velocity_x);
    center_velocity_y_storage = float_to_state_storage(center_velocity_y);
  }
};

struct initialize_particle_fields_functor {
  configuration_type config_;
  configuration_type::initialization_observation_cache initialization_cache_;

  template <typename TupleType>
  PF_TARGET_ONLY_ATTRS void operator()(TupleType tuple) const noexcept {
    sampler_type& sampler_state = cuda::std::get<0>(tuple);

    prediction sampled_particle = config_.sample_from_cache(sampler_state, initialization_cache_);

    cuda::std::get<1>(tuple) = float_to_state_storage(sampled_particle.radius());
    cuda::std::get<2>(tuple) = float_to_state_storage(sampled_particle.z_coordinate_0());
    cuda::std::get<3>(tuple) = float_to_state_storage(sampled_particle.z_coordinate_1());

    cuda::std::get<4>(tuple) = float_to_state_storage(sampled_particle.orientation());
    cuda::std::get<5>(tuple) = float_to_state_storage(sampled_particle.orientation_velocity());

    const Eigen::Vector2f center = sampled_particle.center();
    cuda::std::get<6>(tuple) = float_to_state_storage(center(0));
    cuda::std::get<7>(tuple) = float_to_state_storage(center(1));

    const Eigen::Vector2f center_velocity = sampled_particle.center_velocity();
    cuda::std::get<8>(tuple) = float_to_state_storage(center_velocity(0));
    cuda::std::get<9>(tuple) = float_to_state_storage(center_velocity(1));
  }
};

struct reduction_state_transform_functor {
  const state_storage_type* radius_;
  const state_storage_type* z_coordinate_0_;
  const state_storage_type* z_coordinate_1_;

  const state_storage_type* orientation_;
  const state_storage_type* orientation_velocity_;

  const state_storage_type* center_x_;
  const state_storage_type* center_y_;
  const state_storage_type* center_velocity_x_;
  const state_storage_type* center_velocity_y_;

  PF_TARGET_ATTRS [[nodiscard]] reduction_particle_state operator()(const std::uint32_t index) const noexcept {
    return reduction_particle_state::from_fields(
        state_storage_to_float(radius_[index]),
        state_storage_to_float(z_coordinate_0_[index]),
        state_storage_to_float(z_coordinate_1_[index]),
        state_storage_to_float(orientation_[index]),
        state_storage_to_float(orientation_velocity_[index]),
        state_storage_to_float(center_x_[index]),
        state_storage_to_float(center_y_[index]),
        state_storage_to_float(center_velocity_x_[index]),
        state_storage_to_float(center_velocity_y_[index]));
  }
};

struct reduction_particle_state_reduce_functor {
  struct orientation_and_z_coordinates_and_radius {
    float orientation_;
    float radius_;
    float z_coordinate_0_;
    float z_coordinate_1_;
  };

  PF_TARGET_ATTRS [[nodiscard]] reduction_particle_state operator()(
      const reduction_particle_state& a,
      const reduction_particle_state& b) const noexcept {
    if (a.count_ == 0U) {
      return b;
    }

    if (b.count_ == 0U) {
      return a;
    }

    static constexpr float half_pi = M_PI_2;

    const orientation_and_z_coordinates_and_radius candidates[5] = {
      orientation_and_z_coordinates_and_radius{
          .orientation_ = a.orientation_ + 0.0f * half_pi,
          .radius_ = a.radius_,
          .z_coordinate_0_ = a.z_coordinate_0_,
          .z_coordinate_1_ = a.z_coordinate_1_,
      },
      orientation_and_z_coordinates_and_radius{
          .orientation_ = a.orientation_ + 1.0f * half_pi,
          .radius_ = a.radius_,
          .z_coordinate_0_ = a.z_coordinate_1_,
          .z_coordinate_1_ = a.z_coordinate_0_,
      },
      orientation_and_z_coordinates_and_radius{
          .orientation_ = a.orientation_ - 1.0f * half_pi,
          .radius_ = a.radius_,
          .z_coordinate_0_ = a.z_coordinate_1_,
          .z_coordinate_1_ = a.z_coordinate_0_,
      },
      orientation_and_z_coordinates_and_radius{
          .orientation_ = a.orientation_ + 2.0f * half_pi,
          .radius_ = a.radius_,
          .z_coordinate_0_ = a.z_coordinate_0_,
          .z_coordinate_1_ = a.z_coordinate_1_,
      },
      orientation_and_z_coordinates_and_radius{
          .orientation_ = a.orientation_ - 2.0f * half_pi,
          .radius_ = a.radius_,
          .z_coordinate_0_ = a.z_coordinate_0_,
          .z_coordinate_1_ = a.z_coordinate_1_,
      },
    };

    std::size_t minimum_index = 0;
    float minimum_value = fabsf(b.orientation_ - candidates[0].orientation_);

    for (std::size_t i = 1; i < 5; ++i) {
      const float value = fabsf(b.orientation_ - candidates[i].orientation_);
      if (value < minimum_value) {
        minimum_index = i;
        minimum_value = value;
      }
    }

    const auto selected = candidates[minimum_index];

    const float alpha = static_cast<float>(b.count_) / static_cast<float>(a.count_ + b.count_);
    const float c_alpha = 1.0f - alpha;

    return reduction_particle_state{
      c_alpha * selected.radius_ + alpha * b.radius_,
      c_alpha * selected.z_coordinate_0_ + alpha * b.z_coordinate_0_,
      c_alpha * selected.z_coordinate_1_ + alpha * b.z_coordinate_1_,
      c_alpha * selected.orientation_ + alpha * b.orientation_,
        c_alpha * a.orientation_velocity_ + alpha * b.orientation_velocity_,
        c_alpha * a.center_x_ + alpha * b.center_x_,
        c_alpha * a.center_y_ + alpha * b.center_y_,
        c_alpha * a.center_velocity_x_ + alpha * b.center_velocity_x_,
        c_alpha * a.center_velocity_y_ + alpha * b.center_velocity_y_,
        a.count_ + b.count_};
  }
};

}  // namespace helper

struct particle_filter::impl {
  using configuration_type = helper::configuration_type;
  using sampler_type = helper::sampler_type;
  using state_storage_type = helper::state_storage_type;

  pf::filter::helper::caching_allocator_type caching_allocator_;
  configuration_type config_;

  prediction most_likely_particle_state_;

  std::size_t number_of_particles_;
  pf::filter::systematic_resampler<std::uint32_t, std::uint32_t> index_resampler_;

  pf::target_config::vector<sampler_type> sampler_states_;
  pf::target_config::vector<float> log_particle_weights_;
  pf::target_config::vector<std::uint32_t> particle_indices_;

  // Struct-of-arrays particle state storage.
  pf::target_config::vector<state_storage_type> radius_;
  pf::target_config::vector<state_storage_type> z_coordinate_0_;
  pf::target_config::vector<state_storage_type> z_coordinate_1_;
  pf::target_config::vector<state_storage_type> orientation_;
  pf::target_config::vector<state_storage_type> orientation_velocity_;
  pf::target_config::vector<state_storage_type> center_x_;
  pf::target_config::vector<state_storage_type> center_y_;
  pf::target_config::vector<state_storage_type> center_velocity_x_;
  pf::target_config::vector<state_storage_type> center_velocity_y_;

  // Temporary storage for index-gather materialization during resampling.
  pf::target_config::vector<state_storage_type> temp_radius_;
  pf::target_config::vector<state_storage_type> temp_z_coordinate_0_;
  pf::target_config::vector<state_storage_type> temp_z_coordinate_1_;
  pf::target_config::vector<state_storage_type> temp_orientation_;
  pf::target_config::vector<state_storage_type> temp_orientation_velocity_;
  pf::target_config::vector<state_storage_type> temp_center_x_;
  pf::target_config::vector<state_storage_type> temp_center_y_;
  pf::target_config::vector<state_storage_type> temp_center_velocity_x_;
  pf::target_config::vector<state_storage_type> temp_center_velocity_y_;

  [[nodiscard]] prediction extrapolate_state(const float& time_offset_seconds) const noexcept {
    return most_likely_particle_state_.extrapolate_state(time_offset_seconds);
  }

  void reinitialize(
      const observation& initial_observation,
      const bool refresh_most_likely_particle_state) noexcept {
    const auto initialization_cache = config_.build_initialization_observation_cache(initial_observation);

    thrust::for_each(
        pf::target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(
            sampler_states_.begin(),
            radius_.begin(),
            z_coordinate_0_.begin(),
            z_coordinate_1_.begin(),
            orientation_.begin(),
            orientation_velocity_.begin(),
            center_x_.begin(),
            center_y_.begin(),
            center_velocity_x_.begin(),
            center_velocity_y_.begin()),
        thrust::make_zip_iterator(
            sampler_states_.end(),
            radius_.end(),
            z_coordinate_0_.end(),
            z_coordinate_1_.end(),
            orientation_.end(),
            orientation_velocity_.end(),
            center_x_.end(),
            center_y_.end(),
            center_velocity_x_.end(),
            center_velocity_y_.end()),
          helper::initialize_particle_fields_functor{config_, initialization_cache});

    if (refresh_most_likely_particle_state) {
      update_most_likely_particle_state_();
    }
  }

  void update_state_sans_observation(const float& time_offset_seconds) noexcept {
    const auto process_noise_scales = configuration_type::compute_process_noise_scales(time_offset_seconds);

    thrust::for_each(
        pf::target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(
            sampler_states_.begin(),
            radius_.begin(),
            z_coordinate_0_.begin(),
            z_coordinate_1_.begin(),
            orientation_.begin(),
            orientation_velocity_.begin(),
            center_x_.begin(),
            center_y_.begin(),
            center_velocity_x_.begin(),
            center_velocity_y_.begin()),
        thrust::make_zip_iterator(
            sampler_states_.end(),
            radius_.end(),
            z_coordinate_0_.end(),
            z_coordinate_1_.end(),
            orientation_.end(),
            orientation_velocity_.end(),
            center_x_.end(),
            center_y_.end(),
            center_velocity_x_.end(),
            center_velocity_y_.end()),
          helper::process_particle_fields_functor{config_, process_noise_scales});

    update_most_likely_particle_state_();
  }

  void update_state_with_observation(const float& time_offset_seconds, const observation& observation_state) noexcept {
    const auto process_noise_scales = configuration_type::compute_process_noise_scales(time_offset_seconds);
    const auto observation_cache = configuration_type::build_likelihood_observation_cache(observation_state);

    thrust::for_each(
        pf::target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(
            sampler_states_.begin(),
            log_particle_weights_.begin(),
            radius_.begin(),
            z_coordinate_0_.begin(),
            z_coordinate_1_.begin(),
            orientation_.begin(),
            orientation_velocity_.begin(),
            center_x_.begin(),
            center_y_.begin(),
            center_velocity_x_.begin(),
            center_velocity_y_.begin()),
        thrust::make_zip_iterator(
            sampler_states_.end(),
            log_particle_weights_.end(),
            radius_.end(),
            z_coordinate_0_.end(),
            z_coordinate_1_.end(),
            orientation_.end(),
            orientation_velocity_.end(),
            center_x_.end(),
            center_y_.end(),
            center_velocity_x_.end(),
            center_velocity_y_.end()),
          helper::process_and_weight_particle_fields_functor{config_, process_noise_scales, observation_cache});

    index_resampler_.resample_indices(
        pf::target_config::policy(caching_allocator_),
        log_particle_weights_,
        particle_indices_);

    resample_particle_fields_();
    update_most_likely_particle_state_();
  }

  void initialize_internal_state_(const observation& initial_observation) noexcept {
    const auto initialization_cache = config_.build_initialization_observation_cache(initial_observation);

    thrust::counting_iterator<std::size_t> index_sequence_begin(std::size_t{});

    thrust::for_each(
        pf::target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(index_sequence_begin, sampler_states_.begin()),
        thrust::make_zip_iterator(index_sequence_begin + number_of_particles_, sampler_states_.end()),
        helper::seed_sampler_state_functor<sampler_type>{});

    thrust::for_each(
        pf::target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(
            sampler_states_.begin(),
            radius_.begin(),
            z_coordinate_0_.begin(),
            z_coordinate_1_.begin(),
            orientation_.begin(),
            orientation_velocity_.begin(),
            center_x_.begin(),
            center_y_.begin(),
            center_velocity_x_.begin(),
            center_velocity_y_.begin()),
        thrust::make_zip_iterator(
            sampler_states_.end(),
            radius_.end(),
            z_coordinate_0_.end(),
            z_coordinate_1_.end(),
            orientation_.end(),
            orientation_velocity_.end(),
            center_x_.end(),
            center_y_.end(),
            center_velocity_x_.end(),
            center_velocity_y_.end()),
          helper::initialize_particle_fields_functor{config_, initialization_cache});

    update_most_likely_particle_state_();
  }

  void resample_particle_fields_() noexcept {
    auto source_particle_fields = thrust::make_zip_iterator(
      radius_.begin(),
      z_coordinate_0_.begin(),
      z_coordinate_1_.begin(),
      orientation_.begin(),
      orientation_velocity_.begin(),
      center_x_.begin(),
      center_y_.begin(),
      center_velocity_x_.begin(),
      center_velocity_y_.begin());

    auto destination_particle_fields = thrust::make_zip_iterator(
      temp_radius_.begin(),
      temp_z_coordinate_0_.begin(),
      temp_z_coordinate_1_.begin(),
      temp_orientation_.begin(),
      temp_orientation_velocity_.begin(),
      temp_center_x_.begin(),
      temp_center_y_.begin(),
      temp_center_velocity_x_.begin(),
      temp_center_velocity_y_.begin());

    thrust::gather(
      pf::target_config::policy(caching_allocator_),
      particle_indices_.cbegin(),
      particle_indices_.cend(),
      source_particle_fields,
      destination_particle_fields);

    radius_.swap(temp_radius_);
    z_coordinate_0_.swap(temp_z_coordinate_0_);
    z_coordinate_1_.swap(temp_z_coordinate_1_);
    orientation_.swap(temp_orientation_);
    orientation_velocity_.swap(temp_orientation_velocity_);
    center_x_.swap(temp_center_x_);
    center_y_.swap(temp_center_y_);
    center_velocity_x_.swap(temp_center_velocity_x_);
    center_velocity_y_.swap(temp_center_velocity_y_);
  }

  void update_most_likely_particle_state_() noexcept {
    const auto* radius = thrust::raw_pointer_cast(radius_.data());
    const auto* z_coordinate_0 = thrust::raw_pointer_cast(z_coordinate_0_.data());
    const auto* z_coordinate_1 = thrust::raw_pointer_cast(z_coordinate_1_.data());

    const auto* orientation = thrust::raw_pointer_cast(orientation_.data());
    const auto* orientation_velocity = thrust::raw_pointer_cast(orientation_velocity_.data());

    const auto* center_x = thrust::raw_pointer_cast(center_x_.data());
    const auto* center_y = thrust::raw_pointer_cast(center_y_.data());
    const auto* center_velocity_x = thrust::raw_pointer_cast(center_velocity_x_.data());
    const auto* center_velocity_y = thrust::raw_pointer_cast(center_velocity_y_.data());

    const auto index_begin = thrust::make_counting_iterator<std::uint32_t>(std::uint32_t{});
    const auto index_end = index_begin + static_cast<std::uint32_t>(number_of_particles_);

    const helper::reduction_state_transform_functor transform_functor{
      radius,
      z_coordinate_0,
      z_coordinate_1,
      orientation,
      orientation_velocity,
      center_x,
      center_y,
      center_velocity_x,
      center_velocity_y};

    const helper::reduction_particle_state reduced_particle_state = thrust::transform_reduce(
      pf::target_config::policy(caching_allocator_),
      index_begin,
      index_end,
      transform_functor,
      helper::reduction_particle_state::zero(),
      helper::reduction_particle_state_reduce_functor{});

    most_likely_particle_state_ = prediction(
      reduced_particle_state.radius_,
      reduced_particle_state.z_coordinate_0_,
      reduced_particle_state.z_coordinate_1_,
      reduced_particle_state.orientation_,
      reduced_particle_state.orientation_velocity_,
      Eigen::Vector2f(reduced_particle_state.center_x_, reduced_particle_state.center_y_),
      Eigen::Vector2f(reduced_particle_state.center_velocity_x_, reduced_particle_state.center_velocity_y_));
  }

  impl(
      const std::size_t& number_of_particles,
      const observation& initial_observation,
      const particle_filter_configuration_parameters& params) noexcept
      : caching_allocator_{pf::filter::helper::thread_local_caching_allocator()},
        config_(params),
        number_of_particles_(number_of_particles),
        index_resampler_(number_of_particles),
        sampler_states_(number_of_particles),
        log_particle_weights_(number_of_particles),
        particle_indices_(number_of_particles),
        radius_(number_of_particles),
        z_coordinate_0_(number_of_particles),
        z_coordinate_1_(number_of_particles),
        orientation_(number_of_particles),
        orientation_velocity_(number_of_particles),
        center_x_(number_of_particles),
        center_y_(number_of_particles),
        center_velocity_x_(number_of_particles),
        center_velocity_y_(number_of_particles),
        temp_radius_(number_of_particles),
        temp_z_coordinate_0_(number_of_particles),
        temp_z_coordinate_1_(number_of_particles),
        temp_orientation_(number_of_particles),
        temp_orientation_velocity_(number_of_particles),
        temp_center_x_(number_of_particles),
        temp_center_y_(number_of_particles),
        temp_center_velocity_x_(number_of_particles),
        temp_center_velocity_y_(number_of_particles) {
    initialize_internal_state_(initial_observation);
  }
};

void particle_filter::impl_deleter::operator()(particle_filter::impl* p_impl) { delete p_impl; }

prediction particle_filter::extrapolate_state(const float& time_offset_seconds) const noexcept {
  return p_impl_->extrapolate_state(time_offset_seconds);
}

void particle_filter::reinitialize(
    const observation& initial_observation,
    bool refresh_most_likely_particle_state) noexcept {
  p_impl_->reinitialize(initial_observation, refresh_most_likely_particle_state);
}

void particle_filter::update_state_sans_observation(const float& time_offset_seconds) noexcept {
  p_impl_->update_state_sans_observation(time_offset_seconds);
}

void particle_filter::update_state_with_observation(
    const float& time_offset_seconds,
    const observation& observation_state) noexcept {
  p_impl_->update_state_with_observation(time_offset_seconds, observation_state);
}

particle_filter::particle_filter(
    const std::size_t& number_of_particles,
    const observation& initial_observation,
    const particle_filter_configuration_parameters& params) noexcept
  : p_impl_(new particle_filter::impl(number_of_particles, initial_observation, params)) {}

}  // namespace fast_plate_orbit_with_z_offset
