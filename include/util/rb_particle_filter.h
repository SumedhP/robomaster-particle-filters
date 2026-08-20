#pragma once

#include <pf/config/target_config.h>
#include <pf/filter/particle_reduction_state.h>
#include <pf/filter/systematic_resampler.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>
#include <thrust/mr/disjoint_tls_pool.h>
#include <thrust/mr/new.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>

#include <cstdint>
#include <cuda/std/tuple>
#include <limits>
#include <utility>

namespace util {

namespace helper {

using disjoint_pool_resource_type =
    thrust::mr::disjoint_unsynchronized_pool_resource<thrust::device_memory_resource, thrust::mr::new_delete_resource>;

using caching_allocator_type = thrust::mr::allocator<char, helper::disjoint_pool_resource_type>;

inline caching_allocator_type thread_local_caching_allocator() {
  return {&thrust::mr::tls_disjoint_pool(
      thrust::mr::get_global_resource<thrust::device_memory_resource>(),
      thrust::mr::get_global_resource<thrust::mr::new_delete_resource>())};
}

}  // namespace helper

// A Rao-Blackwellized variant of pf::filter::particle_filter.
//
// Two behavioural differences from the upstream filter, both required once
// particles carry a Gaussian rather than a point:
//
//   1. Log weights accumulate across steps instead of being overwritten each
//      step. The upstream filter can overwrite because it resamples every
//      step, which makes the prior weights uniform by construction.
//
//   2. Resampling is triggered by effective sample size rather than run
//      unconditionally. Resampling copies particles wholesale, and two copies
//      of a particle carry identical Gaussians, so unconditional resampling
//      destroys diversity in the only dimension actually being sampled while
//      recovering nothing. With a well matched model the filter here typically
//      resamples every few frames rather than every frame.
//
// The particle weights also feed the reduction, since between resampling steps
// the population is no longer uniformly weighted.
template <typename ParticleFilterConfiguration>
class rb_particle_filter {
 private:
  using observation_type = typename ParticleFilterConfiguration::observation_type;
  using prediction_type = typename ParticleFilterConfiguration::prediction_type;
  using sampler_type = typename ParticleFilterConfiguration::sampler_type;

  helper::caching_allocator_type caching_allocator_;
  ParticleFilterConfiguration config_;

  std::size_t number_of_particles_;
  float resample_threshold_;

  prediction_type most_likely_particle_state_;
  pf::filter::systematic_resampler<prediction_type, std::uint32_t> resampler_;

  pf::target_config::vector<sampler_type> sampler_states_;
  pf::target_config::vector<float> log_particle_weights_;
  pf::target_config::vector<prediction_type> particle_states_;

  float last_effective_sample_size_;
  bool last_step_resampled_;

 public:
  [[nodiscard]] prediction_type extrapolate_state(const float& time_offset_seconds) const noexcept {
    return most_likely_particle_state_.extrapolate_state(time_offset_seconds);
  }

  [[nodiscard]] const float& effective_sample_size() const noexcept { return last_effective_sample_size_; }
  [[nodiscard]] const bool& resampled() const noexcept { return last_step_resampled_; }

  void update_state_sans_observation(const float& time_offset_seconds) noexcept {
    thrust::for_each(
        pf::target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(sampler_states_.begin(), particle_states_.begin()),
        thrust::make_zip_iterator(sampler_states_.end(), particle_states_.end()),
        [config = config_, time_offset_seconds] PF_TARGET_ONLY_ATTRS(
            cuda::std::tuple<sampler_type&, prediction_type&> tuple) {
          config.apply_process(time_offset_seconds, cuda::std::get<0>(tuple), cuda::std::get<1>(tuple));
        });

    last_step_resampled_ = false;
    reduce_most_likely_particle_();
  }

  void update_state_with_observation(const float& time_offset_seconds, const observation_type& observation_state) noexcept {
    thrust::for_each(
        pf::target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(sampler_states_.begin(), log_particle_weights_.begin(), particle_states_.begin()),
        thrust::make_zip_iterator(sampler_states_.end(), log_particle_weights_.end(), particle_states_.end()),
        [config = config_, time_offset_seconds, observation_state] PF_TARGET_ONLY_ATTRS(
            cuda::std::tuple<sampler_type&, float&, prediction_type&> tuple) {
          sampler_type& sampler_state = cuda::std::get<0>(tuple);
          float& log_particle_weight = cuda::std::get<1>(tuple);
          prediction_type& particle_state = cuda::std::get<2>(tuple);

          config.apply_process(time_offset_seconds, sampler_state, particle_state);
          log_particle_weight += config.conditional_log_likelihood(sampler_state, observation_state, particle_state);
        });

    last_effective_sample_size_ = effective_sample_size_();
    last_step_resampled_ = last_effective_sample_size_ < resample_threshold_;

    if (last_step_resampled_) {
      resampler_.resample(pf::target_config::policy(caching_allocator_), log_particle_weights_, particle_states_);
      thrust::fill(pf::target_config::policy(caching_allocator_), log_particle_weights_.begin(),
                   log_particle_weights_.end(), 0.0f);
    }

    reduce_most_likely_particle_();
  }

  void reinitialize(const observation_type& new_observation) noexcept {
    thrust::transform(
        pf::target_config::policy(caching_allocator_),
        sampler_states_.begin(),
        sampler_states_.end(),
        particle_states_.begin(),
        [config = config_, new_observation] PF_TARGET_ONLY_ATTRS(sampler_type& sampler_state) {
          return config.sample_from(sampler_state, new_observation);
        });

    thrust::fill(pf::target_config::policy(caching_allocator_), log_particle_weights_.begin(),
                 log_particle_weights_.end(), 0.0f);

    reduce_most_likely_particle_();
  }

 private:
  // Normalized effective sample size, computed with the usual max subtraction
  // so that accumulated log weights cannot underflow.
  [[nodiscard]] float effective_sample_size_() noexcept {
    const float maximum_log_weight = thrust::reduce(
        pf::target_config::policy(caching_allocator_),
        log_particle_weights_.cbegin(),
        log_particle_weights_.cend(),
        -std::numeric_limits<float>::infinity(),
        thrust::maximum<float>());

    const float sum = thrust::transform_reduce(
        pf::target_config::policy(caching_allocator_),
        log_particle_weights_.cbegin(),
        log_particle_weights_.cend(),
        [maximum_log_weight] PF_TARGET_ATTRS(const float& value) { return expf(value - maximum_log_weight); },
        0.0f,
        thrust::plus<float>());

    const float sum_of_squares = thrust::transform_reduce(
        pf::target_config::policy(caching_allocator_),
        log_particle_weights_.cbegin(),
        log_particle_weights_.cend(),
        [maximum_log_weight] PF_TARGET_ATTRS(const float& value) {
          const float weight = expf(value - maximum_log_weight);
          return weight * weight;
        },
        0.0f,
        thrust::plus<float>());

    if (!(sum_of_squares > 0.0f)) { return 0.0f; }
    return (sum * sum) / sum_of_squares;
  }

  void reduce_most_likely_particle_() noexcept {
    // The reduction is weighted by an integer count, so particles are
    // discretized into count buckets proportional to their normalized weight.
    // Immediately after a resample every particle lands in bucket one and this
    // degenerates to the uniform reduction the upstream filter performs.
    const float maximum_log_weight = thrust::reduce(
        pf::target_config::policy(caching_allocator_),
        log_particle_weights_.cbegin(),
        log_particle_weights_.cend(),
        -std::numeric_limits<float>::infinity(),
        thrust::maximum<float>());

    const float sum = thrust::transform_reduce(
        pf::target_config::policy(caching_allocator_),
        log_particle_weights_.cbegin(),
        log_particle_weights_.cend(),
        [maximum_log_weight] PF_TARGET_ATTRS(const float& value) { return expf(value - maximum_log_weight); },
        0.0f,
        thrust::plus<float>());

    const float scale = (sum > 0.0f) ? (static_cast<float>(number_of_particles_) / sum) : 1.0f;

    most_likely_particle_state_ =
        thrust::transform_reduce(
            pf::target_config::policy(caching_allocator_),
            thrust::make_zip_iterator(particle_states_.cbegin(), log_particle_weights_.cbegin()),
            thrust::make_zip_iterator(particle_states_.cend(), log_particle_weights_.cend()),
            [maximum_log_weight, scale] PF_TARGET_ATTRS(
                const cuda::std::tuple<const prediction_type&, const float&>& tuple) {
              const float weight = scale * expf(cuda::std::get<1>(tuple) - maximum_log_weight);
              const std::uint32_t count = static_cast<std::uint32_t>(thrust::max(0.0f, weight) + 0.5f);
              return pf::filter::particle_reduction_state<prediction_type>{cuda::std::get<0>(tuple), count};
            },
            pf::filter::particle_reduction_state<prediction_type>::zero(),
            config_.most_likely_particle_reduction())
            .most_likely_particle();
  }

  void initialize_internal_state_(const observation_type& initial_observation) noexcept {
    thrust::counting_iterator<std::size_t> index_sequence_begin(std::size_t{});

    thrust::for_each(
        pf::target_config::policy(caching_allocator_),
        thrust::make_zip_iterator(index_sequence_begin, sampler_states_.begin()),
        thrust::make_zip_iterator(index_sequence_begin + number_of_particles_, sampler_states_.end()),
        [] PF_TARGET_ATTRS(cuda::std::tuple<std::size_t, sampler_type&> tuple) {
          thrust::default_random_engine generator{};
          generator.discard(cuda::std::get<0>(tuple));
          cuda::std::get<1>(tuple).seed(generator());
        });

    reinitialize(initial_observation);
  }

 public:
  template <typename... Ts>
  rb_particle_filter(
      const std::size_t& number_of_particles,
      const float& resample_effective_sample_size_fraction,
      const observation_type& initial_observation,
      Ts&&... params) noexcept
      : caching_allocator_{helper::thread_local_caching_allocator()},
        config_(std::forward<Ts>(params)...),
        number_of_particles_{number_of_particles},
        resample_threshold_{resample_effective_sample_size_fraction * static_cast<float>(number_of_particles)},
        resampler_(number_of_particles),
        sampler_states_(number_of_particles),
        log_particle_weights_(number_of_particles, 0.0f),
        particle_states_(number_of_particles),
        last_effective_sample_size_{static_cast<float>(number_of_particles)},
        last_step_resampled_{false} {
    initialize_internal_state_(initial_observation);
  }
};

}  // namespace util
