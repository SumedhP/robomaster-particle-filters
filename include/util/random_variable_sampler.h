#pragma once

#include <thrust/random.h>

#if defined(PF_TARGET_CUDA)
#include <curand_kernel.h>
#endif

#include <Eigen/Dense>
#include <cassert>
#include <cstdint>
#include <random>

namespace util {

template <typename T, typename RandomNumberGenerator>
class random_variable_sampler {
 private:
  RandomNumberGenerator random_number_generator_{};
  thrust::normal_distribution<T> standard_normal_{};

 public:
  using floating_point_type = T;
  using random_number_generator_type = RandomNumberGenerator;

  PF_TARGET_ATTRS [[nodiscard]] RandomNumberGenerator& random_number_generator() noexcept {
    return random_number_generator_;
  }

  PF_TARGET_ATTRS void seed(const typename RandomNumberGenerator::result_type& seed) noexcept {
    random_number_generator_.seed(seed);
  }

  PF_TARGET_ATTRS void seed_from_index(const std::uint64_t& index) noexcept {
    thrust::default_random_engine generator{};
    generator.discard(index);
    random_number_generator_.seed(generator());
  }

  PF_TARGET_ATTRS [[nodiscard]] T normal_sample(const T& variance) noexcept {
    return sqrt(variance) * standard_normal_(random_number_generator_);
  }

  PF_TARGET_ATTRS [[nodiscard]] T normal_sample_from_standard_deviation(const T& standard_deviation) noexcept {
    return standard_deviation * standard_normal_(random_number_generator_);
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] Eigen::Matrix<T, N, 1> normal_sample(
      const Eigen::Matrix<T, N, 1>& diagonal_covariance) noexcept {
    return diagonal_covariance.cwiseSqrt().cwiseProduct(
        Eigen::Matrix<T, N, 1>{}.unaryExpr([this](auto) { return standard_normal_(random_number_generator_); }));
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] Eigen::Matrix<T, N, 1> normal_sample_from_standard_deviation(
      const Eigen::Matrix<T, N, 1>& diagonal_standard_deviation) noexcept {
    return diagonal_standard_deviation.cwiseProduct(
        Eigen::Matrix<T, N, 1>{}.unaryExpr([this](auto) { return standard_normal_(random_number_generator_); }));
  }

  PF_TARGET_ATTRS [[nodiscard]] T unnormalized_normal_log_density(const T& variance, const T& x) const noexcept {
    return -static_cast<T>(0.5) * (x * x) / variance;
  }

  PF_TARGET_ATTRS [[nodiscard]] T unnormalized_normal_log_density_from_precision(
      const T& inverse_variance,
      const T& x) const noexcept {
    return -static_cast<T>(0.5) * (x * x) * inverse_variance;
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] T unnormalized_normal_log_density(
      const Eigen::Matrix<T, N, 1>& diagonal_covariance,
      const Eigen::Matrix<T, N, 1>& x) const noexcept {
    return -static_cast<T>(0.5) * x.cwiseProduct(diagonal_covariance.cwiseInverse()).dot(x);
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] T unnormalized_normal_log_density_from_diagonal_precision(
      const Eigen::Matrix<T, N, 1>& diagonal_precision,
      const Eigen::Matrix<T, N, 1>& x) const noexcept {
    return -static_cast<T>(0.5) * x.cwiseProduct(diagonal_precision).dot(x);
  }
};

using default_rv_sampler = random_variable_sampler<float, thrust::random::default_random_engine>;

class philox_rv_sampler {
 private:
#if defined(PF_TARGET_CUDA)
  curandStatePhilox4_32_10_t random_number_generator_{};
  float4 standard_normal_cache_{};
  int standard_normal_cache_position_{4};
#else
  thrust::random::default_random_engine random_number_generator_{};
  thrust::normal_distribution<float> standard_normal_{};
#endif

  PF_TARGET_ATTRS [[nodiscard]] float standard_normal_sample_() noexcept {
#if defined(PF_TARGET_CUDA)
  #if defined(__CUDA_ARCH__)
    if (standard_normal_cache_position_ >= 4) {
      standard_normal_cache_ = curand_normal4(&random_number_generator_);
      standard_normal_cache_position_ = 0;
    }

    switch (standard_normal_cache_position_++) {
      case 0:
        return standard_normal_cache_.x;
      case 1:
        return standard_normal_cache_.y;
      case 2:
        return standard_normal_cache_.z;
      default:
        return standard_normal_cache_.w;
    }
  #else
    // Host code path in CUDA translation units is unused for sampling.
    return 0.0f;
  #endif
#else
    return standard_normal_(random_number_generator_);
#endif
  }

 public:
  using floating_point_type = float;
  using random_number_generator_type = std::uint64_t;

  PF_TARGET_ATTRS void seed(const std::uint64_t& seed) noexcept {
#if defined(PF_TARGET_CUDA)
  #if defined(__CUDA_ARCH__)
    curand_init(static_cast<unsigned long long>(seed), 0ULL, 0ULL, &random_number_generator_);
    standard_normal_cache_position_ = 4;
  #else
    (void)seed;
  #endif
#else
    random_number_generator_.seed(static_cast<thrust::random::default_random_engine::result_type>(seed));
#endif
  }

  PF_TARGET_ATTRS void seed_from_index(const std::uint64_t& index) noexcept {
    static constexpr std::uint64_t base_seed = 0x9E3779B97F4A7C15ULL;
    seed(base_seed + index);
  }

  PF_TARGET_ATTRS [[nodiscard]] float normal_sample(const float& variance) noexcept {
    return sqrtf(variance) * standard_normal_sample_();
  }

  PF_TARGET_ATTRS [[nodiscard]] float normal_sample_from_standard_deviation(const float& standard_deviation) noexcept {
    return standard_deviation * standard_normal_sample_();
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] Eigen::Matrix<float, N, 1> normal_sample(
      const Eigen::Matrix<float, N, 1>& diagonal_covariance) noexcept {
    return diagonal_covariance.cwiseSqrt().cwiseProduct(
        Eigen::Matrix<float, N, 1>{}.unaryExpr([this](auto) { return standard_normal_sample_(); }));
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] Eigen::Matrix<float, N, 1> normal_sample_from_standard_deviation(
      const Eigen::Matrix<float, N, 1>& diagonal_standard_deviation) noexcept {
    return diagonal_standard_deviation.cwiseProduct(
        Eigen::Matrix<float, N, 1>{}.unaryExpr([this](auto) { return standard_normal_sample_(); }));
  }

  PF_TARGET_ATTRS [[nodiscard]] float unnormalized_normal_log_density(const float& variance, const float& x) const noexcept {
    return -0.5f * (x * x) / variance;
  }

  PF_TARGET_ATTRS [[nodiscard]] float unnormalized_normal_log_density_from_precision(
      const float& inverse_variance,
      const float& x) const noexcept {
    return -0.5f * (x * x) * inverse_variance;
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] float unnormalized_normal_log_density(
      const Eigen::Matrix<float, N, 1>& diagonal_covariance,
      const Eigen::Matrix<float, N, 1>& x) const noexcept {
    return -0.5f * x.cwiseProduct(diagonal_covariance.cwiseInverse()).dot(x);
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] float unnormalized_normal_log_density_from_diagonal_precision(
      const Eigen::Matrix<float, N, 1>& diagonal_precision,
      const Eigen::Matrix<float, N, 1>& x) const noexcept {
    return -0.5f * x.cwiseProduct(diagonal_precision).dot(x);
  }
};

#if defined(PF_TARGET_CUDA)
using z_offset_rv_sampler = philox_rv_sampler;
#else
using z_offset_rv_sampler = default_rv_sampler;
#endif

}  // namespace util
