#pragma once

#include <thrust/random.h>

#include <Eigen/Dense>
#include <cassert>
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

  PF_TARGET_ATTRS [[nodiscard]] T normal_sample(const T& variance) noexcept {
    return sqrt(variance) * standard_normal_(random_number_generator_);
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] Eigen::Matrix<T, N, 1> normal_sample(
      const Eigen::Matrix<T, N, 1>& diagonal_covariance) noexcept {
    Eigen::Matrix<T, N, 1> result{};
    for (int i = 0; i < N; ++i) {
      result[i] = sqrt(diagonal_covariance[i]) * standard_normal_(random_number_generator_);
    }

    return result;
  }

  PF_TARGET_ATTRS [[nodiscard]] T unnormalized_normal_log_density(const T& variance, const T& x) const noexcept {
    return -static_cast<T>(0.5) * (x * x) / variance;
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] T unnormalized_normal_log_density(
      const Eigen::Matrix<T, N, 1>& diagonal_covariance,
      const Eigen::Matrix<T, N, 1>& x) const noexcept {
    const Eigen::Matrix<T, N, 1> inverse_diagonal_covariance = diagonal_covariance.cwiseInverse();
    return unnormalized_normal_log_density_from_inverse_covariance(inverse_diagonal_covariance, x);
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] T unnormalized_normal_log_density_from_inverse_covariance(
      const Eigen::Matrix<T, N, 1>& inverse_diagonal_covariance,
      const Eigen::Matrix<T, N, 1>& x) const noexcept {
    return -static_cast<T>(0.5) * x.cwiseProduct(inverse_diagonal_covariance).dot(x);
  }
};

using default_rv_sampler = random_variable_sampler<float, thrust::random::default_random_engine>;

}  // namespace util
