#pragma once

#include <pf/config/target_config.h>
#include <pf/filter/particle_reduction_state.h>
#include <pf/util/device_array.h>
#include <plate_orbit_v3/linear_state.h>
#include <plate_orbit_v3/observation.h>
#include <plate_orbit_v3/observed_plate.h>
#include <plate_orbit_v3/observed_plate_orbit.h>
#include <plate_orbit_v3/observed_plate_orbit_builder.h>
#include <plate_orbit_v3/particle_filter_configuration_parameters.h>
#include <plate_orbit_v3/predicted_plate.h>
#include <plate_orbit_v3/prediction.h>
#include <thrust/random.h>
#include <util/random_variable_sampler.h>

#include <Eigen/Dense>
#include <cmath>

namespace plate_orbit_v3 {

namespace helper {

PF_TARGET_ONLY_ATTRS [[nodiscard]] inline float log_sigmoid(const float& x) noexcept { return -logf(1.0f + expf(-x)); }

// Ornstein-Uhlenbeck coefficients for a mean reverting scalar with the given
// stationary variance and time constant, discretized exactly over dt.
struct ou_coefficients {
  float retention;
  float process_variance;

  PF_TARGET_ATTRS [[nodiscard]] static inline ou_coefficients
  from(const float& dt, const float& time_constant, const float& stationary_variance) noexcept {
    const float retention = expf(-dt / thrust::max(1e-6f, time_constant));
    return ou_coefficients{retention, stationary_variance * (1.0f - retention * retention)};
  }
};

}  // namespace helper

// Pairwise reduction producing a moment matched Gaussian over the particle
// population. Unlike v2 this has to align covariances as well as means, which
// the height (common, offset) basis makes trivial: the pi/2 relabeling is a
// signed diagonal similarity transform.
struct most_likely_particle_reduction_impl {
  using state_type = pf::filter::particle_reduction_state<prediction>;
  static constexpr float half_pi = static_cast<float>(M_PI_2);

  PF_TARGET_ONLY_ATTRS [[nodiscard]] inline state_type operator()(const state_type& a, const state_type& b) const noexcept {
    if (a.count() == 0u) { return b; }
    if (b.count() == 0u) { return a; }

    const prediction a_particle = a.most_likely_particle();
    const prediction b_particle = b.most_likely_particle();

    const float target = b_particle.orientation();

    // Candidate branches of the plate relabeling symmetry, expressed as a
    // number of quarter turns applied to a. An odd number of quarter turns
    // negates the height offset mode; a half turn is the identity.
    float best_orientation = a_particle.orientation();
    int best_quarter_turns = 0;
    float best_distance = fabsf(target - a_particle.orientation());

    for (int quarter_turns = -2; quarter_turns <= 2; ++quarter_turns) {
      const float candidate = a_particle.orientation() + static_cast<float>(quarter_turns) * half_pi;
      const float distance = fabsf(target - candidate);
      if (distance < best_distance) {
        best_distance = distance;
        best_orientation = candidate;
        best_quarter_turns = quarter_turns;
      }
    }

    linear_state a_linear = a_particle.linear();
    if (best_quarter_turns % 2 != 0) { a_linear.apply_quarter_turn_relabeling(); }

    const float alpha = static_cast<float>(b.count()) / static_cast<float>(a.count() + b.count());
    const float c_alpha = 1.0f - alpha;

    const linear_state::vector_type mean = c_alpha * a_linear.mean() + alpha * b_particle.linear().mean();

    const linear_state::vector_type a_delta = a_linear.mean() - mean;
    const linear_state::vector_type b_delta = b_particle.linear().mean() - mean;

    // Moment matching over the mixture: E[xx^T] - E[x]E[x]^T, so each
    // component contributes its own covariance plus its displacement.
    const linear_state::matrix_type covariance =
        c_alpha * (a_linear.covariance() + a_delta * a_delta.transpose()) +
        alpha * (b_particle.linear().covariance() + b_delta * b_delta.transpose());

    const float orientation = c_alpha * best_orientation + alpha * b_particle.orientation();

    return state_type{prediction(orientation, linear_state(mean, covariance)), a.count() + b.count()};
  }
};

class particle_filter_configuration {
 private:
  particle_filter_configuration_parameters params_;

  struct plate_measurement {
    Eigen::Matrix<float, 3, linear_state::dimension> jacobian;
    Eigen::Vector3f observed;
    Eigen::Vector3f noise_diagonal;
  };

  // Measurement Jacobian for plate k. Conditional on the orientation this is
  // exact, not a linearization: cos(angle) and sin(angle) are known constants
  // once the particle has committed to a hypothesis.
  //
  //   x = c_x + r * cos(angle)
  //   y = c_y + r * sin(angle)
  //   z = z_c + s * z_o
  PF_TARGET_ATTRS [[nodiscard]] static Eigen::Matrix<float, 3, linear_state::dimension>
  plate_jacobian(const float& angle, const float& sign) noexcept {
    Eigen::Matrix<float, 3, linear_state::dimension> jacobian =
        Eigen::Matrix<float, 3, linear_state::dimension>::Zero();

    const float cos_angle = cosf(angle);
    const float sin_angle = sinf(angle);

    jacobian(0, linear_state::index_center_x) = 1.0f;
    jacobian(0, linear_state::index_radius) = cos_angle;

    jacobian(1, linear_state::index_center_y) = 1.0f;
    jacobian(1, linear_state::index_radius) = sin_angle;

    jacobian(2, linear_state::index_z_common) = 1.0f;
    jacobian(2, linear_state::index_z_offset) = sign;

    return jacobian;
  }

  // One Kalman update against a single plate, returning the marginal log
  // likelihood of the observation with the linear substate integrated out.
  //
  // This is where Rao-Blackwellization earns its keep. The particle is not
  // scored against its current best guess at the center; it is scored against
  // every center position weighted by plausibility, which is what the
  // H * P * H^T term in the innovation covariance accomplishes. A particle is
  // only penalized if its orientation cannot explain the data under any
  // center at all.
  PF_TARGET_ONLY_ATTRS [[nodiscard]] static float
  apply_plate_update(const plate_measurement& measurement, linear_state& state) noexcept {
    const auto& jacobian = measurement.jacobian;

    const Eigen::Vector3f innovation = measurement.observed - jacobian * state.mean_;

    Eigen::Matrix3f innovation_covariance = jacobian * state.covariance_ * jacobian.transpose();
    innovation_covariance.diagonal() += measurement.noise_diagonal;

    const Eigen::Matrix3f inverse_innovation_covariance = innovation_covariance.inverse();
    const float determinant = innovation_covariance.determinant();

    const Eigen::Matrix<float, linear_state::dimension, 3> gain =
        state.covariance_ * jacobian.transpose() * inverse_innovation_covariance;

    state.mean_ += gain * innovation;

    // Joseph form. The straightforward (I - K * H) * P update loses symmetry
    // and positive definiteness quickly in single precision here, because the
    // radius process variance is four orders of magnitude below the center
    // process variance.
    const linear_state::matrix_type factor =
        linear_state::matrix_type::Identity() - gain * jacobian;

    state.covariance_ = factor * state.covariance_ * factor.transpose() +
                        gain * measurement.noise_diagonal.asDiagonal() * gain.transpose();
    state.symmetrize();

    // The normalizing term is mandatory here, unlike in v2. The innovation
    // covariance now differs between particles, so without log|S| a particle
    // whose orientation leaves the plate position poorly constrained would
    // score better than a confident one rather than worse.
    return -0.5f * (innovation.dot(inverse_innovation_covariance * innovation) +
                    logf(thrust::max(1e-30f, determinant)));
  }

 public:
  using observation_type = observation;
  using prediction_type = prediction;
  using sampler_type = util::default_rv_sampler;

  [[nodiscard]] most_likely_particle_reduction_impl most_likely_particle_reduction() const noexcept {
    return most_likely_particle_reduction_impl{};
  }

  // Process step. Draws the new orientation, then conditions the linear
  // substate on that draw, then propagates it.
  PF_TARGET_ONLY_ATTRS void apply_process(
      const float& time_offset_seconds,
      util::default_rv_sampler& sampler,
      prediction& state) const noexcept {
    const float dt = thrust::max(1e-6f, time_offset_seconds);
    linear_state& linear = state.linear();

    const float acceleration_variance = params_.orientation_velocity_process_variance;

    // Exact discretization of continuous white noise acceleration on
    // (orientation, orientation velocity). v2 reproduced this same covariance
    // by construction inside update_state; here it is written down directly.
    const float q_angle = acceleration_variance * dt * dt * dt / 3.0f;
    const float q_cross = acceleration_variance * dt * dt / 2.0f;
    const float q_velocity = acceleration_variance * dt;

    // Step 1: sample the orientation increment from its predictive
    // distribution, marginalizing over the uncertain angular velocity.
    const float predicted_increment = dt * linear.mean_[linear_state::index_orientation_velocity];
    const float increment_variance =
        dt * dt * linear.covariance_(linear_state::index_orientation_velocity,
                                     linear_state::index_orientation_velocity) +
        q_angle;

    const float increment = predicted_increment + sampler.normal_sample(increment_variance);

    // Step 2: the realized increment is itself a noisy measurement of the
    // angular velocity, so condition the linear substate on it. This is what
    // lets v3 keep the angular velocity on the analytic side of the partition
    // rather than sampling it as v2 does.
    {
      const linear_state::vector_type gain =
          (dt / increment_variance) * linear.covariance_.col(linear_state::index_orientation_velocity);

      linear.mean_ += gain * (increment - predicted_increment);
      linear.covariance_ -= increment_variance * gain * gain.transpose();
      linear.symmetrize();
    }

    // Step 3: propagate. The orientation and angular velocity process noises
    // are correlated, and the increment drawn above is already known, so
    // decompose w_velocity = J * w_angle + v with v independent of w_angle:
    //
    //   J = q_cross / q_angle = 3 / (2 dt)
    //
    // Substituting w_angle = increment - dt * omega gives an exact propagation
    // with the correlation folded into the transition:
    //
    //   omega' = (1 - J dt) omega + J * increment + v
    //          = -0.5 omega + (3 / (2 dt)) increment + v
    //
    // Sanity check: a noise free increment of dt * omega returns omega
    // unchanged, as it must.
    const float coupling = q_cross / q_angle;

    // The transition is a diagonal plus exactly two off diagonal entries, both
    // equal to dt, and both sitting in rows whose diagonal entry is one. That
    // means it factors exactly as
    //
    //   T = D * S,   S = I + dt * (e_cx e_vx^T + e_cy e_vy^T)
    //
    // so the propagation T P T^T = D (S P S^T) D is a pair of row/column
    // shears followed by a diagonal scaling, rather than two dense 9x9
    // products that exist to move eleven nonzeros around.
    linear_state::vector_type retention = linear_state::vector_type::Ones();
    linear_state::vector_type offset = linear_state::vector_type::Zero();
    linear_state::vector_type process_variance = linear_state::vector_type::Zero();

    retention[linear_state::index_orientation_velocity] = 1.0f - coupling * dt;
    offset[linear_state::index_orientation_velocity] = coupling * increment;
    process_variance[linear_state::index_orientation_velocity] = q_velocity - (q_cross * q_cross) / q_angle;

    const auto radius = helper::ou_coefficients::from(
        dt, params_.radius_reversion_time_constant, params_.radius_prior_variance_one_plate);
    retention[linear_state::index_radius] = radius.retention;
    offset[linear_state::index_radius] = (1.0f - radius.retention) * params_.radius_prior;
    process_variance[linear_state::index_radius] = params_.radius_process_variance * dt;

    process_variance[linear_state::index_z_common] = params_.z_common_process_variance * dt;

    const auto z_offset = helper::ou_coefficients::from(
        dt, params_.z_offset_reversion_time_constant, params_.z_offset_stationary_variance);
    retention[linear_state::index_z_offset] = z_offset.retention;
    process_variance[linear_state::index_z_offset] = z_offset.process_variance;

    // S, then D. Rows before columns, so the column pass sees S * P.
    linear.mean_[linear_state::index_center_x] += dt * linear.mean_[linear_state::index_center_velocity_x];
    linear.mean_[linear_state::index_center_y] += dt * linear.mean_[linear_state::index_center_velocity_y];
    linear.mean_ = retention.cwiseProduct(linear.mean_) + offset;

    linear.covariance_.row(linear_state::index_center_x) +=
        dt * linear.covariance_.row(linear_state::index_center_velocity_x);
    linear.covariance_.row(linear_state::index_center_y) +=
        dt * linear.covariance_.row(linear_state::index_center_velocity_y);

    linear.covariance_.col(linear_state::index_center_x) +=
        dt * linear.covariance_.col(linear_state::index_center_velocity_x);
    linear.covariance_.col(linear_state::index_center_y) +=
        dt * linear.covariance_.col(linear_state::index_center_velocity_y);

    linear.covariance_ = retention.asDiagonal() * linear.covariance_ * retention.asDiagonal();

    linear.covariance_.diagonal() += process_variance;

    // Center position and velocity share an acceleration, so their block
    // carries the usual off diagonal correlation.
    for (int axis = 0; axis < 2; ++axis) {
      const int position = linear_state::index_center_x + axis;
      const int velocity = linear_state::index_center_velocity_x + axis;
      const float intensity = params_.center_velocity_process_diagonal_covariance[axis];

      linear.covariance_(position, position) += intensity * dt * dt * dt / 3.0f;
      linear.covariance_(position, velocity) += intensity * dt * dt / 2.0f;
      linear.covariance_(velocity, position) += intensity * dt * dt / 2.0f;
      linear.covariance_(velocity, velocity) += intensity * dt;
    }

    linear.symmetrize();
    state.set_orientation(state.orientation() + increment);
  }

  // Measurement step. Returns the marginal log likelihood used as the particle
  // weight, and updates the particle's Gaussian in place.
  //
  // Note the mutable prediction reference: the concept declares the prediction
  // parameter as a non-const lvalue and particle_filter.h passes a non-const
  // reference, so this still satisfies particle_filter_configuration without
  // any change to the upstream library.
  PF_TARGET_ONLY_ATTRS [[nodiscard]] float conditional_log_likelihood(
      const util::default_rv_sampler& sampler,
      const observation& state,
      prediction& given) const noexcept {
    (void)sampler;

    auto plates = given.predicted_plates();
    auto parameters = given.plate_parameters_array();

    // Sort plate indices by distance to the observer, keeping the index so the
    // measurement Jacobian can be built with the correct angle and parity.
    // v2 discarded the index because it only ever needed the position.
    int order[4] = {0, 1, 2, 3};
    float distance[4];
    for (int k = 0; k < 4; ++k) { distance[k] = (state.observer_position() - plates[k].position()).squaredNorm(); }

    for (int i = 0; i < 4; ++i) {
      int best = i;
      for (int j = i + 1; j < 4; ++j) {
        if (distance[order[j]] < distance[order[best]]) { best = j; }
      }
      const int swap = order[i];
      order[i] = order[best];
      order[best] = swap;
    }

    const Eigen::Vector2f observer_delta = state.observer_position().head<2>() - given.center();

    float visibility_logit[4];
    for (int i = 0; i < 4; ++i) {
      const Eigen::Vector2f plate_delta = plates[order[i]].position().head<2>() - given.center();
      const float similarity = observer_delta.dot(plate_delta) /
                               thrust::max(1e-9f, observer_delta.norm() * plate_delta.norm());
      visibility_logit[i] = params_.visibility_logit_coefficient * similarity;
    }

    auto measurement_for = [&](const int& sorted_position, const observed_plate& observed) {
      const auto& parameter = parameters[order[sorted_position]];
      return plate_measurement{
          plate_jacobian(parameter.angle, parameter.sign),
          observed.position(),
          observed.position_diagonal_covariance(),
      };
    };

    if (!state.plate_two().has_value()) {
      const float log_visibility = helper::log_sigmoid(visibility_logit[0]) + helper::log_sigmoid(-visibility_logit[1]) +
                                   helper::log_sigmoid(-visibility_logit[2]) + helper::log_sigmoid(-visibility_logit[3]);

      return log_visibility + apply_plate_update(measurement_for(0, state.plate_one()), given.linear());
    }

    const float log_visibility = helper::log_sigmoid(visibility_logit[0]) + helper::log_sigmoid(visibility_logit[1]) +
                                 helper::log_sigmoid(-visibility_logit[2]) + helper::log_sigmoid(-visibility_logit[3]);

    // Assignment is chosen on point estimate residuals, as in v2. Once the
    // orientation is fixed by the particle the two nearest plates are far
    // apart relative to the observation noise, so this is effectively always
    // the same choice the full marginal comparison would make, and it avoids
    // running the update twice on a copy of a nine by nine covariance.
    const float direct = (state.plate_one().position() - plates[order[0]].position()).squaredNorm() +
                         (state.plate_two()->position() - plates[order[1]].position()).squaredNorm();
    const float swapped = (state.plate_one().position() - plates[order[1]].position()).squaredNorm() +
                          (state.plate_two()->position() - plates[order[0]].position()).squaredNorm();

    const int first = (direct <= swapped) ? 0 : 1;
    const int second = 1 - first;

    // Two sequential three dimensional updates are exactly equivalent to one
    // joint six dimensional update because the plate observation noises are
    // independent, and the accumulated log likelihoods telescope into the
    // joint marginal by the chain rule. This keeps every matrix inverse at
    // three by three, which has a closed form and is safe on device.
    float log_likelihood = apply_plate_update(measurement_for(first, state.plate_one()), given.linear());
    log_likelihood += apply_plate_update(measurement_for(second, *state.plate_two()), given.linear());

    return log_visibility + log_likelihood;
  }

  // Initialization. Every particle draws an orientation, then places the
  // center consistently with that draw so the hypothesis is self consistent
  // from the first frame.
  PF_TARGET_ONLY_ATTRS [[nodiscard]] prediction sample_from(
      util::default_rv_sampler& sampler,
      const observation& state) const noexcept {
    const observed_plate_orbit_builder builder(params_.radius_prior, state.observer_position());

    const observed_plate_orbit orbit = state.plate_two().has_value() ?
                                           builder.from_two_plates(state.plate_one(), *state.plate_two()) :
                                           builder.from_one_plate(state.plate_one());

    const float radius_variance = state.plate_two().has_value() ? params_.radius_prior_variance_two_plates :
                                                                  params_.radius_prior_variance_one_plate;

    // orbit.orientation is the bearing of plate one as seen from the estimated
    // center, so plate one is plate zero by construction. Perturbing that
    // bearing and re-deriving the center from it keeps the pair consistent;
    // wrapping the angle afterward is harmless because it only ever shifts by
    // a multiple of pi, which maps onto a plate with identical radius and
    // height.
    const float bearing = orbit.orientation + sampler.normal_sample(params_.orientation_prior_variance);

    const Eigen::Vector2f center =
        state.plate_one().position().head<2>() - orbit.radius * Eigen::Vector2f(cosf(bearing), sinf(bearing));

    linear_state::vector_type mean = linear_state::vector_type::Zero();
    mean[linear_state::index_center_x] = center[0];
    mean[linear_state::index_center_y] = center[1];
    mean[linear_state::index_radius] = orbit.radius;
    mean[linear_state::index_z_common] = state.plate_one().position()[2];

    const Eigen::Vector3f observation_variance = state.plate_one().position_diagonal_covariance();

    linear_state::vector_type variance = linear_state::vector_type::Zero();
    variance[linear_state::index_orientation_velocity] = params_.orientation_velocity_prior_variance;
    variance[linear_state::index_center_x] = observation_variance[0];
    variance[linear_state::index_center_y] = observation_variance[1];
    variance[linear_state::index_center_velocity_x] = params_.center_velocity_prior_diagonal_covariance[0];
    variance[linear_state::index_center_velocity_y] = params_.center_velocity_prior_diagonal_covariance[1];
    variance[linear_state::index_radius] = radius_variance;
    variance[linear_state::index_z_common] = observation_variance[2];
    variance[linear_state::index_z_offset] = params_.z_offset_stationary_variance;

    linear_state::matrix_type covariance = linear_state::matrix_type::Zero();
    covariance.diagonal() = variance;

    return prediction(bearing, linear_state(mean, covariance));
  }

  [[nodiscard]] const particle_filter_configuration_parameters& params() const noexcept { return params_; }

  particle_filter_configuration(const particle_filter_configuration_parameters& params) noexcept : params_{params} {}
};

}  // namespace plate_orbit_v3
