#pragma once

#include <pf/config/target_config.h>
#include <pf/filter/particle_reduction_state.h>
#include <pf/util/device_array.h>
#include <fast_plate_orbit_with_z_offset/observation.h>
#include <fast_plate_orbit_with_z_offset/observed_plate.h>
#include <fast_plate_orbit_with_z_offset/observed_plate_orbit.h>
#include <fast_plate_orbit_with_z_offset/observed_plate_orbit_builder.h>
#include <fast_plate_orbit_with_z_offset/initialization_prior.h>
#include <fast_plate_orbit_with_z_offset/particle_filter_configuration_parameters.h>
#include <fast_plate_orbit_with_z_offset/predicted_plate.h>
#include <fast_plate_orbit_with_z_offset/prediction.h>
#include <thrust/random.h>
#include <util/random_variable_sampler.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace fast_plate_orbit_with_z_offset {

namespace helper {

PF_TARGET_ONLY_ATTRS [[nodiscard]] inline float log_sigmoid(const float& x) noexcept { return -logf(1.0f + expf(-x)); }

PF_TARGET_ONLY_ATTRS [[nodiscard]] inline float wrap_half_turn(const float& angle_radians) noexcept {
  const float pi = static_cast<float>(M_PI);
  const float pi_2 = static_cast<float>(M_PI_2);
  float value = fmodf(angle_radians + pi_2, pi);
  if (value < 0.0f) {
    value += pi;
  }
  return value - pi_2;
}

PF_TARGET_ONLY_ATTRS [[nodiscard]] inline float log_sum_exp(const float a, const float b) noexcept {
  const float m = fmaxf(a, b);
  return m + log1pf(expf(-fabsf(a - b)));
}

// ---------------------------------------------------------------------------
// Scalar plate-position helper
//
// Computes the 3-D position of one plate directly from raw scalars, without
// constructing a predicted_plate object.  Used by both the sorting network
// and the scalar log-density path so that no Eigen temporaries are allocated
// on the device.
// ---------------------------------------------------------------------------
PF_TARGET_ONLY_ATTRS inline void plate_position_scalars(
    const float cx, const float cy,
    const float radius, const float angle,
    const float z_coordinate,
    float& px, float& py, float& pz) noexcept {
  px = cx + radius * cosf(angle);
  py = cy + radius * sinf(angle);
  pz = z_coordinate;
}

// Squared distance from observer (ox,oy,oz) to plate position (px,py,pz).
PF_TARGET_ONLY_ATTRS [[nodiscard]] inline float sq_dist(
    const float ox, const float oy, const float oz,
    const float px, const float py, const float pz) noexcept {
  const float dx = ox - px;
  const float dy = oy - py;
  const float dz = oz - pz;
  return dx * dx + dy * dy + dz * dz;
}

// Unnormalised normal log-density for a 3-D diagonal-covariance Gaussian,
// computed as plain scalar arithmetic.  Equivalent to the Eigen path:
//   -0.5f * error.cwiseProduct(cov.cwiseInverse()).dot(error)
PF_TARGET_ONLY_ATTRS [[nodiscard]] inline float scalar_log_density_3(
    const float ex, const float ey, const float ez,
    const float vx, const float vy, const float vz) noexcept {
  return -0.5f * (ex * ex / vx + ey * ey / vy + ez * ez / vz);
}

// ---------------------------------------------------------------------------
// 4-element sorting network (Bose-Nelson optimal, 5 compare-swaps)
//
// Sorts four float keys and their associated uint8 index tags in-place.
// Completely branchless on GPU: the conditional swap compiles to a predicated
// move (SELP / FSEL) with no divergence.
// ---------------------------------------------------------------------------
PF_TARGET_ONLY_ATTRS inline void cs(float& ka, uint8_t& ia, float& kb, uint8_t& ib) noexcept {
  const bool swap = kb < ka;
  const float  tmp_k = swap ? ka : kb;  ka = swap ? kb : ka;  kb = tmp_k;
  const uint8_t tmp_i = swap ? ia : ib; ia = swap ? ib : ia; ib = tmp_i;
}

PF_TARGET_ONLY_ATTRS inline void sort4(
    float k0, float k1, float k2, float k3,
    uint8_t& i0, uint8_t& i1, uint8_t& i2, uint8_t& i3) noexcept {
  // Initialise index tags.
  i0 = 0; i1 = 1; i2 = 2; i3 = 3;
  // Bose-Nelson network for n=4 (5 comparators).
  cs(k0, i0, k1, i1);
  cs(k2, i2, k3, i3);
  cs(k0, i0, k2, i2);
  cs(k1, i1, k3, i3);
  cs(k1, i1, k2, i2);
}

}  // namespace helper

// =============================================================================
// Change 2 — Scalar SOA reduction
//
// Instead of storing a full `prediction` object inside particle_reduction_state,
// the reduction accumulates each scalar field as a weighted sum.  The blending
// weight is (count_b / total), exactly as before, but it is applied to raw
// floats rather than to Eigen vectors or prediction objects.
//
// The orientation branch-selection (choosing among the five pi/2-spaced
// candidates to minimise angular distance to b) still happens in the reduction
// operator, but operates on a single float rather than a device_array lookup.
//
// `most_likely_particle()` reconstructs a prediction object exactly once, at
// the point where the caller actually needs it, rather than maintaining a live
// prediction throughout the tree reduction.
// =============================================================================
struct scalar_reduction_state {
  // Each field is the running weighted mean of that scalar across the
  // particles seen so far.  `count` drives the blending weight.
  float radius;
  float z_coordinate_0;
  float z_coordinate_1;
  float orientation;
  float orientation_velocity;
  float center_x;
  float center_y;
  float center_vel_x;
  float center_vel_y;
  std::uint32_t count;

  PF_TARGET_ATTRS [[nodiscard]] static constexpr scalar_reduction_state zero() noexcept {
    return scalar_reduction_state{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, std::uint32_t{}};
  }

  PF_TARGET_ATTRS [[nodiscard]] static scalar_reduction_state from_particle(const prediction& p) noexcept {
    return scalar_reduction_state{
        p.radius(),
        p.z_coordinate_0(),
        p.z_coordinate_1(),
        p.orientation(),
        p.orientation_velocity(),
        p.center().x(),
        p.center().y(),
        p.center_velocity().x(),
        p.center_velocity().y(),
        std::uint32_t{1}};
  }

  // Reconstruct a prediction from the accumulated scalar means.
  PF_TARGET_ATTRS [[nodiscard]] prediction most_likely_particle() const noexcept {
    return prediction(
        radius,
        z_coordinate_0,
        z_coordinate_1,
        orientation,
        orientation_velocity,
        Eigen::Vector2f{center_x, center_y},
        Eigen::Vector2f{center_vel_x, center_vel_y});
  }
};

struct most_likely_particle_reduction_impl {
  using state_type = scalar_reduction_state;

  static constexpr float half_pi = M_PI_2;

  PF_TARGET_ATTRS [[nodiscard]] inline state_type
  operator()(const state_type& a, const state_type& b) const noexcept {
    if (a.count == 0) return b;
    if (b.count == 0) return a;

    // --- Orientation symmetry resolution -----------------------------------
    // The orbit has pi/2 symmetry.  The five candidates cover the full range
    // of equivalent orientations.  When the z-coordinates swap (±pi/2, ±pi),
    // the z_coordinate pair must be swapped to stay geometrically consistent.
    // We select the candidate whose orientation is closest to b's, then carry
    // the corresponding z-coordinates.

    struct candidate { float orientation; float z0; float z1; };
    const candidate candidates[5] = {
        {a.orientation + 0.0f * half_pi, a.z_coordinate_0, a.z_coordinate_1},
        {a.orientation + 1.0f * half_pi, a.z_coordinate_1, a.z_coordinate_0},
        {a.orientation - 1.0f * half_pi, a.z_coordinate_1, a.z_coordinate_0},
        {a.orientation + 2.0f * half_pi, a.z_coordinate_0, a.z_coordinate_1},
        {a.orientation - 2.0f * half_pi, a.z_coordinate_0, a.z_coordinate_1},
    };

    // Find closest candidate to b.orientation — pure scalar comparisons.
    int best = 0;
    float best_dist = fabsf(b.orientation - candidates[0].orientation);
    for (int i = 1; i < 5; ++i) {
      const float d = fabsf(b.orientation - candidates[i].orientation);
      if (d < best_dist) { best_dist = d; best = i; }
    }

    const float alpha   = static_cast<float>(b.count) / static_cast<float>(a.count + b.count);
    const float c_alpha = 1.0f - alpha;

    return scalar_reduction_state{
        c_alpha * a.radius              + alpha * b.radius,
        c_alpha * candidates[best].z0   + alpha * b.z_coordinate_0,
        c_alpha * candidates[best].z1   + alpha * b.z_coordinate_1,
        c_alpha * candidates[best].orientation + alpha * b.orientation,
        c_alpha * a.orientation_velocity + alpha * b.orientation_velocity,
        c_alpha * a.center_x            + alpha * b.center_x,
        c_alpha * a.center_y            + alpha * b.center_y,
        c_alpha * a.center_vel_x        + alpha * b.center_vel_x,
        c_alpha * a.center_vel_y        + alpha * b.center_vel_y,
        a.count + b.count,
    };
  }
};

// =============================================================================
// particle_filter_configuration
// =============================================================================
class particle_filter_configuration {
 private:
  particle_filter_configuration_parameters params_;

 public:
  using observation_type = observation;
  using prediction_type  = prediction;
  using sampler_type     = util::default_rv_sampler;
  using initialization_prior_type = initialization_prior;

  // The reduction type is now scalar_reduction_state rather than
  // particle_reduction_state<prediction>.  particle_filter.h calls
  //   config_.most_likely_particle_reduction()
  // and then ultimately calls .most_likely_particle() on the result;
  // scalar_reduction_state satisfies both.
  [[nodiscard]] most_likely_particle_reduction_impl most_likely_particle_reduction() const noexcept {
    return most_likely_particle_reduction_impl{};
  }

  // ===========================================================================
  // Change 3 — Fixed-size sorting network + scalar scoring
  //
  // Original path:
  //   1. given.predicted_plates()  — builds 4 predicted_plate{Vector3f, Vector3f}
  //   2. selection_sort_by(squaredNorm)  — O(n²) comparison loop on structs
  //   3. transformed(visibility logit)  — another pass over 4 structs
  //   4. unnormalized_normal_log_density(Vector3f error)  — Eigen dot product
  //
  // New path:
  //   1. Compute the 4 plate positions as scalar triples inline (no struct alloc)
  //   2. Compute 4 squared distances as scalars
  //   3. Sort the 4 distance keys + index tags with a 5-comparator Bose-Nelson
  //      network — zero divergence, fully unrolled by the compiler
  //   4. Compute visibility logits as scalar multiplications
  //   5. Compute log-density as plain (ex²/vx + ey²/vy + ez²/vz) arithmetic
  // ===========================================================================
  PF_TARGET_ONLY_ATTRS [[nodiscard]] float conditional_log_likelihood(
      const util::default_rv_sampler& sampler,
      const observation& state,
      const prediction& given) const noexcept {

    // ---- Plate positions (scalars only) ------------------------------------
    // Plates are at angles: θ, θ+π/2, θ+π, θ+3π/2
    // z-coordinates alternate: z0, z1, z0, z1
    const float cx = given.center().x();
    const float cy = given.center().y();
    const float r  = given.radius();
    const float th = given.orientation();

    float px[4], py[4], pz[4];
    helper::plate_position_scalars(cx, cy, r, th + 0.0f,         given.z_coordinate_0(), px[0], py[0], pz[0]);
    helper::plate_position_scalars(cx, cy, r, th + M_PI_2,       given.z_coordinate_1(), px[1], py[1], pz[1]);
    helper::plate_position_scalars(cx, cy, r, th + M_PI,         given.z_coordinate_0(), px[2], py[2], pz[2]);
    helper::plate_position_scalars(cx, cy, r, th + M_PI + M_PI_2,given.z_coordinate_1(), px[3], py[3], pz[3]);

    // ---- Sort by distance to observer (sorting network) -------------------
    const float ox = state.observer_position().x();
    const float oy = state.observer_position().y();
    const float oz = state.observer_position().z();

    float d0 = helper::sq_dist(ox, oy, oz, px[0], py[0], pz[0]);
    float d1 = helper::sq_dist(ox, oy, oz, px[1], py[1], pz[1]);
    float d2 = helper::sq_dist(ox, oy, oz, px[2], py[2], pz[2]);
    float d3 = helper::sq_dist(ox, oy, oz, px[3], py[3], pz[3]);

    uint8_t i0, i1, i2, i3;
    helper::sort4(d0, d1, d2, d3, i0, i1, i2, i3);
    // i0 now holds the index of the closest plate, i1 the second-closest, etc.

    // ---- Visibility logits (scalar) ----------------------------------------
    // observer_delta and plate_delta are 2-D (xy plane only, z ignored for
    // similarity).  We avoid Eigen temporaries by working with raw scalars.
    const float odx = ox - cx;
    const float ody = oy - cy;
    const float od_norm = sqrtf(odx * odx + ody * ody);

    auto visibility_logit = [&](const uint8_t idx) -> float {
      const float pdx = px[idx] - cx;
      const float pdy = py[idx] - cy;
      const float pd_norm = sqrtf(pdx * pdx + pdy * pdy);
      const float similarity = (odx * pdx + ody * pdy) / (od_norm * pd_norm);
      return params_.visibility_logit_coefficient * similarity;
    };

    const float lv0 = visibility_logit(i0);
    const float lv1 = visibility_logit(i1);
    const float lv2 = visibility_logit(i2);
    const float lv3 = visibility_logit(i3);

    // ---- Log-density (scalar) ---------------------------------------------
    // observed_plate carries position and diagonal covariance as Vector3f.
    // We extract x/y/z components and compute the Gaussian score as three
    // scalar divisions — no Eigen allocation, no virtual dispatch.
    auto log_p_of = [&](const observed_plate& obs, const uint8_t pred_idx) -> float {

      // ---- Position likelihood ----------------------------------------------
      const float ex = obs.position().x() - px[pred_idx];
      const float ey = obs.position().y() - py[pred_idx];
      const float ez = obs.position().z() - pz[pred_idx];

      const float vx = obs.position_diagonal_covariance().x();
      const float vy = obs.position_diagonal_covariance().y();
      const float vz = obs.position_diagonal_covariance().z();

      const float pos_log_density =
          helper::scalar_log_density_3(ex, ey, ez, vx, vy, vz);

      // ---- Yaw likelihood ----------------------------------------------------
      const float predicted_yaw =
          atan2f(py[pred_idx] - cy, px[pred_idx] - cx);

      const float view_ray_yaw =
          atan2f(py[pred_idx] - oy, px[pred_idx] - ox);

      const float mirrored_predicted_yaw =
          2.0f * view_ray_yaw - predicted_yaw;

      const float yaw_error =
          helper::wrap_half_turn(obs.yaw() - predicted_yaw);

      const float mirrored_yaw_error =
          helper::wrap_half_turn(obs.yaw() - mirrored_predicted_yaw);

      const float yaw_variance =
          thrust::max(1.0e-6f, obs.yaw_variance());

      const float log_yaw_density =
          sampler.unnormalized_normal_log_density(
              yaw_variance,
              yaw_error);

      const float log_mirrored_yaw_density =
          sampler.unnormalized_normal_log_density(
              yaw_variance,
              mirrored_yaw_error)
          - params_.mirrored_yaw_penalty;

      const float final_yaw_density =
          helper::log_sum_exp(
              log_yaw_density,
              log_mirrored_yaw_density);

      return pos_log_density + final_yaw_density;
    };

    // ---- Combine -----------------------------------------------------------
    if (!state.plate_two().has_value()) {
      const float pr_visibility =
          helper::log_sigmoid( lv0) +
          helper::log_sigmoid(-lv1) +
          helper::log_sigmoid(-lv2) +
          helper::log_sigmoid(-lv3);
      return pr_visibility + log_p_of(state.plate_one(), i0);
    }

    const float log_pr_visibility =
        helper::log_sigmoid( lv0) +
        helper::log_sigmoid( lv1) +
        helper::log_sigmoid(-lv2) +
        helper::log_sigmoid(-lv3);

    const float assignment_one = log_p_of(state.plate_one(), i0) + log_p_of(*state.plate_two(), i1);
    const float assignment_two = log_p_of(state.plate_one(), i1) + log_p_of(*state.plate_two(), i0);
    return log_pr_visibility + fmaxf(assignment_one, assignment_two);
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] prediction sample_from(
      util::default_rv_sampler& sampler,
      const observation& state,
      const std::optional<initialization_prior_type>& initialization_prior) const noexcept {
    const float robot_radius = initialization_prior.has_value() ?
                                   initialization_prior->robot_radius :
                                   params_.initialization_prior.robot_radius;
    const observed_plate_orbit_builder builder(robot_radius, state.observer_position());

    const observed_plate_orbit orbit = state.plate_two().has_value() ?
                                           builder.from_two_plates(state.plate_one(), *state.plate_two()) :
                                           builder.from_one_plate(state.plate_one());

    const float radius_variance =
        state.plate_two().has_value() ? params_.radius_prior_variance_two_plates : params_.radius_prior_variance_one_plate;

    const float radius               = orbit.radius + sampler.normal_sample(radius_variance);
    const float orientation          = orbit.orientation;
    const float orientation_velocity = sampler.normal_sample(params_.orientation_velocity_prior_variance);

    const Eigen::Vector3f center    = orbit.center + sampler.normal_sample(state.plate_one().position_diagonal_covariance());
    const Eigen::Vector2f center_xy = center.head<2>();

    const bool has_two_plates = state.plate_two().has_value();
    const float z_coordinate_0 = has_two_plates ? orbit.z_coordinate_0
                          : orbit.z_coordinate_0 + sampler.normal_sample(params_.z_height_prior_variance);
    const float z_coordinate_1 = has_two_plates ? orbit.z_coordinate_1
                          : orbit.z_coordinate_1 + sampler.normal_sample(params_.z_height_prior_variance);

    const Eigen::Vector2f center_xy_velocity = sampler.normal_sample(params_.center_velocity_prior_diagonal_covariance);

    return prediction(radius, z_coordinate_0, z_coordinate_1, orientation, orientation_velocity, center_xy, center_xy_velocity);
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] prediction sample_from(
      util::default_rv_sampler& sampler,
      const observation& state) const noexcept {
    return sample_from(sampler, state, std::nullopt);
  }

  PF_TARGET_ONLY_ATTRS void apply_process(const float& time_offset_seconds, util::default_rv_sampler& sampler, prediction& state)
      const noexcept {
    const float radius_noise = sampler.normal_sample(params_.radius_process_variance);

    const float z_coordinate_noise_common = sampler.normal_sample(params_.z_coordinate_common_process_variance);
    const float z_coordinate_noise_0      = sampler.normal_sample(params_.z_coordinate_offset_process_variance);
    const float z_coordinate_noise_1      = sampler.normal_sample(params_.z_coordinate_offset_process_variance);

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
