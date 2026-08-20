
# Filter Guide

All filters estimate the pose of a 4-plate robot from 1-2 observed plate positions. 

The following variants of filters exist:

| Filter Name | Estimated Robot State |
| --- | --- |
| `FastPlateOrbit` | x, y, z, vx, vy, radius, orientation, angular velocity |
| `FastPlateOrbitWithZOffset` | x, y, z, vx, vy, radius, orientation, angular velocity, plate height offset |
| `PlateOrbit` | x, v, z, vx, vy, radius1, radius2, orientation, angular velocity |
| `PlateOrbitV2` | x, v, z, vx, vy, radius1, radius2, orientation, angular velocity, plate height offset |
| `PlateOrbitV3` | same state as `PlateOrbitV2`, estimated by a Rao-Blackwellized particle filter |

For the above filters, plate height offset assumes that robots have 2 opposing pairs of plates at two different heights (z0, z1). This same opposing plate grouping is also used when defining radius1 and radius2, for ovular shaped robots. 

# A Note About Parameters

Let's look at the configuration parameters for `FastPlateOrbitWithZOffset`. This model assumes a circular robot, no Z velocity, but 2 Z-heights for plates.

```c++
struct particle_filter_configuration_parameters {
  float radius_prior;
  float visibility_logit_coefficient;

  float radius_prior_variance_one_plate;
  float radius_prior_variance_two_plates;
  float radius_process_variance;

  float orientation_velocity_prior_variance;
  float orientation_velocity_process_variance;

  float z_coordinate_common_process_variance;
  float z_coordinate_offset_process_variance;

  Eigen::Vector2f center_xy_velocity_prior_diagonal_covariance;
  Eigen::Vector2f center_xy_velocity_process_diagonal_covariance;
};
```
Largely the above parameters represent two systems, prior variance which is the initial uncertainty in the system about that state, and the process variance which is how much those parameters could plausibly shift per step.

A few unique parameters that don't follow this paradigm:
- `radius_prior`: For the particle filter, what should be initial estimate for the robot's radius in meters (used in the 1-plate observation)
- `visibilty_logit_coefficient`: As the plates passed into the filter do not have an associated angle, we must threshold somehow the likelihood that a given plate is observable. This parameter is a scalar that weighs how aggressively to assume that the given plate is directly in line with the camera or at an off-axis orientation.
- `radius_prior_variance_{one_plate,two_plate}`: These represent the uncertainty in initial radius estimate, but differ as seeing two plates gives a stronger constraint of radius geometry than one, so is suggested to be smaller than the one plate prior variance.

An example configuration is shown below:

```c++
struct particle_filter_configuration_parameters {
  float radius_prior = 0.3; // Assuems default 0.3m radius
  float visibility_logit_coefficient = 1.0;

  float radius_prior_variance_one_plate = 0.0025; // 5cm standard deviation
  float radius_prior_variance_two_plates = 0.000225; // 1.5cm standard deviation
  float radius_process_variance = 5e-6; // Assume that there is a very small change in radius, to account for measurement noise 

  float orientation_velocity_prior_variance = 100.0; // Highly uncertain of robot initial angular velocity, as they may be already spinning
  float orientation_velocity_process_variance = 5.0; // Limit on how fast they start/stop spinning

  float z_coordinate_common_process_variance = 4e-7; // Assume that robots don't change height much
  float z_coordinate_offset_process_variance = 5e-3; // Allow for some change in estimate to converge to true plate offset 

  Eigen::Vector2f center_xy_velocity_prior_diagonal_covariance = [0.5, 0.5]; // Robots initially may already be moving
  Eigen::Vector2f center_xy_velocity_process_diagonal_covariance = [0.25, 0.25]; // Limit on how fast they start/stop spinning
};
```





# PlateOrbitV3 (Rao-Blackwellized)

`PlateOrbitV3` estimates the same physical state as `PlateOrbitV2` but splits it
into a sampled part and an analytic part instead of sampling all of it.

## Why the split exists

Conditional on the orientation, the plate position measurement is *exactly*
linear in everything else:

```
x = c_x + (r_c + s_k r_o) cos(theta + k pi/2)
y = c_y + (r_c + s_k r_o) sin(theta + k pi/2)
z = z_c + s_k z_o
```

Fix `theta` and `cos(theta)` becomes a known constant, so this is a straight
line in `(c_x, c_y, r_c, r_o)` with Gaussian noise. The posterior is then an
exact Gaussian and a Kalman filter computes it in closed form. The dynamics are
linear in the same variables.

So the orientation is the only quantity that makes the problem hard. V3 guesses
it with particles and solves for the rest:

| | states | dimension |
| --- | --- | --- |
| Sampled (particles) | `theta` | 1 |
| Analytic (Kalman) | `omega, c_x, c_y, v_x, v_y, r_c, r_o, z_c, z_o` | 9 |

The angular velocity stays on the analytic side because `theta' = theta + dt *
omega` is linear: once a particle has drawn its new orientation, the realized
increment is a noisy measurement of `omega` that a Kalman update can absorb.
The orientation and angular velocity process noises are correlated, so the
propagation uses the exact conditional decomposition

```
omega' = -0.5 omega + (3 / (2 dt)) * increment + noise
```

which returns `omega` unchanged for a noise-free increment of `dt * omega`, as
it must.

**V3 does not estimate fewer things than V2.** It estimates the same things,
with the guessing machine searching a 1D space instead of a 12D one. This is
why it needs a few hundred particles where V2 needs on the order of a million.

## Weighting

The particle weight is the *marginal* likelihood, with the linear substate
integrated out:

```
w *= N(y; 0, H P H^T + R)
```

A particle is therefore never penalized for not yet knowing where the center
is. It is penalized only if its orientation cannot explain the observation
under *any* center position. Note that the `log|S|` normalizer is required
here, unlike in V2 where the innovation covariance was shared across particles
and cancelled.

## The (common, offset) basis

V3 stores `r_c, r_o` rather than `r_0, r_1`, with `r_0 = r_c + r_o` and
`r_1 = r_c - r_o` (likewise for `z`). Three consequences:

1. **It fixes a dead parameter in V2.** Working through
   `helper::update_value_offsets` with the clamps removed, `d_value_common`
   cancels identically: `v0_new = v0 + d_0`. So
   `radius_common_process_variance` and `z_coordinate_common_process_variance`
   had no effect in V2 except when the offset clamp was saturated. In the
   common/offset basis they do what the documentation above says they do.
2. **The clamps become linear.** `to_radius`'s hard `[0.2, 1.0]` bound and the
   `offset_limit` clamps are replaced by Ornstein-Uhlenbeck mean reversion,
   which is linear and therefore compatible with the Kalman substate. The
   `*_offset_stationary_variance` parameters replace `offset_limit`: set them
   so two or three standard deviations covers the asymmetry you expect.
3. **The pi/2 relabeling symmetry becomes a signed permutation.** V2's
   reduction enumerated explicit `radius_0`/`radius_1` swaps; in this basis a
   quarter turn simply negates both offset modes, so the same alignment applies
   to the covariance as `T P T^T` with `T` diagonal.

## Resampling

V3 uses `util::rb_particle_filter` rather than `pf::filter::particle_filter`.
Log weights accumulate across steps and resampling is triggered by effective
sample size, controlled by `resample_effective_sample_size_fraction` (0.5 is a
reasonable default). Unconditional resampling is actively harmful here:
duplicating a particle duplicates its Gaussian too, so it destroys diversity in
the one dimension being sampled while recovering nothing. In practice a healthy
track resamples on roughly 10% of frames.

`effective_sample_size()` and `resampled()` are exposed as diagnostics. A track
whose ESS is pinned near 1 is a track whose model is wrong.

## New output: calibrated uncertainty

`Prediction.covariance()` returns the full 9x9 posterior covariance, ordered as
in `plate_orbit_v3::linear_state`. This is genuinely informative rather than
decorative. Measured on a simulated 3 second track:

| | spinning at 6 rad/s | stationary |
| --- | --- | --- |
| sigma(r_offset) | 7 mm | 47 mm |
| sigma(z_offset) | 3 mm | 10 mm |
| sigma(center x) | 4 mm | 46 mm |

The stationary case is the fundamental observability limit, not a filter
defect: a robot that does not rotate never presents its other plate pair, so
the offsets are unobservable. V3 reports this honestly, which makes it usable
for shot gating. V2 had no way to express it.

## Example configuration

```c++
particle_filter_configuration_parameters params{
  .radius_prior = 0.30f,
  .visibility_logit_coefficient = 1.0f,

  .radius_prior_variance_one_plate = 0.0025f,       // 5 cm sigma
  .radius_prior_variance_two_plates = 0.000225f,    // 1.5 cm sigma

  .radius_common_process_variance = 5e-6f,
  .radius_common_reversion_time_constant = 20.0f,   // weak pull to prior

  .radius_offset_stationary_variance = 0.0025f,     // 5 cm sigma, replaces offset_limit
  .radius_offset_reversion_time_constant = 5.0f,

  .z_common_process_variance = 4e-7f,

  .z_offset_stationary_variance = 0.0016f,          // 4 cm sigma
  .z_offset_reversion_time_constant = 5.0f,

  .orientation_prior_variance = 0.25f,              // 0.5 rad sigma; new in v3
  .orientation_velocity_prior_variance = 100.0f,
  .orientation_velocity_process_variance = 5.0f,

  .center_velocity_prior_diagonal_covariance = {0.5f, 0.5f},
  .center_velocity_process_diagonal_covariance = {0.25f, 0.25f},

  .resample_effective_sample_size_fraction = 0.5f,
};
```

`orientation_prior_variance` is new and matters: the orientation is now the
only sampled continuous quantity, so it needs explicit initial spread rather
than inheriting diversity from the center and radius draws as in V2.
