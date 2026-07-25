
# Filter Guide

All filters estimate the pose of a 4-plate robot from 1-2 observed plate positions. 

The following variants of filters exist:

| Filter Name | Estimated Robot State |
| --- | --- |
| `FastPlateOrbit` | x, y, z, vx, vy, radius, orientation, angular velocity |
| `FastPlateOrbitWithZOffset` | x, y, z, vx, vy, radius, orientation, angular velocity, plate height offset |
| `PlateOrbit` | x, v, z, vx, vy, radius1, radius2, orientation, angular velocity |
| `PlateOrbitV2` | x, v, z, vx, vy, radius1, radius2, orientation, angular velocity, plate height offset |

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



