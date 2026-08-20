from typing import List
import numpy as np
import numpy.typing as npt


class ObservedPlate:
    def __init__(self, position: npt.NDArray[np.float64],
                 position_diagonal_covariance: npt.NDArray[np.float64]): ...

    def position(self) -> npt.NDArray[np.float64]: ...
    def position_diagonal_covariance(self) -> npt.NDArray[np.float64]: ...


class PredictedPlate:
    def __init__(
        self, position: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64]): ...

    def position(self) -> npt.NDArray[np.float64]: ...
    def velocity(self) -> npt.NDArray[np.float64]: ...


class Observation:
    @staticmethod
    def from_one_plate(observer_position: npt.NDArray[np.float64],
                       plate_one: ObservedPlate) -> Observation: ...

    @staticmethod
    def from_two_plates(observer_position: npt.NDArray[np.float64], plate_one: ObservedPlate,
                        plate_two: ObservedPlate) -> Observation: ...


class Prediction:
    def predicted_plates(self) -> List[PredictedPlate]: ...
    def radius_0(self) -> float: ...
    def radius_1(self) -> float: ...
    def z_coordinate_0(self) -> float: ...
    def z_coordinate_1(self) -> float: ...
    def orientation(self) -> float: ...
    def orientation_velocity(self) -> float: ...
    def center(self) -> npt.NDArray[np.float64]: ...
    def center_velocity(self) -> npt.NDArray[np.float64]: ...

    # 9x9 posterior covariance over the linear substate, ordered as in
    # plate_orbit_v3::linear_state: orientation_velocity, center (x, y),
    # center_velocity (x, y), radius common/offset, z common/offset.
    def covariance(self) -> npt.NDArray[np.float64]: ...

    def extrapolate_state(
        self, time_offset_seconds: float) -> Prediction: ...


class ParticleFilterConfigurationParameters:
    def __init__(
        self,
        radius_prior: float,
        visibility_logit_coefficient: float,
        radius_prior_variance_one_plate: float,
        radius_prior_variance_two_plates: float,
        radius_common_process_variance: float,
        radius_common_reversion_time_constant: float,
        radius_offset_stationary_variance: float,
        radius_offset_reversion_time_constant: float,
        z_common_process_variance: float,
        z_offset_stationary_variance: float,
        z_offset_reversion_time_constant: float,
        orientation_prior_variance: float,
        orientation_velocity_prior_variance: float,
        orientation_velocity_process_variance: float,
        center_velocity_prior_diagonal_covariance: npt.NDArray[np.float64],
        center_velocity_process_diagonal_covariance: npt.NDArray[np.float64],
        resample_effective_sample_size_fraction: float
    ): ...


class ParticleFilter:
    def __init__(self, number_of_samples: int, initial_observation: Observation,
                 params: ParticleFilterConfigurationParameters): ...

    def extrapolate_state(
        self, time_offset_seconds: float) -> Prediction: ...

    def update_state_sans_observation(self, time_offset_seconds: float): ...

    def update_state_with_observation(
        self, time_offset_seconds: float, state: Observation): ...

    def reinitialize(self, observation: Observation): ...

    # Diagnostics. A healthy track holds a high effective sample size and
    # resamples on roughly 10% of frames.
    def effective_sample_size(self) -> float: ...
    def resampled(self) -> bool: ...
