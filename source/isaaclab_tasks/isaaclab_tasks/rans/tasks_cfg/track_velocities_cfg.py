# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import NoisyObservationsCfg

from .task_core_cfg import TaskCoreCfg


@configclass
class TrackVelocitiesCfg(TaskCoreCfg):
    """Configuration for the TrackVelocityTask task."""

    # Initial conditions
    spawn_min_lin_vel: float = 0.0
    """Minimal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_max_lin_vel: float = 0.0
    """Maximal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_min_ang_vel: float = 0.0
    """Minimal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""
    spawn_max_ang_vel: float = 0.0
    """Maximal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""

    # Goal spawn
    enable_linear_velocity: bool = True
    """Enable linear velocity goal. Defaults to True."""
    goal_min_lin_vel: float = 0.0
    """Minimal linear velocity goal in m/s. Defaults to 0.0 m/s. (a random sign is added)"""
    goal_max_lin_vel: float = 0.45
    """Maximal linear velocity goal in m/s. Defaults to 2.0 m/s. (a random sign is added)"""
    enable_lateral_velocity: bool = True
    """Enable lateral velocity goal. Defaults to False."""
    goal_min_lat_vel: float = 0.0
    """Minimal lateral velocity goal in m/s. Defaults to 0.0 m/s. (a random sign is added)"""
    goal_max_lat_vel: float = 0.45
    """Maximal lateral velocity goal in m/s. Defaults to 0.0 m/s. (a random sign is added)"""
    enable_angular_velocity: bool = True
    """Enable angular velocity goal. Defaults to True."""
    goal_min_ang_vel: float = 0.0
    """Minimal angular velocity goal in rad/s. Defaults to 0.0 rad/s. (a random sign is added)"""
    goal_max_ang_vel: float = 0.9
    """Maximal angular velocity goal in rad/s. Defaults to 0.4 rad/s. (a random sign is added)"""

    # Settings
    resample_at_regular_interval: bool = True
    interval: tuple[int, int] = (60, 80)
    smoothing_factor: tuple[float, float] = (0.0, 0.9)

    # Tolerance
    linear_velocity_tolerance: float = 0.01
    lateral_velocity_tolerance: float = 0.01
    angular_velocity_tolerance: float = 0.05
    maximum_robot_distance: float = 1000.0  # should be plenty enough not to reset
    resample_after_steps_in_tolerance: int = 50

    # Reward Would be good to have a config for each reward type
    lin_vel_exponential_reward_coeff: float = 0.5
    lat_vel_exponential_reward_coeff: float = 0.5
    ang_vel_exponential_reward_coeff: float = 0.5
    linear_velocity_weight: float = 0.33
    lateral_velocity_weight: float = 0.33
    angular_velocity_weight: float = 0.33

    # Visualization
    visualization_linear_velocity_scale: float = 1.0
    visualization_angular_velocity_scale: float = 2.5

    # Randomization
    noisy_observation_cfg: NoisyObservationsCfg = NoisyObservationsCfg(
        enable=False,
        randomization_modes=["uniform"],
        slices=[
            0,
            1,
            2,
            (3, 5),
            5,
        ],
        max_delta=[0.03, 0.03, 0.01, 0.03, 0.01],
    )

    # Spaces
    observation_space: int = 6
    state_space: int = 0
    action_space: int = 0
    gen_space: int = 5
