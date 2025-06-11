# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .go_to_position_cfg import GoToPositionCfg


@configclass
class StabilizationCfg(GoToPositionCfg):
    """Configuration for the GoToPosition task."""

    # Initial conditions
    spawn_max_lin_vel: float = 1.0
    """Maximal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_max_ang_vel: float = 6.2
    """Maximal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""

    # Tolerance
    maximum_robot_distance: float = 15.0
    """Maximal distance between the robot and the target pose. Defaults to 10 m."""