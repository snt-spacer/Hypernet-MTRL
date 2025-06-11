# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.markers import BICOLOR_DIAMOND_CFG, PIN_SPHERE_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveScene

from isaaclab_tasks.rans import StabilizationCfg

from .go_to_position import GoToPositionTask

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class StabilizationTask(GoToPositionTask):
    """
    Implements the Stabilization task. The robot has to reduce it's inital linear and angular velocity to 0.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: StabilizationCfg = StabilizationCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
        decimation: int = 1,
        num_tasks: int = 1,
    ) -> None:
        """
        Initializes the Stabilization task.

        Args:
            task_cfg: The configuration of the task.
            task_uid: The unique id of the task.
            num_envs: The number of environments.
            device: The device on which the tensors are stored.
            task_id: The id of the task.
            env_ids: The ids of the environments used by this task."""

        # Task and reward parameters
        self._task_cfg = task_cfg
        
        super().__init__(scene=scene, task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids, num_tasks=num_tasks)