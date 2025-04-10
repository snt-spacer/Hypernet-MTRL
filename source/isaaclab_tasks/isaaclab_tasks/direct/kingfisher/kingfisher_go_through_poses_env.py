# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from isaaclab_tasks.rans import GoThroughPosesCfg, GoThroughPosesTask, KingfisherRobot, KingfisherRobotCfg


@configclass
class KingfisherGoThroughPosesEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=20.0, replicate_physics=True)

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)

    robot_cfg: KingfisherRobotCfg = KingfisherRobotCfg()
    task_cfg: GoThroughPosesCfg = GoThroughPosesCfg()
    debug_vis: bool = True

    action_space = robot_cfg.action_space + task_cfg.action_space
    observation_space = robot_cfg.observation_space + task_cfg.observation_space
    state_space = robot_cfg.state_space + task_cfg.state_space
    gen_space = robot_cfg.gen_space + task_cfg.gen_space


class KingfisherGoThroughPosesEnv(DirectRLEnv):
    # Workflow: Step
    #   - self._pre_physics_step
    #   - (Loop over N skipped steps)
    #       - self._apply_actions
    #       - self.scene.write_data_to_sim()
    #       - self.sim.step(render=False)
    #       - (Check if rendering is required)
    #           - self.sim.render()
    #       - self.scene.update()
    #   - self._get_dones
    #   - self._get_rewards
    #   - (Check if reset is required)
    #       - self._reset_idx
    #       - (Check if RTX sensors)
    #           - self.scene.render()
    #   - (Check for events)
    #       - self.event_manager.apply()
    #   - self._get_observations
    #   - (Check if noise is required)
    #       - self._add_noise

    cfg: KingfisherGoThroughPosesEnvCfg

    def __init__(
        self,
        cfg: KingfisherGoThroughPosesEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.env_seeds = torch.randint(0, 100000, (self.num_envs,), dtype=torch.int32, device=self.device)
        self.robot_api.run_setup(self.robot)
        self.task_api.run_setup(self.robot_api, self.scene.env_origins)
        self.set_debug_vis(self.cfg.debug_vis)

    @property
    def eval_data_keys(self) -> list[str]:
        task_data_keys = self.task_api.eval_data_keys
        robot_data_keys = self.robot_api.eval_data_keys
        return task_data_keys + robot_data_keys
    
    @property
    def eval_data(self) -> dict:
        task_eval_data = self.task_api.eval_data
        robot_eval_data = self.robot_api.eval_data
        return {**task_eval_data, **robot_eval_data}

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg.robot_cfg)
        self.robot_api = KingfisherRobot(
            self.cfg.robot_cfg, robot_uid=0, num_envs=self.num_envs, device=self.device, decimation=self.cfg.decimation
        )
        self.task_api = GoThroughPosesTask(
            self.cfg.task_cfg, task_uid=0, num_envs=self.num_envs, device=self.device, decimation=self.cfg.decimation
        )

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["kingfisher"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.robot_api.process_actions(actions)

    def _apply_action(self) -> None:
        self.robot_api.compute_physics()
        self.robot_api.apply_actions(self.robot)

    def _get_observations(self) -> dict:
        task_obs = self.task_api.get_observations()
        observations = {"policy": task_obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return self.task_api.compute_rewards()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_early_termination, robot_clean_termination = self.robot_api.get_dones()
        task_early_termination, task_clean_termination = self.task_api.get_dones()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        early_termination = robot_early_termination | task_early_termination
        clean_termination = robot_clean_termination | task_clean_termination | time_out
        return early_termination, clean_termination

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if (env_ids is None) or (len(env_ids) == self.num_envs):
            env_ids = self.robot._ALL_INDICES

        # Logging
        self.task_api.reset_logs(env_ids, self.episode_length_buf)
        task_extras = self.task_api.compute_logs()
        self.robot_api.reset_logs(env_ids, self.episode_length_buf)
        robot_extras = self.robot_api.compute_logs()
        self.extras["log"] = dict()
        self.extras["log"].update(task_extras)
        self.extras["log"].update(robot_extras)

        super()._reset_idx(env_ids)

        self.task_api.reset(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            self.task_api.create_task_visualization()

    def _debug_vis_callback(self, event) -> None:
        if self.cfg.debug_vis:
            self.task_api.update_task_visualization()
