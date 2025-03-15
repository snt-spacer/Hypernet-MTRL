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

from isaaclab_tasks.rans import GoToPositionCfg, GoToPositionTask, GoToPoseCfg, GoToPoseTask, TurtleBot2Robot, TurtleBot2RobotCfg


@configclass
class TurtleBot2DoubleTaskEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 6
    episode_length_s = 30.0

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=7.5, replicate_physics=True)

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)

    robot_cfg: TurtleBot2RobotCfg = TurtleBot2RobotCfg()
    task_cfg_0: GoToPositionCfg = GoToPositionCfg()
    task_cfg_1: GoToPoseCfg = GoToPoseCfg()
    debug_vis: bool = True

    action_space = robot_cfg.action_space
    observation_space = robot_cfg.observation_space + task_cfg_1.observation_space + 1  # TODO We use the task with the largest observation space
    state_space = robot_cfg.state_space
    gen_space = robot_cfg.gen_space + task_cfg_1.gen_space  # TODO We use the task with the largest gen space

    num_tasks = 2


class TurtleBot2DoubleTaskEnv(DirectRLEnv):
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

    cfg: TurtleBot2DoubleTaskEnvCfg

    def __init__(
        self,
        cfg: TurtleBot2DoubleTaskEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        self.num_tasks = cfg.num_tasks
        super().__init__(cfg, render_mode, **kwargs)
        self.env_seeds = torch.randint(0, 100000, (self.num_envs,), dtype=torch.int32, device=self.device)
        self.robot_api.run_setup(self.robot)
        self.task_api_0.run_setup(self.robot_api, self.scene.env_origins[: self.num_envs // self.num_tasks])
        self.task_api_1.run_setup(self.robot_api, self.scene.env_origins[self.num_envs // self.num_tasks :])
        self.set_debug_vis(self.cfg.debug_vis)

        self.observation_buffer = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg.robot_cfg)
        self.robot_api = TurtleBot2Robot(
            self.scene,
            self.cfg.robot_cfg,
            robot_uid=0,
            num_envs=self.num_envs,
            decimation=self.cfg.decimation,
            device=self.device,
            num_tasks=self.num_tasks,
        )

        self.env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        num_envs_per_task = self.num_envs // self.num_tasks
        task_0_env_ids = self.env_ids[:num_envs_per_task]
        task_1_env_ids = self.env_ids[num_envs_per_task:]
        self.task_api_0 = GoToPositionTask(
            scene=self.scene, 
            task_cfg=self.cfg.task_cfg_0, 
            task_uid=1, 
            num_envs=num_envs_per_task, 
            device=self.device,
            num_tasks=self.num_tasks,
            env_ids=task_0_env_ids
        )
        self.task_api_1 = GoToPoseTask(
            scene=self.scene,
            task_cfg=self.cfg.task_cfg_1,
            task_uid=2,
            num_envs=num_envs_per_task,
            device=self.device,
            num_tasks=self.num_tasks,
            env_ids=task_1_env_ids
        )

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations[self.cfg.robot_cfg.robot_name] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.robot_api.process_actions(actions)

    def _apply_action(self) -> None:
        self.robot_api.apply_actions()

    def _get_observations(self) -> dict:
        task_0_obs = self.task_api_0.get_observations()
        task_1_obs = self.task_api_1.get_observations()
        # TODO
        max_len = max(tensor.shape[-1] for tensor in [task_0_obs, task_1_obs]) + 1  # Task uid
        for i, task_obs in enumerate([task_0_obs, task_1_obs]):
            padding_size = max_len - task_obs.shape[-1]
            padded_obs = torch.nn.functional.pad(task_obs, (0, padding_size), "constant", 0)
            padded_obs[:, -1] = i + 1  # Task uid
            self.observation_buffer[i * self.num_envs // self.num_tasks : (i + 1) * self.num_envs // self.num_tasks] = padded_obs
        observations = {"policy": self.observation_buffer}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        task_0_rewards = self.task_api_0.compute_rewards()
        task_1_rewards = self.task_api_1.compute_rewards()
        return torch.cat([task_0_rewards, task_1_rewards])

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_early_termination, robot_clean_termination = self.robot_api.get_dones()
        task_0_early_termination, task_0_clean_termination = self.task_api_0.get_dones()
        task_1_early_termination, task_1_clean_termination = self.task_api_1.get_dones()

        task_early_termination = torch.cat([task_0_early_termination, task_1_early_termination])
        task_clean_termination = torch.cat([task_0_clean_termination, task_1_clean_termination])

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        early_termination = robot_early_termination | task_early_termination
        clean_termination = robot_clean_termination | task_clean_termination | time_out
        return early_termination, clean_termination

    def _reset_idx(self, env_ids: Sequence[int] | None):

        task_0_env_ids_idx = torch.where(env_ids < self.num_envs // 2)[0]
        task_0_env_ids = env_ids[task_0_env_ids_idx]
        task_1_env_ids_idx = torch.where(env_ids >= self.num_envs // 2)[0]
        task_1_env_ids = env_ids[task_1_env_ids_idx] - self.num_envs // 2

        self.task_api_0.reset_logs(task_0_env_ids, self.episode_length_buf[:self.num_envs // 2])
        self.task_api_1.reset_logs(task_1_env_ids, self.episode_length_buf[self.num_envs // 2:])

        task_0_extras = self.task_api_0.compute_logs()
        task_1_extras = self.task_api_1.compute_logs()

        self.robot_api.reset_logs(env_ids, self.episode_length_buf)
        robot_extras = self.robot_api.compute_logs()

        self.extras["log"] = dict()
        self.extras["log"].update(task_0_extras)
        self.extras["log"].update(task_1_extras)
        self.extras["log"].update(robot_extras)

        super()._reset_idx(env_ids)

        self.robot_api.reset(env_ids)

        if len(task_0_env_ids) > 0:
            self.task_api_0.reset(task_0_env_ids)
        if len(task_1_env_ids) > 0:
            self.task_api_1.reset(task_1_env_ids)
        
        

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            self.task_api_0.create_task_visualization()
            self.task_api_1.create_task_visualization()

    def _debug_vis_callback(self, event) -> None:
        if self.cfg.debug_vis:
            self.task_api_0.update_task_visualization()
            self.task_api_1.update_task_visualization()

