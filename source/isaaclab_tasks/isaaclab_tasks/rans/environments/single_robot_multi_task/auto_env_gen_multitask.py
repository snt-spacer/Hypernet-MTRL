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
from isaaclab.envs.utils.spaces import sample_space
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from isaaclab_tasks.rans import ROBOT_CFG_FACTORY, ROBOT_FACTORY, TASK_CFG_FACTORY, TASK_FACTORY


@configclass
class MultiTaskEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 6
    episode_length_s = 20.0

    robot_name = "Leatherback"
    tasks_names = ["GoToPosition"]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=7.5, replicate_physics=True)

    # Steps per episode
    #spe = 1/hz * decumation * episode_length_s

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)
    # Simulation
    # sim = SimulationCfg(
    #     disable_contact_processing=True,
    #     physx=sim_utils.PhysxCfg(
    #         enable_ccd=True,
    #         enable_stabilization=True,
    #         bounce_threshold_velocity=0.0,
    #         friction_correlation_distance=0.005,
    #         min_velocity_iteration_count=2,
    #         # GPU settings
    #         gpu_temp_buffer_capacity=2 ** (24 - 4),
    #         gpu_max_rigid_contact_count=2 ** (22 - 5),
    #         gpu_max_rigid_patch_count=2 ** (13 - 3),
    #         gpu_heap_capacity=2 ** (26 - 3),
    #         gpu_found_lost_pairs_capacity=2 ** (18 - 3),
    #         gpu_found_lost_aggregate_pairs_capacity=2 ** (10 - 2),
    #         gpu_total_aggregate_pairs_capacity=2 ** (10 - 2),
    #         gpu_max_soft_body_contacts=2 ** (20 - 5),
    #         gpu_max_particle_contacts=2 ** (20 - 5),
    #         gpu_collision_stack_size=2 ** (26 - 5),
    #         gpu_max_num_partitions=8,
    #     ),
    #     render=sim_utils.RenderCfg(
    #         enable_reflections=True,
    #     ),
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #     ),
    # )
    debug_vis: bool = True

    action_space = 0
    observation_space = 0
    state_space = 0
    gen_space = 0


class MultiTaskEnv(DirectRLEnv):

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

    cfg: MultiTaskEnvCfg

    def __init__(
        self,
        cfg: MultiTaskEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        cfg = self.edit_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        if self.num_envs % self.num_tasks != 0:
            raise ValueError(f"Number of environments ({self.num_envs}) must be divisible by the number of tasks ({self.num_tasks})")
        
        self.env_seeds = torch.randint(0, 100000, (self.num_envs,), dtype=torch.int32, device=self.device)
        self.robot_api.run_setup(self.robot)
        chunk_size = self.num_envs // self.num_tasks
        for i, task_api in enumerate(self.tasks_apis):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            task_api.run_setup(self.robot_api, self.scene.env_origins[start_idx:end_idx])
            task_api.register_rigid_objects()
        self.set_debug_vis(self.cfg.debug_vis)
        self.observation_buffer = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        super()._configure_gym_env_spaces()
        self.single_action_space, self.action_space = self.robot_api.configure_gym_env_spaces()
        self.actions = sample_space(self.single_action_space, self.sim.device, batch_size=self.num_envs, fill_value=0)

    def edit_cfg(self, cfg: MultiTaskEnvCfg) -> MultiTaskEnvCfg:
        self.robot_cfg = ROBOT_CFG_FACTORY(cfg.robot_name)

        self.tasks_cfgs = []
        max_action_space = 0
        max_observation_space = 0
        max_state_space = 0
        max_gen_space = 0
        for task_name in cfg.tasks_names:
            self.tasks_cfgs.append(TASK_CFG_FACTORY(task_name))
            if self.tasks_cfgs[-1].observation_space > max_observation_space:
                max_observation_space = self.tasks_cfgs[-1].observation_space
            if self.tasks_cfgs[-1].action_space > max_action_space:
                max_action_space = self.tasks_cfgs[-1].action_space
            if self.tasks_cfgs[-1].state_space > max_state_space:
                max_state_space = self.tasks_cfgs[-1].state_space
            if self.tasks_cfgs[-1].gen_space > max_gen_space:
                max_gen_space = self.tasks_cfgs[-1].gen_space

        cfg.action_space = self.robot_cfg.action_space + max_action_space
        cfg.observation_space = self.robot_cfg.observation_space + max_observation_space + 1 # +1 for task uid
        cfg.state_space = self.robot_cfg.state_space + max_state_space
        cfg.gen_space = self.robot_cfg.gen_space + max_gen_space
        self.num_tasks = len(self.tasks_cfgs)
        return cfg

    def _setup_scene(self):
        self.robot = Articulation(self.robot_cfg.robot_cfg)
        self.robot_api = ROBOT_FACTORY(
            self.cfg.robot_name,
            scene=self.scene,
            robot_cfg=self.robot_cfg,
            robot_uid=0,
            num_envs=self.num_envs,
            decimation=self.cfg.decimation,
            device=self.device,
            num_tasks=self.num_tasks,
        )

        self.tasks_apis = []
        num_envs_per_task = self.num_envs // self.num_tasks
        self.env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        chunk_size = self.num_envs // self.num_tasks
        for i, task_name in enumerate(self.cfg.tasks_names):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            task_api = TASK_FACTORY(
                task_name,
                scene=self.scene,
                task_cfg=self.tasks_cfgs[i],
                task_uid=i + 1,
                num_envs=num_envs_per_task,
                device=self.device,
                num_tasks=self.num_tasks,
                env_ids=self.env_ids[start_idx:end_idx]
            )
            self.tasks_apis.append(task_api)

            task_api.register_robot(self.robot_api)
            task_api.register_sensors()

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations[self.cfg.robot_name] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.robot_api.process_actions(actions)

    def _apply_action(self) -> None:
        self.robot_api.apply_actions()

    def _get_observations(self) -> dict:
        tasks_obs = [task_api.get_observations() for task_api in self.tasks_apis]
        max_len = max(tensor.shape[-1] for tensor in tasks_obs) + 1  # +1 for task uid

        for i, task_obs in enumerate(tasks_obs):
            padding_size = max_len - task_obs.shape[-1]
            padded_obs = torch.nn.functional.pad(task_obs, (0, padding_size), "constant", 0)
            padded_obs[:, -1] = i + 1  # Task uid
            self.observation_buffer[i * self.num_envs // self.num_tasks : (i + 1) * self.num_envs // self.num_tasks] = padded_obs

        observations = {"policy": self.observation_buffer}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        task_rewards = [task_api.compute_rewards() for task_api in self.tasks_apis]
        # print("#"*50)
        # print(task_rewards[0])
        # print(task_rewards[1])
        return torch.cat(task_rewards, dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_early_termination, robot_clean_termination = self.robot_api.get_dones()
        
        tasks_early_terminations = []
        tasks_clean_terminations = []
        for task_api in self.tasks_apis:
            early_termination, clean_termination = task_api.get_dones()
            tasks_early_terminations.append(early_termination)
            tasks_clean_terminations.append(clean_termination)

        task_early_termination = torch.cat(tasks_early_terminations)
        task_clean_termination = torch.cat(tasks_clean_terminations)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        early_termination = robot_early_termination | task_early_termination
        clean_termination = robot_clean_termination | task_clean_termination | time_out
        return early_termination, clean_termination

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if (env_ids is None) or (len(env_ids) == self.num_envs):
            env_ids = self.robot._ALL_INDICES

        tasks_env_ids = []
        tasks_extras = []
        chunk_size = self.num_envs // self.num_tasks
        for i in range(self.num_tasks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            tasks_env_ids_indx = torch.where((env_ids >= start_idx) & (env_ids < end_idx))[0]
            task_env_ids = env_ids[tasks_env_ids_indx] - start_idx
            tasks_env_ids.append(task_env_ids)

            # Reset / Compute tasks logs
            self.tasks_apis[i].reset_logs(task_env_ids, self.episode_length_buf[start_idx:end_idx])
            tasks_extras.append(self.tasks_apis[i].compute_logs())

        # Logging
        self.robot_api.reset_logs(env_ids, self.episode_length_buf)
        robot_extras = self.robot_api.compute_logs()
        self.extras["log"] = dict()
        for task_extras in tasks_extras:
            self.extras["log"].update(task_extras)
        self.extras["log"].update(robot_extras)

        # Reset
        super()._reset_idx(env_ids)
        self.robot_api.reset(env_ids)
        for i, task_api in enumerate(self.tasks_apis):

            # Curriculum
            if self.common_step_counter < 10 * 500:
                scale = self.common_step_counter / (10 * 500 )
                gen_actions = torch.zeros((len(env_ids) // self.num_tasks, self.tasks_cfgs[i].gen_space), device=self.device) * scale
            else:
                gen_actions = None

            if len(tasks_env_ids[i]) > 0:
                task_api.reset(tasks_env_ids[i], gen_actions=gen_actions)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            for task_api in self.tasks_apis:
                task_api.create_task_visualization()

    def _debug_vis_callback(self, event) -> None:
        if self.cfg.debug_vis:
            for task_api in self.tasks_apis:
                task_api.update_task_visualization()
