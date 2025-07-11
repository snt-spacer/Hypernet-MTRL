# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticMemoryCfg, RslRlPpoAlgorithmCfg


@configclass
class SingleRobotMultiTaskPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 2000
    save_interval = 50
    experiment_name = "multitask_memory_racing"
    logger = "wandb"
    wandb_kwargs = {
        "project": "multitask_memory_racing",
        "entity": "spacer-rl",
        "group": "zeroG",
    }
    empirical_normalization = False
    policy = RslRlPpoActorCriticMemoryCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
        clip_actions=True,
        clip_actions_range=[-1, 1],
        use_embeddings=True,
        embeddings_size=32,
        generator_size=(32, 32),
        num_memory_obs=46, # Two times the number of max_num_corners on the race_gates task (y,x of points)
        network_type="hybrid", #pure, hybrid
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=4.0e-3,
        schedule="adaptive", # fixed, adaptive
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
