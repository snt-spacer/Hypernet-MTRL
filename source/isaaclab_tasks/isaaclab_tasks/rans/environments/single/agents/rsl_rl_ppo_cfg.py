# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class SinglePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 4000
    save_interval = 50
    experiment_name = "multitask_racing_baseline_noTrackInfo"
    logger = "wandb"
    wandb_kwargs = {
        "project": "multitask_racing_baseline_noTrackInfo",
        "entity": "spacer-rl",
        "group": "zeroG",
    }
    resume = False
    load_run = "2025-09-01_21-55-27_rsl-rl_ppo_RaceGates_Leatherback_r-0_seed-5"
    load_checkpoint = "model_3200.pt"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[256, 256],
        activation="tanh",
        clip_actions=True,
        clip_actions_range=[-1, 1],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive", #adaptive, fixed
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
