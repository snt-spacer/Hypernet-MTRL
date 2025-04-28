# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--runs_per_env",
    type=int,
    default=4,
    help="The number of runs to be performed for each environment.",
)
parser.add_argument(
    "--overload-experiment-cfg", 
    action="store_true", 
    default=True, help="Overload experiment config. If set to True, it will load the cfg of the model that was used for training."
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    help="The RL algorithm used for training the rsl-rl agent.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import copy

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config


from isaaclab_tasks.rans.utils import EvalMetrics

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    
    if args_cli.overload_experiment_cfg:
        if args_cli.checkpoint:
            agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.load_rsl_rl_cfg(args_cli.checkpoint, args_cli.task)
        else:
            raise ValueError("Missing checkpoint path for loading the experiment config.")
    else:
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    if "MultiTask" in args_cli.task:
        robot_name = env.env.get_wrapper_attr('robot_api')._robot_cfg.robot_name
        print(f"[INFO] Evaluation only one task: {env.env.get_wrapper_attr('tasks_apis')[0].__class__.__name__}")
        task_name = env.env.get_wrapper_attr('tasks_apis')[0].__class__.__name__[:-4] # remove "Task" suffix
        num_tasks = len(env.env.get_wrapper_attr('tasks_apis'))
        task_chunk = env_cfg.scene.num_envs // num_tasks
    
    else:
        robot_name = env.env.get_wrapper_attr('robot_api')._robot_cfg.robot_name
        task_name = env.env.get_wrapper_attr('task_api').__class__.__name__[:-4] # remove "Task" suffix
    
    metrics_dir = os.path.join(log_dir, "metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    plots_dir = os.path.join(log_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    eval_metrics = EvalMetrics(
        env=env, 
        robot_name=robot_name, 
        task_name=task_name, 
        folder_path=log_dir, 
        device=env.unwrapped.device,
        num_runs_per_env=args_cli.runs_per_env,
    )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    dt = env.unwrapped.physics_dt

    data = {k: [] for k in env.env.get_wrapper_attr('eval_data_keys')}
    data["dones"] = []

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            new_data = copy.deepcopy(env.env.unwrapped.eval_data)
            for k, v in new_data.items():
                data[k].append(v)

            if "MultiTask" in args_cli.task:
                data["dones"].append(dones[:task_chunk])

                # Check if the number of runs per env is reached
                if torch.all(torch.sum(torch.cat(data["dones"], dim=-1).view(-1, task_chunk), dim=0) >= args_cli.runs_per_env).item():
                    print(f"[INFO] Collected {args_cli.runs_per_env} runs per env.")
                    break
            else:
                data["dones"].append(dones)

                # Check if the number of runs per env is reached
                if torch.all(torch.sum(torch.cat(data["dones"], dim=-1).view(-1, env_cfg.scene.num_envs), dim=0) >= args_cli.runs_per_env).item():
                    print(f"[INFO] Collected {args_cli.runs_per_env} runs per env.")
                    break

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

    data = {k: torch.stack(v, dim=0) for k, v in data.items()}
    eval_metrics.calculate_metrics(data=data)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
