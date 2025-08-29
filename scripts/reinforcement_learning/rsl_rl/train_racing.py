# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import os

# Helper function for boolean argument parsing
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--run_num",type=int,default=0,help="The run number for the current experiment.")
parser.add_argument("--fixed_track_id",type=int,default=-1,help="The fixed track id for the racing task. -1 means random track.")
parser.add_argument("--same_track_for_all_envs",type=str2bool,default=False,help="If True, all environments will use the same track. If False, each environment will use a different track.")
parser.add_argument(
    "--type_of_training",
    type=str,
    default="padd",
    help="The type of training to use. Options: 'hyper' or 'padd'.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

def modify_racing_config():
        # Modify dynamically the task configuration. IsaacLab do something about this.
    new_content = []
    eval_racing_cfg_path = os.path.normpath(os.path.join(os.getcwd(), "source", "isaaclab_tasks", "isaaclab_tasks", "rans", "tasks_cfg", "race_gates_cfg.py"))
    with open(eval_racing_cfg_path, 'r') as file:
        for line in file:
            if "loop:" in line:
                new_content.append("    loop: bool = True\n")
            elif "num_laps:" in line:
                new_content.append("    num_laps: int = 5\n")
            elif "fixed_track_id:" in line:
                new_content.append(f"    fixed_track_id: int = {args_cli.fixed_track_id}\n")
            elif "spawn_at_random_gate:" in line:
                new_content.append("    spawn_at_random_gate: bool = True\n")
            elif "same_track_for_all_envs:" in line:
                new_content.append(f"    same_track_for_all_envs: bool = {args_cli.same_track_for_all_envs}\n")
            elif "type_of_training:" in line:
                new_content.append(f"    type_of_training: str = \"{args_cli.type_of_training}\"\n")
            elif "max_num_corners:" in line:
                new_content.append("    max_num_corners: int = 40\n")
            elif "min_num_corners:" in line:
                new_content.append("    min_num_corners: int = 4\n")
            else:
                new_content.append(line)
    with open(eval_racing_cfg_path, 'w') as file:
        file.writelines(new_content)

modify_racing_config()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "rsl_rl_cfg_entry_point" if algorithm in ["ppo"] else f"rsl_rl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_run-{agent_cfg.run_name}"

    if "Single" in args_cli.task:
        robot_name = env.env.cfg.robot_name
        task_name = env.env.cfg.task_name
    elif "MultiTask" in args_cli.task:
        robot_name = env.env.cfg.robot_name
        task_name = "-".join(env.env.cfg.tasks_names)
    elif "DoubleTask" in args_cli.task:
        robot_name = env_cfg.robot_cfg.robot_name
        task_name = env_cfg.task_cfg_0.__class__.__name__[:-3] + "-" + env.env.cfg.task_cfg_1.__class__.__name__[:-3] # remove the last 3 characters "cfg"
    else:
        robot_name = env_cfg.robot_cfg.robot_name
        task_name = env_cfg.task_cfg.__class__.__name__[:-3] # remove the last 3 characters "cfg"

    log_dir += f"_rsl-rl_{algorithm}_{task_name}_{robot_name}_r-{args_cli.run_num}_seed-{agent_cfg.seed}"
    log_dir = os.path.join(log_root_path, log_dir)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
