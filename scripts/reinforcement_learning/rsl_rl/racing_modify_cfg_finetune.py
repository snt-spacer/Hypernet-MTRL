import argparse
import sys
import os

parser = argparse.ArgumentParser(description="Modify cfg an RL agent with RSL-RL.")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file.")
args = parser.parse_args()


def modify_racing_rsl_rl_config():
        # Modify dynamically the task configuration. IsaacLab do something about this.
    new_content = []
    eval_racing_cfg_path = os.path.normpath(os.path.join(os.getcwd(), "source", "isaaclab_tasks", "isaaclab_tasks", "rans", "environments", "single", "agents", "rsl_rl_ppo_cfg.py"))
    
    load_run = args.checkpoint_path.split('/')[0]
    load_checkpoint = args.checkpoint_path.split('/')[1]
    with open(eval_racing_cfg_path, 'r') as file:
        for line in file:
            if "resume" in line:
                new_content.append("    resume = True\n")
            elif "max_iterations" in line:
                new_content.append(f"    max_iterations = 1000\n")
            elif "save_interval" in line:
                new_content.append(f"    save_interval = 10\n")
            elif "load_run" in line:
                new_content.append(f"    load_run = \"{load_run}\"\n")
            elif "load_checkpoint" in line:
                new_content.append(f"    load_checkpoint = \"{load_checkpoint}\"\n")
            else:
                new_content.append(line)
    with open(eval_racing_cfg_path, 'w') as file:
        file.writelines(new_content)

if __name__ == "__main__":
    modify_racing_rsl_rl_config()