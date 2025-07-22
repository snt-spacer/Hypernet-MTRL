import torch
from .tasks import TaskMetricsFactory
from .robots import RobotMetricsFactory

import pandas as pd
import os
import datetime
import yaml
import numpy as np

class EvalMetrics:
    def __init__(self, env, robot_name: str, task_name: str, folder_path: str, device: str = "cuda", num_runs_per_env: int = 1, task_index: int = 0):
        self._env = env
        self.device = device
        self.num_runs_per_env = num_runs_per_env
        self.task_name = task_name
        self.robot_name = robot_name
        self.save_path = folder_path
        self.task_index = task_index

        self.task_metrics_factory = TaskMetricsFactory.create(
            task_name, 
            env=self._env,
            folder_path=folder_path, 
            physics_dt=self._env.env.unwrapped.scene.physics_dt, 
            step_dt=self._env.env.unwrapped.robot_api._step_dt,
            task_name=self.task_name,
            task_index=self.task_index
            
        )
        self.robot_metrics_factory = RobotMetricsFactory.create(
            robot_name, 
            env=self._env,
            folder_path=folder_path,
            physics_dt=self._env.env.unwrapped.scene.physics_dt,
            step_dt=self._env.env.unwrapped.robot_api._step_dt,
            robot_name=self.robot_name
        )

    def convert_metrics_to_pd(self) -> pd.DataFrame:
        """Converts the metrics to pandas DataFrames"""
        metrics = self.task_metrics_factory.metrics | self.robot_metrics_factory.metrics

        data = {}
        for k,v in metrics.items():
            numpy_log = v.cpu().numpy()
            data[k] = numpy_log
        return pd.DataFrame(data)
    
    def save_env_info(self) -> None:
        """Saves the env info into a yaml file"""
        env_info = self.task_metrics_factory.env_info | self.robot_metrics_factory.env_info
        env_info["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        env_info["num_runs_per_env"] = self.num_runs_per_env

        save_path = os.path.join(self.save_path, "metrics", "env_info.yaml")
        # Ensure the metrics directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(env_info, f)

    def calculate_metrics(self, data: dict)->None:

        self.data = data

        print("[INFO] Processing trajectories...")
        truncated_data, dones_tensor = self.cutoff_indices_per_env()
        self.trajectory_lengths, self.extracted_trajectories = self.env_trajectory_extraction(truncated_data, dones_tensor)
        trajectories, trajectories_mask = self.pad_trajectories(self.trajectory_lengths, self.extracted_trajectories)
        
        print("[INFO] Evaluating metrics...")
        self.task_metrics_factory.generate_metrics(
            trajectories=trajectories, 
            trajectories_masks=trajectories_mask, 
        )
        self.task_metrics_factory.populate_env_info()
        self.robot_metrics_factory.generate_metrics(
            trajectories=trajectories, 
            trajectories_masks=trajectories_mask, 
        )
        self.robot_metrics_factory.populate_env_info()
        
        if "MultiTask" in self._env.unwrapped.__class__.__name__:
            # For multitask, use a simpler naming pattern without task lists
            # Extract the base model name from the save_path
            base_model_name = os.path.basename(self.save_path)
            base_model_list = base_model_name.split("_")
            base_model_list[4] = self.task_name
            name = "_".join(base_model_list)
            filename = f"{name}_metrics.csv"
            save_path = os.path.join(self.save_path, "metrics", filename)
        else:
            name = self.save_path.split("/")[-1]
            save_path = os.path.join(self.save_path, "metrics", f"{name}_metrics.csv")
        
        # Ensure the metrics directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df = self.convert_metrics_to_pd()
        df.to_csv(save_path, index=False)

        self.save_env_info()

    def cutoff_indices_per_env(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Calculates the cutoff indices for each environment based on the number of runs per environment.
            Args:
                data (dict): Data tensors for each environment.
            Returns:
                tuple[dict[str, torch.Tensor], torch.Tensor]: Truncated data tensors and a tensor indicating done steps.
        """
        all_dones = self.data["dones"]
        self.num_steps = all_dones.shape[0]
        self.num_envs = all_dones.shape[1]
        cumulative_dones = torch.cumsum(all_dones, dim=0)
        self.cutoff_indices = torch.full((self.num_envs,), fill_value=self.num_steps, dtype=torch.long, device=self.device)

        self.cutoff_indices = torch.full((self.num_envs,), fill_value=self.num_steps, dtype=torch.long, device=self.device)
        for i in range(self.num_envs):
            valid_indices = torch.where(cumulative_dones[:, i] >= self.num_runs_per_env)[0]
            if len(valid_indices) > 0:
                self.cutoff_indices[i] = valid_indices[0] # The step where target is met
            else:
                # If target is never met, the cutoff is the last possible step index
                self.cutoff_indices[i] = self.num_steps - 1

        max_cutoff_idx = torch.max(self.cutoff_indices).item()
        max_len = max_cutoff_idx + 1
        truncated_data = {k: v[:max_len] for k, v in self.data.items()}
        dones_tensor = all_dones[:max_len]

        return truncated_data, dones_tensor
    
    def env_trajectory_extraction(self, truncated_data: dict, dones_tensor: torch.Tensor)-> tuple[list, dict[str, list[torch.Tensor]]]:
        """Extracts trajectories from the data based on the cutoff indices.
            Args:
                truncated_data (dict): Data tensors truncated to the cutoff indices.
                dones_tensor (torch.Tensor): Tensor indicating done steps.
            Returns:
                tuple[list, dict[str, list[torch.Tensor]]]: List of trajectory lengths and a dictionary of extracted trajectories.
        """
        all_extracted_trajectories = {k: [] for k in truncated_data.keys()}
        all_trajectory_lengths = []
        final_num_valid_trajectories_per_env = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        for env_idx in range(self.num_envs):

            # Find all 'done' steps for this env within the considered range (0 to max_len-1)
            env_all_done_steps = torch.where(dones_tensor[:, env_idx])[0]

            # Filter done steps to only include those *at or before* this env's specific cutoff
            env_cutoff_step = self.cutoff_indices[env_idx]
            valid_done_steps = env_all_done_steps[env_all_done_steps <= env_cutoff_step]

            # Determine the end steps for the trajectories we will actually extract
            end_steps_for_env = []

            # Take the first num_runs_per_env completed trajectories
            end_steps_for_env = valid_done_steps[:self.num_runs_per_env]
            final_num_valid_trajectories_per_env[env_idx] = self.num_runs_per_env

            # Extract trajectories based on these end_steps
            last_end_step = -1
            for i, current_end_step in enumerate(end_steps_for_env):
                start_step = last_end_step + 1
                length = current_end_step - start_step 

                # If dones happen on consecutive steps
                if length <= 0:
                    continue

                all_trajectory_lengths.append(length.item()) # Store length as int

                # Extract data slice for this trajectory from each data tensor
                for key, data_tensor in truncated_data.items():
                    trajectory_slice = data_tensor[start_step : current_end_step, env_idx]
                    all_extracted_trajectories[key].append(trajectory_slice)

                last_end_step = current_end_step # Update for the next trajectory's start

        return all_trajectory_lengths, all_extracted_trajectories

    def save_extracted_trajectories_to_csv(self):
        """Saves the extracted trajectories to a CSV file in the metrics directory, one row per time step per trajectory. Vectorized for speed."""
        import numpy as np
        import pandas as pd
        import os

        save_path = os.path.join(self.save_path, "metrics", f"extracted_trajectories_{self.task_name}.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not hasattr(self, 'extracted_trajectories') or self.extracted_trajectories is None:
            print("[WARNING] No extracted_trajectories to save.")
            return

        dim_names = ['x', 'y', 'z']
        keys = list(self.extracted_trajectories.keys())
        if not keys:
            print("[WARNING] extracted_trajectories is empty.")
            return

        if len(self.extracted_trajectories[keys[0]]) > 64:
            num_trajectories = 64
        else:
            num_trajectories = len(self.extracted_trajectories[keys[0]])
        all_dfs = []
        for traj_idx in range(num_trajectories):
            # Find the length of this trajectory (use the first key)
            first_tensor = self.extracted_trajectories[keys[0]][traj_idx]
            if not (hasattr(first_tensor, 'shape') and hasattr(first_tensor, '__getitem__')):
                print(f"[WARNING] Trajectory {traj_idx} first key is not array-like, skipping.")
                continue
            traj_len = first_tensor.shape[0]
            data = {
                'trajectory': np.full(traj_len, traj_idx, dtype=int),
                'step': np.arange(traj_len, dtype=int)
            }
            for key in keys:
                tensor = self.extracted_trajectories[key][traj_idx]
                arr = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else np.array(tensor)
                if arr.ndim == 1:
                    data[key] = arr
                elif arr.ndim == 2:
                    for d in range(arr.shape[1]):
                        dim_label = dim_names[d] if arr.shape[1] <= 3 else str(d)
                        data[f"{key}_{dim_label}"] = arr[:, d]
                else:
                    # flatten higher dims
                    flat = arr.reshape(arr.shape[0], -1)
                    for d in range(flat.shape[1]):
                        data[f"{key}_{d}"] = flat[:, d]
            df_traj = pd.DataFrame(data)
            all_dfs.append(df_traj)
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
            df.to_csv(save_path, index=False, float_format='%.4f')
            print(f"[INFO] Saved extracted trajectories to {save_path}")
        else:
            print("[WARNING] No valid trajectories to save.")

    def pad_trajectories(self, all_trajectory_lengths: list, all_extracted_trajectories:dict[str, list[torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Pads the extracted trajectories to the same length.
            Args:
                all_trajectory_lengths (torch.Tensor): Lengths of all trajectories.
                all_extracted_trajectories (dict[str, list[torch.Tensor]]): Extracted trajectories.
            Returns:
                tuple[dict[str, torch.Tensor], torch.Tensor]: Padded trajectories and their mask.
        """
        extracted_trajectories = {}
        max_traj_len = max(all_trajectory_lengths) if all_trajectory_lengths else 0

        for key, traj_list in all_extracted_trajectories.items():

            padding_value = 0.0 if traj_list[0].dtype == torch.float32 else 0

            padded_tensor = torch.nn.utils.rnn.pad_sequence(
                traj_list,
                batch_first=True,
                padding_value=padding_value
            )
            extracted_trajectories[key] = padded_tensor

        traj_lengths_tensor = torch.tensor(all_trajectory_lengths, dtype=torch.long, device=self.device)
        traj_range = torch.arange(max_traj_len, device=self.device)
        trajectories_padding_mask = traj_range[None, :] < traj_lengths_tensor[:, None]

        return extracted_trajectories, trajectories_padding_mask
