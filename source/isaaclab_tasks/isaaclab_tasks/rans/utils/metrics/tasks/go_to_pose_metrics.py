from . import BaseTaskMetrics, Registerable
import torch

class GoToPoseMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name)

    def get_reached_idx(self) -> torch.Tensor:
        # Position threshold
        pos_threshold = self.env.unwrapped.task_api._task_cfg.position_tolerance
        masked_distances = self.trajectories['position_distance'] * self.trajectories_masks
        pos_reached_threshold = masked_distances <= pos_threshold

        # Heading threshold
        heading_threshold = self.env.unwrapped.task_api._task_cfg.heading_tolerance
        heading_error = torch.arctan2(
            torch.sin(self.trajectories['target_headings'] - self.trajectories['heading']),
            torch.cos(self.trajectories['target_headings'] - self.trajectories['heading']),
        ) * self.trajectories_masks
        heading_reached_threshold = torch.abs(heading_error) <= heading_threshold

        # Combine position and heading thresholds
        reached_threshold = pos_reached_threshold & heading_reached_threshold

        # First index where the threshold is reached
        len_trajec = self.trajectories['position_distance'].shape[1] # argmax returns 0 if all values are false
        reached_idx = torch.argmax(reached_threshold.int(), dim=1)
        all_false = ~torch.any(reached_threshold, dim=1)
        reached_idx[all_false] = len_trajec

        return reached_idx
    
    @BaseTaskMetrics.register
    def time_to_reach_position_threshold(self):
        print("[INFO][METRICS][TASK] Time to reach position threshold")
        reached_idx = self.get_reached_idx()
        episode_length_in_s = reached_idx * self.step_dt
        self.metrics[f"{self.task_name}/time_to_reach_position_threshold"] = episode_length_in_s


    @BaseTaskMetrics.register
    def final_position(self):
        print("[INFO][METRICS][TASK] Final position")
        masked_distances = self.trajectories['position'] * self.trajectories_masks.unsqueeze(-1)
        final_positions = torch.stack([row[index.unsqueeze(-1) - 1][-1] for row, index in zip(masked_distances, self.last_true_index)])
        
        self.metrics[f"{self.task_name}/final_position_x"] = final_positions[:, 0]
        self.metrics[f"{self.task_name}/final_position_y"] = final_positions[:, 1]
        self.metrics[f"{self.task_name}/final_position_z"] = final_positions[:, 2]

    @BaseTaskMetrics.register
    def final_orientation(self):
        print("[INFO][METRICS][TASK] Final orientation")
        masked_heading = self.trajectories['heading'] * self.trajectories_masks
        final_orientations = torch.stack([row[index.unsqueeze(-1) - 1][-1] for row, index in zip(masked_heading, self.last_true_index)])
        self.metrics[f"{self.task_name}/final_orientation"] = final_orientations

    @BaseTaskMetrics.register
    def convergence_time(self):
        #TODO: how diff from time_to_reach_threshold?
        pass

    @BaseTaskMetrics.register
    def position_overshoot(self):
        print("[INFO][METRICS][TASK] Position overshoot")
        threshold = 0.1 #self.env.unwrapped.task_api._task_cfg.position_tolerance
        masked_distances = self.trajectories['position_distance'] * self.trajectories_masks
        reached_threshold = masked_distances <= threshold
        
        len_trajec = self.trajectories['position_distance'].shape[1] # argmax returns 0 if all values are false
        reached_idx = torch.argmax(reached_threshold.int(), dim=1)
        all_false = ~torch.any(reached_threshold, dim=1)
        reached_idx[all_false] = len_trajec - 1

        overshoot_idx = self.get_indx_of_n_true_values(num_consecutive=5, bool_tensor=reached_threshold, init_indx=reached_idx) #TODO: Hardcoded 5
        overshoot_idx = torch.where(overshoot_idx == -1, 0, overshoot_idx)
        overshoot_num_steps = overshoot_idx - reached_idx 

        self.metrics[f"{self.task_name}/num_steps_position_overshoot"] = overshoot_num_steps
        # Calculate the overshoot distance
        indx = torch.arange(masked_distances.shape[0], device=masked_distances.device)
        self.metrics[f"{self.task_name}/overshoot_distance_error"] = masked_distances[indx, overshoot_idx] - masked_distances[indx, reached_idx]
        

    @BaseTaskMetrics.register
    def trajectory_efficiency(self):
        print("[INFO][METRICS][TASK] Trajectory efficiency")
        # Calculate path
        masked_positions = self.trajectories['position'] * self.trajectories_masks.unsqueeze(-1)
        diff_positions = masked_positions[:, 1:] - masked_positions[:, :-1]
        segment_lengths = torch.linalg.vector_norm(diff_positions, dim=-1)
        total_distance = torch.sum(segment_lengths, dim=1)

        # Calculate shortest path
        masked_targets = self.trajectories['target_position'] * self.trajectories_masks.unsqueeze(-1)
        start_positions = masked_positions[:, 0]
        final_targets = masked_targets[:, 0]
        shortest_path = torch.linalg.vector_norm(final_targets - start_positions[:, :2], dim=-1)

        efficiency = shortest_path / total_distance
        self.metrics[f"{self.task_name}/trajectory_efficiency"] = efficiency

    @BaseTaskMetrics.register
    def jerckiness(self):
        print("[INFO][METRICS][ROBOT] Jerkiness")
        last_n_steps = 20
        masked_pos_dist = self.trajectories['position_distance'] * self.trajectories_masks
        last_pos = torch.stack([row[start_idx - last_n_steps: start_idx] for row, start_idx in zip(masked_pos_dist, self.last_true_index)])

        vel = last_pos[:, 1:] - last_pos[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        jerk = torch.mean(acc[:, 1:] - acc[:, :-1], dim=1)

        self.metrics[f"{self.task_name}/mean_last_{last_n_steps}_steps_jerk"] = jerk