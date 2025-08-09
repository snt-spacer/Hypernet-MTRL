from . import BaseTaskMetrics, Registerable
import torch

class RendezvousMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env=env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    @BaseTaskMetrics.register
    def avg_orientation_error(self):
        print("[INFO][METRICS][TASK] Average orientation error")
        mean_avg_angle_errors = []
        failed_trajectories = 0
        for traj_idx, trajectory_target_index in enumerate(self.trajectories["target_index"]):
            cut_idx = torch.where(trajectory_target_index > 6)[0]

            if cut_idx.numel() > 1:
                cut_idx = cut_idx[0].unsqueeze(-1)  # Use the first index if there are multiple 
            if cut_idx.numel() == 0:
                failed_trajectories += 1
                continue  # Skip if no index found

            diffs = torch.diff(trajectory_target_index[:cut_idx.item()])
            increment_indices = torch.where(diffs > 0)[0] + 1
            increment_indices = torch.cat([increment_indices, cut_idx])

            headings = torch.tensor_split(self.trajectories["heading"][traj_idx].cpu(), increment_indices.cpu())
            target_headings = self.trajectories["target_headings"][traj_idx]

            avg_angle_errors = []
            for segment_idx, (heading, target_heading) in enumerate(zip(headings, target_headings)):
                heading_error = target_heading[segment_idx].repeat(heading.shape[0]).cpu() - heading
                angle_error = torch.abs(torch.arctan2(torch.sin(heading_error), torch.cos(heading_error)))  # Normalize the angle error to [-pi, pi] and take the absolute value
                avg_angle_errors.append(torch.mean(angle_error.float()))
            
            mean_avg_angle_errors.append(torch.mean(torch.stack(avg_angle_errors)))

        for i in range(failed_trajectories):
            mean_avg_angle_errors.append(torch.tensor(float('nan'), device=self.trajectories["target_index"].device))

        self.metrics["mean_orientation_error.rad"] = torch.tensor(mean_avg_angle_errors, device=self.trajectories["target_index"].device)
        self.env_info["failed_trajectories"] = torch.tensor(failed_trajectories, device=self.trajectories["target_index"].device)

    @BaseTaskMetrics.register
    def avg_orientation_error_path(self):
        """Calculate the average orientation error along the path after the robot arrived to the first marker."""
        print("[INFO][METRICS][TASK] Average orientation error path")
        mean_avg_angle_errors = []
        failed_trajectories = 0
        for traj_idx, trajectory_target_index in enumerate(self.trajectories["target_index"]):
            cut_idx = torch.where(trajectory_target_index > 6)[0]

            if cut_idx.numel() > 1:
                cut_idx = cut_idx[0].unsqueeze(-1)  # Use the first index if there are multiple 
            if cut_idx.numel() == 0:
                failed_trajectories += 1
                continue  # Skip if no index found

            diffs = torch.diff(trajectory_target_index[:cut_idx.item()])
            increment_indices = torch.where(diffs > 0)[0] + 1
            first_increment = increment_indices[0]
            increment_indices = increment_indices[1:] - first_increment
            increment_indices = torch.cat([increment_indices, cut_idx - first_increment])

            headings = torch.tensor_split(self.trajectories["heading"][traj_idx][first_increment:].cpu(), increment_indices.cpu())
            target_headings = self.trajectories["target_headings"][traj_idx][increment_indices[0]:]

            avg_angle_errors = []
            for segment_idx, (heading, target_heading) in enumerate(zip(headings, target_headings)):
                heading_error = target_heading[segment_idx].repeat(heading.shape[0]).cpu() - heading
                angle_error = torch.abs(torch.arctan2(torch.sin(heading_error), torch.cos(heading_error)))  # Normalize the angle error to [-pi, pi] and take the absolute value
                avg_angle_errors.append(torch.mean(angle_error.float()))

            if torch.any(torch.isnan(torch.stack(avg_angle_errors))):
                breakpoint()
            mean_avg_angle_errors.append(torch.mean(torch.stack(avg_angle_errors)))

        for i in range(failed_trajectories):
            mean_avg_angle_errors.append(torch.tensor(float('nan'), device=self.trajectories["target_index"].device))

        self.metrics["mean_orientation_error_path.rad"] = torch.tensor(mean_avg_angle_errors, device=self.trajectories["target_index"].device)