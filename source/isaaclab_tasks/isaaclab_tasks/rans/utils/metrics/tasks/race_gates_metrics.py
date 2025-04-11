from . import BaseTaskMetrics, Registerable
import torch

class RaceGatesMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name)

    @BaseTaskMetrics.register
    def num_goals_reached_in_fixed_steps(self):
        print("[INFO][METRICS][TASK] Number of gates reached in fixed steps")

        fix_num_steps = self.trajectories['target_index'].shape[1] // 2 #TODO: Hardcoded
        # fix_num_steps = torch.min(torch.sum(self.trajectories_masks.int(), dim=1))
        masked_target_index = (self.trajectories['target_index'] * self.trajectories_masks)[:, :fix_num_steps]
        max_target_index_idx = torch.argmax(masked_target_index, dim=1)
        
        env_idx = torch.arange(masked_target_index.shape[0], device=masked_target_index.device)
        max_target_reached = masked_target_index[env_idx, max_target_index_idx]

        self.metrics["num_gates_reached_in_{fix_num_steps}_steps"] = max_target_reached