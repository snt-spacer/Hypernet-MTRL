from . import BaseTaskMetrics, Registerable
import torch

class GoThroughPositionsMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, folder_path: str, physics_dt: float, step_dt: float) -> None:
        super().__init__(folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt)
    
    @BaseTaskMetrics.register
    def time_to_reach_goal(self):

        print("[INFO][METRICS] Time to reach goal")

    @BaseTaskMetrics.register
    def num_goals_reached(self):
        print("[INFO][METRICS] Number of goals reached")
        breakpoint()
        position_dist = self.trajectories['position_distance'] < 0.1
        num_goals_reached = torch.cumsum(position_dist, dim=1)
        
        breakpoint()
        self.trajectories
        self.trajectories_masks
        torch.where(self.trajectories['trajectory_completed'] >0)
        self.trajectories['position'][0][:,:1]
        self.trajectories['position'][1][:,:1]

        self.trajectories['position_distance'][0][:,:1]

    @BaseTaskMetrics.register
    def action_rate(self):
        print("[INFO][METRICS] Action rate")

    @BaseTaskMetrics.register
    def jerkiness(self):
        print("[INFO][METRICS] Jerkiness")

    @BaseTaskMetrics.register
    def energy_consumption(self):
        print("[INFO][METRICS] Energy consumption")