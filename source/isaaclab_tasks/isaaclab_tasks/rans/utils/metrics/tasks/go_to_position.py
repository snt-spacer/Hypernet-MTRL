from . import BaseTaskMetrics, Registerable
import torch

class GoToPositionMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, folder_path: str, physics_dt: float, step_dt: float) -> None:
        super().__init__(folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt)
    
    @BaseTaskMetrics.register
    def final_position(self):

        print("[INFO][METRICS] Time to reach goal")

    @BaseTaskMetrics.register
    def time_to_reach_threshold(self):
        print("[INFO][METRICS] Time to reach threshold")
        breakpoint()
        # torch.where(self.trajectories['position_distance'] < )

        self.trajectories['target_position'][0][0]
        self.trajectories['position'][0][-1][:2]
        self.trajectories['position_distance'][0][-1]