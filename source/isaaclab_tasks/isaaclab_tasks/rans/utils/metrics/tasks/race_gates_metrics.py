from . import BaseTaskMetrics, Registerable
import torch

class RaceGatesMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    @BaseTaskMetrics.register
    def num_goals_reached_in_fixed_steps(self):
        print("[INFO][METRICS][TASK] Number of gates reached in fixed steps")

        breakpoint()