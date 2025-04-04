from . import BaseRobotMetrics, Registerable

class FloatingPlatformMetrics(BaseRobotMetrics, Registerable):
    def __init__(self, folder_path: str, physics_dt: float, step_dt: float) -> None:
        super().__init__(folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt)
    
    @BaseRobotMetrics.register
    def some_metrics_here(self):
        print("[INFO][METRICS][ROBOT] Test metrics")