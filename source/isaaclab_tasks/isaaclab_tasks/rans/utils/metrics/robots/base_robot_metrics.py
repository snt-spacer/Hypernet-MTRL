import torch

class AutoRegister:
    def __init_subclass__(cls, **kwargs):
        """Ensure each subclass gets its own independent registry."""
        super().__init_subclass__(**kwargs)
        cls._registry: dict[str, callable] = {}  # Unique for each subclass

        for name, value in cls.__dict__.items():
            # If an attribute is a function and has our marker, register it.
            if callable(value) and getattr(value, '_auto_register', False):
                cls._registry[name] = value

    @staticmethod
    def register(func: callable) -> callable:
        """Decorator that simply marks a function so that __init_subclass__
        knows it should be placed in the registry.
        """
        func._auto_register = True
        return func

    @classmethod
    def get_registered_methods(cls) -> dict[str, callable]:
        """Retrieve registered methods."""
        return cls._registry


class BaseRobotMetrics(AutoRegister):
    def __init__(self, folder_path: str, physics_dt: float, step_dt: float) -> None:
        self.folder_path = folder_path
        self.physics_dt = physics_dt
        self.step_dt = step_dt

    def generate_metrics(
            self, 
            trajectories: dict, 
            trajectories_masks: torch.Tensor, 
        ) -> None:

        self.trajectories = trajectories
        self.trajectories_masks = trajectories_masks

        for metric_fnc in self.get_registered_methods().values():
            metric_fnc(self)