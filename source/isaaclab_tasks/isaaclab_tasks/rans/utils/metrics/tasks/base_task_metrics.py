import torch

class AutoRegister:
    def __init_subclass__(cls, **kwargs):
        """Ensure each subclass gets its own independent registry."""
        super().__init_subclass__(**kwargs)

        # Unique for each subclass + inherit from parent class
        cls._registry = getattr(super(cls, cls), '_registry', {}).copy()

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


class BaseTaskMetrics(AutoRegister):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name:str) -> None:
        self.env = env
        self.folder_path = folder_path
        self.physics_dt = physics_dt
        self.step_dt = step_dt
        self.task_name = task_name

        self.metrics = {}

    def generate_metrics(
            self, 
            trajectories: dict, 
            trajectories_masks: torch.Tensor, 
        ) -> None:

        self.trajectories = trajectories
        self.trajectories_masks = trajectories_masks

        for metric_fnc in self.get_registered_methods().values():
            metric_fnc(self)

    @property
    def last_true_index(self) -> torch.Tensor:
        """Returns the last true index of a masked tensor along the first dimension.
            Args:
                masked_tensor (torch.Tensor): The tensor to find the last true index for.
            Returns:
                torch.Tensor: A tensor containing the last true indices for each row.
        """
        traj_len = self.trajectories_masks.shape[1]
        last_true_idx = torch.argmax(~self.trajectories_masks.int(), dim=1)
        all_true = torch.all(self.trajectories_masks, dim=1)
        last_true_idx[all_true] = traj_len
        return last_true_idx

    def get_indx_of_n_true_values(self, num_consecutive: int, bool_tensor: torch.Tensor, init_indx: torch.Tensor) -> torch.Tensor:
        """ Check that there are N consecutive True values and return the index of the first one
            Args:
                num_consecutive (int): Number of consecutive True values to check for.
                bool_tensor (torch.Tensor): The tensor to check for consecutive True values.
                init_indx (torch.Tensor): The initial indexs to start checking from.
            Returns:
                torch.Tensor: A tensor containing the index of the first occurrence of N consecutive True values.
        """
        len_trajec = bool_tensor.shape[1]
        results = torch.zeros(bool_tensor.shape[0], dtype=torch.int64, device=bool_tensor.device)
        for row_idx, row in enumerate(bool_tensor):
            start_idx = init_indx[row_idx].item()
            if start_idx == len_trajec:
                results[row_idx] = len_trajec - 1
                continue

            found = False
            for i in range(start_idx, len(row) - num_consecutive + 1):
                if all(row[i + j] for j in range(num_consecutive)):
                    results[row_idx] = i
                    found = True
                    break
            if not found:
                results[row_idx] = -1

        return results