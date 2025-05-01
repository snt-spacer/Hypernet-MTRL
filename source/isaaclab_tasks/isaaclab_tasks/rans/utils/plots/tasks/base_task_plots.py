import matplotlib.pyplot as plt
import pandas as pd
import os

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


class BaseTaskPlots(AutoRegister):
    def __init__(self, dfs: dict, labels: dict, env_info: dict, folder_path: list) -> None:
        self._dfs = dfs
        self._labels = labels
        self._env_info = env_info
        self._save_plots_folder_path = folder_path

    def plot(self):
        raise NotImplementedError("Subclasses should implement this method.")
    

    def boxplot(self, key_to_plot: str):
        key_name, units = key_to_plot.split(".")

        fig, ax = plt.subplots(figsize=(14, 6))

        data_to_plot = []
        label_names = []

        for group_key, group_dfs in self._dfs.items():
            # Concatenate all values of the key across the group's dataframes
            group_values = pd.concat([df[key_to_plot] for df in group_dfs], ignore_index=True)
            data_to_plot.append(group_values)
            label_names.append(group_key)

        ax.boxplot(
            data_to_plot,
            labels=label_names,
            patch_artist=True,
            boxprops=dict(facecolor='skyblue'),
            medianprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=5),
            widths=0.6,
        )

        ax.set_title(f"Boxplots of {key_to_plot}")
        ax.set_ylabel(key_to_plot)
        ax.set_xticks(range(1, len(label_names) + 1))
        ax.set_xticklabels(label_names, rotation=20, ha='right')
        ax.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_{key_name}.svg")
        plt.savefig(save_path)
        plt.close()