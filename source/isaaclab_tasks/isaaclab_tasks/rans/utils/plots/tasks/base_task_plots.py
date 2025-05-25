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
    def __init__(self, dfs: dict, labels: dict, env_info: dict, folder_path: list, plot_cfg:dict) -> None:
        self._dfs = dfs
        self._labels = labels
        self._env_info = env_info
        self._save_plots_folder_path = folder_path
        self._plot_cfg = plot_cfg

    def plot(self):
        raise NotImplementedError("Subclasses should implement this method.")
    

    def boxplot(self, key_to_plot: str):
        key_name, units = key_to_plot.split(".")

        fig, ax = plt.subplots(figsize=(14, 6))

        data_to_plot = []
        label_names = []

        
        colors_indx = []
        for group_key, group_dfs in self._dfs.items():
            # Concatenate all values of the key across the group's dataframes
            # for group_idx, df in enumerate(group_dfs):
            #     print("#" * 20)
            #     print(f"Group {group_idx}: {df.columns}")

            try:
                group_values = pd.concat([df[key_to_plot] for df in group_dfs], ignore_index=True)
            except KeyError as e:
                print(f"KeyError: {e}. The key '{key_to_plot}' does not exist in the DataFrames.")
                print(self.__class__.__name__)
                return
            
            colors_indx.append(self._plot_cfg['runs_names'].index(group_key.split("_")[-1]))
            
            data_to_plot.append(group_values)
            label_names.append(group_key.split("_")[-1])

        box = ax.boxplot(
            data_to_plot,
            labels=label_names,
            patch_artist=True,
            boxprops=dict(facecolor='skyblue'),
            medianprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=5),
            # showfliers=False,
            widths=0.6,
        )

        if self._plot_cfg["zoom_in"]:
            all_values = pd.concat(data_to_plot)
            ax.set_ylim(top=all_values.quantile(0.95) )

        # Set the color of the boxes
        colors = [self._plot_cfg["box_colors"][index] for index in colors_indx]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        y_label, units = key_to_plot.split(".")
        y_label = y_label.replace("_", " ").capitalize()
        ax.set_title(f"{y_label}")
        ax.set_ylabel(f"{y_label} ({units})")
        ax.set_xticks(range(1, len(label_names) + 1))
        ax.set_xticklabels(label_names, rotation=20, ha='right')
        ax.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_{key_name}.svg")
        plt.savefig(save_path)
        plt.close()