from . import BaseTaskPlots, Registerable
import torch

class TrackVelocitiesPlots(BaseTaskPlots, Registerable):
    def __init__(self, dfs: dict, labels: dict, env_info:dict, folder_path:list) -> None:
        super().__init__(dfs=dfs, labels=labels, env_info=env_info, folder_path=folder_path)

        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                keys_set.update(
                    key for key in df.columns if key.startswith("linear_velocity_error")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("lateral_velocity_error")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("angular_velocity_error")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("overshoot_linear_velocity")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("overshoot_lateral_velocity")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("overshoot_angular_velocity")
                )

        self.labels_to_plot = list(keys_set)

    def plot(self):
        for label_to_plot in self.labels_to_plot:
            self.boxplot(label_to_plot)