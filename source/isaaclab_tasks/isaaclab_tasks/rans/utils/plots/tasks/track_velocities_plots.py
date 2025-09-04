from . import BaseTaskPlots, Registerable
import torch

class TrackVelocitiesPlots(BaseTaskPlots, Registerable):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info:dict, folder_path:list, plot_cfg:dict) -> None:
        super().__init__(dfs=dfs, trajectories_dfs=trajectories_dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        self.task_name = "track_velocities"

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
                # keys_set.update(
                #     key for key in df.columns if key.startswith("overshoot_linear_velocity")
                # )
                # keys_set.update(
                #     key for key in df.columns if key.startswith("overshoot_lateral_velocity")
                # )
                # keys_set.update(
                #     key for key in df.columns if key.startswith("overshoot_angular_velocity")
                # )

        self.labels_to_plot = list(keys_set)

    def plot(self):
        for label_to_plot in self.labels_to_plot:
            self.boxplot(label_to_plot)

        # if len(self._trajectories_dfs) > 0:
        #     self.plot_linear_velocity_over_time()
        #     self.plot_angular_velocity_over_time()
        #     self.plot_linear_velocity_error_over_time()
        #     self.plot_angular_velocity_error_over_time()
        #     self.plot_actions_over_time()