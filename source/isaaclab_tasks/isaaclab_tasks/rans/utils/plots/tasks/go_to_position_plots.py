from . import BaseTaskPlots, Registerable
import matplotlib.pyplot as plt

class GoToPositionPlots(BaseTaskPlots, Registerable):
    def __init__(self, dfs: dict, labels: dict, env_info:dict, folder_path:list, plot_cfg:dict) -> None:
        super().__init__(dfs=dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        self.task_name = "go_to_position"

        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                keys_set.update(
                    key for key in df.columns if "_steps_jerk" in key
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("time_to_reach_position_threshold")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("num_steps_position_overshoot")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("overshoot_distance_error")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("trajectory_efficiency")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("final_position_distance")
                )

        self.labels_to_plot = list(keys_set)

    def plot(self):
        for label_to_plot in self.labels_to_plot:
            self.boxplot(label_to_plot)