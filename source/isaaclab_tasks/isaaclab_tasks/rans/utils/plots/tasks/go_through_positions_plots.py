from . import BaseTaskPlots, Registerable
import torch

class GoThroughPositionsPlots(BaseTaskPlots, Registerable):
    def __init__(self, dfs: dict, labels: dict, env_info:dict, folder_path:list, plot_cfg:dict) -> None:
        super().__init__(dfs=dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                keys_set.update(
                    key for key in df.columns if "num_goals_reached_in_" in key
                )
                for i in range(env_info["num_goals"]):
                    label = f"time_to_reach_goal_num_{i}"
                    keys_set.update(
                        key for key in df.columns if label in key
                    )
                

        self.labels_to_plot = list(keys_set)

    def plot(self):
        for label_to_plot in self.labels_to_plot:
            self.boxplot(label_to_plot)