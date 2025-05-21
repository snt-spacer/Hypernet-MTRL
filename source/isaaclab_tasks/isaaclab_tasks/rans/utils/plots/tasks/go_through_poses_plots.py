from . import BaseTaskPlots, Registerable
import torch

class GoThroughPosesPlots(BaseTaskPlots, Registerable):
    def __init__(self, dfs: dict, labels: dict, env_info:dict, folder_path:list, plot_cfg:dict) -> None:
        super().__init__(dfs=dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)