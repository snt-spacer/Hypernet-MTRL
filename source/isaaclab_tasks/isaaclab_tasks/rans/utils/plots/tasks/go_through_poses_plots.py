from . import BaseTaskPlots, Registerable
import torch

class GoThroughPosesPlots(BaseTaskPlots, Registerable):
    def __init__(self):
        super().__init__()