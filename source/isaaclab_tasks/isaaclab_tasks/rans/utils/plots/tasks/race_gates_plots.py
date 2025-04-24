from . import BaseTaskPlots, Registerable
import torch

class RaceGatesPlots(BaseTaskPlots, Registerable):
    def __init__(self):
        super().__init__()