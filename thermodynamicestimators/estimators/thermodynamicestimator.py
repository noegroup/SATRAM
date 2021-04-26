import torch

class ThermodynamicEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def residue(self, data):
        return NotImplemented