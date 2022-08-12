import torch
import math
from torch.utils.data import RandomSampler, SequentialSampler


def _compute_max_batch_size(datarow):
    # TODO: implement
    return 8192


class Dataset:

    def __init__(self, data, state_counts, transition_counts=None, device='cpu', batch_size=256, is_stochastic=False):
        self._dataloader = None
        self._data = data

        self.device = device

        self.state_counts = state_counts.to(device)
        self.transition_counts = transition_counts.to(device)
        self.log_N = torch.log(state_counts.sum())

        self.n_therm_states = state_counts.shape[0]
        self.n_markov_states = state_counts.shape[1]

        N_k = torch.sum(state_counts, 1)
        self.N_k_log = torch.log(N_k).to(device)
        self.normalized_N_k = (N_k / N_k.sum()).to(device)

        if transition_counts is not None:
            self.log_C_sym = torch.log(transition_counts + torch.transpose(transition_counts, 1, 2)).to(device)
            diag = torch.diagonal(self.log_C_sym, dim1=1, dim2=2)
            diag -= math.log(2.)

        self.max_batch_size = min(_compute_max_batch_size(data[0]), len(data))
        self.init_dataloader(batch_size, is_stochastic)


    @property
    def dataloader(self):
        return self._dataloader


    @property
    def deterministic_dataloader(self):
        return torch.utils.data.DataLoader(self._data, batch_size=self.max_batch_size, drop_last=False, shuffle=False)


    def init_dataloader(self, batch_size, is_stochastic):
        batch_size = min(self.max_batch_size, batch_size)
        batch_size = batch_size if is_stochastic else self.max_batch_size
        if is_stochastic:
            sampler = RandomSampler(self._data, replacement=True)
        else:
            sampler = SequentialSampler(self._data)
        self._dataloader = torch.utils.data.DataLoader(dataset=self._data, sampler=sampler, batch_size=batch_size)
