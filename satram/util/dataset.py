import torch


class Dataset:
    def __init__(self, data, state_counts, transition_counts = None, device='cpu', batch_size=256, is_stochastic=False):

        self._dataloader = None
        self._data = data

        self.device=device

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

        self.init_dataloader(batch_size, is_stochastic)
    
    @property
    def dataloader(self):
        return self._dataloader


    def init_dataloader(self, batch_size, is_stochastic):
        self._dataloader = torch.utils.data.DataLoader(self._data, batch_size=batch_size,
                                                       drop_last=is_stochastic, shuffle=is_stochastic)