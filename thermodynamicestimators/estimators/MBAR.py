from thermodynamicestimators.estimators.thermodynamicestimator import ThermodynamicEstimator
import torch
from functools import reduce



class MBAR(ThermodynamicEstimator):
    def __init__(self, n_states):
        super().__init__()
        self.n_states = n_states
        self.b = torch.nn.Parameter(torch.ones(n_states))


    # free energy estimate per bias
    @property
    def free_energy(self):
        return - self.b


    # compute the loss function for gradient descent
    # data has shape: (n_simulations, observed_potentials)
    def residue(self, data):
        N = data.numel()
        N_i = data.shape[2]

        eps = 10e-2
        log_values = torch.log(eps + torch.sum(torch.exp(-(data + (torch.ones_like(data.T) * self.b).T)), axis=0))

        log_likelihood = 1/N * torch.sum(log_values) + torch.sum(self.b * N_i/N)

        return - log_likelihood
