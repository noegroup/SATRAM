import math
from thermodynamicestimators.estimators.thermodynamicestimator import ThermodynamicEstimator
import numpy as np
import torch
from functools import reduce



class WHAM(ThermodynamicEstimator):
    def __init__(self,  bias_coefficients, n_bins):
        super().__init__()

        assert torch.tensor(bias_coefficients).shape[1:] == n_bins

        self.n_bins = n_bins

        self.bias_coefficients = torch.tensor(bias_coefficients)
        self.n_biases = self.bias_coefficients.shape[0]

        if isinstance(n_bins, tuple) or isinstance(n_bins, list):
            self.total_histogram_bins = reduce(lambda x, y: x*y, n_bins)
        else:
            self.total_histogram_bins = n_bins

        self.g = torch.nn.Parameter(torch.ones(self.n_biases))



    # free energy estimate per bias
    @property
    def free_energy(self):
        # return 1 / torch.sum(self.bias_coefficients * self.probabilities, axis=1)
        return torch.exp(self.g)


    # estimated potential energy function
    def get_potential(self, data):
        # return -torch.log(self.probabilities)
        M = torch.sum(data, axis=0)  # total count per histogram bin summed over all simulations
        N = torch.sum(data.view(data.shape[0], self.total_histogram_bins), axis=1)  # total samples per simulations
        return - torch.log(M / torch.sum(N * self.free_energy * self.bias_coefficients.T, axis=-1).T).T


    # compute the loss function for gradient descent
    # data has shape: (n_simulations, n_bins)
    def residue(self, data):

        N = torch.sum(data.view(data.shape[0], self.total_histogram_bins), axis=1)  # total samples per simulations
        M = torch.sum(data, axis=0)  # total count per histogram bin summed over all simulations

        # small epsilon value to avoid taking the log of zero
        eps = 1e-10
        log_val = torch.log((M + eps) / torch.sum(eps + N * self.bias_coefficients.T * torch.exp(self.g), axis=-1).T)

        log_likelihood = torch.sum(N * self.g) + \
                         torch.sum(M * log_val)

        return - log_likelihood
