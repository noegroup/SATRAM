from thermodynamicestimators.estimators.thermodynamicestimator import ThermodynamicEstimator
import torch
from functools import reduce



class WHAM(ThermodynamicEstimator):
    def __init__(self,  dataset):
        super().__init__()

        self.bias_coefficients = torch.tensor(dataset.bias_coefficients)
        self.n_biases = self.bias_coefficients.shape[0]

        assert torch.tensor(self.bias_coefficients).shape[1:] == dataset.histogram_shape

        if isinstance(dataset.histogram_shape, tuple) or isinstance(dataset.histogram_shape, list):
            self.total_histogram_bins = reduce(lambda x, y: x*y, dataset.histogram_shape)
        else:
            self.total_histogram_bins = dataset.histogram_shape

        self.free_energy_log = torch.nn.Parameter(torch.zeros(self.n_biases))


    # free energy estimate per thermodynamic state
    @property
    def free_energy(self):
        return torch.exp(self.free_energy_log.detach())


    # estimated potential energy function based on observed data
    def get_potential(self, data):
        N_bin = torch.sum(data, axis=0)  # total count per histogram bin summed over all simulations
        N_state = torch.sum(data.view(data.shape[0], self.total_histogram_bins), axis=1)  # total samples per thermodynamic state
        return - torch.log(N_bin / torch.sum(N_state * self.free_energy * self.bias_coefficients.T, axis=-1).T).T


    ''' compute the loss function for gradient descent
     data has shape: (M, b1, b2, ...) where M is the number of thermodynamic states, and b1, b2,... are the number
     of histogram bins in each dimensional axis.'''
    def residue(self, data):
        N_state = torch.sum(data.view(data.shape[0], self.total_histogram_bins), axis=1)  # total samples per thermodynamic state
        N_bin = torch.sum(data, axis=0)  # total count per histogram bin summed over all simulations

        # small epsilon value to avoid taking the log of zero
        eps = 1e-10
        log_val = torch.log((N_bin + eps) / torch.sum(eps + N_state * self.bias_coefficients.T * torch.exp(self.free_energy_log), axis=-1).T)

        log_likelihood = torch.sum(N_state * self.free_energy_log) + \
                         torch.sum(N_bin * log_val)

        return - log_likelihood
