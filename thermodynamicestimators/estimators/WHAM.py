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

        self._free_energy_log = torch.nn.Parameter(torch.zeros(self.n_biases))


    ''' free energy estimate per thermodynamic state '''
    @property
    def free_energy(self):
        return torch.exp(self._free_energy_log.detach())


    ''' Subtract the first free energy from all free energies such that the first is zero and all other energies are
    relative to the first.
    For WHAM, we optimize for the log of the free energy so we divide by the first free energy in stead of subtracting it.'''
    def shift_free_energies_relative_to_zero(self):
        with torch.no_grad():
            self._free_energy_log -= self._free_energy_log[0].clone()


    ''' Takes input histogram or list of histograms and sums and normalizes these, returning the normalized sample 
    count per state and per bin. '''
    #TODO: move this somewhere else. WHAM should just recieve one histogram, not make one.
    def to_normalized_sample_counts(self, data):
        # check if this is one big histogram, or if its a tensor with a histogram for each sample
        if len(data.shape) - len(self.bias_coefficients.shape) > 0:
            # sum over the separate histograms to form one histogram per state. Data is now of shape (M, b1, b2,...)
            data = torch.sum(data, axis=0)

        N_bin = torch.sum(data, axis=0)  # total count per histogram bin summed over all simulations
        N_state = torch.sum(data, axis=1)  # total samples per thermodynamic state

        return N_bin, N_state


    ''' estimated potential energy function based on observed data '''
    def get_potential(self, data):
        N_bin, N_state = self.to_normalized_sample_counts(data)
        return - torch.log(N_bin / torch.sum(N_state * self.free_energy * self.bias_coefficients.T, axis=-1).T).T


    ''' compute the loss function for gradient descent
     data has shape: (N, M, b1, b2, ...) where N is the number of samples, M is the number of thermodynamic states, 
     and b1, b2,... are the number of histogram bins in each dimensional axis.
     One sample consists of M datapoints, one from each thermodynamic state, '''
    def residue(self, data):

        N_bin, N_state = self.to_normalized_sample_counts(data)

        # small epsilon value to avoid taking the log of zero
        eps = 1e-10
        log_val = torch.log((N_bin + eps) / torch.sum(eps + N_state * self.bias_coefficients.T * torch.exp(self._free_energy_log), axis=-1).T)

        log_likelihood = torch.sum(N_state * self._free_energy_log) + \
                         torch.sum(N_bin * log_val)

        return - log_likelihood
