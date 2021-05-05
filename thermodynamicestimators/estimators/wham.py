from functools import reduce
import torch
from thermodynamicestimators.estimators.thermodynamic_estimator import ThermodynamicEstimator



class WHAM(ThermodynamicEstimator):
    def __init__(self, dataset):
        super().__init__()

        self.bias_coefficients = dataset.bias_coefficients
        self.n_biases = self.bias_coefficients.shape[0]

        if isinstance(dataset.histogram_shape, tuple) or isinstance(dataset.histogram_shape, list):
            self.total_histogram_bins = reduce(lambda x, y: x*y, dataset.histogram_shape)
        else:
            self.total_histogram_bins = dataset.histogram_shape

        self._free_energy_log = torch.nn.Parameter(torch.zeros(self.n_biases))


    @property
    def free_energies(self):
        """See base class"""
        return torch.exp(self._free_energy_log.detach())


    def shift_free_energies_relative_to_zero(self):
        """See base class"""
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

        # keep summing over remaining dimensions of the histogram until we have a one-dimensional tensor with the number
        # of samples per state.
        while(len(N_state.shape) > 1):
            N_state = torch.sum(N_state, axis=1)

        return N_bin, N_state


    def get_potential(self, data):
        """estimate potential energy function based on observed data

        Parameters
        ----------

        data : torch.Tensor
            Tensor of shape (S, d1, d2, ....) Where S is the number of thermody-
            namic states and d1,d2,... are the number of bins across each dimension.

        Returns
        -------
        potential energy : torch.Tensor
            Tensor of shape (d1, d2, ...) containing the estimated potential energy
            at each histogram bin.
        """
        N_bin, N_state = self.to_normalized_sample_counts(data)
        return - torch.log(N_bin / torch.sum(N_state * self.free_energies * self.bias_coefficients.T, axis=-1).T)


    #TODO: implement
    def self_consistent_step(self, data):
        pass


    def residue(self, data):
        """ compute the loss function for gradient descent
        data has shape: (N, M, b1, b2, ...) where N is the number of samples, M is the number of thermodynamic states,
        and b1, b2,... are the number of histogram bins in each dimensional axis.
        One sample consists of M datapoints, one from each thermodynamic state,
        """

        N_bin, N_state = self.to_normalized_sample_counts(data)

        # small epsilon value to avoid taking the log of zero
        eps = 1e-10
        log_val = torch.log((N_bin + eps) / torch.sum(eps + N_state * self.bias_coefficients.T * torch.exp(self._free_energy_log), axis=-1).T)

        log_likelihood = torch.sum(N_state * self._free_energy_log) + \
                         torch.sum(N_bin * log_val)

        return - log_likelihood
