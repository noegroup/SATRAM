import math
import numpy as np
import torch

class TestProblem:
    def __init__(self, potential=None, biases=None, histogram_range=None, data=None, bias_coefficients=None):
        self._potential = potential
        self._bias_functions = biases
        self._data = data
        self._histogram_range = histogram_range

        self._bias_coefficients = bias_coefficients
        if bias_coefficients is None and biases is not None:
            bias_coefficients_shape = tuple([len(biases)] + [d_range[1]- d_range[0]for d_range in self._histogram_range])

            self._bias_coefficients = np.zeros(bias_coefficients_shape)

            for i in range(len(self._bias_coefficients )):
                for hist_coords, _ in np.ndenumerate(self._bias_coefficients [i]):
                    self._bias_coefficients[i][hist_coords] = math.exp(-biases[i](hist_coords + histogram_range[:, 0]))


    ''' The number of thermodynamic states that were sampled '''
    @property
    def n_states(self):
        return len(self.bias_functions)


    ''' The unbiased potential function '''
    @property
    def potential(self):
        return self._potential


    ''' The bias coefficient matrix for a discrete estimator (WHAM) '''
    @property
    def bias_coefficients(self):
        return self._bias_coefficients


    ''' The bias functions '''
    @property
    def bias_functions(self):
        return self._bias_functions


    ''' The bias potentials added to the unbiased potential function. 
    These functions govern the thermodynamic states that are sampled. '''
    @property
    def biased_potentials(self):
        return [lambda x, bias=_bias: self.potential(x) + bias(x) for _bias in self.bias_functions]


    ''' The sampled data points. These are the coordinates of the MD/MC simulation'''
    @property
    def data(self):
        return torch.tensor(self._data, dtype=torch.float64)


    ''' For WHAM: the histogram range over each dimensional axis.
    eg. for a two-dimensional histogram: [[min_x, max_x], [min_y, max_y]]'''
    @property
    def histogram_range(self):
        return self._histogram_range


    ''' For WHAM: the shape of the histogram in which to bin the data. Calculated from self.histogram_range. 
    All bins are assumed to be of size 1.'''
    @property
    def histogram_shape(self):
        return tuple([dimension_range[1]-dimension_range[0] for dimension_range in self.histogram_range])



    ''' The potential energy of the observed trajectories, evaluated at all thermodynamic states. 
    Since MBAR doesn't care which sample was sampled at which thermodynamic state, the data array is flattened 
    and all samples are evaluated at each and every thermodynamic state. 
    The resulting matrix is of shape (M, N) where M is the number of thermodynamic states, and N the total number 
    of samples taken.'''
    @property
    def sampled_potentials_at_all_states(self):
        return torch.stack([bias(torch.flatten(self.data)) for bias in self.bias_functions]).squeeze(-1) + self.sampled_unbiased_potentials


    ''' The unbiased potential values for each sample. This is needed for calculating an expectation value with MBAR.'''
    @property
    def sampled_unbiased_potentials(self):
        return torch.tensor(self.potential(torch.flatten(self.data)))