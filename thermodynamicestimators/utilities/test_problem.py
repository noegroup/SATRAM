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


    @property
    def potential(self):
        return self._potential


    @property
    def bias_coefficients(self):
        return self._bias_coefficients

    @property
    def bias_functions(self):
        return self._bias_functions

    @property
    def biased_potentials(self):
        return [lambda x, bias=_bias: self.potential(x) + bias(x) for _bias in self.bias_functions]

    @property
    def data(self):
        return torch.tensor(self._data, dtype=torch.float64)


    @property
    def histogram_range(self):
        return self._histogram_range


    @property
    def histogram_shape(self):
        return tuple([dimension_range[1]-dimension_range[0] for dimension_range in self.histogram_range])


    # for MBAR: the potential energy of the observed trajectories.
    @property
    def sampled_potentials(self):
        data_potential = []

        for i, bias in enumerate(self.bias_functions):

            biased_potential = lambda r: self.potential(r) + bias(r)

            data_potential.append(np.asarray(biased_potential(self.data)).flatten())

        return np.asarray(data_potential)



    # for MBAR: the potential energy of the observed trajectories, evaluated at all thermodynamic states.
    @property
    def data_at_all_states(self):
        return torch.stack([bias(torch.flatten(self.data)) for bias in self.bias_functions]).squeeze(-1)