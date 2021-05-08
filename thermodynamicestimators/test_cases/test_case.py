import torch
import math
from thermodynamicestimators.data_sets import dataset


class TestCase:
    """Wrapper for test case data that build a thermodynamicestimators.data_sets.dataset
    object out of the test case."""
    def __init__(self, potential, biases, sampled_coordinates, histogram_range):
        self.potential_fn = potential
        self.bias_fns = biases
        self.histogram_range = histogram_range
        self.sampled_coordinates = sampled_coordinates


    def to_wham_dataset(self):
        bias_coefficients = self._construct_bias_coefficients()
        samples = self.sampled_coordinates.long().flatten(0,1)
        N_i = torch.Tensor([len(self.sampled_coordinates) for i in self.sampled_coordinates])

        return dataset.Dataset(samples=samples, N_i=N_i, bias_coefficients=bias_coefficients)


    def to_mbar_dataset(self):
        sampled_potentials = self.potential_at_all_states()
        N_i = torch.Tensor([len(self.sampled_coordinates) for i in self.sampled_coordinates])

        return dataset.Dataset(sampled_potentials, N_i=N_i)


    def _construct_bias_coefficients(self):
        """Construct the tensor of bias coefficients from the available bias functions
        and the histogram range.

        Returns
        -------
        bias coefficient matrix : torch.Tensor
            The matrix of shape (B, d_1, d_2, ...) where B is the number of thermo-
            dynamic states, and d_n is the size of the nth dimension of the histogram.
            each coefficient c_ij is the given by the boltzmann factor of the potential
            of state i evaluated at bin j:

            .. math::

                c_{ij} = e^{-u_i(x_j)}
        """
        bias_coefficients_shape = tuple([self.n_states]) + self.histogram_shape
        bias_coefficients = torch.zeros(bias_coefficients_shape)

        # fill an array with all indices of the bias coefficients.
        indices = (bias_coefficients == 0).nonzero()

        # iterate over the indices to fill the bias coefficients array.
        # The array is filled in this way because we don't know the shape of the histogram beforehand and there is
        # no equivalent of numpy.ndenumerate in pytorch.
        for idx in indices:
            bias_coefficients[tuple(idx)] = math.exp(
                -self.bias_fns[idx[0]](idx[1:] + self._histogram_range[:, 0]))

        return bias_coefficients


    def potential_at_all_states(self):
        """The potential energy of the observed trajectories, evaluated at all thermodynamic states."""
        return torch.stack(
            [bias(torch.flatten(self.sampled_coordinates, 0, 1).squeeze(-1)) for bias in self.bias_fns]) + \
               self.unbiased_potential()


    def unbiased_potential(self):
        """The unbiased potential values for each sample. This is needed for calculating an expectation value with MBAR."""
        return self.potential_fn(torch.flatten(self.sampled_coordinates, 0, 1).squeeze(-1)).clone()