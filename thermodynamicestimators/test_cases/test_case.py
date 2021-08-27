import torch
import math
from thermodynamicestimators.utilities.helper_function import to_histogram


class TestCase:
    """Wrapper for test case data that build a thermodynamicestimators.data_sets.dataset
    object out of the test case."""


    def __init__(self, potential, biases, sampled_coordinates, histogram_range, ground_truth=None):
        self.potential_fn = potential
        self.bias_fns = biases
        self.histogram_range = histogram_range
        self.N_i = torch.Tensor([len(state_samples) for state_samples in sampled_coordinates])
        self.sampled_coordinates = sampled_coordinates.flatten(0, 1)
        self.ground_truth = ground_truth


    def to_wham_dataset(self):
        # in stead of binning, for now, assume that these are sampled integers
        # within the histogram range. Subtract the lower limit to turn this into
        # indices
        # TODO: allow for definition of bins so that the samples are binned.
        samples = self.sampled_coordinates.round().long() - self.histogram_range[:, 0].long()

        bias_coefficients = self._construct_bias_coefficients()
        M_l = to_histogram(samples, bias_coefficients.shape[1:])

        state_indices = torch.hstack([torch.full([int(self.N_i[i].item())], i) for i in range(len(self.N_i))])
        samples = torch.vstack((state_indices, samples.T)).T

        return self.N_i, M_l, bias_coefficients, samples


    def to_mbar_dataset(self):
        sampled_potentials = self.biased_potentials().double().T
        return self.N_i, sampled_potentials


    def _construct_bias_coefficients(self):
        """Construct the tensor of bias coefficients from the available bias functions
        and the histogram range.

        Returns
        -------
        bias coefficient matrix : torch.Tensor
            The matrix of shape (S, d_1, d_2, ...) where S is the number of thermo-
            dynamic states, and d_n is the size of the nth dimension of the histogram.
            each coefficient c_ij is the given by the boltzmann factor of the potential
            of state i evaluated at bin j:

            .. math::

                c_{ij} = e^{-u_i(x_j)}
        """

        # compute the shape of the histogram from the given bin ranges.
        histogram_shape = tuple(
            [int((dimension_range[1] - dimension_range[0]).item()) for dimension_range in self.histogram_range])
        n_states = len(self.bias_fns)

        bias_coefficients_shape = tuple([n_states]) + histogram_shape
        bias_coefficients = torch.zeros(bias_coefficients_shape)

        # fill an array with all indices of the bias coefficients.
        indices = (bias_coefficients == 0).nonzero()

        # iterate over the indices to fill the bias coefficients array.
        # The array is filled in this way because we don't know the shape of the histogram beforehand and there is
        # no equivalent of numpy.ndenumerate in pytorch.
        for idx in indices:
            bias_coefficients[tuple(idx)] = -self.bias_fns[idx[0]](idx[1:] + self.histogram_range[:, 0])

        return bias_coefficients


    def potential_at_all_states(self):
        """The total potential energy of the observed trajectories, evaluated at all thermodynamic states."""
        return self.biased_potentials() + self.unbiased_potential()


    def biased_potentials(self):
        """The bias energy of the observed trajectories, evaluated at all thermodynamic states."""
        return torch.stack(
            [bias(self.sampled_coordinates.squeeze(-1)) for bias in self.bias_fns])


    def unbiased_potential(self):
        """The unbiased potential values for each sample. This is needed for calculating an expectation value with MBAR."""
        return self.potential_fn(self.sampled_coordinates.squeeze(-1)).clone()
