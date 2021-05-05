import math
import torch
import thermodynamicestimators.utilities.helper_function as helpers
from thermodynamicestimators.data_sets import dataset


class WHAMDataset(dataset.Dataset):
    """ The dataset to use with WHAM. Samples are returned in the shape of unnormalized histograms specifying bin counts for
    each thermodynamic state."""


    def __init__(self, potential, biases, histogram_range, sampled_positions=None, bias_coefficients=None):
        super().__init__(potential, biases)

        sampled_positions = sampled_positions.long()
        super().add_data(sampled_positions)

        self._histogram_range = histogram_range.int()

        # compute the shape of the histogram from the given bin ranges.
        self._histogram_shape = tuple(
            [int((dimension_range[1] - dimension_range[0]).item()) for dimension_range in self._histogram_range])

        if bias_coefficients is not None:
            self._bias_coefficients = helpers.to_high_precision_tensor(bias_coefficients)

        else:
            if biases is None:
                raise ValueError("Either the bias functions or bias coefficients need to be passed to construct a "
                                 "WHAM_dataset")
            self._bias_functions = biases
            self._bias_coefficients = self.construct_bias_coefficients()


    @property
    def bias_coefficients(self):
        """The bias coefficient matrix for a discrete estimator (WHAM)"""
        return self._bias_coefficients


    @property
    def histogram_range(self):
        """The histogram range over each dimensional axis.

        Returns
        -------
        histogram range : torch.Tensor
            eg. for a two-dimensional histogram: [[min_x, max_x], [min_y, max_y]]'"""
        return self._histogram_range


    @property
    def histogram_shape(self):
        """The shape of the histogram in which to bin the data. Calculated from self.histogram_range.
        All bins are assumed to be of size 1."""
        return self._histogram_shape


    def __len__(self):
        return len(self._sampled_positions[0])


    def construct_bias_coefficients(self):
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
                -self._bias_functions[idx[0]](idx[1:] + self._histogram_range[:, 0]))

        return bias_coefficients


    def __getitem__(self, item):
        """Get sample by index.

        WHAM uses histograms, but does not care which sample was sampled at which
        state. The samples are therefore returned in the shape of two tensors,
        one specifying the number of samples per state, the other specifying the
        number of samples per bin.

        Returns
        -------
        N_per_state : torch.Tensor
            Tensor of shape (S) where S is the number of thermodynamic states.
            Contains the number of samples per state (when used with the data-
            loader, this will be 1 for each state).
        N_per_bin : torch.Tensor
            Unnormalized histogram of shape (d1, d2, ...) where di is the number
            of bins in the i'th dimension that was sampled.
        """
        sample = self._sampled_positions[:, item]

        N_per_state = torch.ones(self.n_states)

        # if multiple items per state were sampled
        if len(sample.squeeze(-1).shape) > len(self.histogram_shape):
            # assume equal number of samples per state
            N_per_state *= sample.shape[1]

            # flatten it so we are left with a d-dimenstional tensor of coordinates
            sample = sample.flatten(1)

        # make a histogram
        N_per_bin = torch.zeros(self.histogram_shape)

        # subtract left of histogram range so the bins can start at 0
        sample = sample - self.histogram_range[:, 0]

        # if more than 1-dimensional
        if len(self.histogram_shape) > 1:
            # flatten indices to 1D so we can use torch bincount
            sample = helpers.ravel_index(sample, self.histogram_shape).int()

        N_per_bin = torch.bincount(sample, minlength=N_per_bin.numel())

        if len(self.histogram_shape) > 1:
            N_per_bin = N_per_bin.reshape(self.histogram_shape)

        return N_per_state, N_per_bin
