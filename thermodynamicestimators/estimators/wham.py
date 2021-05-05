from functools import reduce
import torch
from thermodynamicestimators.estimators.thermodynamic_estimator import ThermodynamicEstimator


class WHAM(ThermodynamicEstimator):

    def __init__(self, dataset):
        super().__init__(dataset.n_states)

        self.bias_coefficients = dataset.bias_coefficients

        if isinstance(dataset.histogram_shape, tuple) or isinstance(dataset.histogram_shape, list):
            self.total_histogram_bins = reduce(lambda x, y: x * y, dataset.histogram_shape)
        else:
            self.total_histogram_bins = dataset.histogram_shape


    def to_one_sample(self, data):
        """ Takes input list of histograms and sums these, returning the one histogram
        with the counts per bin, and a tensor with the sample count per state.
        :param data:
        :return:
        """
        N_per_state, N_per_bin = data

        # check if this is one big histogram, or if its a tensor with a histogram for each sample
        if N_per_bin.numel() > self.total_histogram_bins:
            # sum over the separate histograms to form one histogram per state. Data is now of shape (b1, b2,...)
            N_per_bin = torch.sum(N_per_bin, axis=0)

        N_per_state = N_per_state[0] * N_per_state.shape[0]  # total samples per thermodynamic state

        return N_per_state, N_per_bin


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
        (N_per_state, N_per_bin) = self.to_one_sample(data)

        return - torch.log(
            N_per_bin / torch.sum(N_per_state * self._free_energies * self.bias_coefficients.T, axis=-1).T)


    def self_consistent_step(self, data):
        """Update the free energies by calculating the self-consistent MBAR
        equations:

            .. math::

                p_b = \\frac{M_b}{\sum_i N_i f_i c_{ib}}

                f_i = \\frac{1}{\sum_b c_{ib}{p_b}}

        Where :math:`p_b` is the probability of drawing a sample from bin :math:`b`,
        :math:`M_b` is the number of samples in bin :math:`b` summed over all states, :math:`N_i`
        is the total number of samples from state :math:`i`, and :math:`f_i` is the free
        energy at state :math:`i`.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape (S, d1, d2, ....) Where S is the number of thermody-
            namic states and d1,d2,... are the number of bins across each dimension.
        """
        (N_per_state, N_per_bin) = self.to_one_sample(data)

        new_p = N_per_bin / torch.sum(N_per_state * torch.exp(self._free_energies) * self.bias_coefficients.T, axis=-1)

        new_free_energy = 1 / torch.sum(self.bias_coefficients * new_p, axis=-1)

        # keep summing over histogram dimensions until we have only the state dimension left
        while len(new_free_energy.shape) > 1:
            new_free_energy = torch.sum(new_free_energy, axis=-1)

        new_state_dict = self.state_dict()
        new_state_dict['_free_energies'] = torch.log(new_free_energy)

        self.load_state_dict(new_state_dict, strict=False)


    def residue(self, data):
        """Compute the loss function for gradient descent, given by:

        .. math::
            \hat{A}(g_1,...g_S)=-\sum_{i=1}^S N_i g_i - \sum_{b=1}^B M_b \\text{ln}\\frac{M_b}{\sum_i N_i c_{ib} e^{g_i}}

        Where :math:`g_i = \\text{ln}\;f_i` is the log of the free energy at state :math:`i`,
        :math:`M_b` is the number of samples in bin :math:`b` summed over all states, :math:`N_i`
        is the total number of samples from state :math:`i`.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape (S, d1, d2, ....) Where S is the number of thermody-
            namic states and d1,d2,... are the number of bins across each dimension.
        """

        (N_per_state, N_per_bin) = self.to_one_sample(data)

        # small epsilon value to avoid taking the log of zero
        eps = 1e-10
        log_val = torch.log((N_per_bin + eps) / torch.sum(
            eps + N_per_state * self.bias_coefficients.T * torch.exp(self._free_energies), axis=-1).T)

        log_likelihood = torch.sum(N_per_state * self._free_energies) + \
                         torch.sum(N_per_bin * log_val)

        return - log_likelihood
