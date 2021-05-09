from functools import reduce
import torch
from thermodynamicestimators.estimators.thermodynamic_estimator import ThermodynamicEstimator
from thermodynamicestimators.utilities.helper_function import ravel_index, unravel_index


class WHAM(ThermodynamicEstimator):
    """Free energy estimator based on the WHAM equations.

       Estimates the free energies of multiple biased thermodynamic states using
       either a likelihood formulation of the WHAM equations. The data can be fed
       to the estimator batch-wise so that stochastic optimizers can be used to
       optimize convergence.

       Example::

           $ dataset = test_case_factory.make_test_case("double_well_1D", 'WHAM')
           $ dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=128, shuffle=True)
           $ estimator = wham.WHAM(dataset.n_states)
           $ optimizer = torch.optim.SGD(estimator.parameters(), lr=0.1)
           $ free_energies, errors = estimator.estimate(dataloader, optimizer)

       """
    def __init__(self, dataset):
        super().__init__(dataset.n_states)

        self.bias_coefficients = dataset.bias_coefficients
        self._histogram_shape = self.bias_coefficients.shape[1:]

        # if isinstance(dataset.histogram_shape, tuple) or isinstance(dataset.histogram_shape, list):
        #     self.total_histogram_bins = reduce(lambda x, y: x * y, dataset.histogram_shape)
        # else:
        #     self.total_histogram_bins = dataset.histogram_shape


    def _to_histogram(self, samples):
        """ Takes input list of samples and returns one histogram with the counts
        per bin summed over all states.

        Parameters
        ----------
        samples : torch.Tensor
            Tensor of shape (N, D) Where N is the number of samples
            and D is the dimensionality of the coordinates.

        Returns
        -------
        N_per_bin : torch.Tensor
            Histogram of shape (d1, d2, ...) where di is the number of bins accross
            dimension i. N_per_bin[i] contains the number of samples binned at index
            i, summed over all simulations.

        """

        # if the coordinates are 1d, get rid of the last dimension (of size 1)
        samples = samples.squeeze(-1)

        # make a histogram
        N_per_bin = torch.zeros(self._histogram_shape)

        # if more than 1-dimensional
        if len(self._histogram_shape) > 1:
            # flatten indices to 1D so we can use torch bincount
            samples = ravel_index(samples, self._histogram_shape).int()

        N_per_bin = torch.bincount(samples, minlength=N_per_bin.numel())

        # if originally multi-dimensional: restore dimensions
        if len(self._histogram_shape) > 1:
            N_per_bin = N_per_bin.reshape(self._histogram_shape)

        return N_per_bin


    def get_potential(self, samples, normalized_N_i):
        """estimate potential energy function based on observed data

        Parameters
        ----------
        samples : torch.Tensor
            Tensor of shape (N, D) Where N is the number of samples
            and D is the dimensionality of the coordinates.
        normalized_N_i : torch.Tensor
            Tensor of shape (S) where S is the number of thermodynamic states.
            normalized_N_i[i] represents the number of samples taken at state i,
            divided by the total number of samples taken.

        Returns
        -------
        potential energy : torch.Tensor
            Tensor of shape (d1, d2, ...) containing the estimated potential energy
            at each histogram bin.
        """
        N_per_bin = self._to_histogram(samples)

        return - torch.log(
            N_per_bin / torch.sum(normalized_N_i * len(samples) * torch.exp(self.free_energies) * self.bias_coefficients.T, axis=-1).T)


    def self_consistent_step(self, samples, normalized_N_i):
        """Update the free energies by calculating the self-consistent MBAR
        equations:

            .. math::

                p_b = \\frac{M_b}{\sum_i N_i f_i c_{ib}}

                f_i = \\frac{1}{\sum_b c_{ib}{p_b}}

        Where :math:`p_b` is the probability of drawing a sample from bin :math:`b`,
        :math:`M_b` is the number of samples in bin :math:`b` summed over all states, :math:`N_i`
        is the total number of samples from state :math:`i`, and :math:`f_i = e^{F_i}` is the
        log of the free energy at state :math:`i`.

        Parameters
        ----------
        samples : torch.Tensor
            Tensor of shape (N, D) Where N is the number of samples
            and D is the dimensionality of the coordinates.
        normalized_N_i : torch.Tensor
            Tensor of shape (S) where S is the number of thermodynamic states.
            normalized_N_i[i] represents the number of samples taken at state i,
            divided by the total number of samples taken.
        """
        N_per_bin = self._to_histogram(samples)
        N_i = samples.shape[0] * normalized_N_i

        new_p = N_per_bin / torch.sum(N_i * torch.exp(self.free_energies) * self.bias_coefficients.T, axis=-1)

        new_f = torch.sum(self.bias_coefficients * new_p, axis=-1)

        # keep summing over histogram dimensions until we have only the state dimension left
        while len(new_f.shape) > 1:
            new_f = torch.sum(new_f, axis=-1)

        new_state_dict = self.state_dict()
        new_state_dict['_free_energies'] = -torch.log(new_f)

        self.load_state_dict(new_state_dict, strict=False)


    def residue(self, samples, normalized_N_i):
        """Compute the loss function for gradient descent, given by:

        .. math::
            \hat{A}(g_1,...g_S)=-\sum_{i=1}^S N_i g_i - \sum_{b=1}^B M_b \\text{ln}\\frac{M_b}{\sum_i N_i c_{ib} e^{g_i}}

        Where :math:`g_i = \\text{ln}\;f_i` is the log of the free energy at state :math:`i`,
        :math:`M_b` is the number of samples in bin :math:`b` summed over all states, :math:`N_i`
        is the total number of samples from state :math:`i`.

        Parameters
        ----------
        samples : torch.Tensor
            Tensor of shape (N, D) Where N is the number of samples
            and D is the dimensionality of the coordinates.
        normalized_N_i : torch.Tensor
            Tensor of shape (S) where S is the number of thermodynamic states.
            normalized_N_i[i] represents the number of samples taken at state i,
            divided by the total number of samples taken.
        """

        N_per_bin = self._to_histogram(samples)
        N_i = samples.shape[0] * normalized_N_i

        # small epsilon value to avoid taking the log of zero
        eps = 1e-10
        log_val = torch.log((N_per_bin + eps) / torch.sum(
            eps + N_i * self.bias_coefficients.T * torch.exp(self._free_energies), axis=-1).T)

        log_likelihood = torch.sum(N_i * self._free_energies) + \
                         torch.sum(N_per_bin * log_val)

        return - log_likelihood
