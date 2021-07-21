import torch
from thermodynamicestimators.estimators.thermodynamic_estimator import ThermodynamicEstimator
from thermodynamicestimators.utilities.helper_function import to_histogram


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


    def __init__(self, N_i, M_l, bias_coefficients, device=None):
        super().__init__(n_states=bias_coefficients.shape[0])

        self.bias_coefficients = bias_coefficients

        self.M_l = M_l
        self.N_i = N_i

        self.normalized_N_i = N_i / torch.sum(N_i)


    def get_potential(self, samples):
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
        N_per_bin = to_histogram(samples[:, 1], self.bias_coefficients.shape[1:])

        return - torch.log(
            N_per_bin / torch.sum(
                self.normalized_N_i * len(samples) * torch.exp(self.free_energies) * self.bias_coefficients.T, axis=-1).T)


    def self_consistent_step(self, samples):
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
        N_i = samples.shape[0] * self.normalized_N_i

        new_p = N_per_bin / torch.sum(N_i * torch.exp(self.free_energies) * self.bias_coefficients.T, axis=-1)

        new_f = torch.sum(self.bias_coefficients * new_p, axis=-1)

        # keep summing over histogram dimensions until we have only the state dimension left
        while len(new_f.shape) > 1:
            new_f = torch.sum(new_f, axis=-1)

        new_state_dict = self.state_dict()
        new_state_dict['_free_energies'] = -torch.log(new_f)

        self.load_state_dict(new_state_dict, strict=False)


    def residue(self, samples):
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
        p_l = torch.log(self.M_l) - torch.log(
            torch.sum((self.N_i * torch.exp(self._free_energies)).unsqueeze(1) * self.bias_coefficients, axis=0))

        f_i_samples = self._free_energies[samples[:, 0].long()]
        p_l_samples = p_l[samples[:, 1].long()]

        log_likelihood = -torch.sum(f_i_samples) - torch.sum(p_l_samples)
        return log_likelihood
