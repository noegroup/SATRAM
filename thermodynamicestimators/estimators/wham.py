import torch
from thermodynamicestimators.estimators.thermodynamic_estimator import ThermodynamicEstimator


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


    def __init__(self, N_i, M_b, bias_coefficients_log, device=None):
        super().__init__(device=device)

        self.bias_coefficients_log = bias_coefficients_log

        self.M_b = M_b
        self.N_i = N_i

        # work in log space for better precision
        self.N_i_log = torch.log(self.N_i)
        self.M_b_log = torch.log(self.M_b)

        self.normalized_N_i = N_i / torch.sum(N_i)

        self.n_states = bias_coefficients_log.shape[0]
        self._free_energies = torch.nn.Parameter(torch.ones(self.n_states, dtype=torch.float64))




    def get_potential(self):
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
        return - self.M_b_log - torch.logsumexp(self.N_i_log + self.free_energies + self.bias_coefficients_log.T, axis=-1)


    def self_consistent_step(self):
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
        new_p_log = self.M_b_log - torch.logsumexp(self.N_i_log + self.free_energies + self.bias_coefficients_log.T, axis=1)

        new_f = torch.logsumexp(self.bias_coefficients_log + new_p_log, axis=1)

        new_state_dict = self.state_dict()
        new_state_dict['_free_energies'] = -new_f

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


        p_b = self.M_b_log - torch.logsumexp(self.N_i_log + self._free_energies + self.bias_coefficients_log.T, axis=1)

        # pure gradient descent:
        log_likelihood = torch.sum(self.N_i * self._free_energies) + torch.sum(self.M_b * p_b)
        return -torch.sum(log_likelihood) / torch.sum(self.N_i)

        # stochastic gradient descent:
        f_i_samples = self._free_energies[samples[:, 0]]
        p_b_samples = p_b[samples[:, 1]]

        log_likelihood = torch.sum(f_i_samples) + torch.sum(p_b_samples)
        return - log_likelihood/len(samples)
