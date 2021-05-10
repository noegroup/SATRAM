import torch
from thermodynamicestimators.estimators.thermodynamic_estimator import ThermodynamicEstimator


class MBAR(ThermodynamicEstimator):
    """Free energy estimator based on the MBAR equations.

    Estimates the free energies of multiple biased thermodynamic states using
    either a likelihood formulation of the MBAR equations. The data can be fed
    to the estimator batch-wise so that stochastic optimizers can be used to
    optimize convergence.

    Example::

        $ dataset = test_case_factory.make_test_case("double_well_1D", 'MBAR')
        $ dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=128, shuffle=True)
        $ estimator = mbar.MBAR(dataset.n_states)
        $ optimizer = torch.optim.SGD(estimator.parameters(), lr=0.1)
        $ free_energies, errors = estimator.estimate(dataloader, optimizer)

    """


    def __init__(self, n_states):
        super().__init__(n_states)


    def _get_unbiased_partition_function(self, sampled_potentials, N_i):
        """ get the unbiased partition function based on the sampled data

        The estimate of the unbiased partition function is given by

        .. math::

            \hat{Z} = \sum_{i=1}^S \sum_{j=1}^{n_i} \\text{exp} [-u_i(x_{ij})] \hat{G}(u_i(x_{ij}))

        where :math:`u_i(x_{ij})` is the potential of the j'th sample of state
        i, evaluated at state i, :math:`n_i` is the number of samples taken at
        state i, and :math:`G(u_i(x_{ij}))` is the sample weight.

        Parameters
        ----------
        sampled_potentials : torch.Tensor
            Tensor of shape (S,N) where S is the number of thermodynamic states
            and N is the total number of samples taken. sampled_potentials[i,j]
            is the potential energy of the j'th sample evaluated at state i.

        N_i : torch.Tensor
            Tensor of shape (S) where S is the number of thermodynamic states.
            N_i[i] is the total number of samples taken at state i.

        Returns
        ----------
        partition function : torch.Tensor
            A Tensor of shape (1,) containing the partition function
            for the unbiased potential.


        See Also
        --------
        get_sample_weights
        """

        return torch.sum(self._get_sample_weights(sampled_potentials, N_i))
            # torch.exp(-unbiased_potentials) * )


    def _get_sample_weights(self, sampled_potentials, N_i):
        """ Gets the weights of all samples.

        The weight of a sample is the inverse of sum over all thermodynamic
        states of the sample probability at that thermodynamic state.
        The sample weight of sample j from state i is given by:

        .. math::

            \hat{G}(x_{ij}) = \\frac{1}{\sum_{l=1}^S N_l \\text{exp}[-u_l(x_{ij}) + f_l] }

        where :math:`u_l(x_{ij})` is the potential of the j'th sample of state i,
        evaluated at state l, and :math:`f_l` is the free energy of state l.

        Parameters
        ----------
        sampled_potentials : torch.Tensor
            Tensor of shape (S,N) where S is the number of thermodynamic states
            and N is the total number of samples taken. sampled_potentials[i,j]
            is the potential energy of the j'th sample evaluated at state i.

        N_i : torch.Tensor
            Tensor of shape (S) where S is the number of thermodynamic states.
            N_i[i] is the total number of samples taken at state i.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape (N) containing the sample weight for each data point.
        """
        return 1 / torch.sum(N_i * torch.exp(-sampled_potentials.T + self._free_energies), axis=1)


    def get_equilibrium_expectation(self, sampled_potentials, N_i, observable_values):
        """ Gets the expectation value of an observable function based on the
        observed data, at the unbiased state.

        Parameters
        ----------
        sampled_potentials : torch.Tensor
            Tensor of shape (S,N) where S is the number of thermodynamic states
            and N is the total number of samples taken. sampled_potentials[i,j]
            is the potential energy of the j'th sample evaluated at state i.

        N_i : torch.Tensor
            Tensor of shape (S) where S is the number of thermodynamic states.
            N_i[i] is the total number of samples taken at state i.

        observable_values: torch.Tensor
            Tensor of shape (N, D) where N is the total number of samples, and D
            is the dimensionality of the observable.


        Returns
        -------
        Expectation value : torch.Tensor
            The expectation value of the observable function given the dataset.
            The shape of the output depends on the shape of the observable
            function, e.g. if the observable function outputs a histogram, the
            output expectation value has the shape of the histogram.
        """

        # Weight the observed values by multiplying with the sample probabilities.
        weighted_observables = observable_values.T * self._get_sample_weights(sampled_potentials, N_i) \
                               / self._get_unbiased_partition_function(sampled_potentials, N_i)

        return torch.sum(weighted_observables, axis=-1).T


    def residue(self, sampled_potentials, normalized_N_i):
        """ Computes the value of the optimization function for gradient descent.

        Finding the minimum of the derivative of this function is equivalent to
        solving the MBAR equations for the free energies.

        .. math::

            \phi(f_1, ... f_S) = \\frac{1}{N} \sum_{i=1}^S \sum_{n=1}^{N_i}
            \\text{ln} \\left( \sum_{j=1}^S \\text{exp} [-u_j(x_i^n) + f_j +
            \\text{ln}\\frac{N_j}{N}] \\right) + \sum_{i=1}^S \\frac{N_i}{N}f_i

        Parameters
        ----------
        sampled_potentials : torch.Tensor
            Tensor of shape (S,N) where S is the number of thermodynamic states
            and N is the total number of samples taken. sampled_potentials[i,j]
            is the potential energy of the j'th sample evaluated at state i.

        normalized_N_i : torch.Tensor
            Tensor of shape (S) containing the normalized count of samples evaluated
            at each state, over the entire dataset. The assumption is that the distribution
            of origin state of the samples in the batch is, on average, equal to that
            of the entire dataset.


        Returns
        -------
        Residue : torch.Tensor

        Notes
        -----
            This is the value of equation (7) in paper: 10.1021/acs.jctc.8b01010
            eq. (7) is a function of :math:`\{b_1, ..., b_M\}`. Here it is
            implemented as a function of the free energies :math:`\{f_1,... f_S\}`
            Additive constants are ignored since they don't affect the gradient.
        """

        log_sum_arg = -sampled_potentials + self._free_energies + torch.log(normalized_N_i)

        logsum = torch.logsumexp(log_sum_arg, dim=1)

        objective_function = torch.mean(logsum) - torch.sum(self._free_energies * normalized_N_i)

        return objective_function


    def self_consistent_step(self, sampled_potentials, normalized_N_i):
        """ Update the free energies by calculating the self-consistent MBAR
        equations:

        .. math::

            \hat{Z}_l = \sum_{i=1}^S \sum_{n=1}^{N_i} \\text{exp}[-u_i(x_{in})] \hat{G}(x_{in})

        Where :math:`\hat{G}(x_{in})` is the sample weight

        Parameters
        ----------
        sampled_potentials : torch.Tensor
            sampled_potentials : torch.Tensor
            Tensor of shape (S,N) where S is the number of thermodynamic states
            and N is the total number of samples taken. sampled_potentials[i,j]
            is the potential energy of the j'th sample evaluated at state i.

        normalized_N_i : torch.Tensor
            Tensor of shape (S) containing the normalized count of samples evaluated
            at each state, over the entire dataset. The assumption is that the distribution
            of origin state of the samples in the batch is, on average, equal to that
            of the entire dataset.

        See also
        --------
        get_sample_weights
        """

        # the total number of samples
        N = sampled_potentials.shape[0]

        # number of samples per state in this batch
        N_i = N * normalized_N_i

        new_free_energy = - torch.log(
            torch.sum(torch.exp(- sampled_potentials.T) \
                      * self._get_sample_weights(sampled_potentials.T, N_i), axis=1)).clone()

        new_state_dict = self.state_dict()
        new_state_dict['_free_energies'] = new_free_energy

        self.load_state_dict(new_state_dict, strict=False)
