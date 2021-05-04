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


    Attributes
    ----------
        n_states : int
            The number of thermodynamic states
        _free_energies : torch.Tensor
            A Tensor of shape (n_states) containing the estimated free energies.
            These are the parameters of the estimator and automatically updated
            by torch Autograd.

    """


    def __init__(self, n_states):
        super().__init__()
        self.n_states = n_states
        self._free_energies = torch.nn.Parameter(torch.zeros(self.n_states, dtype=torch.float64))


    @property
    def free_energies(self):
        """ Free energy estimate per thermodynamic state.

        Returns
        -------
        Free energies : torch.Tensor
        """
        return self._free_energies.detach().clone()


    @property
    def partition_functions(self):
        """ The partition function for each (biased) thermodynamic state.

        The partition function for each biased state follows directly from the
        free energy estimate according to

        .. math::

            \hat{Z}_i = e^{-f_i}

        Returns
        -------
        partition functions : torch.Tensor
            A Tensor of shape (S), containing the partition function for each
            of the S thermodynamic states.

        See also
        --------
        free_energies
        """
        return torch.exp(-self.free_energies)


    def get_unbiased_partition_function(self, potentials, N_i):
        """ get the unbiased partition function based on the sampled data

        The estimate of the unbiased partition function is given by

        .. math::

            \hat{Z} = \sum_{i=1}^S \sum_{j=1}^{n_i} \\text{exp} [-u_i(x_{ij})] \hat{G}(u_i(x_{ij}))

        where :math:`u_i(x_{ij})` is the potential of the j'th sample of state
        i, evaluated at state i, :math:`n_i` is the number of samples taken at
        state i, and :math:`G(u_i(x_{ij}))` is the sample weight.

        Parameters
        ----------
        potentials : torch.Tensor
            Tensor containing potentials of all sampled data points evaluated at
            each thermodynamic state. Potentials is of shape (S,N)
            where S is the number of thermodynamic states and N is the total number
            of samples taken. data[i,j] is the potential energy of the j'th sample
            evaluated at state i.

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
        return torch.sum(torch.exp(-potentials) * self.get_sample_weights(potentials, N_i))


    def get_sample_weights(self, potentials, N_i):
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
        potentials : torch.Tensor
            Tensor containing potentials of all sampled data points evaluated at
            each thermodynamic state. Potentials is of shape (S,N)
            where S is the number of thermodynamic states and N is the total number
            of samples taken. data[i,j] is the potential energy of the j'th sample
            evaluated at state i.
        N_i : torch.Tensor
            Tensor of shape (S) where S is the number of thermodynamic states.
            N_i[i] is the total number of samples taken at state i.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape (N) containing the sample weight for each data point.
        """

        return 1 / torch.sum(N_i * torch.exp(-potentials + self._free_energies), axis=1)


    def get_unbiased_expectation_value(self, potentials, N_i, unbiased_potentials,
                                       observable_function, sampled_positions):
        """ Gets the expectation value of an observable function based on the
        observed data, at the unbiased state.

        Parameters
        ----------

        potentials : torch.Tensor
            Tensor containing potentials of all sampled data points evaluated at
            each thermodynamic state. Potentials is of shape (S,N)
            where S is the number of thermodynamic states and N is the total number
            of samples taken. potentials[i,j] is the potential energy of the j'th
            sample evaluated at state i.
        N_i : torch.Tensor
            Tensor of shape (S) where S is the number of thermodynamic states.
            N_i[i] is the total number of samples taken at state i.
        unbiased_potentials : torch.Tensor
            Tensor of shape (N) containing the potentials of each sample evaluated
            at the unbiased (reference) state.
        sampled_positions : torch.Tensor
            Tensor of shape (N, D) where N is the total number of samples taken,
            and D is the number of sampled dimensions
        observable_function: callable
            a function that takes one position from sampled_positions and outputs
            the observable value.

        Returns
        -------
        Expectation value : torch.Tensor
            The expectation value of the observable function given the dataset.
            The shape of the output depends on the shape of the observable
            function, e.g. if the observable function outputs a histogram, the
            output expectation value has the shape of the histogram.
        """
        # samples = dataset.sampled_positions.flatten(0, 1)

        # construct a matrix to store the computed observables
        result_shape = observable_function(sampled_positions[0]).shape
        observable_values = torch.zeros(sampled_positions.shape[:1] + result_shape)

        # fill it with the observed values
        for s_i in range(len(sampled_positions)):
            observable_values[s_i] = observable_function(sampled_positions[s_i])

        # Weight the observed values by multiplying with the sample probabilities.
        weighted_observables = observable_values.T * torch.exp(-unbiased_potentials) \
                               * self.get_sample_weights(potentials, N_i) \
                               / self.get_unbiased_partition_function(potentials, N_i)

        return torch.sum(weighted_observables, axis=0)


    def shift_free_energies_relative_to_zero(self):
        """ Subtract the first free energy from all free energies such that the
        first is zero and all other energies are relative to the first."""
        with torch.no_grad():
            self._free_energies -= self._free_energies[0].clone()


    def residue(self, potentials):
        """ Computes the value of the optimization function for gradient descent.

        Finding the minimum of the derivative of this function is equivalent to
        solving the MBAR equations for the free energies.

        .. math::

            \phi(f_1, ... f_S) = \\frac{1}{N} \sum_{i=1}^S \sum_{n=1}^{N_i}
            \\text{ln} \\left( \sum_{j=1}^S \\text{exp} [-u_j(x_i^n) + f_j +
            \\text{ln}\\frac{N_j}{N}] \\right) + \sum_{i=1}^S \\frac{N_i}{N}f_i

        Parameters
        ----------

        potentials : torch.Tensor
            Tensor containing potentials of all sampled data points evaluated at
            each thermodynamic state. Potentials is of shape (S,N)
            where S is the number of thermodynamic states and N is the total number
            of samples taken. potentials[i,j] is the potential energy of the j'th
            sample evaluated at state i.

        Notes
        -----
        This method assumes an equal number of samples is taken at each thermo-
        dynamic state!

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

        N = potentials.shape[1]

        # assume an equal number of samples was taken at each state.
        N_i = N / potentials.shape[0]

        log_sum_arg = -potentials + self._free_energies + torch.log(N_i / N)

        logsum = torch.logsumexp(log_sum_arg, dim=1)

        objective_function = torch.mean(torch.sum(logsum) - torch.sum(self._free_energies * N_i))

        return objective_function


    def self_consistent_step(self, potentials, N_i):
        """ Update the free energies by calculating the self-consistent MBAR
        equations:

        .. math::

            \hat{Z}_l = \sum_{i=1}^S \sum_{n=1}^{N_i} \\text{exp}[-u_i(x_{in})] \hat{G}(x_{in})

        Where :math:`\hat{G}(x_{in})` is the sample weight

        Parameters
        ----------
        potentials : torch.Tensor
            Tensor containing potentials of all sampled data points evaluated at
            each thermodynamic state. Potentials is of shape (S,N)
            where S is the number of thermodynamic states and N is the total number
            of samples taken. potentials[i,j] is the potential energy of the j'th
            sample evaluated at state i.
        N_i : torch.Tensor
            Tensor of shape (S) where S is the number of thermodynamic states.
            N_i[i] is the total number of samples taken at state i.

        See also
        --------
        get_sample_weights
        """

        new_free_energy = - torch.log(
            torch.sum(torch.exp(- potentials.T) \
                      * self.get_sample_weights(potentials, N_i), axis=1)).clone()

        new_state_dict = self.state_dict()
        new_state_dict['_free_energies'] = new_free_energy

        self.load_state_dict(new_state_dict, strict=False)
