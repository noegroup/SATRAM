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


    def _get_unbiased_partition_function(self, dataset):
        """ get the unbiased partition function based on the sampled data

        The estimate of the unbiased partition function is given by

        .. math::

            \hat{Z} = \sum_{i=1}^S \sum_{j=1}^{n_i} \\text{exp} [-u_i(x_{ij})] \hat{G}(u_i(x_{ij}))

        where :math:`u_i(x_{ij})` is the potential of the j'th sample of state
        i, evaluated at state i, :math:`n_i` is the number of samples taken at
        state i, and :math:`G(u_i(x_{ij}))` is the sample weight.

        Parameters
        ----------
        dataset : thermodynamicestimators.data_sets.mbar_dataset.MBARDataset
            Dataset containing sampled potentials and unbiased potentials

        Returns
        ----------
        partition function : torch.Tensor
            A Tensor of shape (1,) containing the partition function
            for the unbiased potential.


        See Also
        --------
        get_sample_weights
        """

        return torch.sum(
            torch.exp(-dataset.unbiased_potentials) * self._get_sample_weights(dataset.sampled_potentials, dataset.N_i))


    def _get_sample_weights(self, potentials, N_i):
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
        data : tuple(torch.Tensor, torch.Tensor)
            containing:
            potentials : torch.Tensor
                Tensor containing potentials of all sampled data points evaluated at
                each thermodynamic state. Potentials is of shape (S,N)
                where S is the number of thermodynamic states and N is the total number
                of samples taken. data[i,j] is the potential energy of the j'th sample
                evaluated at state i.
            N_i : torch.Tensor
                Tensor of shape (S,N) where S is the number of thermodynamic states.
                N_i[i] is the total number of samples taken at state i.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape (N) containing the sample weight for each data point.
        """
        return 1 / torch.sum(N_i * torch.exp(-potentials.T + self._free_energies), axis=1)


    def get_equilibrium_expectation(self, dataset, observable_function):
        """ Gets the expectation value of an observable function based on the
        observed data, at the unbiased state.

        Parameters
        ----------

        dataset : thermodynamicestimators.data_sets.mbar_dataset.MBARDataset
            Dataset containing sampled potentials and unbiased potentials
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
        samples = dataset.sampled_positions.flatten(0, 1)

        # construct a matrix to store the computed observables
        result_shape = observable_function(samples[0]).shape
        observable_values = torch.zeros(samples.shape[:1] + result_shape)

        # fill it with the observed values
        for s_i in range(len(samples)):
            observable_values[s_i] = observable_function(samples[s_i])

        # Weight the observed values by multiplying with the sample probabilities.
        weighted_observables = observable_values.T * torch.exp(-dataset.unbiased_potentials) \
                               * self._get_sample_weights(dataset.sampled_potentials, dataset.N_i) \
                               / self._get_unbiased_partition_function(dataset)

        return torch.sum(weighted_observables, axis=-1)


    def residue(self, data):
        """ Computes the value of the optimization function for gradient descent.

        Finding the minimum of the derivative of this function is equivalent to
        solving the MBAR equations for the free energies.

        .. math::

            \phi(f_1, ... f_S) = \\frac{1}{N} \sum_{i=1}^S \sum_{n=1}^{N_i}
            \\text{ln} \\left( \sum_{j=1}^S \\text{exp} [-u_j(x_i^n) + f_j +
            \\text{ln}\\frac{N_j}{N}] \\right) + \sum_{i=1}^S \\frac{N_i}{N}f_i

        Parameters
        ----------

        data : tuple(torch.Tensor, torch.Tensor)

            data[0] is a Tensor containing potentials of all sampled data points
            evaluated at each thermodynamic state. Potentials is of shape (S,N)
            where S is the number of thermodynamic states and N is the total number
            of samples taken. potentials[i,j] is the potential energy of the j'th
            sample evaluated at state i.
            data[1] is a Tensor of shape (S) containing the total count of samples
            evaluated at each state.


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

        potentials = data[0]

        # the total number of samples
        N = potentials.shape[0]

        # the number of samples per thermodynamic state. This is based on the
        # total number of samples in the entire dataset. This batch does not
        # necessarily contain this amount of samples, it is an average.
        N_i = N * data[1][0] / torch.sum(data[1][0])

        log_sum_arg = -potentials + self._free_energies + torch.log(N_i / N)

        logsum = torch.logsumexp(log_sum_arg, dim=1)

        objective_function = torch.mean(torch.sum(logsum) - torch.sum(self._free_energies * N_i))

        return objective_function


    def self_consistent_step(self, data):
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

        See also
        --------
        get_sample_weights
        """

        potentials = data[0]

        # the total number of samples
        N = potentials.shape[0]

        N_i = N * data[1][0] / torch.sum(data[1][0])

        new_free_energy = - torch.log(
            torch.sum(torch.exp(- potentials.T) \
                      * self.get_sample_weights(potentials.T, N_i), axis=1)).clone()

        new_state_dict = self.state_dict()
        new_state_dict['_free_energies'] = new_free_energy

        self.load_state_dict(new_state_dict, strict=False)
