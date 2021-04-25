from thermodynamicestimators.estimators.thermodynamicestimator import ThermodynamicEstimator
import torch
import math
import pymbar


class MBAR(ThermodynamicEstimator):
    def __init__(self, data):
        super().__init__()
        self.n_states = len(data)

        self.G = torch.zeros(self.n_states, dtype=torch.float64)
        # self.self_consistent_step(data)
        # self.self_consistent_step(data)

        self._f = torch.nn.Parameter(self.G.clone())

    '''free energy estimate per bias '''
    @property
    def free_energy(self):
        # TODO: make N_i/N variable
        return self._f.detach()

    def free_energy_test(self, data):
        N = data.shape[1]
        log_denom = torch.sum(torch.exp(-data + self._b.detach().unsqueeze(1)))
        return -torch.log(1/N * torch.sum(torch.exp(-data + self._b.detach().unsqueeze(1))/log_denom, axis=1))

    '''get the partition function for each (biased) thermodynamic state'''
    @property
    def partition_functions(self):
        return torch.exp(-self.free_energy)


    '''get the unbiased partition function'''
    def get_unbiased_partition_function(self, data):
        return torch.sum(1/self.get_sample_weights(data))


    '''Get the weights of all samples.
    The weight of a sample is the inverse of sum over all thermodynamic states of the sample probability at that thermodynamic state
    This is the denominator of eqn. (21) in the 2012 paper: https://doi.org/10.1063/1.3701175 '''
    def get_sample_weights(self, data):
        # total samples per simulation.
        # TODO: make variable
        N_i = 1000

        return 1/ torch.sum(((N_i/self.partition_functions) * torch.exp(-data).T).T, axis=0)


    '''Get the expectation value of an observable function based on the observed data'''
    def get_expectation_value(self, sample_positions, sample_potentials, observable_function):
        sample_positions = torch.flatten(sample_positions)
        result_shape = torch.tensor(observable_function(1)).shape
        values = torch.zeros(sample_positions.shape + result_shape)

        for s_i in range(len(sample_positions)):
            values[s_i] = torch.tensor(observable_function(sample_positions[s_i]))

        # weigh each observed value with the probability of the sample
        res = (1 / self.get_unbiased_partition_function(sample_potentials)) * \
            torch.sum(values.T/self.get_sample_weights(sample_potentials), axis=1)
        return res


    '''
    compute the loss function for gradient descent
    data has shape: [M, N] where M is the number of thermodynamic states, and N the total number of observed samples.
    data[m,n] is sample n evaluated at state m.
    This returns the value of equation (7) in paper: 10.1021/acs.jctc.8b01010
     '''
    def residue(self, data):

        # total number of samples
        N = data.shape[1]

        # Number of samples taken per state. For now: assume equal number of samples.
        # TODO: make this variable.
        N_i = int(N/data.shape[0])

        log_sum_arg = -data.T + self._f

        logsum = torch.log(torch.sum(1000 * torch.exp(log_sum_arg), axis=1))

        objective_function = (torch.sum(logsum) - torch.sum(self._f * N_i))


        return objective_function



    def self_consistent_step(self, data):
        N_l = 1000

        denominator = torch.sum(N_l * torch.exp(- data.T + self.G), axis=1)

        self.G = -1. * torch.log(torch.sum(torch.exp(- data) / denominator, axis=1))
        self.G -= self.G.clone()[0]