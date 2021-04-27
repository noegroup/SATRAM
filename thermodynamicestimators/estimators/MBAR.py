import torch
from thermodynamicestimators.estimators.thermodynamicestimator import ThermodynamicEstimator


class MBAR(ThermodynamicEstimator):
    def __init__(self, n_states):
        super().__init__()
        self.n_states = n_states
        self._free_energy = torch.nn.Parameter(torch.zeros(self.n_states, dtype=torch.float64))


    '''free energy estimate per bias '''
    @property
    def free_energy(self):
        return self._free_energy.detach().clone()


    '''get the partition function for each (biased) thermodynamic state'''
    @property
    def partition_functions(self):
        return torch.exp(-self.free_energy)


    '''get the unbiased partition function'''
    def get_unbiased_partition_function(self, data):
        return torch.sum(torch.exp(-data) * self.get_sample_weights(data))


    '''Get the weights of all samples.
    The weight of a sample is the inverse of sum over all thermodynamic states of the sample probability at that thermodynamic state
    This is the denominator of eqn. (21) in the 2012 paper: https://doi.org/10.1063/1.3701175 '''
    def get_sample_weights(self, data):
        # total samples per simulation.
        # TODO: make variable
        N_i = 1000

        return 1/ torch.sum(((N_i/self.partition_functions) * torch.exp(-data).T), axis=1)


    '''Get the expectation value of an observable function based on the observed data'''
    def get_expectation_value(self, dataset, observable_function):

        samples = dataset.sampled_positions.flatten(0,1)

        # construct a matrix to store the computed observables
        result_shape = observable_function(samples[0]).shape
        observable_values = torch.zeros(samples.shape[:1] + result_shape)

        # fill it with the observed values
        for s_i in range(len(samples)):
            observable_values[s_i] = observable_function(samples[s_i])

        # weight the observed values with the sample probabilities and sum over all samples
        return  torch.sum(self.get_weighted_observables(observable_values, dataset), axis=0)



    ''' Weight the observed values by multiplying with the sample probabilities '''
    def get_weighted_observables(self, observable_values, dataset):

        unbiased_potentials = dataset.evaluate_unbiased_potential(dataset.sampled_positions)

        # weigh each observed value with the probability of the sample
        res= observable_values.T *  torch.exp(-unbiased_potentials) * self.get_sample_weights(dataset[:]) /\
               self.get_unbiased_partition_function(dataset[:])
        return res.T


    ''' Subtract the first free energy from all free energies such that the first is zero and all other energies are
    relative to the first. '''
    def shift_free_energies_relative_to_zero(self):
        with torch.no_grad():
            self._free_energy -= self._free_energy[0].clone()


    '''
    compute the loss function for gradient descent
    data has shape: [M, N] where M is the number of thermodynamic states, and N the total number of observed samples.
    data[m,n] is sample n evaluated at state m.
    This returns the value of equation (7) in paper: 10.1021/acs.jctc.8b01010
    Note: eq. (7) is f as a function of {b_1, ..., b_M}. Here f is written as a function of {G*_1,... G*_M}
    Additive constants are ignored since they don't affect the gradient.
     '''
    def residue(self, data):

        # total number of samples
        N = data.shape[0]

        # Number of samples taken per state. For now: assume equal number of samples.
        # TODO: make this variable.
        N_i = int(N/data.shape[1])

        log_sum_arg = -data + self._free_energy

        logsum = torch.log(torch.sum(N_i * torch.exp(log_sum_arg), axis=1))

        objective_function = (torch.sum(logsum) - torch.sum(self._free_energy * N_i))

        return objective_function
