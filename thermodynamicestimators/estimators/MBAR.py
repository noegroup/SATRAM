from thermodynamicestimators.estimators.thermodynamicestimator import ThermodynamicEstimator
import torch
from functools import reduce



class MBAR(ThermodynamicEstimator):
    def __init__(self, n_states, biased_potentials):
        super().__init__()
        self.n_states = n_states
        self.f = torch.nn.Parameter(torch.zeros(n_states))
        self.biased_potentials = biased_potentials


    # free energy estimate per bias
    @property
    def free_energy(self):
        return self.f


    # get the partition function Z, or normalization constant of the boltzmann distribution
    @property
    def partition_function(self):
        return -torch.exp(self.free_energy)


    # get probability of observing a histogram of data.
    def get_probability_distribution(self, histogram, bin_centers):
        N_i = torch.sum(histogram, axis=1)  # total samples per simulations
        N = torch.sum(N_i).item()

        dG = torch.tensor([self.get_measure_of_states(histogram, bin_center) for bin_center in bin_centers])

        boltzman_factors = torch.tensor([[-potential(bin) for potential in self.biased_potentials] for bin in bin_centers] )

        F = torch.sum((N_i/(N * self.partition_function) * torch.exp(boltzman_factors)).T * dG, axis=0)
        return F/dG


    def get_measure_of_states(self, histogram, bin):
        N_i = torch.sum(histogram, axis=1)  # total samples per simulations
        N_j = torch.sum(histogram[:,bin])  # total samples over all simulations in this bin

        boltzman_factors = torch.tensor([-potential(bin) for potential in self.biased_potentials])
        denom = torch.sum(N_i * torch.exp(boltzman_factors) / self.partition_function)
        return 1/denom


    # compute the loss function for gradient descent
    # data has shape: (n_simulations, observed_potentials)
    def residue(self, data):
        with torch.no_grad():
            self.f -= self.f[0].clone()

        N = data.shape[1]
        N_i = data.shape[1]/data.shape[0]

        eps = 10e-2
        log_values = torch.log(eps + torch.sum(torch.exp(-data + (torch.ones_like(data.T) * self.f).T), axis=0))

        log_likelihood = (1/N) * torch.sum(log_values) - torch.sum(self.f * N_i/N)

        return log_likelihood
