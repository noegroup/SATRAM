import math
import torch
from thermodynamicestimators.estimators.thermodynamicestimator import ThermodynamicEstimator


class WHAM(ThermodynamicEstimator):
    def __init__(self, biases=None, n_bins=10):
        super().__init__()
        self.n_bins = n_bins
        self.n_biases = len(biases)

        self.probabilities = 1 / n_bins * torch.ones(n_bins)

        self.normalization_constants = torch.ones(self.n_biases)
        self.bias_coefficients = torch.zeros((self.n_biases, n_bins))

        for i in range(self.n_biases):
            for j in range(n_bins):
                self.bias_coefficients[i,j] = math.exp(-biases[i](j))


    # free energy profile estimate
    @property
    def free_energy(self):
        return -torch.log(self.probabilities)


    # compute normalization constants based on the probabilities
    def normalize(self):
        const = torch.sum(self.bias_coefficients * self.probabilities, axis=-1)

        # keep summing over all axes (dimensions of the histogram) until back at the desired shape.
        while self.normalization_constants.shape != const.shape:
            const = torch.sum(self.bias_coefficients * self.probabilities, axis=-1)

        self.normalization_constants = 1 / const


    # Perform one iteration over a batch of data
    def step(self, data, samples_per_bias, lr=0.01):

        # Satisfies the constraint that everything is normalized
        self.normalize()

        # compute new probabilities using new normalization constants
        self.probabilities = (1 - lr) * self.probabilities  +\
                             lr * data / (torch.sum(samples_per_bias * self.normalization_constants * self.bias_coefficients.T, axis=1))
