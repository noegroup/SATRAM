import math
from thermodynamicestimators.estimators.thermodynamicestimator import ThermodynamicEstimator
import numpy as np

class WHAM_nd(ThermodynamicEstimator):
    def __init__(self, biases=None, n_bins=10, d=1):
        super().__init__()
        self.n_bins = n_bins
        self.n_biases = len(biases)

        hist_shape = [n_bins for _ in range(d)]

        self.probabilities = 1 / n_bins * np.ones(tuple(hist_shape))

        self.normalization_constants = np.ones(self.n_biases)
        self.bias_coefficients = np.zeros(tuple( [self.n_biases] + hist_shape))

        for i in range(self.n_biases):
            for hist_coords, _ in np.ndenumerate(self.bias_coefficients[i]):
                self.bias_coefficients[i,hist_coords] = math.exp(-biases[i](hist_coords))


    # free energy profile estimate
    @property
    def free_energy(self):
        return -np.log(self.probabilities)

        # compute normalization constants based on the probabilities

    def normalize(self):
        const = np.sum(self.bias_coefficients * self.probabilities, axis=-1)

        # keep summing over all axes (dimensions of the histogram) until back at the desired shape.
        while self.normalization_constants.shape != const.shape:
            const = np.sum(self.bias_coefficients * self.probabilities, axis=-1)

        self.normalization_constants = 1 / const


    # Perform one iteration over a batch of data
    def step(self, data, samples_per_bias, lr=0.01):

        # Satisfies the constraint that everything is normalized
        self.normalize()

        # compute new probabilities using new normalization constants
        self.probabilities = (1 - lr) * self.probabilities  +\
                             lr * data / (np.sum(samples_per_bias * self.normalization_constants * self.bias_coefficients.T, axis=-1))
