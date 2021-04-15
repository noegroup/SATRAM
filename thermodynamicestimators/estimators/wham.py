import math
import torch
from thermodynamicestimators.estimators.thermodynamicestimator import ThermodynamicEstimator


class WHAM(ThermodynamicEstimator):
    def __init__(self, biases=None, n_bins=10):
        super().__init__()
        self.n_bins = n_bins
        self.n_biases = len(biases)

        self.probabilities = torch.nn.Parameter(1 / n_bins * torch.ones(n_bins))

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
            self.normalization_constants = 1 / torch.sum(self.bias_coefficients * self.probabilities, axis=1)


    # compute the loss function for gradient descent
    def residue(self, data, samples_per_bias):

        with torch.no_grad():
            # Satisfies the constraint that everything is normalized
            self.normalize()

        # compute new probabilities using new normalization constants
        p_new = data / (torch.sum(samples_per_bias * self.normalization_constants * self.bias_coefficients.T, axis=1))

        # Return loss function: relative squared difference between old and new probabilities.
        # If WHAM equations have converged, this should no longer change and optimum is reached (and loss=0)
        return torch.square(torch.sub(p_new, self.probabilities.clone())).mean()