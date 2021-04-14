import math
import torch
from thermodynamicestimators.estimators.thermodynamicestimator import ThermodynamicEstimator


class WHAM(ThermodynamicEstimator):
    def __init__(self, biases=None, n_bins=10):
        super().__init__()
        self.n_bins = n_bins
        self.n_simulations = len(biases)

        self.probabilities = torch.nn.Parameter(1 / n_bins * torch.ones(n_bins))

        self.normalization_constants = torch.ones(self.n_simulations)
        self.bias_coefficients = torch.zeros((self.n_simulations, n_bins))

        for i in range(self.n_simulations):
            for j in range(n_bins):
                self.bias_coefficients[i,j] = math.exp(-biases[i](j))


    def get_free_energy(self):
        return -torch.log(self.probabilities)


    def residue(self, data):
        with torch.no_grad():
            # compute new normalization constants based on the updated probabilities
            self.normalization_constants = 1 / torch.sum(self.bias_coefficients * self.probabilities, axis=1)

        p_old = self.probabilities.clone()
        p_new = data / (data.nelement() * torch.sum(self.normalization_constants * self.bias_coefficients.T, axis=1))

        return torch.square(torch.sub(p_new, p_old)).mean() / p_new.mean()

        # compute the log-likelyhood of the data given the probabilities (this is the loss)
        # under the constraint that everything is normalized

        # return torch.sum(torch.mul(data, torch.log(self.normalization_constants * (self.bias_coefficients * self.probabilities).T).T))