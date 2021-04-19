import numpy as np
from thermodynamicestimators.utilities import potential as potentials
from thermodynamicestimators.utilities import MCMC


class data_generator():
    def __init__(self, args):

        self.n_simulations = args.n_simulations

        if args.potential == "double_well_1D":
            self._potential = potentials.double_well_1D()

            bias_centers = np.linspace(args.x_min, args.x_max, 10)
            self._biases = [lambda x, r_0 = bias_center: potentials.harmonic(x, k=0.1, r_0=r_0) for bias_center in bias_centers]

            self.initial_coordinates = [None for _ in self.biases]

        if args.potential == "double_well_2D":
            self._potential = potentials.double_well_2D()

            bias_centers = [(10 * k + 5) / 3 for k in range(1,8)]
            self._biases = [(lambda r, r_0=bias_center : potentials.harmonic(r[0], k=0.2, r_0=r_0)) for bias_center in bias_centers]

            self.initial_coordinates = [(c, np.random.randint(args.y_min, args.y_max)) for c in bias_centers for _ in range(args.n_simulations)]


        self.sampler = MCMC.MCMC(args)

    @property
    def potential(self):
        return self._potential


    @property
    def biases(self):
        return np.repeat(self._biases, self.n_simulations)


    def get_data(self):
        results = []

        for k, bias in enumerate(self._biases):
            biased_potential = lambda r: self.potential(r) + bias(r)

            for i in range(self.n_simulations):
                samples = self.sampler.get_trajectory(biased_potential, r_initial=self.initial_coordinates[i])
                results.append(samples)

        return np.asarray(results)