import numpy as np
from thermodynamicestimators.utilities import potential as potentials
from thermodynamicestimators.utilities import MCMC
import thermodynamicestimators.utilities.test_problem as test_problem
import math


def make_test_problem(test_name):

    simulations_per_bias = 1

    if test_name == "double_well_1D":
        potential = potentials.double_well_1D()

        n_biases = 20
        bias_centers = np.linspace(0, 100, n_biases)

        biases = [lambda x, r_0=bias_center: potentials.harmonic(x, k=0.1, r_0=r_0) for bias_center in bias_centers]
        bias_coefficients = np.zeros((n_biases, 100))

        initial_coordinates = [None for _ in biases]
        histogram_range = np.asarray([[0, 100]])

        sampler = MCMC.MCMC(histogram_range , max_step=3, n_dimensions=1, n_samples=1000)


    if test_name == "double_well_2D":
        potential = potentials.double_well_2D()

        bias_centers = [(10 * k + 5) / 3 for k in range(1,8)]
        biases = [(lambda r, r_0=bias_center : potentials.harmonic(r[0], k=0.2, r_0=r_0)) for bias_center in bias_centers]
        bias_coefficients = np.zeros((7,20,19))

        initial_coordinates = [(c, np.random.randint(5, 24)) for c in bias_centers for _ in range(simulations_per_bias)]
        histogram_range = np.asarray([[5, 25], [5, 24]])

        sampler = MCMC.MCMC(histogram_range , max_step=3, n_dimensions=2, n_samples=1000)


    for i in range(len(bias_coefficients)):
        for hist_coords, _ in np.ndenumerate(bias_coefficients[i]):
            bias_coefficients[i][hist_coords] = math.exp(-biases[i](hist_coords + histogram_range[:,0]))

    data = get_data(sampler, potential, biases, simulations_per_bias, initial_coordinates)

    return test_problem.TestProblem(potential=potential, bias_coefficients=bias_coefficients, histogram_range=histogram_range, data=data)



def get_data(sampler, potential, biases, n_simulations, initial_coordinates):
    results = []

    for k, bias in enumerate(biases):
        biased_potential = lambda r: potential(r) + bias(r)
        bias_results = []

        for i in range(n_simulations):
            samples = sampler.get_trajectory(biased_potential, r_initial=initial_coordinates[i])
            bias_results.extend(samples)

        results.append(bias_results)

    return np.asarray(results)