from thermodynamicestimators.utilities import potential as potentials
from thermodynamicestimators.utilities import MCMC
import thermodynamicestimators.data_helpers.MBAR_dataset as MBAR_dataset
import torch


def make_test_case(test_name):

    if test_name == "double_well_1D":
        potential = potentials.double_well_1D()

        n_biases = 20
        bias_centers = torch.linspace(0, 100, n_biases)

        biases = [lambda x, r_0=bias_center: potentials.harmonic(x, k=0.1, r_0=r_0) for bias_center in bias_centers]

        simulations_per_bias = 1

        initial_coordinates = [None for _ in biases]
        histogram_range = torch.tensor([[0, 100]])

        sampler = MCMC.MCMC(histogram_range , max_step=3, n_dimensions=1, n_samples=1000)


    if test_name == "double_well_2D":
        potential = potentials.double_well_2D()

        bias_centers = [(10 * k + 5) / 3 for k in range(1,8)]
        biases = [(lambda r, r_0=bias_center : potentials.harmonic(r[0], k=0.2, r_0=r_0)) for bias_center in bias_centers]

        simulations_per_bias = 5

        initial_coordinates = [(c, torch.randint(5, 24)) for c in bias_centers for _ in range(simulations_per_bias)]
        histogram_range = torch.tensort([[5, 25], [5, 24]])


        sampler = MCMC.MCMC(histogram_range , max_step=3, n_dimensions=2, n_samples=1000)



    data = torch.tensor(get_data(sampler, potential, biases, simulations_per_bias, initial_coordinates))

    return MBAR_dataset.MBAR_dataset(potential=potential, biases=biases, sampled_positions=data)



def get_data(sampler, potential, biases, n_simulations, initial_coordinates):
    results = []

    for k, bias in enumerate(biases):
        biased_potential = lambda r: potential(r.item()) + bias(r.item())
        bias_results = []

        for i in range(n_simulations):
            samples = sampler.get_trajectory(biased_potential, r_initial=initial_coordinates[i])
            bias_results.extend(samples)

        results.append(bias_results)

    return torch.tensor(results)