from thermodynamicestimators.test_cases import potential as potentials
from thermodynamicestimators.utilities import MCMC
import thermodynamicestimators.data_sets.MBAR_dataset as MBAR_dataset
import thermodynamicestimators.data_sets.WHAM_dataset as WHAM_dataset
import torch
import thermodynamicestimators.test_cases.data_file_manager as file_manager

def make_test_case(test_name, method):

    dataset = file_manager.load_when_available(test_name, method)
    if dataset is not None:
        return dataset

    if test_name == "double_well_1D":
        potential = potentials.double_well_1D()

        n_biases = 20
        bias_centers = torch.linspace(0, 100, n_biases)

        biases = [lambda x, r_0=bias_center: potentials.harmonic(x, k=0.1, r_0=r_0) for bias_center in bias_centers]

        simulations_per_bias = 1

        initial_coordinates = [None for _ in biases]
        sampling_range = torch.tensor([[0, 100]])

        sampler = MCMC.MCMC(sampling_range , max_step=3, n_dimensions=1, n_samples=1000)


    if test_name == "double_well_2D":
        potential = potentials.double_well_2D()

        bias_centers = [(10 * k + 5) / 3 for k in range(1,8)]

        def bias(r, r_0):
            if len(r.shape) == 1:
                r = r.unsqueeze(0)
            # bias depends only on x value
            x = r[:, 0]

            return potentials.harmonic(x, r_0, k=0.2)

        biases = [(lambda r, r_0=bias_center : bias(r, r_0=r_0)) for bias_center in bias_centers]

        simulations_per_bias = 1

        initial_coordinates = torch.tensor([[c, torch.randint(5, 26, size=[1]).item()] for c in bias_centers for _ in range(simulations_per_bias)])
        sampling_range = torch.tensor([[5., 25.], [5., 25.]])


        sampler = MCMC.MCMC(sampling_range , max_step=3, n_dimensions=2, n_samples=20000)



    data = get_data(sampler, potential, biases, simulations_per_bias, initial_coordinates)

    if method == 'WHAM':
        dataset = WHAM_dataset.WHAM_dataset(potential=potential, biases=biases, sampled_positions=data, histogram_range=sampling_range)
    if method == "MBAR":
        dataset = MBAR_dataset.MBAR_dataset(potential=potential, biases=biases, sampled_positions=data)

    file_manager.save_dataset(dataset, test_name, method)
    return dataset


def get_data(sampler, potential, biases, n_simulations, initial_coordinates):
    results = []

    for k, bias in enumerate(biases):
        biased_potential = lambda r: potential(r) + bias(r)
        bias_results = []

        for i in range(n_simulations):
            samples = sampler.get_trajectory(biased_potential, r_initial=initial_coordinates[i])
            bias_results.extend(samples)

        results.append(torch.stack(bias_results))

    return torch.stack(results)