import torch
from scipy import integrate
from thermodynamicestimators.test_cases import data_file_manager, test_case, potential
from thermodynamicestimators.utilities import mcmc


def make_test_case(test_name, load_from_disk=True):
    # dataset = data_file_manager.load_when_available(test_name)
    # if dataset is not None:
    #     return dataset

    if test_name == "double_well_1D":
        test_case = double_well_1d()

    if test_name == "double_well_2D":
        test_case = double_well_2d()

    data_file_manager.save_dataset(test_case, test_name)
    return test_case


def double_well_1d():
    ground_truth = torch.Tensor([ 0.0000e+00, -4.2250e+00, -6.5877e+00, -7.3749e+00, -6.9108e+00,
        -5.5602e+00, -3.7249e+00, -1.8271e+00, -2.7441e-01,  5.9830e-01,
         5.9830e-01, -2.7441e-01, -1.8271e+00, -3.7249e+00, -5.5602e+00,
        -6.9108e+00, -7.3749e+00, -6.5877e+00, -4.2250e+00,  1.9073e-06])
    potential_fn = potential.double_well_1d()

    bias_centers = torch.linspace(0, 100, 20)
    biases = [lambda x, r_0=bias_center: potential.harmonic(x, k=0.1, r_0=r_0) for bias_center in bias_centers]

    # ground_truth = calculate_ground_truth(potential_fn, biases)

    sampling_range = torch.tensor([[0, 100]])

    sampler = mcmc.MCMC(sampling_range, max_step=3, n_dimensions=1, n_samples=1000)
    sampled_coordinates = get_data(sampler, potential_fn, biases)

    return test_case.TestCase(potential_fn, biases, sampled_coordinates, sampling_range, ground_truth=ground_truth)


def double_well_2d():
    ground_truth = torch.tensor([0.0000, -2.5042, -1.9476, 0.3180, -0.0461, -0.0088, 2.1938])

    potential_fn = potential.double_well_2d()

    bias_centers = [(10 * k + 5) / 3 for k in range(1, 8)]


    def bias(r, r_0):
        if len(r.shape) == 1:
            r = r.unsqueeze(0)
        # bias depends only on x value
        x = r[:, 0]

        return potential.harmonic(x, r_0, k=0.2)


    biases = [(lambda r, r_0=bias_center: bias(r, r_0=r_0)) for bias_center in bias_centers]

    initial_coordinates = torch.tensor([[c, torch.randint(5, 26, size=[1]).item()] for c in bias_centers])
    sampling_range = torch.tensor([[5., 25.], [5., 25.]])
    sampler = mcmc.MCMC(sampling_range, max_step=3, n_dimensions=2, n_samples=1000)
    sampled_coordinates = get_data(sampler, potential_fn, biases, initial_coordinates)

    return test_case.TestCase(potential_fn, biases, sampled_coordinates, histogram_range=sampling_range,
                              ground_truth=ground_truth)


def get_data(sampler, potential_fn, biases, initial_coordinates=None):
    sampled_coordinates = []

    for k, bias in enumerate(biases):
        biased_potential = lambda r: potential_fn(r) + bias(r)

        if initial_coordinates is not None:
            samples = sampler.get_trajectory(biased_potential, r_initial=initial_coordinates[k])
        else:
            samples = sampler.get_trajectory(biased_potential)

        sampled_coordinates.append(samples)

    return torch.stack(sampled_coordinates)


def calculate_ground_truth(potential_fn, bias_fns):
    partition_sums = torch.zeros(len(bias_fns))

    for i, bias in enumerate(bias_fns):
        U = lambda x: torch.exp(-potential_fn(torch.Tensor([x])) - bias(torch.Tensor([x])))
        partition_sum = integrate.quad(U, 0, 100, epsabs=1e-3, epsrel=1e-3)
        partition_sums[i] = partition_sum[0]

    free_energies_ground_truth = -torch.log(partition_sums)
    free_energies_ground_truth -= free_energies_ground_truth[0].clone()

    return free_energies_ground_truth
