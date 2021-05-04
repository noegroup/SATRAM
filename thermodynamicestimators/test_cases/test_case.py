import torch
from thermodynamicestimators.data_sets import wham_dataset, mbar_dataset


class TestCase():

    def __init__(self, potential, biases, sampler, sampling_range, initial_coordinates=None):
        self.potential = potential
        self.biases = biases
        self.sampling_range = sampling_range

        self.data = self.get_data(sampler, potential, biases, initial_coordinates)


    def get_data(self, sampler, potential, biases, initial_coordinates):
        results = []

        for k, bias in enumerate(biases):
            biased_potential = lambda r: potential(r) + bias(r)

            if initial_coordinates is not None:
                samples = sampler.get_trajectory(biased_potential, r_initial=initial_coordinates[k])
            else:
                samples = sampler.get_trajectory(biased_potential)

            results.append(samples)

        return torch.stack(results)


    def to_wham_dataset(self):
        return wham_dataset.WHAMDataset(self.potential, self.biases, self.data, self.sampling_range)


    def to_mbar_dataset(self):
        return mbar_dataset.MBARDataset(self.potential, self.biases, self.data)
