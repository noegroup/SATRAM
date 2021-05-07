from thermodynamicestimators.data_sets import wham_dataset, mbar_dataset


class TestCase:
    """Wrapper for test case data that build an actual dataset object out of the
    test case."""
    def __init__(self, potential, biases, sampled_coordinates, histogram_range):
        self.potential = potential
        self.biases = biases
        self.histogram_range = histogram_range
        self.sampled_coordinates = sampled_coordinates


    def to_wham_dataset(self):
        return wham_dataset.WHAMDataset(self.potential, self.biases, sampled_positions=self.sampled_coordinates,
                                        histogram_range=self.histogram_range)


    def to_mbar_dataset(self):
        return mbar_dataset.MBARDataset(self.potential, self.biases, self.sampled_coordinates)
