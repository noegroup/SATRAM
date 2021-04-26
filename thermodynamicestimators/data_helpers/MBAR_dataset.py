import  torch
import thermodynamicestimators.data_helpers.dataset as dataset
import thermodynamicestimators.utilities.helper_function as helpers


''' Class encapsulating the data for use with MBAR method.
Data are stored in tensor of shape [M, N] where M is the number of
thermodynamic states, and N the total number of data points. 
Each x_mn represents the potential energy of the n'th sample evaluated at the m'th thermodynamic state.
(MBAR does not care at which state a sample was sampled to calculate the free energies.)
N_m is the number of samples taken at thermodynamic state m. All N_m sum up to N'''

class MBAR_dataset(dataset.dataset):
    # TODO: write such that potentials can be added in stead of only positions

    def __init__(self, potential, biases, sampled_positions, N_m = None):
        super().__init__(potential, biases)

        self._sampled_potentials = None
        self._N_m = None

        self.add_data(sampled_positions, N_m)


    def __len__(self):
        return len(self._sampled_potentials[0])


    def __getitem__(self, item):
        return self._sampled_potentials[:, item]


    def shuffle(self):
        self._sampled_potentials = self._sampled_potentials[:,torch.randperm(self.__len__())]


    ''' Allow adding data on the fly '''
    def add_data(self, sampled_positions, N_m):
        sampled_positions = helpers.to_high_precision_tensor(sampled_positions)

        if N_m is None:
            N_m = [len(sampled_positions[i]) for i in range(len(sampled_positions))]

        assert len(sampled_positions) == len(N_m)
        assert [len(sampled_positions[i]) == N_m[i] for i in range(len(N_m))]

        super().add_data(sampled_positions)

        if self._sampled_potentials is None:
            self._sampled_potentials = self.evaluate_at_all_states(sampled_positions)
            self._N_m = N_m

        else:
            assert len(sampled_positions) == len(self.sampled_positions)

            torch.cat(self._sampled_potentials, self.evaluate_at_all_states(sampled_positions))
            self.N_m += N_m


    ''' The potential energy of the observed trajectories, evaluated at all thermodynamic states. '''
    def evaluate_at_all_states(self, sampled_positions):
        return torch.stack([bias(torch.flatten(sampled_positions)) for bias in self._bias_functions]).squeeze(-1) + \
               self.evaluate_unbiased_potential(sampled_positions)


    ''' The unbiased potential values for each sample. This is needed for calculating an expectation value with MBAR.'''
    def evaluate_unbiased_potential(self, sampled_positions):
        return torch.tensor(self._potential_function(torch.flatten(sampled_positions)), dtype=torch.float64)