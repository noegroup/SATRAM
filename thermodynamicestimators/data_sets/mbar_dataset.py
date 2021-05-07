import  torch
from thermodynamicestimators.data_sets import dataset
import thermodynamicestimators.utilities.helper_function as helpers


''' Class encapsulating the data for use with MBAR method.
Data are stored in tensor of shape [M, N] where M is the number of
thermodynamic states, and N the total number of data points. 
Each x_mn represents the potential energy of the n'th sample evaluated at the m'th thermodynamic state.
(MBAR does not care at which state a sample was sampled to calculate the free energies.)
N_m is the number of samples taken at thermodynamic state m. All N_i sum up to N'''

class MBARDataset(dataset.Dataset):
    # TODO: write such that potentials can be added in stead of only positions

    def __init__(self, potential, biases, sampled_positions, N_i = None):
        super().__init__(potential, biases)

        self._sampled_potentials = None
        self._unbiased_potentials = None

        self.add_data(sampled_positions, N_i)


    @property
    def unbiased_potentials(self):
        return self._unbiased_potentials

    @property
    def sampled_potentials(self):
        return self._sampled_potentials


    def __len__(self):
        return len(self._sampled_potentials[0])


    def __getitem__(self, item):
        return self._sampled_potentials[:, item], self.N_i


    ''' Allow adding data on the fly. Ideally, the data is added once or in large batches. '''
    def add_data(self, sampled_positions, N_i):
        sampled_positions = helpers.to_high_precision_tensor(sampled_positions)

        super().add_data(sampled_positions, N_i)

        if self._sampled_potentials is None:
            self._sampled_potentials = self.evaluate_at_all_states(sampled_positions)

        else:
            torch.cat(self._sampled_potentials, self.evaluate_at_all_states(sampled_positions))

        self._unbiased_potentials = self.evaluate_unbiased_potential(self.sampled_coordinates)


    ''' The potential energy of the observed trajectories, evaluated at all thermodynamic states. '''
    def evaluate_at_all_states(self, sampled_positions):
        return torch.stack([bias(torch.flatten(sampled_positions, 0, 1).squeeze(-1)) for bias in self._bias_functions]) + \
               self.evaluate_unbiased_potential(sampled_positions)


    ''' The unbiased potential values for each sample. This is needed for calculating an expectation value with MBAR.'''
    def evaluate_unbiased_potential(self, sampled_positions):
        return self._potential_function(torch.flatten(sampled_positions, 0, 1).squeeze(-1)).clone()