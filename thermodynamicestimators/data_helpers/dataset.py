import torch
import thermodynamicestimators.data_helpers.helpers as helpers

class dataset(torch.utils.data.Dataset):
    def __init__(self, potential=None, biases=None):
        self._potential_function = potential
        self._bias_functions = biases
        self._sampled_positions = None


    ''' The number of thermodynamic states that were sampled '''
    @property
    def n_states(self):
        return len(self.bias_functions)


    ''' The sampled data points. These are the coordinates of the MD/MC simulation'''
    @property
    def sampled_positions(self):
        return self._sampled_positions


    ''' The unbiased potential function '''
    @property
    def potential_function(self):
        return self._potential_function


    ''' The bias functions '''
    @property
    def bias_functions(self):
        return self._bias_functions


    ''' The bias potentials added to the unbiased potential function. 
    These functions govern the thermodynamic states that are sampled. '''
    @property
    def biased_potentials(self):
        return [lambda x, bias=_bias: self.potential(x) + bias(x) for _bias in self.bias_functions]


    def add_data(self, sampled_positions):
        sampled_positions = helpers.to_high_precision_tensor(sampled_positions)

        if self._sampled_positions is None:
            self._sampled_positions = sampled_positions
        else:
            torch.cat(self._sampled_positions, sampled_positions, dim=1, out=self._sampled_positions)
