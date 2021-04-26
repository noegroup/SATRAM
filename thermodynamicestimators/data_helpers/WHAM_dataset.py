import math
import torch
import thermodynamicestimators.data_helpers.dataset as dataset


class WHAM_dataset(dataset):
    def __init__(self, potential=None, biases=None, histogram_range=None, sampled_positions=None, bias_coefficients=None):
        super().__init__(potential, biases, sampled_positions)

        self._histogram_range = histogram_range

        self._bias_coefficients = bias_coefficients
        if bias_coefficients is None and biases is not None:
            bias_coefficients_shape = tuple([len(biases)] + [d_range[1]- d_range[0]for d_range in self._histogram_range])

            self._bias_coefficients = torch.zeros(bias_coefficients_shape)

            for i in range(len(self._bias_coefficients )):
                for hist_coords, item in torch.ndenumerate(self._bias_coefficients[i]):
                    item = math.exp(-biases[i](hist_coords + histogram_range[:, 0]))


    ''' The bias coefficient matrix for a discrete estimator (WHAM) '''
    @property
    def bias_coefficients(self):
        return self._bias_coefficients


    ''' The histogram range over each dimensional axis.
    eg. for a two-dimensional histogram: [[min_x, max_x], [min_y, max_y]]'''
    @property
    def histogram_range(self):
        return self._histogram_range


    ''' The shape of the histogram in which to bin the data. Calculated from self.histogram_range. 
    All bins are assumed to be of size 1.'''
    @property
    def histogram_shape(self):
        return tuple([dimension_range[1]-dimension_range[0] for dimension_range in self.histogram_range])


    def __len__(self):
        return len(self._sampled_positions[0])


    ''' One sample consists of one sampled position for each thermodynamic state, returned in the shape of M histograms,
    where M is the number of states.
    The histogram tensor is of shape (M, d1, d2,...) with M the number of thermodynamic states and d1, d2,... the sizes 
    of the dimensions. The histogram values for one state are set to zero everywhere, except for the bin in which the 
    sample falls, which is set to 1. The total number of entries in all histograms is M, the number of states. '''
    def __getitem__(self, item):
        assert item < self.__len__()
        sample = self._sampled_positions[:, item]

        hist = torch.zeros(self.n_states + self.histogram_shape)

        state_indices = range(self.n_states)

        hist[state_indices][sample[state_indices]] = 1

        return torch.tensor(hist)