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


    def __getitem__(self, item):
        assert item < self.__len__()
        return self._sampled_positions[:, item]