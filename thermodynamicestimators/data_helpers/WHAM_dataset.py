import math
import torch
import thermodynamicestimators.data_helpers.dataset as dataset


class WHAM_dataset(dataset.dataset):
    def __init__(self, potential, biases, histogram_range, sampled_positions=None, bias_coefficients=None):
        super().__init__(potential, biases)
        super().add_data(sampled_positions)

        self._histogram_range = histogram_range

        self._bias_coefficients = bias_coefficients
        if bias_coefficients is None and biases is not None:
            bias_coefficients_shape = tuple([len(biases)] + [d_range[1]- d_range[0]for d_range in self._histogram_range])

            self._bias_coefficients = torch.zeros(bias_coefficients_shape)

            for i in range(len(self._bias_coefficients )):
                for hist_coords, _ in enumerate(self._bias_coefficients[i]):
                    self._bias_coefficients[i, hist_coords] = math.exp(-biases[i](hist_coords + histogram_range[:, 0]))


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
    where M is the number of states. The selected samples are binned in a separate histogram belonging to their state.
    The histogram tensor is of shape (M, d1, d2,...) with M the number of thermodynamic states and d1, d2,... the sizes 
    of the dimensions. 
    This method is written to be used with a data loader so that one item is indexed at a time. If a range index is used,
    the method loops over the samples to construct a histogram, which is very. Do not use a range index!'''
    def __getitem__(self, item):
        sample = self._sampled_positions[:, item]

        #TODO: n-dimensional histogram
        hist = torch.zeros(tuple([self.n_states]) + self.histogram_shape)

        # if multiple items were sampled
        if len(sample.shape) > 1:
            for i in range(sample.shape[1]):
                hist[torch.tensor(range(20)), sample[:,i]] += 1
        else:
            hist[torch.tensor(range(20)), sample] = 1

        return torch.tensor(hist)