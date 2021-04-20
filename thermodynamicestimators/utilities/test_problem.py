


class TestProblem:
    def __init__(self, potential=None, bias_coefficients=None, histogram_range=None, data=None):
        self._potential = potential
        self._bias_coefficients = bias_coefficients
        self._data = data
        self._histogram_range = histogram_range

    @property
    def potential(self):
        return self._potential


    @property
    def bias_coefficients(self):
        return self._bias_coefficients


    @property
    def data(self):
        return self._data


    @property
    def histogram_range(self):
        return self._histogram_range


    @property
    def histogram_shape(self):
        return tuple([dimension_range[1]-dimension_range[0] for dimension_range in self.histogram_range])