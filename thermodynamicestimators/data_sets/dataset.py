import torch


class Dataset(torch.utils.data.Dataset):
    """Data wrapper for using the thermodynamicestimator with a torch dataloader.

    Parameters
    ----------
    potential : callable
        Potential function of the reference state. Takes an input coordinate and
        outputs the potential at those coordinates.
    biases : list(callable)
        List of bias functions that take an input coordinate and output the bias
        potential at those coordinates.

    Attributes
    ----------
    _samples : torch.Tensor
        tensor of shape (S, N, d) where S is the number of states and N is the total
        number of samples, and d is the dimensionality of the sample. The sample
        at index [i,n] is either the bin index of the n'th sample taken at state
        i (in case of WHAM) or the potential energy value of the n'th sample evaluated
        at state i (in case of MBAR). In the latter case, d must 1.
    _N_i : torch.Tensor
        Tensor of shape (S) containing the total number of samples taken per state.

    """


    def __init__(self, samples=None, N_i=None, sampled_coordinates=None, bias_coefficients=None,
                 unbiased_potentials=None):
        assert (bias_coefficients is None or len(bias_coefficients) == len(N_i))
        assert (sum(N_i) == len(samples))
        self._samples = samples.double()
        self._sampled_coordinates = sampled_coordinates
        self._N_i = N_i.double()
        self._normalized_N_i = (self._N_i / torch.sum(N_i)).double()
	
        print(self._normalized_N_i)
        # for WHAM
        if bias_coefficients is not None:
            self._bias_coefficients = bias_coefficients.double()


    @property
    def n_states(self):
        """The number of thermodynamic states that were sampled (`int`)"""
        return len(self._N_i)


    @property
    def normalized_N_i(self):
        """The relative number of samples taken per state (`torch.Tensor`)

        This is the normalized N_i, so that the number of samples per state sum
        up to 1.
        For use with batch-wise iteration, to not have to re-calculating this on
        every batch.
        """
        return self._normalized_N_i


    @property
    def N_i(self):
        """The number of samples taken per thermodynamic state"""
        return self._N_i


    @property
    def samples(self):
        """The samples. Either potentials or bin indices (`torch.Tensor`)
        In case of sampled potentials (MBAR):
            Tensor of shape (S, N) where the element at index [i, j] is the bias
            energy of the j'th sample evaluated at the i'th thermodynamic state.
        In case of sampled coordinates (WHAM):
            Tensor of shape (N, D) Where N is the number of samples
            and D is the dimensionality of the coordinates."""
        return self._samples


    @property
    def sampled_coordinates(self):
        """The sampled coordinates (`torch.Tensor`)
        Used for calculating observables. """
        return self._sampled_coordinates


    @property
    def bias_coefficients(self):
        """The bias coefficient matrix for a discrete estimator (WHAM)"""
        return self._bias_coefficients


    def __getitem__(self, item):
        return self._samples[item]


    def __len__(self):
        return len(self._samples)


    def add_data(self, samples, N_i=None):
        """Adds data to the dataset.

        Parameters
        ----------
        samples : torch.Tensor
            Tensor of shape (S,N,d) containing all samples.
            S is the number of thermodynamic states, N is the maximum number of
            samples per state, and d is the sample dimension.
            For WHAM, a sample is a bin index and d >= 1. For MBAR, a sample is
            a potential value and d=1.
        N_i : torch.Tensor
            Tensor of shape (S) containing the total number of samples taken per
            state.
        """
        if N_i is None:
            N_i = torch.Tensor([len(samples[i]) for i in range(len(samples))])

        if self._samples is None:
            self._samples = samples
            self._N_i = N_i
        else:
            torch.cat(self._potentials, samples, dim=1, out=self._samples)
            self.N_i += N_i

        self._normalized_N_i = self._N_i / torch.sum(N_i)
