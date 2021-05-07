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
    _potential_function : callable
        Takes an input coordinate and outputs the potential at those coordinates.
    _bias_functions : list(callable)
        List of callables that take an input coordinate and output the bias potential
        at those coordinates.
        len(_bias_functions) = S, there is one bias function specifying each
        thermodynamic state.
    _sampled_coordinates : torch.Tensor
        Tensor containing all sampled coordinates. Is of shape (S, N, D) where S
        is the number of thermodynamic states, N is the maximum number of samples
        per state, and D is the dimension of the coordinates.
    _N_i : torch.Tensor
        Tensor of shape (S) containing the total number of samples taken per state.

    """
    def __init__(self, potential=None, biases=None):
        self._potential_function = potential
        self._bias_functions = biases

        # add sampled positions manually by calling add_data()
        self._sampled_coordinates = None
        self._N_i = None


    @property
    def n_states(self):
        """The number of thermodynamic states that were sampled (`int`)"""
        return len(self.bias_functions)


    @property
    def sampled_coordinates(self):
        """The sampled data points (`torch.Tensor`)
        These are the coordinates of the MD/MC simulation"""
        return self._sampled_coordinates


    @property
    def N_i(self):
        """The number of samples taken per state (`torch.Tensor`)"""
        return self._N_i


    @property
    def potential_function(self):
        """The unbiased potential function (`callable`)"""
        return self._potential_function


    @property
    def bias_functions(self):
        """ The bias potential functions. (`list(callable)`)
        These are added to the unbiased potential to define a thermodynamic state.
        """
        return self._bias_functions


    @property
    def biased_potentials(self):
        """The bias potentials added to the unbiased potential function. (`list(callable)`)
        These functions govern the thermodynamic states that are sampled. """
        return [lambda x, bias=_bias: self.potential_function(x) + bias(x) for _bias in self.bias_functions]


    def add_data(self, sampled_coordinates, N_i=None):
        """Adds data to the dataset.

        Parameters
        ----------
        sampled_coordinates : torch.Tensor
            Tensor of shape (S,N,D) containing all sampled coordinates.
            S is the number of thermodynamic states, N is the maximum number of
            samples per state, and D is the dimension of the coordinates.
        N_i : torch.Tensor
            Tensor of shape (S) containing the total number of samples taken per
            state.
        """
        if N_i is None:
            N_i = torch.Tensor([len(sampled_coordinates[i]) for i in range(len(sampled_coordinates))])

        if self._sampled_coordinates is None:
            self._sampled_coordinates = sampled_coordinates
            self._N_i = N_i
        else:
            torch.cat(self._sampled_coordinates, sampled_coordinates, dim=1, out=self._sampled_coordinates)
            self.N_i += N_i