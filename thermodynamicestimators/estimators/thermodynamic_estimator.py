import time
import torch
import abc


class ThermodynamicEstimator(torch.nn.Module):
    """Base class for a thermodynamic estimator.

    Thermodynamic Estimator handles running estimation epoch until the desired
    convergence criterium has been achieved.

    Attributes
    ----------
    n_states : int
        The number of thermodynamic states
    _free_energies : torch.Tensor
        A Tensor of shape (n_states) containing the estimated free energies.
        These are the parameters of the estimator and automatically updated
        by torch Autograd.
    """


    def __init__(self, n_states):
        super().__init__()
        self.n_states = n_states
        self._free_energies = torch.nn.Parameter(torch.zeros(self.n_states, dtype=torch.float64))


    @property
    def free_energies(self):
        """ Free energy estimate per thermodynamic state.

        Returns
        -------
        Free energies : torch.Tensor
        """
        return self._free_energies.detach().clone()


    @property
    def partition_functions(self):
        """ The partition function for each (biased) thermodynamic state.

        The partition function for each biased state follows directly from the
        free energy estimate according to

        .. math::

            \hat{Z}_i = e^{-f_i}

        Returns
        -------
        partition functions : torch.Tensor
            A Tensor of shape (S), containing the partition function for each
            of the S thermodynamic states.

        See also
        --------
        free_energies
        """
        return torch.exp(-self.free_energies)


    @abc.abstractmethod
    def residue(self, samples, normalized_N_i):
        """The value of the objective function that needs to be minimized to obtain
        the free energies.

        Parameters
        ----------
        samples : object
            The samples, either binned coordinates (WHAM) or potential energies
            (MBAR). The shape of the data depends on what the concrete implementation
            of the estimator expects.

        Returns
        -------
        loss : torch.Tensor
            Tensor of shape (1) containing the value of the optimization function.
            The gradient of the loss needs to be accesible.
         """
        return


    @abc.abstractmethod
    def self_consistent_step(self, samples, normalized_N_i):
        """Performs one direct iteration step solving the self consistent equations
        of the estimators.

        Parameters
        ----------
        data : object
            The shape of the data depends on what the concrete implementation of
            the estimator expects. The correct data object should be passed by the
            dataset corresponding to the estimator.
         """
        return


    def shift_free_energies_relative_to_zero(self):
        """Subtract the minimum free energy from all free energies such all
        energies are relative to the minimum at zero."""
        with torch.no_grad():
            self._free_energies -= torch.min(self._free_energies.clone())


    def estimate(self, data_loader, dataset, optimizer=None, scheduler=None, tolerance=1e-2, max_iterations=100,
                 direct_iterate=False, ground_truth=None):
        """Estimate the free energies.

        Parameters
        ----------
        data_loader : torch.torch.utils.data.DataLoader
            The dataloader holds a thermodynamicestimators.data_sets.dataset object
            that matches the estimator.
        optimizer : torch.optim.Optimizer
        scheduler : torch.optim.lr_scheduler._LRScheduler
        tolerance : float, default = 1e-2
            The error tolerance. When the MSE of the estimated energies is below
            this value, the free energies are returned. The exact implementation
            of the error calculation of the MSE depends on the availability of the
            ground_truth
        max_iterations : int, default=1000
            The maximum number of iterations allowed for convergence.
        direct_iterate : bool, default=False
            When True, use direct iteration.
        ground_truth: torch.Tensor, default=None
            When the ground truth is given, the error is computed as an MSE with
            respect to the ground truth. Otherwise, the MSE with respect to the
            estimate from the previous epoch is calculted. When the latter is the
            case, the tolerance should be set to a smaller number (e.g. 1e-8) to
            ensure convergence.

        Returns
        -------
        free_energies : torch.Tensor
            Tensor containing the estimated free energies for each state.
        errors : list of floats
            The MSE at each epoch.

        """
        errors = []
        running_times = []

        previous_estimate = torch.zeros_like(self.free_energies)

        # start with error higher than tolerance so epoch loop begins
        error = tolerance + 1
        epoch = 0

        while epoch < max_iterations and error > tolerance:
            t0 = time.time()

            epoch += 1

            for (idx, batch) in enumerate(data_loader):

                if direct_iterate:
                    self.self_consistent_step(batch, dataset.normalized_N_i)

                else:
                    optimizer.zero_grad()
                    loss = self.residue(batch, dataset.normalized_N_i)
                    loss.backward()
                    optimizer.step()

            t1 = time.time()
            running_times.append(t1 - t0)

            if not scheduler is None:
                scheduler.step()

            # avoid free energies getting to large by shifting them back towards zero.
            self.shift_free_energies_relative_to_zero()

            if ground_truth is not None:
                error = torch.abs(torch.square(self.free_energies - ground_truth).mean() / ground_truth.mean())
            else:
                lr = optimizer.param_groups[0]['lr']
                error = torch.abs(
                    torch.square(self.free_energies - previous_estimate).mean() / (lr * previous_estimate.mean()))
                previous_estimate = self.free_energies

            print('Error at epoch {}: {}'.format(epoch, error.item()))
            errors.append(error.item())

        print('average running time per epoch: {}'.format(torch.tensor(running_times).mean().item()))

        return self.free_energies, errors
