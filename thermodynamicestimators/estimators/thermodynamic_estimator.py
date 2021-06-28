import time
import torch
import abc
import pickle


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
    def __init__(self, n_states, free_energy_log=None, device=None):
        super().__init__()

        self.n_states = n_states
        self._free_energies = torch.nn.Parameter(torch.zeros(self.n_states, dtype=torch.float64))
        self.epoch = 0

        self.free_energy_log = free_energy_log

        if self.free_energy_log is None:
            self.free_energy_log = "Stoch_F_per_iteration_{}.pkl".format(time.time())

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.to(self.device)


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

    def _shift_free_energies_relative_to_zero(self):
        """Subtract the minimum free energy from all free energies such all
        energies are relative to the minimum at zero."""
        with torch.no_grad():
            self._free_energies -= self._free_energies.clone()[0]


    def _handle_scheduler(self, scheduler, error):
        if not scheduler is None:
            if type(scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(error)
            else:
                scheduler.step()


    def _get_error(self, ground_truth, previous_estimate):
        if ground_truth is not None:
            return torch.max(torch.abs(self.free_energies - ground_truth))
        else:
            return torch.max(torch.abs(self.free_energies - previous_estimate))


    # TODO: get rid of dataset here
    def estimate(self, data_loader, dataset, optimizer=None, epoch_scheduler=None, batch_scheduler=None, tolerance=1e-8,
                 max_iterations=100, direct_iterate=False, ground_truth=None, log_interval=100):
        """Estimate the free energies.

        Parameters
        ----------
        data_loader : torch.torch.utils.data.DataLoader
            The dataloader holds a thermodynamicestimators.data_sets.dataset object
            that matches the estimator.
        optimizer : torch.optim.Optimizer
        epoch_scheduler : torch.optim.lr_scheduler._LRScheduler, default = None
            scheduler.step() is called after every epoch
        batch_scheduler : torch.optim.lr_scheduler._LRScheduler, default = None
            scheduler.step() is called after every batch
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
        log_interval: int, default=100
            Interval in which to log the current free energy estimate

        Returns
        -------
        free_energies : torch.Tensor
            Tensor containing the estimated free energies for each state.
        errors : list of floats
            The MSE at each epoch.

        """
        print(("Starting MBAR estimation... \n"
               "   batch size: {}\n"
               "   batches per epoch: {}\n"
               "   Initial learning rate: {}\n"
               "   Logging free energy estimate every {} batches.")
              .format(data_loader.batch_size, len(data_loader), optimizer.param_groups[0]['lr'], log_interval))

        errors = []

        previous_estimate = torch.zeros_like(self.free_energies).to(self.device)
        ground_truth = ground_truth.to(self.device)
        normalized_N_i = dataset.normalized_N_i.to(self.device)

        # start with error higher than tolerance so epoch loop begins
        error = tolerance + 1

        # extra counter for number of iterations. One might want to run estimate() multiple times with different
        # parameters, so keep track of total number of epochs with self.epochs, and for this run with i
        i = 0
        while error > tolerance:

            i += 1

            if i > max_iterations:
                print("Stochastic MBAR did not converge to tolerance {} after {} iterations.".format(tolerance,
                                                                                                     max_iterations))
                return False

            if direct_iterate:
                self.self_consistent_step(dataset)

            else:
                for batch_idx, batch in enumerate(data_loader):
                    optimizer.zero_grad()
                    loss = self.residue(batch.to(self.device), normalized_N_i)
                    loss.backward()
                    optimizer.step()

                    self._shift_free_energies_relative_to_zero()

                    error = self._get_error(ground_truth, previous_estimate)
                    previous_estimate = self.free_energies

                    if batch_idx % log_interval == 0:
                        with open(self.free_energy_log, 'ab+') as f:
                            x = self.epoch + batch_idx / len(data_loader)
                            pickle.dump((x, self.free_energies), f)

                    self._handle_scheduler(batch_scheduler, error)

            self.epoch += 1
            self._handle_scheduler(epoch_scheduler, error)

            print('Max abs error at epoch {}: {}'.format(self.epoch, error.item()))

        print('Stochastic MBAR converged to tolerance of {} after {} epochs'.format(tolerance, self.epoch))

        return True 
