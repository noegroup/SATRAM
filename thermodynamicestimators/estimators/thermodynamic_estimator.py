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


    def __init__(self, device=None):
        super().__init__()

        self.epoch = 0

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
    def residue(self, samples):
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
    def self_consistent_step(self, samples=None):
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
            self._free_energies -= self._free_energies.clone().min()


    def _handle_schedulers(self, schedulers, error):
        for scheduler in schedulers:
            if type(scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(error)
            else:
                scheduler.step()


    def _get_error(self, previous_estimate):
        return torch.abs(self.free_energies - previous_estimate).max()


    def estimate(self, data_loader, optimizer=None, schedulers=None, tolerance=1e-10,
                 max_iterations=100, use_self_consistent_iteration=False, log_interval=1):
        """Estimate the free energies.

        Parameters
        ----------
        data_loader : torch.torch.utils.data.DataLoader
            The dataloader holds a thermodynamicestimators.data_sets.dataset object
            that matches the estimator.
        optimizer : torch.optim.Optimizer
        schedulers : list(torch.optim.lr_scheduler._LRScheduler), default = None
            scheduler.step() is called after every batch for every scheduler in
            the list of schedulers.
        tolerance : float, default = 1e-2
            The error tolerance. When the MSE of the estimated energies is below
            this value, the free energies are returned. The exact implementation
            of the error calculation of the MSE depends on the availability of the
            ground_truth
        max_iterations : int, default=1000
            The maximum number of iterations allowed for convergence.
        use_self_consistent_iteration : bool, default=False
            When True, use direct iteration.
        log_interval: int, default=100
            Interval in which to log the current free energy estimate

        Returns
        -------
        converged : bool

        """
        print(f"Starting free energy estimation... \n"
              f"   batch size: {data_loader.batch_size}\n"
              f"   batches per epoch: {len(data_loader)}\n"
              f"   Logging free energy estimate every {log_interval} batches.\n")

        if not optimizer is None:
            print(f"   Initial learning rate: {optimizer.param_groups[0]['lr']}")

        previous_estimate = torch.zeros_like(self.free_energies).to(self.device)

        # extra counter for number of iterations. One might want to run estimate() multiple times with different
        # parameters, so keep track of total number of epochs with self.epochs, and for this run with i
        i = 0
        while i < max_iterations:
            i += 1

            if use_self_consistent_iteration:
                self.self_consistent_step(data_loader.dataset)
                self._shift_free_energies_relative_to_zero()

            else:
                for batch_idx, batch in enumerate(data_loader):

                    if isinstance(optimizer, torch.optim.LBFGS):
                        def closure():
                            if torch.is_grad_enabled():
                                optimizer.zero_grad()
                            loss = self.residue(None)
                            if loss.requires_grad:
                                loss.backward()
                            return loss

                        loss = optimizer.step(closure)

                    else:
                        optimizer.zero_grad()
                        loss = self.residue(batch)
                        loss.backward()
                        optimizer.step()

                    self._shift_free_energies_relative_to_zero()

                    if not schedulers is None and len(schedulers) > 0:
                        self._handle_schedulers(schedulers, loss)


            error = self._get_error(previous_estimate)

            if self.epoch % log_interval == 0:
                print(f'Max abs increment at epoch {self.epoch}: {error}')

            if error < tolerance:
                print(f'Estimator converged to tolerance of {tolerance} after {self.epoch} epochs.')
                return True

            previous_estimate = self.free_energies

            self.epoch += 1

        print(f"Estimator did not converge to tolerance {tolerance} after {max_iterations} iterations.")
        return False