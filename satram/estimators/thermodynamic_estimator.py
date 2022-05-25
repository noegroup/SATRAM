import torch
from ._common import *
from .implementation_manager import *
from satram.util import *


class ThermodynamicEstimator():
    """Base class for a thermodynamic estimator.

    Thermodynamic Estimator handles running estimation epoch until the desired
    convergence criterium has been achieved.

    Attributes
    ----------

    device : torch.device
        device on which model lives.
    """


    def __init__(self, lagtime=1,
                 maxiter=1000, maxerr: float = 1e-8,
                 track_log_likelihoods=False, callback_interval=1,
                 progress=None, device="cpu", *args, **kwargs):

        self.lagtime = lagtime
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.track_log_likelihoods = track_log_likelihoods
        self.callback_interval = callback_interval
        self.progress = handle_progress(progress)
        self.device = torch.device(device)

        self._log_v = None
        self._f = None

        self._prev_f_therm = None
        self._prev_stat_vec = None

        self.dataset = None


    @property
    def free_energies(self):
        """ Free energy estimate.

        Returns
        -------
        Free energies : torch.Tensor
        """
        return self._f.cpu()


    @property
    def free_energies_per_thermodynamic_state(self):
        return compute_f_therm(self._f).cpu()


    def _initialize_f(self):
        self._f += self.dataset.dataloader.dataset[:, : self.dataset.n_therm_states].mean(axis=0).to(self.device)[:, None]


    def _get_iteration_error(self):
        f_therm = self.free_energies_per_thermodynamic_state
        stat_vec = torch.exp(f_therm[:, None] - self._f.cpu())

        err1 = (f_therm - self._prev_f_therm).abs().max().item()
        err2 = (self._prev_stat_vec - stat_vec).abs().max().item()

        self._prev_f_therm = f_therm
        self._prev_stat_vec = stat_vec
        return max(err1, err2)


    def _initialize_results_matrices(self, n_therm_states, n_markov_states):
        self._f = torch.zeros([n_therm_states, n_markov_states], dtype=torch.double).to(self.device)
        self._log_v = torch.zeros_like(self._f)
        self._prev_f_therm = torch.zeros([n_therm_states], dtype=torch.double)
        self._prev_stat_vec = torch.zeros([n_therm_states, n_markov_states], dtype=torch.double)


    @property
    def sample_weights(self):
        if self.dataset is not None:
            _, log_R = compute_v_R(self._f, self._log_v, self.dataset.log_C_sym, self.dataset.state_counts,
                               self.dataset.log_N)
            return torch.logsumexp(compute_sample_weights(self._f, log_R, self.dataset.dataloader, device=self.device), 1)
        else:
            return None


    def compute_pmf(self, binned_trajs, n_bins):
        weights = self.sample_weights

        pmf = torch.zeros(n_bins)

        for i in range(len(pmf)):
            indices = torch.where(torch.Tensor(binned_trajs) == i)
            if len(indices[0]) > 0:
                pmf[i] = -torch.logsumexp(-weights[indices], 0)
            else:
                pmf[i] = float("Inf")
        return pmf - pmf.min()


    def fit(self, data, callback=None, solver_type='SATRAM', initial_batch_size=256,
            batch_size_increase=None, delta_f_max=1.):
        """Estimate the free energies.

        Parameters
        ----------
        batch_size_increase : int
            double the batch size every batch_size_increase epochs. If the batch size equals the dataset size,
            TRAM is used.
        """
        data, state_counts, transition_counts = process_input(data, self.lagtime)

        implementation_manager = ImplementationManager(solver_type=solver_type,
                                                       initial_batch_size=initial_batch_size,
                                                       batch_size_increase=batch_size_increase,
                                                       total_dataset_size=len(data))

        self.dataset = Dataset(data, state_counts, transition_counts, device=self.device,
                               batch_size=implementation_manager.batch_size,
                               is_stochastic=implementation_manager.is_stochastic)

        self._initialize_results_matrices(state_counts.shape[0], state_counts.shape[1])
        self._initialize_f()

        for i in self.progress(range(self.maxiter)):

            self._f, self._log_v = implementation_manager.solver(self.dataset, self._f, self._log_v,
                                                                 lr=implementation_manager.learning_rate,
                                                                 batch_size=implementation_manager.batch_size,
                                                                 delta_f_max=delta_f_max)

            print(self.dataset.dataloader.batch_size)
            error = self._get_iteration_error()

            if implementation_manager.step(i):
                self.dataset.init_dataloader(
                    min(implementation_manager.batch_size, implementation_manager.batch_size_memory_limit),
                    implementation_manager.is_stochastic)

            if i % self.callback_interval == 0 and callback is not None:
                callback(self._f.cpu(), self._log_v.cpu())

            if error < self.maxerr:
                return
