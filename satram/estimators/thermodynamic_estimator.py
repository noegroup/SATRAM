import torch
from ._common import *
from .implementation_manager import *
from satram.util import *


class ThermodynamicEstimator():
    """Estimator of free energies.

    Thermodynamic Estimator handles estimation of free energies. The specific
    implementation is chosen by the user

    Attributes
    ----------

    device : torch.device
        device on which the parameters live.
    lagtime : int, default=1
        chosen lagtime for counting transitions (for (SA)TRAM)
    maxiter : int, default=1000
        maximum number of iterations (default: 1000)
    maxerr : float, default=1e-8
        maximum error for achieving convergence
    callback_interval : int, default=1
        callback function is called every `callback_interval` epochs
    progress : object or None
        a progress bar such as tqdm for indicating estimation progress

    """


    def __init__(self, lagtime=1,
                 maxiter=1000, maxerr: float = 1e-8,
                 callback_interval=1,
                 progress=None, device="cpu", *args, **kwargs):

        self.lagtime = lagtime
        self.maxiter = maxiter
        self.maxerr = maxerr
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
        """ Free energies per thermodynamic state, :math:`f^k`.

        Returns
        -------
        Free energies : torch.Tensor
        """
        return compute_f_therm(self._f).cpu()


    @property
    def sample_weights(self):
        """ The unbiased sample weight per sample, :math:`\mu(x)`.

        Returns
        -------
        sample weights L torch.Tensor
        """
        if self.dataset is not None:
            _, log_R = compute_v_R(self._f, self._log_v, self.dataset.log_C_sym, self.dataset.state_counts,
                               self.dataset.log_N)
            return torch.logsumexp(compute_sample_weights(self._f, log_R, self.dataset.deterministic_dataloader,
                                                          device=self.device), 1)
        else:
            return None


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


    def compute_pmf(self, binned_trajs, n_bins):
        """ Compute the potential of mean force (PMF) over the given bins.

        Parameters
        ----------
        binned_trajs : torch.Tensor (N)
            The trajectories binned into the bins over which the PMF is to be
            computed.
        n_bins : int
            The total number of bins

        Returns
        -------
        PMF : torch.Tensor
            Tensor of shape (n_bins) containing the estimated PMF.
        """
        weights = self.sample_weights

        pmf = torch.zeros(n_bins)

        for i in range(len(pmf)):
            indices = torch.where(torch.Tensor(binned_trajs) == i)
            if len(indices[0]) > 0:
                pmf[i] = -torch.logsumexp(weights[indices], 0)
            else:
                pmf[i] = float("Inf")
        return pmf - pmf.min()


    def fit(self, data, callback=None, solver_type='SATRAM', initial_batch_size=256,
            batch_size_increase=None, delta_f_max=1.):
        """Estimate the free energies.

        Parameters
        ----------

        data : tuple
            data is a tuple containing (ttrajs, dtrajs, bias_matrices)
            * `ttrajs`: `ttrajs[i]` indicates for each sample in the $i$-th
              trajectory what thermodynamic state that sample was sampled at.
            * `dtrajs`: The discrete trajectories in the form of a list or array
              of numpy arrays. `dtrajs[i]` contains one trajectory.
              `dtrajs[i][n]` equals the Markov state index that the $n$-th
              sample from the $i$-th trajectory was binned into. Each of the
              `dtrajs` thus has the same length as the corresponding `traj`.
            * `bias_list`: The bias energy matrices. `bias_matrices[i][n, k]`
              equals the bias energy of the $n$-th sample from the $i$-th
              trajectory, evaluated at thermodynamic state $k$, $b^k(x_{i,n})$.
              The bias energy matrices should have the same size as dtrajs in
              both the first and second dimensions. The third dimension is of
              size `n_therm_states`, i.e. for each sample, the bias energy in
              every thermodynamic state is calculated and stored in the
              `bias_matrices`.
        callback : callable(f : torch.Tensor, log_v : torch.Tensor) -> void
            called every `self.callback_interval` epochs.
        solver_type : string, default='SATRAM'
            type of solver to estimate free energies with.
            one of 'TRAM', 'SATRAM', 'MBAR', 'SAMBAR'.
            If 'SATRAM' of 'SAMBAR' is used with a batch size increase, the
            solver will revert to 'TRAM' or 'MBAR' respectively, once the batch
            size reaches the total dataset size.
        initial_batch_size : int, default=256
            Initial batch size for stochastic approximation.
            Not used for MBAR and TRAM.
        batch_size_increase : int
            double the batch size every batch_size_increase epochs. If the batch
            size equals the dataset size, the estimator reverts to
            a deterministic implementation.
        delta_f_max : float, default=1.
            The maximum size of the free energy update for SATRAM, to avoid
             explosion of the free energy. The free energy update will be
            :math:`\Delta f_i^k = \mathrm{max}(delta\_f\_max, \eta \Delta f_i^k)
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

            error = self._get_iteration_error()

            if implementation_manager.step(i):
                self.dataset.init_dataloader(
                    min(implementation_manager.batch_size, implementation_manager.batch_size_memory_limit),
                    implementation_manager.is_stochastic)

            if i % self.callback_interval == 0 and callback is not None:
                callback(self._f.cpu(), self._log_v.cpu())

            if error < self.maxerr:
                return

            i += 1
