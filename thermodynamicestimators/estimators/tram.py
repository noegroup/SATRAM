import torch
from thermodynamicestimators.estimators.thermodynamic_estimator import ThermodynamicEstimator


epsilon = 1e-40


class TRAM(ThermodynamicEstimator):
    """Free energy estimator based on the TRAM equations. """


    def __init__(self, n_markov_states, discretized_trajectories, bias_energies, lag_time=1, device='cpu',
                 parameter_log_filepath=None):
        super().__init__()
        print(f"Initializing TRAM on {device}", flush=True)
        self.device = device
        self._lag = lag_time
        self.parameter_log_filepath = parameter_log_filepath

        self.n_therm_states = len(discretized_trajectories)
        self.n_markov_states = n_markov_states

        self._free_energies = torch.nn.Parameter(
            torch.ones((self.n_therm_states, self.n_markov_states), dtype=torch.float64))
        self._lagrangian_mult_log = torch.nn.Parameter(
            torch.ones((self.n_therm_states, self.n_markov_states), dtype=torch.float64))

        self.bias_energies = bias_energies

        print(f"Initializing count matrices with lag time {self._lag}...", flush=True)
        self._effective_state_counts_log = torch.zeros((self.n_therm_states, self.n_markov_states), dtype=torch.float64)
        self._transition_matrix_log = torch.zeros((self.n_therm_states, self.n_markov_states, self.n_markov_states),
                                                  dtype=torch.float64)

        # count transitions with lag time
        self._total_transition_counts = self._compute_transition_counts(discretized_trajectories).to(device)

        # count number of occurences in each markov state per therm. state
        self._total_state_counts = self._compute_state_counts(discretized_trajectories).to(device)
        self._states_with_counts = torch.where(self._total_state_counts > 0)

        # matrix containing log(Ck_ij + Ck_ji) --> total transition counts coming in and going out of the state.
        self._transitions_in_and_out_log = torch.log(self._compute_transitions_in_and_out() + epsilon).to(device)

        self.to(self.device)

        print("Initialization done", flush=True)


    @property
    def free_energies(self):
        free_energies = self._free_energies.detach()
        return free_energies


    @property
    def lagrangians(self):
        return torch.exp(self._lagrangian_mult_log.detach())


    @property
    def free_energies_per_therm_state(self):
        f_therm = -torch.logsumexp(-self.free_energies, dim=1)
        return f_therm - f_therm.min()


    def free_energies_per_markov_state(self, transitions):
        free_energies_per_markov_state = torch.zeros(self.n_markov_states)

        # collect all indices of samples in 1D-tensors
        ii = transitions[:, 1].type(torch.LongTensor)

        # get the sample weight for all visited states
        for i in range(self.n_markov_states):
            samples_in_i = torch.where(ii == i)
            mu_i = -torch.logsumexp(
                self._effective_state_counts_log[:, ii] + self._free_energies[:, ii] - self.bias_energies[samples_in_i],
                dim=0)

            free_energies_per_markov_state[i] = torch.logsumexp(mu_i, dim=0)
        return free_energies_per_markov_state


    def _compute_transition_counts(self, d_trajectories):
        transition_counts = torch.zeros((self.n_therm_states, self.n_markov_states, self.n_markov_states),
                                        dtype=torch.int64)

        for k, traj in enumerate(d_trajectories):

            for sample_idx in range(len(traj) - self._lag):
                transition_counts[k, traj[sample_idx], traj[sample_idx + self._lag]] += 1

        return transition_counts


    def _compute_state_counts(self, d_trajectories):
        state_counts = torch.zeros((self.n_therm_states, self.n_markov_states), dtype=torch.int64)

        for k, traj in enumerate(d_trajectories):
            for sample_idx in range(len(traj)):
                state_counts[k, traj[sample_idx]] += 1
        return state_counts


    def _compute_transitions_in_and_out(self):
        return self._total_transition_counts + torch.transpose(self._total_transition_counts, dim0=1, dim1=2)


    def _compute_effective_state_counts_log(self):
        vj_exp_fj = (self._free_energies + self._lagrangian_mult_log).unsqueeze(1).expand(-1, self.n_markov_states, -1)

        denominator = torch.logsumexp(
            torch.stack(
                [(self._free_energies + self._lagrangian_mult_log).unsqueeze(2).expand(-1, -1, self.n_markov_states),
                 vj_exp_fj]), dim=0)
        numerator = self._transitions_in_and_out_log + vj_exp_fj

        effective_counts = torch.logsumexp(numerator - denominator, dim=2)

        extra_counts = self._total_state_counts - torch.sum(self._total_transition_counts, dim=1)
        extra_counts[torch.where(extra_counts < 0)] = 0
        extra_counts = torch.log(extra_counts + epsilon)

        self._effective_state_counts_log = torch.logsumexp(torch.stack((effective_counts, extra_counts)), dim=0)


    def _compute_transition_matrix_log(self):

        numerator = self._transitions_in_and_out_log + self._free_energies.unsqueeze(2)

        f_plus_log_l = (self._free_energies + self._lagrangian_mult_log)
        f_plus_log_l = f_plus_log_l.unsqueeze(2).expand(-1, -1, self.n_markov_states)

        denominator = torch.logsumexp(
            torch.stack([f_plus_log_l, torch.transpose(f_plus_log_l, 1, 2)]), dim=0)

        transition_matrix = torch.exp(numerator - denominator)

        # # NORMALIZE
        sum_j = torch.sum(transition_matrix, dim=2, keepdim=True)
        max_sum = torch.max(sum_j, dim=1, keepdim=True)[0]

        # create a full matrix to add to transition_matrix; multi-indexing the transition matrix leads to an inplace
        # operation error on gradient evaluation.
        diag_addon = torch.zeros_like(transition_matrix).to(self.device)
        diag_addon += (torch.eye(self.n_markov_states).unsqueeze(0).to(self.device) * (max_sum - sum_j))

        transition_matrix = (transition_matrix + diag_addon) / max_sum

        self._transition_matrix_log = torch.log(transition_matrix)


    def _get_error(self, previous_estimate):
        f_k = self.free_energies_per_therm_state
        f_k -= f_k.min()

        previous_f_k = -torch.logsumexp(-previous_estimate, dim=1)
        previous_f_k -= previous_f_k.min()

        return torch.abs(f_k - previous_f_k).max()


    def _compute_new_lagrangians(self):

        denominator = self._free_energies.unsqueeze(1) - self._free_energies.unsqueeze(2) + \
                      self._lagrangian_mult_log.unsqueeze(1) - self._lagrangian_mult_log.unsqueeze(2)

        denominator = torch.logsumexp(torch.stack([denominator, torch.zeros_like(denominator)]), dim=0)

        new_l = torch.logsumexp(self._transitions_in_and_out_log - denominator, dim=2)
        new_l[self._states_without_counts] = float("-inf")

        return new_l


    def _compute_new_free_energies(self):
        new_f = torch.zeros_like(self._free_energies)

        for i in range(self.n_markov_states):
            samples_in_state = torch.where(self.discretized_trajectories == i)
            numerator = -self.bias_potentials[samples_in_state]
            denominator = torch.logsumexp(self._effective_state_counts_log[:, i] +
                                          self._free_energies[:, i] - self.bias_potentials[samples_in_state], dim=1)
            new_f[:, i] = -torch.logsumexp(numerator - denominator.unsqueeze(1), dim=0)

        return new_f


    def self_consistent_step(self):
        '''

        :param sampled_potentials: bias energies for each sample at each state. Shape (S,N)
        :param d_trajectories: discretized trajectories. Assume one trajectory per state for now;
                shape (S, N_i) where N_i is trajectory length
        :param normalized_N_i: N_i/N
        :return:
        '''
        state_dict = self.state_dict()

        state_dict['_lagrangian_mult_log'] = self._update_lagrangians()
        self._compute_effective_state_counts_log()
        state_dict['_free_energies'] = self._update_free_energies()
        self.load_state_dict(state_dict)

        self._shift_free_energies_relative_to_zero()


    def residue(self, transitions):
        '''

        :param sampled_potentials: bias energies for each sample at each state. Shape (S,N)
        :param d_trajectories: discretized trajectories. Assume one trajectory per state for now;
                shape (S, N_i) where N_i is trajectory length
        :param normalized_N_i: N_i/N
        :return:
        '''

        # re-compute effective state counts (R_k_i) and transition (p_k_ij) matrices
        # with the updated free energies and lagrangians.
        # TODO: per sample? This is slow!
        self._compute_effective_state_counts_log()
        self._compute_transition_matrix_log()

        # collect all indices of samples in 1D-tensors
        kk = transitions[:, 0].type(torch.LongTensor)
        ii = transitions[:, 1].type(torch.LongTensor)
        jj = transitions[:, 2].type(torch.LongTensor)
        # and the biases are everything left at the end
        bias_indices = transitions[:, 3]

        # get markov likelihood based on transition counts over all samples
        ll = torch.sum(self._transition_matrix_log[kk, ii, jj])

        # get the sum of all free energies of visited states
        ll += torch.sum(self._free_energies[kk, ii])

        # get the sample weight for all visited states
        mu = -torch.logsumexp(
            self._effective_state_counts_log[:, ii] + self._free_energies[:, ii] - self.bias_energies[kk, bias_indices].T,
            dim=0)

        ll += torch.sum(mu)  #

        return -ll / len(transitions)
