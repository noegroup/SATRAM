import torch
from thermodynamicestimators.estimators.thermodynamic_estimator import ThermodynamicEstimator


class TRAM(ThermodynamicEstimator):
    """Free energy estimator based on the TRAM equations. """


    def __init__(self, n_therm_states, n_markov_states, lag_time=1):
        super().__init__(n_therm_states)
        self.n_markov_states = n_markov_states
        self._free_energies = torch.nn.Parameter(
            torch.ones((self.n_states, self.n_markov_states), dtype=torch.float64))
        self._lagrangian_mult = torch.nn.Parameter(
            torch.ones((self.n_states, self.n_markov_states), dtype=torch.float64))
        self._lag = lag_time


    def get_potential(self, samples, normalized_N_i):
        return NotImplemented


    def _get_transition_counts(self, d_trajectories, N_i):
        transition_counts = torch.zeros((self.n_states, self.n_markov_states, self.n_markov_states))

        counted = 0
        for k, n_i in enumerate(N_i):

            for sample_idx in range(int(counted), int(counted + n_i - self._lag)):
                transition_counts[k, d_trajectories[sample_idx], d_trajectories[sample_idx + self._lag]] += 1
            counted += n_i

        return transition_counts


    def _get_state_counts(self, d_trajectories, N_i):
        state_counts = torch.zeros((self.n_states, self.n_markov_states))

        counted = 0
        for k, n_i in enumerate(N_i):
            for sample_idx in range(int(counted), int(counted + n_i)):
                state_counts[k, d_trajectories[sample_idx]] += 1
            counted+=n_i
        return state_counts


    def self_consistent_step(self, dataset):
        '''

        :param sampled_potentials: bias energies for each sample at each state. Shape (S,N)
        :param d_trajectories: discretized trajectories. Assume one trajectory per state for now;
                shape (S, N_i) where N_i is trajectory length
        :param normalized_N_i: N_i/N
        :return:
        '''
        # count transitions with lag time
        transition_counts = self._get_transition_counts(dataset.discretized_coordinates, dataset.N_i)

        # count number of occurences in each markov state per therm. state
        state_counts = self._get_state_counts(dataset.discretized_coordinates, dataset.N_i)

        old_l = self._lagrangian_mult.detach()
        old_f = self._free_energies.detach()

        new_l = torch.zeros_like(old_l)
        new_f = torch.zeros_like(old_f)

        for k in range(self.n_states):
            for i in range(self.n_markov_states):

                denom = torch.exp(old_f[k] - old_f[k, i]) * old_l[k] + old_l[k, i]
                new_l[k, i] = old_l[k, i] * torch.sum(
                    transition_counts[k, i, :] + transition_counts[k, :, i] / denom)

        weighted_samples = torch.exp(-dataset.samples) / torch.sum(
            state_counts * torch.exp(old_f - dataset.samples), axis=0)

        for k in range(self.n_states):
            for i in range(self.n_markov_states):
                sample_indices = torch.where(dataset.discretized_coordinates == i)
                new_f[k, i] = -torch.log(torch.sum(weighted_samples[sample_indices]))

        new_state_dict = self.state_dict()
        new_state_dict['_free_energies'] = new_f
        new_state_dict['_lagrangian_mult'] = new_l
        self.load_state_dict(new_state_dict, strict=False)

    def residue(self, samples, normalized_N_i):
        return NotImplemented
