import numpy as np
import torch


def _determine_n_states(dtrajs):
    return max(np.max(d) for d in dtrajs) + 1


def _determine_n_therm_states(dtrajs, ttrajs):
    if ttrajs is None:
        return len(dtrajs)
    else:
        return _determine_n_states(ttrajs)


def _to_tensor(arr):
    if isinstance(arr, list):
        tensor_arr = []
        for item in arr:
            tensor_arr.append(_to_tensor(item))

    if isinstance(arr, np.ndarray):
        tensor_arr = torch.from_numpy(arr)
    return tensor_arr



def process_input(data, lagtime):
    ttrajs, dtrajs, bias_matrices = data

    n_therm_states = _determine_n_therm_states(dtrajs, ttrajs)
    n_markov_states = _determine_n_states(dtrajs)

    transition_counts = np.zeros([n_therm_states, n_markov_states, n_markov_states])
    state_counts = np.zeros([n_therm_states, n_markov_states])

    ind_trajs = []

    for i, traj in enumerate(dtrajs):
        ind_traj = np.zeros((len(traj), n_markov_states))
        sample_idx = np.arange(0, len(traj))

        ind_traj[(sample_idx, traj)] = 1

        C = ind_traj[:-lagtime].T.dot(ind_traj[lagtime:])
        transition_counts[i] = C
        state_counts[i] = ind_traj.sum(0)

        ind_trajs.append(ind_traj)

    ind_trajs = np.concatenate(ind_trajs)

    transition_counts = torch.from_numpy(transition_counts).double()
    state_counts = torch.from_numpy(state_counts).double()

    ind_trajs = torch.from_numpy(ind_trajs).float()
    bias_matrices = torch.from_numpy(np.concatenate(bias_matrices)).double()

    data = torch.cat([bias_matrices, ind_trajs], 1)


    return data, state_counts, transition_counts
