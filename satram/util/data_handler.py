import torch


def _determine_n_states(dtrajs):
    return int(max(torch.max(d).item() for d in dtrajs) + 1)


def _determine_n_therm_states(dtrajs, ttrajs):
    if ttrajs is None:
        return len(dtrajs)
    else:
        return _determine_n_states(ttrajs)


def _to_tensor(arr):
    if isinstance(arr, torch.Tensor):
        return arr

    if isinstance(arr, list):
        tensor_arr = []
        for item in arr:
            tensor_arr.append(_to_tensor(item))
        return tensor_arr

    tensor_arr = torch.from_numpy(arr)
    return tensor_arr


def process_input(data, lagtime):
    ttrajs, dtrajs, bias_matrices = data

    ttrajs = _to_tensor(ttrajs)
    dtrajs = _to_tensor(dtrajs)
    bias_matrices = _to_tensor(bias_matrices)

    n_therm_states = _determine_n_therm_states(dtrajs, ttrajs)
    n_markov_states = _determine_n_states(dtrajs)

    transition_counts = torch.zeros([n_therm_states, n_markov_states, n_markov_states])
    state_counts = torch.zeros([n_therm_states, n_markov_states])

    ind_trajs = []

    for i, traj in enumerate(dtrajs):
        ind_traj = torch.zeros((len(traj), n_markov_states))
        sample_idx = torch.arange(0, len(traj))

        ind_traj[(sample_idx, traj)] = 1

        C = torch.tensordot(ind_traj[:-lagtime].T, ind_traj[lagtime:], 1)
        # C = ind_traj[:-lagtime].T.dot(ind_traj[lagtime:])
        transition_counts[i] = C
        state_counts[i] = ind_traj.sum(0)

        ind_trajs.append(ind_traj)

    ind_trajs = torch.cat(ind_trajs)

    transition_counts = transition_counts.double()
    state_counts = state_counts.double()

    ind_trajs = ind_trajs.float()
    bias_matrices = torch.cat(bias_matrices).double()

    data = torch.cat([bias_matrices, ind_trajs], 1)


    return data, state_counts, transition_counts
