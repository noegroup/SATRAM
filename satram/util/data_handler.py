import torch
from deeptime.markov.msm.tram import TRAMDataset


def _determine_n_states(dtrajs):
    if isinstance(dtrajs, list):
        return max([_determine_n_states(traj) for traj in dtrajs])
    else:
        return int(max(torch.max(d).item() for d in dtrajs) + 1)


def _determine_n_therm_states(dtrajs, ttrajs):
    if ttrajs is None:
        return len(dtrajs)
    else:
        return _determine_n_states(ttrajs)


def process_input(data, lagtime=1):
    """ Process the input trajectories to construct count matrices, accounting
     for replica exchange swaps.

    Uses deeptime TRAMDataset to process the replica exchange swaps.

    Parameters
    ----------
    data : tuple
            data is a tuple containing (ttrajs, dtrajs, bias_matrices)
            * `ttrajs`: `ttrajs[i]` indicates for each sample in the $i$-th
              trajectory what thermodynamic state that sample was sampled at.
              May be None if no replica-exchange swaps were performed.
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
    lagtime : int
        lagtime for which to construct the count matrices.

    Returns
    -------
    data : torch.Tensor (N, S + M)
        data contains one row for each sample (for the total N samples) and each
        data row consists of two parts. The first S columns data[:, :S] contains
        the bias energies, i.e. data[n, :S] contains for the n-th sample the
        bias energy evaluated at each thermodynamic state.
        The last M columns contain the index of the Markov state the sample was
        binned into in the form of one-hot encoding, i.e. if the sample was
        binned into Markov state index 2, data[n, S:] looks like [0, 0, 1, 0, ...]
    state_counts : torch.Tensor (S, M)
        Tensor containing for each thermodynamic state (from 0 to S) and each
        Markov state (from 0 to M) the number of samples sampled at that state.
    transition_counts : torch.Tensor (S, M, M)
        Tensor containing for each thermodynamic state (from 0 to S) and each
        combination of Markov states i, j (from 0 to M) the number of observed
        transitions from i to j under the chosen lagtime.
    """
    ttrajs, dtrajs, bias_matrices = data

    dataset = TRAMDataset(dtrajs, bias_matrices, ttrajs, lagtime=lagtime)

    transition_counts = torch.from_numpy(dataset.transition_counts)
    state_counts = torch.from_numpy(dataset.state_counts)

    # bias_list = []
    ind_trajs = []

    for dtraj in dataset.dtrajs:
        dtraj = torch.from_numpy(dtraj)
        ind_traj = torch.zeros((len(dtraj), dataset.n_markov_states))
        sample_idx = torch.arange(0, len(dtraj))
        ind_traj[(sample_idx.long(), dtraj.long())] = 1
        ind_trajs.append(ind_traj)

    ind_trajs = torch.cat(ind_trajs).double()
    bias_list = torch.cat([torch.from_numpy(bias) for bias in dataset.bias_matrices]).double()

    data = torch.cat([bias_list, ind_trajs], 1)

    return data, state_counts, transition_counts
