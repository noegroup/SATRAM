import pytest
import torch
from satram.util.data_handler import process_input


def random_input_data(n_therm_states, n_markov_states, traj_lengths, has_RE=False):
    dtrajs = [torch.randint(low=0, high=n_markov_states, size=[l]) for l in traj_lengths]
    # ensure we get the correct number of states by hard-coding a sample of the max index
    dtrajs[0][0] = n_markov_states-1

    ttrajs = [torch.Tensor([i] * l).int() for i, l in enumerate(traj_lengths)]
    if has_RE:
        pass
    biases = [torch.rand([l, n_therm_states]) for l in traj_lengths]
    return ttrajs, dtrajs, biases


@pytest.mark.parametrize(
    "n_therm_states, n_markov_states, traj_lengths, lagtime",
    [(3, 5, [4, 4, 4], 1),
     (3, 5, [4, 4, 4], 2),
     (5, 4, [3, 3, 3, 4, 5], 2)],
)
def test_process_input_no_RE(n_therm_states, n_markov_states, traj_lengths, lagtime):
    N = torch.Tensor(traj_lengths).sum()
    input = random_input_data(n_therm_states, n_markov_states, traj_lengths)

    data, state_counts, transition_counts = process_input(input, lagtime=lagtime)
    assert data.shape == (N, n_therm_states + n_markov_states)
    assert state_counts.shape == (n_therm_states, n_markov_states)
    assert transition_counts.shape == (n_therm_states, n_markov_states, n_markov_states)

    assert transition_counts.sum() == N - lagtime * len(traj_lengths)
    assert (transition_counts.sum(1).sum(1) == torch.Tensor(traj_lengths) - lagtime).all()

    assert (state_counts.sum(1) == torch.Tensor(traj_lengths)).all()
    assert (torch.cat(input[2]) == data[:, :n_therm_states]).all()


@pytest.mark.parametrize(
    "n_therm_states, n_markov_states, traj_lengths",
    [(5, 4, [10, 10, 9, 11, 8]), ],
)
@pytest.mark.parametrize(
    "lagtime",
    [2, 3, 4],
)
@pytest.mark.parametrize(
    "swap_position",
    [2, 3],
)
@pytest.mark.parametrize(
    "n_swaps",
    [0, 1, 2, 3, 4]
)
def test_process_input_with_RE(n_therm_states, n_markov_states, traj_lengths, lagtime, n_swaps, swap_position):
    N = torch.Tensor(traj_lengths).sum()

    input = random_input_data(n_therm_states, n_markov_states, traj_lengths)

    for swap in range(n_swaps):
        input[0][swap][swap_position] += 1

    data, state_counts, transition_counts = process_input(input, lagtime=lagtime)
    assert data.shape == (N, n_therm_states + n_markov_states)
    assert state_counts.shape == (n_therm_states, n_markov_states)
    assert transition_counts.shape == (n_therm_states, n_markov_states, n_markov_states)

    assert state_counts.sum() == N

    if lagtime <= swap_position:
        assert transition_counts.sum() == (N - len(traj_lengths) * lagtime - (lagtime * n_swaps))
    else:
        N -= n_swaps * swap_position
        assert transition_counts.sum() == (N - len(traj_lengths) * lagtime)
