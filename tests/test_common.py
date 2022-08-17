import torch
from satram.estimators._common import compute_v_R


def test_compute_v_R():
    n_markov_states = 10
    n_therm_states = 5
    f = torch.zeros([n_therm_states, n_markov_states]) # exp will produce ones
    log_v = torch.zeros_like(f)

    transition_counts = torch.ones([n_therm_states, n_markov_states, n_markov_states])
    for i in range(1, n_therm_states):
        transition_counts[i:] += 1

    state_counts = torch.sum(transition_counts, 2)
    log_C_sym = torch.log(transition_counts.transpose(1, 2) + transition_counts) # filled with log of 2's
    log_v, log_R = compute_v_R(f, log_v, log_C_sym, state_counts, transition_counts)

    for i in range(n_therm_states):
        assert torch.allclose(torch.exp(log_v[i]), torch.ones_like(log_v) * (1+i) * 10.)
        assert torch.allclose(torch.exp(log_R[i]), torch.ones_like(log_R) * (1+i) * 10.)
