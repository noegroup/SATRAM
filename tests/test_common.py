import torch
from satram.estimators._common import compute_v_R


def test_compute_v_R():
    n_markov_states = 10
    n_therm_states = 5
    f = torch.zeros([n_therm_states, n_markov_states]) # exp will produce ones
    log_v = torch.zeros_like(f)

    transition_counts = torch.ones([n_therm_states, n_markov_states, n_markov_states])
    state_counts = torch.sum(transition_counts, 2)
    log_C_sym = torch.log(transition_counts.transpose(1, 2) + transition_counts) # filled with log of 2's
    log_N = 0
    log_v, log_R = compute_v_R(f, log_v, log_C_sym, log_N, state_counts, transition_counts)

    assert (torch.exp(log_v) == 10.).all()
    assert (torch.exp(log_R) == 10.).all()

