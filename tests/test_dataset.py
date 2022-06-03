import pytest
import torch.utils.data

from satram.util.dataset import Dataset
from satram.util.data_handler import process_input
from examples.datasets.toy_problem import *


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"],
)
def test_properties_set(device):
    data, state_counts, transition_counts = process_input(get_tram_input())
    dataset = Dataset(data=data, state_counts=state_counts, transition_counts=transition_counts, device=device)

    assert (dataset.state_counts == state_counts.to(device)).all()
    assert (dataset.transition_counts == transition_counts.to(device)).all()
    assert dataset.log_N == math.log(n_therm_states * T)

    assert dataset.n_therm_states == n_therm_states
    assert dataset.n_markov_states == n_conf_states

    N_k = torch.tensor([T] * n_therm_states, device=device)
    assert (dataset.N_k_log == torch.log(N_k)).all()
    assert (dataset.normalized_N_k == (N_k / (n_therm_states * T))).all()

    assert dataset.log_C_sym.shape == transition_counts.shape


@pytest.mark.parametrize(
    "is_stochastic", [True, False],
)
def test_dataloader_set_correctly(is_stochastic):
    data, state_counts, transition_counts = process_input(get_tram_input())
    dataset = Dataset(data=data, state_counts=state_counts, transition_counts=transition_counts,
                      is_stochastic=is_stochastic)
    assert isinstance(dataset.dataloader, torch.utils.data.DataLoader)
    assert dataset.dataloader.drop_last is is_stochastic
    if is_stochastic:
        assert dataset.dataloader.batch_size == dataset.max_batch_size
        assert isinstance(dataset.dataloader.sampler, torch.utils.data.RandomSampler)
    else:
        assert isinstance(dataset.dataloader.sampler, torch.utils.data.SequentialSampler)
        assert dataset.dataloader.batch_size == 256


@pytest.mark.parametrize(
    "is_stochastic", [True, False],
)
def test_dataset_deterministic_dataloader(is_stochastic):
    data, state_counts, transition_counts = process_input(get_tram_input())
    dataset = Dataset(data=data, state_counts=state_counts, transition_counts=transition_counts,
                      is_stochastic=is_stochastic)
    assert isinstance(dataset.deterministic_dataloader, torch.utils.data.DataLoader)
    assert isinstance(dataset.deterministic_dataloader.sampler, torch.utils.data.SequentialSampler)
    assert dataset.deterministic_dataloader.batch_size == dataset.max_batch_size

