"""
Unit and regression test for the SATRAM package.
"""
import pytest
import torch
from satram import ThermodynamicEstimator
from examples.datasets import toy_problem


@pytest.mark.parametrize(
    "solver_type",
    ["MBAR", "SAMBAR", "TRAM", "SATRAM"],
)
def test_fit(solver_type, device):
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator(maxiter=1000, device=device)
    estimator.fit((ttrajs, dtrajs, bias), solver_type=solver_type, initial_batch_size=256)

    f_k = estimator.free_energies_per_thermodynamic_state
    assert f_k[0] == 0
    assert (f_k[:-1] < f_k[1:]).all()
    assert f_k[-1] > 1


@pytest.mark.parametrize(
    "solver_type",
    ["MBAR", "SAMBAR", "TRAM", "SATRAM"],
)
def test_sample_weights(solver_type):
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator()
    estimator.fit((ttrajs, dtrajs, bias), solver_type=solver_type, initial_batch_size=256)

    weights = estimator.sample_weights()
    assert not weights.isinf().any()
    assert not weights.isnan().any()

    weights_2 = estimator.sample_weights(3)
    assert not weights_2.isinf().any()
    assert not weights_2.isnan().any()
    assert (weights_2 < weights).all()


def test_compute_pmf_with_ndarray_bins():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator(maxiter=10)
    estimator.fit((ttrajs, dtrajs, bias), solver_type="SATRAM", initial_batch_size=256)

    bins = [traj.numpy() for traj in dtrajs]
    pmf = estimator.compute_pmf(bins)
    assert pmf.shape[0] == 5
    assert (pmf >= 0).all()


def test_compute_pmf_with_tensor_bins():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator(maxiter=10)
    estimator.fit((ttrajs, dtrajs, bias), solver_type="SATRAM", initial_batch_size=256)

    pmf = estimator.compute_pmf(torch.cat(dtrajs))
    assert pmf.shape[0] == 5


def test_compute_pmf_with_too_many_bins():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator(maxiter=10)
    estimator.fit((ttrajs, dtrajs, bias), solver_type="SATRAM", initial_batch_size=256)

    pmf = estimator.compute_pmf(torch.cat(dtrajs), n_bins=10)
    assert pmf.shape[0] == 10
    assert (pmf[:5] >= 0).all()
    assert (pmf[5:] == float("Inf")).all()


def test_fit_progress():
    max_iter = 7
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    class ProgressMock():

        def __init__(self, *args, **kwargs):
            self.i = 0
            self.max_iter = 0


        def __iter__(self):
            return self


        def __next__(self):
            self.i += 1
            if self.i < self.max_iter:
                return self.i
            else:
                raise StopIteration

    progress = ProgressMock()

    class ProgressFactory:

        def __new__(cls, *args, **kwargs):
            progress.max_iter = args[0].stop
            return progress

    estimator = ThermodynamicEstimator(maxiter=max_iter, progress=ProgressFactory)
    estimator.fit((ttrajs, dtrajs, bias))
    assert progress.i == max_iter


def test_callback_called():
    fs = []
    vs = []
    iterations = []

    def callback(i, f, log_v):
        fs.append(f)
        vs.append(log_v)
        iterations.append(i)

    ttrajs, dtrajs, bias = toy_problem.get_tram_input()
    estimator = ThermodynamicEstimator(maxiter=10, callback_interval=3)
    estimator.fit((ttrajs, dtrajs, bias), callback=callback)

    assert len(fs) == 4
    assert len(vs) == 4
    assert iterations == [0, 3, 6, 9]
    assert torch.Tensor([isinstance(f, torch.Tensor) for f in fs]).all()
    assert torch.Tensor([isinstance(v, torch.Tensor) for v in vs]).all()


def test_batch_size_doubling():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()
    estimator = ThermodynamicEstimator(maxiter=2)
    estimator.fit((ttrajs, dtrajs, bias), solver_type="SATRAM", initial_batch_size=64, patience=1)
    assert estimator.dataset.dataloader.batch_size == 128
