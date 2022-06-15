"""
Unit and regression test for the SATRAM package.
"""
import pytest
from satram import ThermodynamicEstimator
from examples.datasets import toy_problem
import torch

@pytest.mark.parametrize(
    "solver_type",
    ["MBAR", "SAMBAR", "TRAM", "SATRAM"],
)
def test_fit(solver_type):
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator()
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
