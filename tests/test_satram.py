"""
Unit and regression test for the SATRAM package.
"""

from satram import ThermodynamicEstimator
from examples.datasets import toy_problem


def test_mbar():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator()
    estimator.fit((ttrajs, dtrajs, bias), solver_type="MBAR")

    f_k = estimator.free_energies_per_thermodynamic_state
    assert f_k[0] == 0
    assert (f_k[:-1] < f_k[1:]).all()
    assert f_k[-1] > 1


def test_tram():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator()
    estimator.fit((ttrajs, dtrajs, bias), solver_type="TRAM")

    f_k = estimator.free_energies_per_thermodynamic_state
    assert f_k[0] == 0
    assert (f_k[:-1] < f_k[1:]).all()
    assert f_k[-1] > 1


def test_satram():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator()
    estimator.fit((ttrajs, dtrajs, bias), solver_type="SATRAM", initial_batch_size=16)

    f_k = estimator.free_energies_per_thermodynamic_state
    assert f_k[0] == 0
    assert (f_k[:-1] < f_k[1:]).all()
    assert f_k[-1] > 1


def test_sambar():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator()
    estimator.fit((ttrajs, dtrajs, bias), solver_type="SAMBAR", initial_batch_size=256)

    f_k = estimator.free_energies_per_thermodynamic_state
    assert f_k[0] == 0
    assert (f_k[:-1] < f_k[1:]).all()
    assert f_k[-1] > 1
