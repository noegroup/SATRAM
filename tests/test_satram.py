"""
Unit and regression test for the SATRAM package.
"""

from satram import ThermodynamicEstimator
from examples.datasets import toy_problem


def test_mbar():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator()
    estimator.fit((ttrajs, dtrajs, bias), solver_type="MBAR")

    assert estimator.free_energies_per_thermodynamic_state[0] == 0
    assert (estimator.free_energies_per_thermodynamic_state[1:] > 0).all()
    assert estimator.free_energies_per_thermodynamic_state[-1] > 1


def test_tram():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator()
    estimator.fit((ttrajs, dtrajs, bias), solver_type="TRAM")

    assert estimator.free_energies_per_thermodynamic_state[0] == 0
    assert (estimator.free_energies_per_thermodynamic_state[1:] > 0).all()
    assert estimator.free_energies_per_thermodynamic_state[-1] > 1


def test_satram():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator()
    estimator.fit((ttrajs, dtrajs, bias), solver_type="SATRAM", initial_batch_size=16)

    assert estimator.free_energies_per_thermodynamic_state[0] == 0
    assert (estimator.free_energies_per_thermodynamic_state[1:] > 0).all()
    assert estimator.free_energies_per_thermodynamic_state[-1] > 1


def test_sambar():
    ttrajs, dtrajs, bias = toy_problem.get_tram_input()

    estimator = ThermodynamicEstimator()
    estimator.fit((ttrajs, dtrajs, bias), solver_type="SAMBAR", initial_batch_size=256)

    assert estimator.free_energies_per_thermodynamic_state[0] == 0
    assert (estimator.free_energies_per_thermodynamic_state[1:] > 0).all()
    assert estimator.free_energies_per_thermodynamic_state[-1] > 1
