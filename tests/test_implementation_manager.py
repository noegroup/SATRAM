import pytest
from satram.estimators.implementation_manager import ImplementationManager
from satram.estimators.tram import TRAM
from satram.estimators.satram import SATRAM
from satram.estimators.mbar import MBAR
from satram.estimators.sambar import SAMBAR


@pytest.mark.parametrize(
    "stoch_solver, dtrm_solver, initial_solver_type",
    [(SAMBAR, MBAR, "SAMBAR"), (SATRAM, TRAM, "SATRAM")],
)
@pytest.mark.parametrize(
    "patience",
    [1, 2, 10],
)
def test_stochastic_becomes_deterministic(stoch_solver, dtrm_solver, initial_solver_type, patience):
    impl = ImplementationManager(solver_type=initial_solver_type, initial_batch_size=256, patience=patience, total_dataset_size=1000)
    assert impl.solver == stoch_solver
    assert impl.is_stochastic
    assert impl.batch_size == 256

    for i in range(patience):
        impl.step(i)
        assert impl.is_stochastic
        assert impl.batch_size == 256

    impl.step(patience)
    assert impl.is_stochastic
    assert impl.batch_size == 512

    for i in range(patience):
        impl.step(i)
        assert impl.is_stochastic
        assert impl.batch_size == 512
    impl.step(patience)
    assert not impl.is_stochastic
    assert impl.batch_size == 1000
    assert impl.solver == dtrm_solver