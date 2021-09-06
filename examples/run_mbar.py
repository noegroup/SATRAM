import torch
import matplotlib.pyplot as plt
from examples.test_cases import test_case_factory
from thermodynamicestimators.estimators.mbar import MBAR
from thermodynamicestimators.data_sets.infinite_dataloader import InfiniteDataLoader

"""
main.py

Uses MBAR to estimate free energies of the 1D double well.
"""


def main():
    test_name = 'double_well_1D'

    # generate a test problem.
    # N_i are the total counts per thermodynamic state,
    # samples are the matrix of shape (N, N_i)  containing the bias potentials of
    # each sample, evaluated at every thermodynamic state.
    test_case = test_case_factory.make_test_case(test_name)
    N_i, samples = test_case.to_mbar_dataset()

    dataloader = InfiniteDataLoader(samples, batch_size=256, shuffle=True)
    estimator = MBAR(N_i=N_i)
    optimizer = torch.optim.SGD(estimator.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=10*len(dataloader), verbose=True)

    # estimate the free energies
    estimator.estimate(dataloader, optimizer, [scheduler])

    # plot the estimate
    plt.title('Estimated free energies')
    plt.plot(estimator.free_energies, label=r'Stochastic MBAR')
    plt.ylabel(r'$f$')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
