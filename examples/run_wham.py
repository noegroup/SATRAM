import torch
import matplotlib.pyplot as plt
from examples.test_cases import test_case_factory
from thermodynamicestimators.estimators.wham import WHAM
from thermodynamicestimators.data_sets.infinite_dataloader import InfiniteDataLoader

"""
main.py

Uses MBAR to estimate free energies of the 1D double well.
"""


def main():
    test_name = 'double_well_1D'

    # generate a test problem.
    # N_i are the total counts per thermodynamic state,
    # M_b are the total bin counts over all states
    # bias_coefficients is the (log of the) bias coefficient matrix
    # Samples are pf shape (2). sample[n][0] is the thermodynamic state index of the sample.
    # samples[n][1] is the bin the sample falls into.
    test_case = test_case_factory.make_test_case(test_name)
    N_i, M_b, bias_coefficients, samples = test_case.to_wham_dataset()

    dataloader = InfiniteDataLoader(samples, batch_size=256, shuffle=True)
    estimator = WHAM(N_i=N_i, M_b = M_b, bias_coefficients_log=bias_coefficients)
    optimizer = torch.optim.SGD(estimator.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=10*len(dataloader), verbose=True)

    # estimate the free energies
    estimator.estimate(dataloader, optimizer, [scheduler])

    # plot the estimate
    plt.title('Estimated free energies')
    plt.plot(estimator.free_energies, label=r'Stochastic WHAM')
    plt.ylabel(r'$f$')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
