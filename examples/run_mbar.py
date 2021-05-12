import torch
import matplotlib.pyplot as plt
from ..thermodynamicestimators.test_cases import test_case_factory
from ..thermodynamicestimators.estimators.mbar import MBAR
from ..thermodynamicestimators.data_sets.infinite_dataloader import InfiniteDataLoader

"""
main.py

Prepares histogram data by setting up a potential with biases and using MCMC for sampling.
Uses WHAM to return an estimate of the potential function.
"""


def run_with_optimizer(optimizer, dataset, ground_truth, direct_iterate=False, lr=0.1, batch_size=128,
                       use_scheduler=True):
    dataloader = InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True)
    estimator = MBAR(dataset.n_states)
    optimizer = optimizer(estimator.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    free_energies, errors = estimator.estimate(dataloader,
                                               dataset,
                                               optimizer,
                                               scheduler,
                                               tolerance=1e-3,
                                               max_iterations=20,
                                               direct_iterate=direct_iterate,
                                               ground_truth=ground_truth)
    return estimator, free_energies, errors


def main():
    test_name = 'double_well_1D'

    # generate a test problem with potential, biases, data and histogram bin range
    test_case = test_case_factory.make_test_case(test_name)
    # For the free energy estimates, turn the test case into a dataset than contains
    # only the necessary data (the potentials)
    dataset = test_case.to_mbar_dataset()

    ground_truth = test_case.ground_truth


    estimator_sgd, free_energies_sgd, errors_sgd = run_with_optimizer(torch.optim.SGD, dataset, ground_truth)
    estimator_adam, free_energies_adam, errors_adam = run_with_optimizer(torch.optim.Adam, dataset, ground_truth)
    estimator_sc, free_energies_sc, errors_sc = run_with_optimizer(torch.optim.SGD, dataset, ground_truth,
                                                                   use_scheduler=False,
                                                                   direct_iterate=True, batch_size=len(dataset))

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "font.size": 16
    })

    # plt.title('Relative MSE per epoch')
    # plt.plot(errors_sgd, label='SGD, lr $= 0.1 \cdot 0.95^t$')
    # plt.plot(errors_adam, label='ADAM, lr $= 0.1 \cdot 0.95^t$')
    # plt.plot(errors_sc, label='Self-consistent iteration')
    #
    # plt.ylabel(r'$\frac{(f - f^{\circ})^2 }{ \langle \;|f^{\circ}|\; \rangle}$')
    # plt.xlabel(r'Epoch $t$')
    # plt.yscale('log')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # # xs = [(10 * k + 5) / 3 for k in range(1, 8)]
    #
    # plt.title('Estimated free energies')
    # plt.plot(free_energies_sgd, label=r'SGD, lr $= 0.1\cdot 0.95^t$')
    # plt.plot(free_energies_adam, label=r'Adam, lr $= 0.1\cdot 0.95^t$')
    # plt.plot(free_energies_sc, label='Self-consistent iteration')
    # if ground_truth is not None:
    #     plt.plot(ground_truth, 'k--', label='Ground truth')
    #
    # plt.ylabel(r'$f$')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    if test_name == 'double_well_1D':
        # to obtain a probability distribution, we discretize the space into bins and define a binning function to bin each
        # sample (position) in the correct bin
        def bin_sample(x):
            hist = torch.zeros(100)
            hist[int(x)] = 1
            return hist


        observable_values = get_observable_values(dataset.sampled_coordinates, bin_sample)

        # now get the expectation value of the bin function to obtain a probability distribution over bins.
        # The negative log of this is the potential function.
        potential_SGD = -torch.log(
            estimator_sgd.get_equilibrium_expectation(dataset.samples.T, dataset.N_i, observable_values).detach())
        potential_Adam = -torch.log(
            estimator_adam.get_equilibrium_expectation(dataset.samples.T, dataset.N_i, observable_values).detach())
        potential_sc = -torch.log(
            estimator_sc.get_equilibrium_expectation(dataset.samples.T, dataset.N_i, observable_values).detach())

        plt.plot([test_case.potential_fn(x) for x in range(100)], label="Real potential function")
        plt.plot(potential_SGD, label="SGD, lr=0.1")
        plt.plot(potential_Adam, label="ADAM, lr=0.1")
        plt.plot(potential_sc, label="Self-consistent iteration")
        plt.legend()
        plt.show()

    if test_name == 'double_well_2D':
        def bin_sample(x):
            hist = torch.zeros(21, 21)
            hist[int(x[0]) - 5, int(x[1]) - 5] = 1
            return hist


        observable_values = get_observable_values(dataset.sampled_coordinates, bin_sample)

        potential_SGD = -torch.log(
            estimator_sgd.get_equilibrium_expectation(dataset.samples.T, dataset.N_i, observable_values).detach())
        potential_Adam = -torch.log(
            estimator_adam.get_equilibrium_expectation(dataset.samples.T, dataset.N_i, observable_values).detach())
        potential_sc = -torch.log(estimator_sc.get_equilibrium_expectation(dataset, bin_sample).detach())

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = torch.tensor(range(len(potential_SGD)), dtype=torch.float)
        y = torch.tensor(range(len(potential_SGD[0])), dtype=torch.float)

        X, Y = torch.meshgrid(x, y)

        real_potential = torch.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                real_potential[i, j] = test_case.potential_fn(torch.Tensor([X[i, j], Y[i, j]]) + 5)

        ax.plot_wireframe(X, Y, real_potential - real_potential.mean())
        ax.plot_wireframe(X, Y, potential_SGD - potential_SGD[~torch.isinf(potential_SGD)].mean(), label="SGD",
                          color='b')
        ax.plot_wireframe(X, Y, potential_Adam - potential_Adam[~torch.isinf(potential_Adam)].mean(), label="Adam",
                          color='r')
        ax.plot_wireframe(X, Y, potential_sc - potential_sc[~torch.isinf(potential_sc)].mean(),
                          label="self-consistent iteration",
                          color='g')
        plt.legend()
        plt.show()


def get_observable_values(sampled_coordinates, observable_function):
    # construct a matrix to store the computed observables
    result_shape = observable_function(sampled_coordinates[0]).shape
    observable_values = torch.zeros(sampled_coordinates.shape[:1] + result_shape)

    # fill it with the observed values
    for s_i in range(len(sampled_coordinates)):
        observable_values[s_i] = observable_function(sampled_coordinates[s_i])

    return observable_values


if __name__ == "__main__":

    main()
