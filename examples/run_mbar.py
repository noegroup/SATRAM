import thermodynamicestimators.test_cases.test_case_factory as problem_factory
import thermodynamicestimators.estimators.MBAR as mbar
import torch
from pylab import *
from scipy import integrate

"""
main.py

Prepares histogram data by setting up a potential with biases and using MCMC for sampling.
Uses WHAM to return an estimate of the potential function.
"""


def run_with_optimizer(optimizer, dataset, ground_truth, direct_iterate=False):
    batch_size = 128

    if direct_iterate:
        batch_size = len(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=True)
    estimator = mbar.MBAR(dataset.n_states)
    optimizer = optimizer(estimator.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = None

    free_energies, errors = estimator.estimate(dataloader, optimizer, scheduler, dataset, tolerance=1e-3,
                                               max_iterations=100, direct_iterate=direct_iterate,
                                               ground_truth=ground_truth)
    return estimator, free_energies, errors


def calculate_ground_truth(dataset):
    partition_sums = torch.zeros(dataset.n_states)

    for i, potential in enumerate(dataset.biased_potentials):
        U = lambda y, x: torch.exp(-potential(torch.Tensor([x,y])))
        partition_sum = integrate.dblquad(U, 5, 25, lambda x: 5, lambda x: 25, epsabs=1e-3, epsrel=1e-3)
        partition_sums[i] = partition_sum[0]

    free_energies_ground_truth = -torch.log(partition_sums)
    free_energies_ground_truth -= free_energies_ground_truth[0].clone()

    return free_energies_ground_truth


def main():
    test_case = 'double_well_2D'

    # generate a test problem with potential, biases, data and histogram bin range
    dataset = problem_factory.make_test_case(test_case, 'MBAR')
    # ground_truth = calculate_ground_truth(dataset)
    ground_truth = torch.tensor([ 0.0000, -2.5042, -1.9476,  0.3180, -0.0461, -0.0088,  2.1938])

    estimator_SGD, free_energies_SGD, errors_SGD = run_with_optimizer(torch.optim.SGD, dataset, ground_truth)
    estimator_Adam, free_energies_Adam, errors_Adam = run_with_optimizer(torch.optim.Adam, dataset, ground_truth)
    estimator_sc, free_energies_sc, errors_sc = run_with_optimizer(torch.optim.SGD, dataset, ground_truth, direct_iterate=True)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "font.size": 16
    })

    plt.title('Relative MSE per epoch')
    plt.plot(errors_SGD, label='SGD, lr $= 0.1 \cdot 0.95^t$')
    plt.plot(errors_Adam, label='Adam, lr $= 0.1 \cdot 0.95^t$')
    plt.plot(errors_sc, label='Self-consistent iteration')

    plt.ylabel(r'$\frac{(f - f^{\circ})^2 }{ \langle \;|f^{\circ}|\; \rangle}$')
    plt.xlabel(r'Epoch $t$')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()

    xs = [(10 * k + 5) / 3 for k in range(1,8)]

    plt.title('Estimated free energies')
    plt.plot(xs, free_energies_SGD, label=r'SGD, lr $= 0.1\cdot 0.95^t$')
    plt.plot(xs, free_energies_Adam, label=r'Adam, lr $= 0.1\cdot 0.95^t$')
    plt.plot(xs, free_energies_sc, label='Self-consistent iteration')
    plt.plot(xs, ground_truth, 'k--', label='Ground truth')
    plt.ylabel(r'$f$')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.tight_layout()
    plt.show()

    if test_case == 'double_well_1D':
        # to obtain a probability distribution, we discretize the space into bins and define a binning function to bin each
        # sample (position) in the correct bin
        def bin_sample(x):
            hist = torch.zeros(100)
            hist[int(x)] = 1
            return hist

        # now get the expectation value of the bin function to obtain a probability distribution over bins.
        # The negative log of this is the potential function.
        potential_SGD = -np.log(estimator_SGD.get_expectation_value(dataset, bin_sample).detach())

        plt.plot([dataset.potential_function(x) for x in range(100)], label="Real potential function")
        plt.plot(potential_SGD, label="SGD, lr=0.1")

        plt.legend()
        plt.show()

    if test_case == 'double_well_2D':
        def bin_sample(x):
            hist = torch.zeros(20, 20)
            hist[int(x[0]) - 5, int(x[1]) - 5] = 1
            return hist

        potential_SGD = -np.log(estimator_SGD.get_expectation_value(dataset, bin_sample).detach())
        potential_Adam = -np.log(estimator_Adam.get_expectation_value(dataset, bin_sample).detach())
        potential_sc = -np.log(estimator_sc.get_expectation_value(dataset, bin_sample).detach())

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = torch.tensor(range(len(potential_SGD)))
        y = torch.tensor(range(len(potential_SGD[0])))

        X, Y = torch.meshgrid(x, y)
        ax.plot_wireframe(X, Y, potential_SGD - potential_SGD[~torch.isinf(potential_SGD)].mean(), label="SGD",
                          color='b')
        ax.plot_wireframe(X, Y, potential_Adam - potential_Adam[~torch.isinf(potential_Adam)].mean(), label="Adam",
                          color='r')
        ax.plot_wireframe(X, Y, potential_sc - potential_sc[~torch.isinf(potential_sc)].mean(),
                          label="self-consistent iteration",
                          color='g')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
