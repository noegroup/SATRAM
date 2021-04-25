import argparse
import thermodynamicestimators.utilities.test_problem_factory as problem_factory
import thermodynamicestimators.estimators.WHAM as wham
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D


"""
main.py

Prepares histogram data by setting up a potential with biases and using MCMC for sampling.
Uses WHAM to return an estimate of the potential function.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", default='double_well_1D', help="The name of the test problem")
    parser.add_argument("--tolerance", default=10e-2, help="Error tolerance for convergence")
    parser.add_argument("--max_iterations", default=10, help="Maximum number of iterations allowed to converge")

    args = parser.parse_args()


    # generate a test problem with potential, biases, data and histogram bin range
    test_problem = problem_factory.make_test_problem(args.test_name)

    estimator = wham.WHAM(test_problem.bias_coefficients, test_problem.histogram_shape)
    optimizer_SGD = torch.optim.SGD(estimator.parameters(), lr=0.1)
    potential_SGD, errors_SGD = estimate_free_energy(estimator,
                                             optimizer_SGD,
                                     test_problem.data,
                                     histogram_bin_range=test_problem.histogram_range,
                                     histogram_shape=test_problem.histogram_shape,
                                     args=args)

    plt.plot(estimator.free_energy.detach().numpy())
    plt.show()

    estimator = wham.WHAM(test_problem.bias_coefficients, test_problem.histogram_shape)
    optimizer_ADAM = torch.optim.Adam(estimator.parameters(), lr=0.1)
    potential_ADAM, errors_ADAM = estimate_free_energy(estimator,
                                     optimizer_ADAM,
                                     test_problem.data,
                                     histogram_bin_range=test_problem.histogram_range,
                                     histogram_shape=test_problem.histogram_shape,
                                     args=args)


    plt.yscale('log')
    plt.plot(errors_SGD, label='SGD error, lr=0.1')
    plt.plot(errors_ADAM, label='ADAM error, lr=0.1')

    plt.legend()
    plt.show()


    if args.test_name == "double_well_1D":
        plt.plot(test_problem.potential(range(100)), label="real potential function", color='g')
        plt.plot(potential_SGD, label="SGD")
        plt.plot(potential_ADAM, label="SGD")

    if args.test_name == "double_well_2D":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = range(test_problem.histogram_range[0][0], test_problem.histogram_range[0][1])
        y = range(test_problem.histogram_range[1][0], test_problem.histogram_range[1][1])
        X, Y = np.meshgrid(x, y)

        real_potential = np.zeros_like(X)
        for r, _ in np.ndenumerate(real_potential):
            real_potential[r] = test_problem.potential((X[r], Y[r]))

        ax.plot_wireframe(X, Y, potential_SGD - np.ma.masked_invalid(potential_SGD).mean(), label="SGD", color='b')
        ax.plot_wireframe(X, Y, potential_ADAM - np.ma.masked_invalid(potential_ADAM).mean(), label="ADAM", color='r')
        ax.plot_wireframe(X, Y, real_potential - real_potential.mean(), label="Real potential function", color='g')

    plt.legend()
    plt.show()



def estimate_free_energy(estimator, optimizer, data, histogram_shape, histogram_bin_range, args):

    histograms_all_data = torch.tensor(
        [np.histogramdd(simulation_data, histogram_shape, range=histogram_bin_range, density=True)[0] for
         simulation_data in data])

    epoch = 0
    error = args.tolerance + 1
    free_energy = 0
    errors = []

    batch_size = 100
    n_batches = int(len(data[0])/batch_size)

    while epoch < args.max_iterations and error > args.tolerance:
        epoch += 1

        [np.random.shuffle(d) for d in data]


        for j in range(n_batches):
            batch_data = [set[j * batch_size: (j + 1) * batch_size] for set in data]

            hist = torch.tensor([np.histogramdd(simulation_data, density=True, bins=histogram_shape, range=histogram_bin_range)[0] for simulation_data in batch_data])

            optimizer.zero_grad()
            loss = estimator.residue(hist)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            error = torch.max(torch.square(estimator.free_energy - free_energy) / estimator.free_energy.mean())
            free_energy = estimator.free_energy
        print(error)
        errors.append(error)

    plt.show()
    return estimator.get_potential(histograms_all_data).detach().numpy(), errors

if __name__ == "__main__":
    main()