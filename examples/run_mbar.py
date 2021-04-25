import argparse
import thermodynamicestimators.utilities.test_problem_factory as problem_factory
import thermodynamicestimators.estimators.MBAR as mbar
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *


"""
main.py

Prepares histogram data by setting up a potential with biases and using MCMC for sampling.
Uses WHAM to return an estimate of the potential function.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", default='double_well_1D', help="The name of the test problem")
    parser.add_argument("--tolerance", default=1e-4, help="Error tolerance for convergence")
    parser.add_argument("--max_iterations", default=200, help="Maximum number of iterations allowed to converge")

    args = parser.parse_args()


    # generate a test problem with potential, biases, data and histogram bin range
    test_problem = problem_factory.make_test_problem(args.test_name)

    estimator = mbar.MBAR(test_problem.n_states)
    optimizer_SGD = torch.optim.SGD(estimator.parameters(), lr=0.001)
    free_energy_SGD, errors_SGD = estimate_free_energy(estimator,
                                             optimizer_SGD,
                                            test_problem,
                                     args=args)

    plt.plot(free_energy_SGD)
    plt.show()

    plt.yscale('log')
    plt.ylabel(r'max_i (F(t)_i - F(t-1)_i)^2 / |avg(F(t))|$')
    plt.xlabel(r'Epoch t')
    plt.plot(errors_SGD, label="error")
    plt.show()


    def bin_sample(x):
        hist = [0] * 100
        hist[int(x)]=1
        return hist

    potential_SGD = -np.log(estimator.get_expectation_value(test_problem.data,
                                                            test_problem.sampled_potentials_at_all_states,
                                                            test_problem.sampled_unbiased_potentials,
                                                            bin_sample).detach())

    plt.plot(test_problem.potential(range(100)), label="Real potential function")
    plt.plot(potential_SGD, label="SGD, lr=0.001")

    plt.legend()
    plt.show()



def estimate_free_energy(estimator, optimizer, data_manager, args):

    epoch = 0
    error = args.tolerance + 1
    free_energy = 0
    errors = []

    data =data_manager.data.numpy()

    batch_size = 100
    n_batches = int(data.shape[1]/batch_size)


    while epoch < args.max_iterations and error > args.tolerance:

        epoch += 1

        [np.random.shuffle(d) for d in data]

        for b in range(n_batches):
            batch_data = torch.flatten(torch.tensor(data[:,b*batch_size:(b+1)*batch_size]))

            batch_data = torch.stack([bias(batch_data) for bias in data_manager.bias_functions]).squeeze(
                -1) + torch.tensor(data_manager.potential(batch_data))

            optimizer.zero_grad()
            loss = estimator.residue(batch_data)
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            estimator._f -= estimator._f[0].clone()

            error = torch.max(torch.square((estimator.free_energy - free_energy) / (0.1 * torch.abs(estimator.free_energy.mean()))))
            free_energy = estimator.free_energy

        print(error)
        errors.append(error)

    return estimator.free_energy, errors

if __name__ == "__main__":
    main()