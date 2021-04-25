import argparse
import thermodynamicestimators.utilities.test_problem_factory as problem_factory
import thermodynamicestimators.estimators.MBAR as mbar
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
    parser.add_argument("--max_iterations", default=200, help="Maximum number of iterations allowed to converge")

    args = parser.parse_args()


    # generate a test problem with potential, biases, data and histogram bin range
    test_problem = problem_factory.make_test_problem(args.test_name)


    estimator = mbar.MBAR(test_problem.data_at_all_states)


    optimizer_SGD = torch.optim.SGD(estimator.parameters(), lr=0.001)
    potential_SGD, errors_SGD = estimate_free_energy(estimator,
                                             optimizer_SGD,
                                            test_problem.data_at_all_states,
                                     args=args)

    def bin_sample(x):
        hist = [0] * 100
        hist[x]=1
        return hist

    probability_SGD = estimator.get_expectation_value(test_problem.data, test_problem.data_at_all_states, bin_sample)


    # plt.yscale('log')
    # plt.plot(errors_SGD, label='SGD error, lr=0.1')

    # plt.legend()
    # plt.show()


    # plt.plot(test_problem.potential(range(100)), label="real potential function", color='g')
    plt.plot(-np.log(probability_SGD.detach().numpy()), label="SGD")

    plt.legend()
    plt.show()



def estimate_free_energy(estimator, optimizer, data, args):

    epoch = 0
    error = args.tolerance + 1
    free_energy = 0
    errors = []

    while epoch < args.max_iterations:# and error > args.tolerance:

        epoch += 1

        optimizer.zero_grad()
        loss = estimator.residue(data)
        loss.backward()
        optimizer.step()

        estimator.self_consistent_step(data)


        with torch.no_grad():
            with torch.no_grad():
                estimator._f -= estimator._f[0].clone()
                estimator.G -= estimator.G[0].clone()

            error = torch.abs(torch.max(torch.square(estimator.free_energy - free_energy) / estimator.free_energy.mean()))
            free_energy = estimator.free_energy
        print(error)
        errors.append(error)

        if epoch % 100 ==0:
            plt.plot(estimator.free_energy)
            plt.plot(estimator.G)
    plt.show()
    return estimator.free_energy.numpy(), errors

if __name__ == "__main__":
    main()