import argparse
import thermodynamicestimators.utilities.test_case_factory as problem_factory
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
    dataset = problem_factory.make_test_case(args.test_name, 'MBAR')
    dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=100, shuffle=True)

    estimator = mbar.MBAR(dataset.n_states)
    optimizer_SGD = torch.optim.SGD(estimator.parameters(), lr=0.001)


    free_energy_SGD, errors_SGD = estimator.estimate(dataloader, optimizer_SGD)


    plt.plot(free_energy_SGD)
    plt.show()

    plt.yscale('log')
    plt.ylabel(r'max_i (F(t)_i - F(t-1)_i)^2 / |avg(F(t))|$')
    plt.xlabel(r'Epoch t')
    plt.plot(errors_SGD, label="error")
    plt.show()


    # to obtain a probability distribution, we discretize the space into bins and define a binning function to bin each
    # sample (position) in the correct bin
    def bin_sample(x):
        hist = [0] * 100
        hist[int(x)]=1
        return hist

    # now get the expectation value of the bin function to obtain a probability distribution over bins.
    # The negative log of this is the potential function.
    potential_SGD = -np.log(estimator.get_expectation_value(dataset, bin_sample).detach())

    plt.plot([dataset.potential_function(x) for x in range(100)], label="Real potential function")
    plt.plot(potential_SGD, label="SGD, lr=0.001")

    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()