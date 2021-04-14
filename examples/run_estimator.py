import argparse
import thermodynamicestimators.estimators as estimators
import thermodynamicestimators.potential as potential
import thermodynamicestimators.MCMC as MCMC
import matplotlib.pyplot as plt
import torch
import numpy as np


"""
main.py

Prepares histogram data by setting up a potential with biases and using MCMC for sampling.
Uses WHAM to return an estimate of the potential function.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--potential", default='double_well', help="The potential function")
    parser.add_argument("-d", "--n_dimensions", default=1, help="The number of dimensions to sample from")
    # parser.add_argument("-s", "--n_simulations", default=2, help="The number of simulations PER BIAS")
    parser.add_argument("-n", "--n_samples", default=1000, help="The number of samples per simulation")
    parser.add_argument("-m", "--n_bins", default=100, help="The number of histogram buckets (in case of WHAM)")
    parser.add_argument("-b", "--n_biases", default=10, help="The number of bias potentials")
    parser.add_argument("--hist_min", default=0, help="Minimum of the leftmost histogram bin")
    parser.add_argument("--hist_max", default=100, help="Maximum of the rightmost histogram bin")
    parser.add_argument("--tolerance", default=1e-5, help="Error tolerance for convergence")
    parser.add_argument("--max_iterations", default=1000, help="Maximum number of iterations allowed to converge")

    args = parser.parse_args()

    # construct a potential and a list of bias potentials.
    U = potential.get_potential(args.potential)
    biases = potential.get_biases(args)

    # Generate data for the potential using the MCMC sampling method
    sampler = MCMC.MCMC(args)
    data = np.asarray(sampler.sample(U, biases)).flatten()
    data = torch.tensor(np.histogram(data, range=(args.hist_min, args.hist_max), bins=args.n_bins)[0], dtype=float)

    # instantiate estimator and optimizers
    estimator = estimators.wham.WHAM(biases, args.n_bins)
    SGD = torch.optim.SGD(estimator.parameters(), lr=0.1)
    ADAM = torch.optim.Adam(estimator.parameters(), lr=0.001)

    # estimate free energy based on sampled data and plot
    for (optimizer, label) in [(SGD, "Stochastic gradient descent"), (ADAM, "ADAM")]:
        plt.plot(estimate_free_energy(estimator, optimizer, data).detach().numpy(), label=label)
    plt.legend()
    plt.show()



def estimate_free_energy(estimator, optimizer, data):
    for i in range(1000):
        optimizer.zero_grad()
        loss = estimator.residue(data)
        print(loss.item())
        loss.backward()
        optimizer.step()
    return estimator.get_free_energy()


if __name__ == "__main__":
    main()