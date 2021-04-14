import argparse
import thermodynamicestimators.estimators as estimators
import thermodynamicestimators.utilities.potential as potential
import thermodynamicestimators.utilities.MCMC as MCMC
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
    # data = torch.tensor(np.histogram(data, range=(args.hist_min, args.hist_max), bins=args.n_bins)[0], dtype=float)

    # instantiate estimator and optimizers
    estimator = estimators.wham.WHAM(biases, args.n_bins)
    SGD = torch.optim.SGD(estimator.parameters(), lr=0.1)
    ADAM = torch.optim.Adam(estimator.parameters(), lr=0.001)

    # estimate free energy based on sampled data and plot
    for (optimizer, label) in [(SGD, "SGD"), (ADAM, "ADAM")]:
        estimate_free_energy(estimator, optimizer, data, n_batches=10, args=args)
        # plt.plot(estimate_free_energy(estimator, optimizer, data, n_batches=1, args=args).detach().numpy(), label=label)

    plt.plot(U(range(args.hist_max)), label="Real potential function")
    plt.legend()
    plt.show()



def estimate_free_energy(estimator, optimizer, data, n_batches, args):
    losses = []
    batch_size = int(len(data)/n_batches)

    for i in range(100):
        np.random.shuffle(data)

        for b in range(n_batches):
            batch_data = torch.tensor(
                np.histogram(data[b*batch_size:(b+1)*batch_size], range=(args.hist_min, args.hist_max), bins=args.n_bins)[0],
                dtype=float)

            optimizer.zero_grad()
            loss = estimator.residue(batch_data)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

    plt.plot(losses)
    plt.show()
    return estimator.get_free_energy()


if __name__ == "__main__":
    main()