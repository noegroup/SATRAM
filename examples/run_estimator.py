import argparse
import SGD_free_energy_estimators.estimators.WHAM as WHAM
import SGD_free_energy_estimators.potential as potential
import SGD_free_energy_estimators.MCMC as MCMC
import matplotlib.pyplot as plt


"""
main.py

Prepares histogram data by setting up a potential with biases and using MCMC for sampling.
Uses WHAM to return an estimate of the potential function.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--potential", default='double_well', help="The potential function")
    parser.add_argument("-d", "--n_dimensions", default=1, help="The number of dimensions to sample from")
    parser.add_argument("-s", "--n_simulations", default=2, help="The number of simulations PER BIAS")
    parser.add_argument("-n", "--n_samples", default=10000, help="The number of samples per simulation")
    parser.add_argument("-m", "--n_hist_bins", default=100, help="The number of histogram buckets (in case of WHAM)")
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
    data = sampler.sample(U, biases)

    # Use WHAM to estimate the original potential function
    plt.plot(range(args.hist_max), WHAM.run(args, data, biases, method='iterative'), label='Iterative')
    plt.plot(range(args.hist_max), WHAM.run(args, data, biases, method='SGD'), label='SGD')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()