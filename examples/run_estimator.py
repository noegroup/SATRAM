import argparse
import thermodynamicestimators.utilities.data_generator as data_generator
import thermodynamicestimators.estimators.wham_nd as wham_nd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


"""
main.py

Prepares histogram data by setting up a potential with biases and using MCMC for sampling.
Uses WHAM to return an estimate of the potential function.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--potential", default='double_well_2D', help="The potential function")
    parser.add_argument("-d", "--n_dimensions", default=2, help="The number of dimensions to sample from")
    parser.add_argument("-n", "--n_samples", default=1000, help="The number of samples per simulation")
    parser.add_argument("-m", "--n_bins", default=20, help="The number of histogram buckets (in case of WHAM)")
    parser.add_argument("--n_simulations", default=5, help="The number of simulations to run per bias potential")
    # parser.add_argument("-b", "--n_biases", default=10, help="The number of bias potentials")
    parser.add_argument("--x_min", default=5, help="Minimum of the leftmost histogram bin")
    parser.add_argument("--x_max", default=25, help="Maximum of the rightmost histogram bin")
    parser.add_argument("--y_min", default=5, help="Minimum of the leftmost histogram bin")
    parser.add_argument("--y_max", default=25, help="Maximum of the rightmost histogram bin")
    parser.add_argument("--tolerance", default=1e-5, help="Error tolerance for convergence")
    parser.add_argument("--max_iterations", default=100, help="Maximum number of iterations allowed to converge")

    args = parser.parse_args()

    # generate data
    _data_generator = data_generator.data_generator(args)
    data = _data_generator.get_data()
    biases = _data_generator.biases


    estimator = wham_nd.WHAM_nd(biases, args.n_bins, args.n_dimensions)
    estimate_free_energy_nd(estimator, data, n_batches=10, args=args)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = range(args.x_min, args.x_max)
    y = range(args.x_min, args.x_max)
    X, Y = np.meshgrid(x, y)

    real_potential = np.zeros_like(X)
    for idx, _ in np.ndenumerate(real_potential):
        real_potential[idx] = _data_generator.potential((X[idx], Y[idx]))

    ax.plot_surface(X, Y, estimator.free_energy)
    ax.plot_wireframe(X, Y, real_potential, label="Real potential function", color='r')

    plt.show()



def estimate_free_energy_nd(estimator, data, n_batches, args):
    lr = 0.01  # learning rate

    samples_per_bias = np.asarray([len(data[i]) for i in range(len(data))])

    batch_size = int(sum(samples_per_bias) / n_batches)

    # leave only the dimension of the coordinates intact, flatten otherwise
    data = data.reshape(-1, data.shape[-1])

    epoch = 0

    error = 1
    while epoch < args.max_iterations and error > args.tolerance:
        epoch += 1

        np.random.shuffle(data)

        p_old = estimator.probabilities

        for b in range(n_batches):
            batch_data = data[b * batch_size:(b + 1) * batch_size]
            hist = np.histogramdd(batch_data
                                  , range=[[args.x_min, args.x_max], [args.y_min,args.y_max]][:args.n_dimensions]
                                  , bins=args.n_bins)[0]

            estimator.step(hist, samples_per_bias/n_batches, lr)

        error = np.square(np.subtract(p_old, estimator.probabilities)).mean() / estimator.probabilities.mean()
        print(error)
    print("SGD done after {} epochs.".format(epoch))


if __name__ == "__main__":
    main()