import numpy as np
import matplotlib.pyplot as plt


def get_biases(args):
    bias = lambda x, x_0: 0.05 * (x-x_0)**2
    bias_centers = np.linspace(args.hist_min, args.hist_max, args.n_biases)
    biases = [(lambda x, x_0=bias_center: bias(x,x_0)) for bias_center in bias_centers]

    return np.asarray(biases)


def get_potential(name):
    if name =='double_well':
        x = np.linspace(0, 100, 5)
        y = [10, 0, 10, 0, 10]

        z = np.polyfit(x, y, 6)
        U = np.poly1d(z)
        return U