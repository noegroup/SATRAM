import numpy as np


def harmonic(r, r_0, k):
    return k * (r - r_0)**2


def double_well_1D():
    x = np.linspace(0, 100, 5)
    y = [10, 0, 10, 0, 10]

    z = np.polyfit(x, y, 6)
    U = np.poly1d(z)
    return U


def double_well_2D():
    gaussian_params = [(8,15,15,10,10), (4.8,9,9,2.5,2.5), (8,9,21,2.5,2.5), (4,21,13,2.5,2.5)]

    def gauss(x, y, params):
        (a, h_1, h_2, s_1, s_2) = params
        return a * np.exp(-(x-h_1)**2/(2*s_1**2) - (y-h_2)**2/(2*s_2**2))

    def potential(r):
        (x, y) = r
        return -np.sum([gauss(x,y,params) for params in gaussian_params])

    return potential