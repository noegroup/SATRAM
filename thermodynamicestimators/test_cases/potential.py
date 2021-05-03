import torch


def harmonic(r, r_0, k):
    return k * (r - r_0)**2


def double_well_1D():
    U = lambda x: 0.0001 * ((0.5*(x - 50))**4 - (12 * (x - 50))**2)
    return U


def double_well_2D():
    gaussian_params = [(8,15,15,10,10), (4.8,9,9,2.5,2.5), (8,9,21,2.5,2.5), (4,21,13,2.5,2.5)]

    def gauss(x, y, params):
        (a, h_1, h_2, s_1, s_2) = params
        return a * torch.exp(-(x-h_1)**2/(2*s_1**2) - (y-h_2)**2/(2*s_2**2))

    def potential(r):
        if len(r.shape) == 1:
            r = r.unsqueeze(0)

        x = r[:, 0];
        y = r[:, 1];

        return -torch.sum(torch.stack([gauss(x,y,params) for params in gaussian_params]), axis=0)

    return potential