import math
import torch
import numpy as np

import scipy.spatial

np.random.seed(1000)

n_therm_states = 4
n_conf_states = 5

mu = np.linspace(-1, 1, n_therm_states)
s = np.array([0, 0.5, 1, 2])  # np.linspace(0,num_therm_states-1,num_therm_states)
centers = np.linspace(-1, 1, n_conf_states).reshape(-1, 1)
sigma2 = 0.05
T = 100 #int(1e4)

def get_ground_truth(*args, **kwargs):
    return torch.Tensor(s)

# perform simulation
def OU_simulation(mu, sigma2, b, delta_t, T):
    a = b * b / 2 / sigma2
    x = np.random.randn() * math.sqrt(sigma2)
    traj = np.empty([T, 1])
    r = math.exp(-a * delta_t)
    v = math.sqrt(b * b * (1 - math.exp(-2 * a * delta_t)) / 2 / a)
    for t in range(T):
        x = x * r + v * np.random.randn()
        traj[t] = x
    traj += mu
    return traj

lag=1


def generate_data():
    trajs = []
    ttrajs = []
    dtrajs = []
    bias_list = []

    for i in range(n_therm_states):
        traj = OU_simulation(mu[i], sigma2, 1, 10, T)

        tmp_d = scipy.spatial.distance.cdist(traj, centers) ** 2
        dtraj = np.argmin(tmp_d, 1)

        trajs.append(traj)
        dtrajs.append(dtraj)
        ttrajs.append(np.asarray([i] * T))

        bias = (traj - mu) ** 2 / (2 * sigma2) + s
        bias_list.append(bias)

    return trajs, ttrajs, dtrajs, bias_list
