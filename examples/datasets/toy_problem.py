import math
import torch
import scipy.spatial


torch.random.manual_seed(100)

n_therm_states = 4
n_conf_states = 5

mu = torch.linspace(-1, 1, n_therm_states)
s = torch.Tensor([0, 0.5, 1, 2])  # np.linspace(0,num_therm_states-1,num_therm_states)
centers = torch.linspace(-1, 1, n_conf_states).reshape(-1, 1)
sigma2 = 0.05
T = 100 #int(1e4)

def get_ground_truth(*args, **kwargs):
    return torch.Tensor(s)


# perform simulation
def OU_simulation(mu, sigma2, b, delta_t, T):
    a = b * b / 2 / sigma2
    x = torch.randn(1) * math.sqrt(sigma2)
    traj = torch.zeros([T, 1])
    r = math.exp(-a * delta_t)
    v = math.sqrt(b * b * (1 - math.exp(-2 * a * delta_t)) / 2 / a)
    for t in range(T):
        x = x * r + v * torch.randn(1)
        traj[t] = x
    traj += mu
    return traj

lag=1


def get_data():
    trajs = []
    energies = []

    for i in range(n_therm_states):
        traj = OU_simulation(mu[i], sigma2, 1, 10, T)

        trajs.append(traj)
        energies.append((traj - mu) ** 2 / (2 * sigma2))

    return trajs, energies


def get_tram_input():
    trajs, energies = get_data()

    ttrajs = []
    dtrajs = []
    bias_list = []

    for i in range(n_therm_states):
        traj = trajs[i]
        tmp_d = torch.Tensor(scipy.spatial.distance.cdist(traj, centers) ** 2)
        dtraj = torch.argmin(tmp_d, 1)

        trajs.append(traj)
        dtrajs.append(dtraj.int())
        ttrajs.append(torch.Tensor([i] * T).int())

        bias = energies[i] + s
        bias_list.append(bias)

    return ttrajs, dtrajs, bias_list


