import os
import math
import cProfile
import numpy as np
import torch
import simtk.unit as unit
import matplotlib.pyplot as plt
from FastMBAR import *
from pymbar import mbar as pymbar
from thermodynamicestimators.estimators import mbar
from thermodynamicestimators.data_sets import dataset, infinite_dataloader

import time


if __name__ =='__main__':
    dirname = os.path.dirname(__file__)

    M = 10

    ## read dihedral values from umbrella sampling
    thetas = []
    num_conf = []
    for theta0_index in range(M):
        theta = np.loadtxt(os.path.join(dirname, f"output/dihedral/dihedral_{theta0_index}.csv"), delimiter=",")
        thetas.append(theta)
        num_conf.append(len(theta))
    thetas = np.concatenate(thetas)
    num_conf = np.array(num_conf).astype(np.float64)
    N = len(thetas)

    ## compute reduced energy matrix A
    A = np.zeros((M, N))
    K = 100
    T = 298.15 * unit.kelvin
    kbT = unit.BOLTZMANN_CONSTANT_kB * 298.15 * unit.kelvin * unit.AVOGADRO_CONSTANT_NA
    kbT = kbT.value_in_unit(unit.kilojoule_per_mole)

    theta0 = np.loadtxt(os.path.join(dirname, "./output/theta0.csv"), delimiter=",")

    for theta0_index in range(M):
        current_theta0 = theta0[theta0_index]
        diff = np.abs(thetas - current_theta0)
        diff = np.minimum(diff, 2 * math.pi - diff)
        A[theta0_index, :] = 0.5 * K * diff ** 2 / kbT

    ## solve MBAR equations using FastMBAR
    t0 = time.perf_counter()
    fastmbar = FastMBAR(energy=A, num_conf=num_conf, cuda=False, verbose=True)

    t1 = time.perf_counter()
    print(t1 - t0)
    print("Relative free energies: ", fastmbar.F)

    t0 = time.perf_counter()
    py_mbar = pymbar.MBAR(u_kn=A, N_k=num_conf, verbose=True)
    t1 = time.perf_counter()
    print(t1-t0)
    pymbar_free_energies = py_mbar.getFreeEnergyDifferences()[0][0]

    t0 = time.perf_counter()
    n_batches = int(len(A)/ 64)
    print(n_batches)
    start = 0
    stop = 0
    for i in range(n_batches):
        start = stop
        stop += 64
        a = A[start:stop].copy()
    t1 = time.perf_counter()

    print("Batch indexing time: {}".format(t1 - t0))

    A = torch.Tensor(A).T
    N_i = torch.Tensor(num_conf)
    dataset = dataset.Dataset(samples=A, N_i=N_i)
    dataloader = infinite_dataloader.InfiniteDataLoader(dataset, batch_size=64, shuffle=True, num_workers=1, prefetch_factor=256)
    slowmbar = mbar.MBAR(dataset.n_states)
    optimizer = torch.optim.SGD(slowmbar.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    cProfile.run('slowmbar.estimate(dataloader, dataset, optimizer, scheduler, tolerance=1e-3, ground_truth=torch.Tensor(fastmbar.F), max_iterations=100)')

    # free_energies, errors = slowmbar.estimate(dataloader, dataset, optimizer, scheduler,
    #                                           tolerance=1e-3, ground_truth=torch.Tensor(fastmbar.F), max_iterations=200)
    print("Relative free energies: ", free_energies)
    plt.plot(pymbar_free_energies, label="PyMBAR")
    plt.plot(fastmbar.F, label="FastMBAR")
    plt.plot(free_energies, label="SlowMBAR")
    plt.legend()
    plt.show()

