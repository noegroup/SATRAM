import os
import math
import simtk.unit as unit
from FastMBAR import *
import numpy as np
from thermodynamicestimators.estimators import mbar
from thermodynamicestimators.data_sets import dataset
from torch.utils.data import dataloader


dirname = os.path.dirname(__file__)

M = 10

## read dihedral values from umbrella sampling
thetas = []
num_conf = []
for theta0_index in range(M):
    theta = np.loadtxt(os.path.join(dirname, f"output/dihedral/dihedral_{theta0_index}.csv"), delimiter = ",")
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

theta0 = np.loadtxt(os.path.join(dirname, "./output/theta0.csv"), delimiter = ",")

for theta0_index in range(M):
    current_theta0 = theta0[theta0_index]
    diff = np.abs(thetas - current_theta0)
    diff = np.minimum(diff, 2*math.pi-diff)
    A[theta0_index, :] = 0.5*K*diff**2/kbT

## solve MBAR equations using FastMBAR
fastmbar = FastMBAR(energy = A, num_conf = num_conf, cuda=False, verbose = True)
print("Relative free energies: ", fastmbar.F)

dataset = dataset.Dataset(samples=A, N_i= num_conf)
loader = dataloader(dataset, batch_size=128, shuffle=True)
slowmbar = mbar(dataset.n_states)
free_energies, errors = slowmbar.estimate(dataloader,
                                               dataset,
                                               tolerance=1e-6,
                                               max_iterations=100)
print("Relative free energies: ", free_energies)