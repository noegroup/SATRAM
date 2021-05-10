import os
import math
import matplotlib.pyplot as plt
import simtk.unit as unit
from FastMBAR import *
import numpy as np


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
