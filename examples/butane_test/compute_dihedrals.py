import os
import math
import numpy as np
import mdtraj

dirname = os.path.dirname(__file__)

M = 10
theta0 = np.linspace(-math.pi, math.pi, M, endpoint = False)

topology = mdtraj.load_psf(os.path.join(dirname,"data/butane.psf"))
for theta0_index in range(M):
    traj = mdtraj.load_dcd(os.path.join(dirname, f"output/traj/traj_{theta0_index}.dcd"), topology)
    theta = mdtraj.compute_dihedrals(traj, [[3, 6, 9, 13]])
    np.savetxt(os.path.join(dirname, f"output/dihedral/dihedral_{theta0_index}.csv"), theta, fmt = "%.5f", delimiter = ",")

