import os, math, sys
import numpy as np
import simtk.openmm.app  as omm_app
import simtk.openmm as omm
import simtk.unit as unit


dirname = os.path.dirname(__file__)


## read the OpenMM system of butane
with open(os.path.join(dirname, "output/system.xml"), 'r') as file_handle:
    xml = file_handle.read()
system = omm.XmlSerializer.deserialize(xml)

## read psf and pdb file of butane
psf = omm_app.CharmmPsfFile(os.path.join(dirname, 'data/butane.psf'))
pdb = omm_app.PDBFile(os.path.join(dirname, 'data/butane.pdb'))

#### setup an OpenMM context

## platform
platform = omm.Platform.getPlatformByName('CPU')

## intergrator
T = 298.15 * unit.kelvin  ## temperature
fricCoef = 10/unit.picoseconds ## friction coefficient 
stepsize = 1 * unit.femtoseconds ## integration step size
integrator = omm.LangevinIntegrator(T, fricCoef, stepsize)

## construct an OpenMM context
context = omm.Context(system, integrator, platform)

## set force constant K for the biasing potential. 
## the unit here is kJ*mol^{-1}*nm^{-2}, which is the default unit used in OpenMM
K = 100
context.setParameter("K", K)

## M centers of harmonic biasing potentials
M = 10
theta0 = np.linspace(-math.pi, math.pi, M, endpoint = False)
np.savetxt(os.path.join(dirname,"output/theta0.csv"), theta0, delimiter = ",")

## the main loop to run umbrella sampling window by window
for theta0_index in range(M):
    print(f"sampling at theta0 index: {theta0_index} out of {M}")

    ## set the center of the biasing potential
    context.setParameter("theta0", theta0[theta0_index])

    ## minimize
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy = True)
    energy = state.getPotentialEnergy()
    for i in range(50):
        omm.LocalEnergyMinimizer_minimize(context, 1, 20)
        state = context.getState(getEnergy = True)
        energy = state.getPotentialEnergy()

    ## initial equilibrium
    integrator.step(5000)
    
    fpath = os.path.join(dirname, f"output/traj/traj_{theta0_index}.dcd")
    ## sampling production. trajectories are saved in dcd files
    file_handle = open(fpath, 'bw')
    dcd_file = omm_app.dcdfile.DCDFile(file_handle, psf.topology, dt = stepsize)
    for i in range(1000):
        integrator.step(100)
        state = context.getState(getPositions = True)
        positions = state.getPositions()
        dcd_file.writeModel(positions)
    file_handle.close()
