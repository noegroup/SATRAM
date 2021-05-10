import os, math
import simtk.openmm.app as omm_app
import simtk.openmm as omm


dirname = os.path.dirname(__file__)

psf = omm_app.CharmmPsfFile(os.path.join(dirname, 'data/butane.psf'))
pdb = omm_app.PDBFile(os.path.join(dirname, 'data/butane.pdb'))

params = omm_app.CharmmParameterSet(os.path.join(dirname, 'data/top_all35_ethers.rtf'),
                                    os.path.join(dirname, 'data/par_all35_ethers.prm'))

## creay an OpenMM system
system = psf.createSystem(params, nonbondedMethod=omm_app.NoCutoff)

## add a harmonic biasing potential on butane dihedral to the OpenMM system
bias_torsion = omm.CustomTorsionForce("0.5*K*dtheta^2; dtheta = min(diff, 2*Pi-diff); diff = abs(theta - theta0)")
bias_torsion.addGlobalParameter("Pi", math.pi)
bias_torsion.addGlobalParameter("K", 1.0)
bias_torsion.addGlobalParameter("theta0", 0.0)
## 3, 6, 9, 13 are indices of the four carton atoms in butane, between which
## the dihedral angle is biased.
bias_torsion.addTorsion(3, 6, 9, 13)
system.addForce(bias_torsion)

## save the OpenMM system of butane
with open(os.path.join(dirname, "output/system.xml"), 'w') as file_handle:
    file_handle.write(omm.XmlSerializer.serialize(system))
