from bgmol.systems import MiniPeptide

system = MiniPeptide("A", solvated=True, download=True)
system.system # is an openmm system instance