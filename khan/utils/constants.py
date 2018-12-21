from scipy import constants
ANGSTROM_IN_BOHR = constants.physical_constants['Bohr radius'][0]*1.0e10
MOLECULES_IN_MOL = constants.physical_constants['Avogadro constant'][0]
JOULES_IN_CALORIE = constants.calorie
KCAL_MOL_IN_HARTREE = 1.0e-3*constants.physical_constants['Hartree energy'][0]*MOLECULES_IN_MOL/JOULES_IN_CALORIE

# highest energy geometries used for training
ENERGY_CUTOFF = 100.0/KCAL_MOL_IN_HARTREE
