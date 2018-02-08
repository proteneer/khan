import os
import time
import numpy as np

from khan.data import pyanitools as pya

HARTREE_TO_KCAL_PER_MOL = 627.509
CALIBRATION_FILE = "/home/yutong/roitberg_data/ANI-1_release/results_QM_M06-2X.txt"

params_list = [ -3.61495025e+02,  -2.38566440e+04,  -3.43234157e+04,
        -4.71784651e+04,  -1.35927769e+02,  -1.74835391e+02,
        -2.66558100e+02,   2.36476049e+01,   1.00037527e+02,
         1.27547041e+01,   1.72507376e+02,  -7.21578715e+01,
        -2.77910695e+02,   2.10919757e+03,   2.79778489e+03,
         3.47283119e+03,   3.22775414e+02,   4.65919734e+02,
         2.06357637e+03,   1.51680516e+03,   2.66909212e+03,
         3.29117774e+03,  -6.80491343e+03,  -8.27595549e+03,
        -9.40328190e+03,  -3.46198683e+03,  -4.85364601e+03,
        -1.00000000e+04,  -9.99999939e+03,  -9.99934500e+03,
        -1.37665219e+03,   5.88608669e+03,   6.60117401e+03,
         7.06803912e+03,   5.16847643e+03,   6.35979202e+03,
         1.11347193e+04,   1.26617984e+04,   1.03047549e+04,
         1.68165667e+02] # from 1.2M run

n_atom_types = 4

pair_indices = {
    ('H','C'):0,
    ('H','N'):1,
    ('H','O'):2,
    ('C','C'):3,
    ('C','N'):4,
    ('C','O'):5,
    ('N','N'):6,
    ('N','O'):7,
    ('O','O'):8
}

def convert_species_to_atomic_nums(s):
    PERIODIC_TABLE = {"H": 0, "C": 1, "N": 2, "O": 3}
    res = []
    for k in s:
        res.append(PERIODIC_TABLE[k])
    return np.array(res, dtype=np.int32)

n_pairs = len(pair_indices)

pair_indices = {
    **pair_indices,
    ('C','H'):0,
    ('N','H'):1,
    ('O','H'):2,
    ('N','C'):4,
    ('O','C'):5,
    ('O','N'):7
} # add reversed pairs

def jamesPairwiseCorrection(coords_list, species_list):
    X, S = coords_list, species_list
    # print(X, S)
    params = params_list
    E_self = params[0]*S.count('H') + params[1]*S.count('C') + params[2]*S.count('N') + params[3]*S.count('O')



    # for z in convert_species_to_atomic_nums(species_list):
        # print(params_list[z])


    E_pair = 0
    # print("---")
    for i, a in enumerate(X): # i is index, a is (x,y,z)
        for j, b in enumerate(X[i+1:]): # j is index, b is (x,y,z)
            j += i+1
            if S[i]=='H' and S[j]=='H':
                continue # ignore H for speed and physics
            r2 = (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2

            # print(r2)

            basis1 = np.exp(-0.5*r2)
            basis2 = basis1*basis1
            basis3 = basis2*basis1
            basis4 = basis3*basis1
            input_index = pair_indices[ (S[i],S[j]) ]

            # print(basis1, basis2, basis3, basis4)           

            # print(params[ n_atom_types + n_pairs*0 + input_index ],
            #     params[ n_atom_types + n_pairs*1 + input_index ],
            #     params[ n_atom_types + n_pairs*2 + input_index ],
            #     params[ n_atom_types + n_pairs*3 + input_index ])

            energy = basis1 * params[ n_atom_types + n_pairs*0 + input_index ]
            energy += basis2 * params[ n_atom_types + n_pairs*1 + input_index ]
            energy += basis3 * params[ n_atom_types + n_pairs*2 + input_index ]
            energy += basis4 * params[ n_atom_types + n_pairs*3 + input_index ]


            E_pair += energy

            # print(energy)


    # print(E_self, E_pair)
    # assert 0

    return E_self + E_pair

def load_calibration_file(calibration_file):
    with open(calibration_file, 'r') as fh:
        mapping = {}
        for line in fh.readlines():
            parts = line.split()
            path = parts[0].split(".")[0]
            energy = parts[-1]
            mapping[path] = float(energy)
        return mapping

def load_hdf5_files(
    hdf5files,
    calibration_map=None,
    energy_cutoff=100.0/HARTREE_TO_KCAL_PER_MOL):

    Xs = []
    ys = []

    print("Loading...")

    for hdf5file in hdf5files:
        adl = pya.anidataloader(hdf5file)
        for data in adl:

            # Extract the data
            P = data['path']
            R = data['coordinates']
            E = data['energies']
            S = data['species']
            smi = data['smiles']


            print("Processing: ", P)

            path = P.split("/")[-1]

            Z = convert_species_to_atomic_nums(S)
            minimum = np.amin(E)

            calibration_offset = 0

            if calibration_map:
                calibration_offset = calibration_map[path] - minimum 

            for k in range(len(E)):
                if energy_cutoff is not None and E[k] - minimum > energy_cutoff:
                    continue

                js18pairwiseOffset = jamesPairwiseCorrection(R[k], S)/HARTREE_TO_KCAL_PER_MOL

                y = E[k] - js18pairwiseOffset + calibration_offset

                # y = E[k] - wb97offset + calibration_offset
                ys.append(y)
                X = np.concatenate([np.expand_dims(Z, 1), R[k]], axis=1)
                Xs.append(X)

    return Xs, ys

if __name__ == "__main__":

    ROITBERG_ANI_DIR = "/home/yutong/roitberg_data/ANI-1_release"

    cal_map = load_calibration_file(CALIBRATION_FILE)

    st = time.time()

    Xs, ys = load_hdf5_files([
        os.path.join(ROITBERG_ANI_DIR, "ani_gdb_s01.h5"),
        os.path.join(ROITBERG_ANI_DIR, "ani_gdb_s02.h5"),
        os.path.join(ROITBERG_ANI_DIR, "ani_gdb_s03.h5"),
    ],  calibration_map=cal_map)

    print(time.time() - st, "seconds")