import os

import numpy as np

from khan.data import pyanitools as pya


HARTREE_TO_KCAL_PER_MOL = 627.509

selfIxnNrgWB97 = np.array([
    -0.500607632585,
    -37.8302333826,
    -54.5680045287,
    -75.0362229210], dtype=np.float32)

# DFT-calculated self-interaction, not linear fitted
selfIxnNrgMO62x = np.array([
  -0.498135,
  -37.841399,
  -54.586413,
  -75.062826,
], dtype=np.float32)

# linear fitted self-interaction
selfIxnNrgFitted = np.array([
    -374.85 / HARTREE_TO_KCAL_PER_MOL, 
    -23898.1 / HARTREE_TO_KCAL_PER_MOL, 
    -34337.6 / HARTREE_TO_KCAL_PER_MOL, 
    -47188.0 / HARTREE_TO_KCAL_PER_MOL 
], dtype=np.float32)

# import correction
# import featurizer

MAX_ATOM_LIMIT = 32

def convert_species_to_atomic_nums(s):
  PERIODIC_TABLE = {"H": 0, "C": 1, "N": 2, "O": 3}
  res = []
  for k in s:
    res.append(PERIODIC_TABLE[k])
  res =  np.array(res, dtype=np.int32)
  np.ascontiguousarray(res)
  return res


def filter(xyz_file, use_fitted):
    with open(xyz_file, "r") as fh:

        header = fh.readline()
        comment = fh.readline()

        cols = comment.split()

        c = None
        # print(len(cols))

        y = float(cols[-1])

        body = fh.readlines()
        elems = []
        coords = []

        for line in body:
            res = line.split()
            elem = res[0]
            elems.append(elem)
            coords.append((float(res[1]),float(res[2]),float(res[3])))

        if len(elems) > MAX_ATOM_LIMIT:
            return True

        coords = np.array(coords, dtype=np.float32)

        # Z = convert_species_to_atomic_nums(elems)

        PERIODIC_TABLE = {"H": 0, "C": 1, "N": 2, "O": 3}
        res = []
        for k in elems:
            if k not in PERIODIC_TABLE:
                return True

        return False


def parse_xyz(xyz_file, use_fitted):
    """
    If use_fitted is False, return the mo62x atomization energies. Otherwise, return a fitted energy.
    """

    # charge_offsets_triplets = {
    #     -2: -0.025669747949388432,
    #     -1: -0.06299837847398589,
    #     0: -0.006866110783921851,
    #     1: 0.23970742577917747,
    #     2: 0.613121455830691
    # }

    charge_offsets_pairwise = {
       -2: -0.1351248548906346,
       -1: -0.13134710542420283,
       0: -0.06481444455289967,
       1: 0.1661214579933956,
       2: 0.5260559975736232,
    }

    with open(xyz_file, "r") as fh:

        header = fh.readline()
        comment = fh.readline()

        cols = comment.split()

        c = None
        # print(len(cols))

        y = float(cols[-1])

        body = fh.readlines()
        elems = []
        coords = []

        for line in body:
            res = line.split()
            elem = res[0]
            elems.append(elem)
            coords.append((float(res[1]),float(res[2]),float(res[3])))

        coords = np.array(coords, dtype=np.float32)

        Z = convert_species_to_atomic_nums(elems)

        mo62xoffset = 0

        for z in Z:
            mo62xoffset += selfIxnNrgMO62x[z]

        if use_fitted:
            js18pairwiseOffset = correction.jamesPairwiseCorrection_C(coords, Z)/HARTREE_TO_KCAL_PER_MOL
            y -= js18pairwiseOffset
            if len(cols) == 9:
                c = float(cols[-2])
                y -= charge_offsets_pairwise[c]

        else:
            y -= mo62xoffset 

        R = np.array(coords, dtype=np.float32)
        X = np.concatenate([np.expand_dims(Z, 1), R], axis=1)

        return X, y, c


def load_ff_files(ff_dir, use_fitted=False, names=[], group_names=[]):
    Xs = []
    ys = []
    g_ys = []
    # cs = []
    for root, dirs, files in os.walk(ff_dir):
        group_ys = []
        for filename in files:
            rootname, extension = os.path.splitext(filename)
            if extension == ".xyz":
                filepath = os.path.join(root, filename)

                if filter(filepath, use_fitted):
                    # print("Filtered")
                    continue

                X, y, c = parse_xyz(filepath, use_fitted)

                Xs.append(X)
                ys.append(y)
                group_ys.append(y)
                names.append(filepath)

            else:
                print("Unknown filetype:", filename)

        if len(group_ys) > 0:
            g_ys.append(group_ys)
            group_names.append(root)

    # for charge in [-2, -1, 0, 1, 2]:
        # m_idxs = np.argwhere(np.array(cs, dtype=np.int32) == charge)
        # mean_per_charge = np.mean(np.array(ys)[m_idxs])
        # print(charge, mean_per_charge)

    # import matplotlib.mlab as mlab
    # import matplotlib.pyplot as plt

    # plt.plot(ys, cs, 'r+')
    # plt.xlabel('energy')
    # plt.ylabel('formal charge')
    # plt.show()
    # assert 0

    # n, bins, patches = plt.hist(ys, 300, facecolor='green', alpha=0.75)
    # plt.show()

    # assert 0
    # print(len(Xs), len(ys))
    return Xs, ys, g_ys

def load_hdf5_files(
    hdf5files,
    calibration_map=None,
    energy_cutoff=100.0/HARTREE_TO_KCAL_PER_MOL,
    use_fitted=False):
    """
    Load the ANI dataset.

    Parameters
    ----------
    hdf5files: list of str
        List of paths to hdf5 files that will be used to generate the dataset. The data should be
        in the format used by the ANI-1 dataset.

    use_fitted: bool
        If use_fitted is False, return the mo62x atomization energies. Otherwise, return a fitted energy.


    Returns
    -------
    Dataset, list of int
        Returns a Dataset object and a list of integers corresponding to the groupings of the
        respective atoms.

    """

    # zs = []
    Xs = []
    ys = []

    print("Loading...")

    num_samples = 0

    for hdf5file in hdf5files:
        print("Processing", hdf5file)
        adl = pya.anidataloader(hdf5file)
        for data in adl:

            # Extract the data
            P = data['path']
            R = data['coordinates']
            E = data['energies']
            S = data['species']
            smi = data['smiles']

            path = P.split("/")[-1]

            Z = convert_species_to_atomic_nums(S)

            if len(Z) > MAX_ATOM_LIMIT:
                print("skippng", P, 'n_atoms too large:', len(Z), '>', MAX_ATOM_LIMIT)
                continue

            minimum_wb97 = np.amin(E)

            if use_fitted:

                calibration_offset = 0

                if calibration_map:
                    calibration_offset = calibration_map[path] - minimum_wb97 

                for k in range(len(E)):
                    if energy_cutoff is not None and E[k] - minimum_wb97 > energy_cutoff:
                        continue


                    # BROKEN FOR NOW
                    js18pairwiseOffset = correction.jamesPairwiseCorrection_C(R[k], Z)/HARTREE_TO_KCAL_PER_MOL
                    y = E[k] - js18pairwiseOffset + calibration_offset

                    ys.append(y)
                    X = featurizer.ANI1(R[k], Z)
                    Xs.append(X)
                    # BROKEN FOR NOW

            else:
                wb97offset = 0
                mo62xoffset = 0

                for z in Z:
                    wb97offset += selfIxnNrgWB97[z]
                    mo62xoffset += selfIxnNrgMO62x[z]

                calibration_offset = 0

                if calibration_map:
                    min_atomization_wb97 = minimum_wb97 - wb97offset
                    min_atomization_mo62x = calibration_map[path] - mo62xoffset
                    # difference between the wb97_min and the mo62x_min
                    calibration_offset = min_atomization_mo62x - min_atomization_wb97

                for k in range(len(E)):
                    if energy_cutoff is not None and E[k] - minimum_wb97 > energy_cutoff:
                        continue


                    y = E[k] - wb97offset + calibration_offset
                    ys.append(y)

                    X = np.concatenate([np.expand_dims(Z, 1), R[k]], axis=1)
                    Xs.append(X)

    return Xs, ys

