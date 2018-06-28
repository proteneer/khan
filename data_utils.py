import os

import numpy as np

from khan.data import pyanitools as pya
import json
from sklearn.model_selection import train_test_split


HARTREE_TO_KCAL_PER_MOL = 627.509

# these energies are incorrect, see https://github.com/isayev/ANI1_dataset/issues/2
# the "correct" energies are 

selfIxnNrgWB97X = np.array([
    -0.499321232710,
    -37.8338334397,
    -54.5732824628,
    -75.0424519384], dtype=np.float32)

selfIxnNrgWB97 = np.array([
    -0.500607632585,
    -37.8302333826,
    -54.5680045287,
    -75.0362229210], dtype=np.float32)

# computed by jaguar uhf with default settings
selfIxnNrgWB97X_631gdp = np.array([
    -0.49932123901, # doublet hydrogen
    -37.83271996200, # triplet carbon
    -54.57325122225, # quartet nitrogen
    -75.04147134502], # triplet oxygen
    dtype=np.float32)

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

def atomic_number_to_atom_id(atno):
    """
    Return an atom index (ANI atom type)  given an atomic number
    atomic number must be convertable to an int
    """
    return {1: 0, 6: 1, 7: 2, 8: 3}[int(atno)]

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
                wb97Xoffset = 0
                mo62xoffset = 0

                for z in Z:
                    wb97offset += selfIxnNrgWB97[z]
                    wb97Xoffset += selfIxnNrgWB97X[z]
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


                    # LDJ: using wb97x offset 
                    y = E[k] - wb97Xoffset + calibration_offset
                    ys.append(y)

                    X = np.concatenate([np.expand_dims(Z, 1), R[k]], axis=1)
                    Xs.append(X)

    return Xs, ys

def read_all_reactions(reactivity_dir):
    """
    Read reactivity data and return as a dict 

    Returns two dicts for the "small" and "big" examples defined
    by MAX_ATOM_LIMIT. The keys are filenames and values are lists of
    examples.  Example is a tuple (X, Y) for that reaction
    """

    skipped = 0
    cnt = 0
    reactions = {} 
    big_reactions = {}
    for root, dirs, files in os.walk(reactivity_dir):
        for fname in files:
            if fname.endswith(".json"):
                X, Y = _read_reactivity_data(os.path.join(reactivity_dir, fname))
                natoms = len(X[0])
                if natoms < MAX_ATOM_LIMIT:
                    reactions[fname[:-5]] = (X, Y)
                else:
                    big_reactions[fname[:-5]] = (X, Y)
                    skipped += 1
                cnt += 1

    print("Found %d out of %d reactions with less than %d atoms" % (cnt-skipped, cnt, MAX_ATOM_LIMIT))

    return reactions, big_reactions


def load_reactivity_data(reactivity_dir, percent_test=0.5):
    """
    Load all reactivity data from a directory
    split the reactions into testing/training sets (randomly)

    returns two lists of data, the training and test set.
    each element of the list is a tuple (X, Y) for that reaction
    """

    reactions_dict, big_reactions_dict = read_all_reactions(reactivity_dir)

    reactions = list(reactions_dict.values())
    big_reactions = list(big_reactions_dict.values())

    # split by reaction type
    if percent_test == 1.0:
        test = reactions
        train = []
    elif percent_test == 0.0:
        test = []
        train = reactions
    else:
        train, test = train_test_split(reactions, test_size=percent_test)
    Xtrain, Ytrain, Xtest, Ytest, Xbigtest, Ybigtest = ([], [], [], [], [], [])

    # unpack each reaction into a list of X, Y values
    for A, B in train:
        Xtrain.extend(A)
        Ytrain.extend(B)

    for A, B in test:
        Xtest.extend(A)
        Ytest.extend(B)

    for A, B in big_reactions:
        Xbigtest.extend(A)
        Ybigtest.extend(B)

    return Xtrain, Ytrain, Xtest, Ytest, Xbigtest, Ybigtest

def _read_reactivity_data(fname):
    """
    Read data from json file prepared for QM data
    there are two fields X and Y which hold the molecule definition
    and a total energy respectively.
    """

    with open(fname) as fin:
        data = json.load(fin)

    X = data.get("X")
    Y = list(map(np.float64, data.get("Y")))

    # remove atomic energies
    for i, mol in enumerate(X):
        self_interaction = sum(selfIxnNrgWB97X_631gdp[at] for at, x, y, z in mol)
        Y[i] -= self_interaction

    X_np = [np.array(molecule, dtype=np.float32) for molecule in X]

    return X_np, Y

def load_hdf5_minima_gradients(
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
    fs = []

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
            minimum_arg = np.argmin(E)

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

                if k != minimum_arg:
                    continue

                if energy_cutoff is not None and E[k] - minimum_wb97 > energy_cutoff:
                    continue

                y = E[k] - wb97offset + calibration_offset
                ys.append(y)

                # print("asdf", np.zeros_like(R[k]))

                fs.append(np.zeros_like(R[k]))

                X = np.concatenate([np.expand_dims(Z, 1), R[k]], axis=1)
                Xs.append(X)

    return Xs, ys, fs