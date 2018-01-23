import numpy as np

from khan.data import pyanitools as pya

from khan.data.dataset import RawDataset, FeaturizedDataset

HARTREE_TO_KCAL_PER_MOL = 627.509

def convert_species_to_atomic_nums(s):
  PERIODIC_TABLE = {"H": 0, "C": 1, "N": 2, "O": 3}
  res = []
  for k in s:
    res.append(PERIODIC_TABLE[k])
  return np.array(res, dtype=np.int32)

def load_hdf5_files(hdf5files, energy_cutoff=100.0/HARTREE_TO_KCAL_PER_MOL):
    """
    Load the ANI dataset.

    Parameters
    ----------
    hdf5files: list of str
        List of paths to hdf5 files that will be used to generate the dataset. The data should be
        in the format used by the ANI-1 dataset.

    batch_size: int
        Used to determined the shard_size, where shard_size is batch_size * 4096

    data_dir: str
        Directory in which we save the resulting data

    mode: str
        Accepted modes are "relative", "atomization", or "absolute". These settings are used
        to adjust the dynamic range of the model, with absolute having the greatest and relative
        having the lowest. Note that for atomization we approximate the single atom energy
        using a different level of theory

    max_atoms: int
        Total number of atoms we allow for.

    energy_cutoff: int or None
        A cutoff to use for pruning high energy conformations from a dataset. Units are in
        hartrees. Default is set to 100 kcal/mol or ~0.16 hartrees.

    selection_size: int or None
        Subsample of conformations that we want to choose from gdb-8

    Returns
    -------
    Dataset, list of int
        Returns a Dataset object and a list of integers corresponding to the groupings of the
        respective atoms.

    """

    atomizationEnergies = np.array([
        0,
        -0.500607632585,
        -37.8302333826,
        -54.5680045287,
        -75.0362229210], dtype=np.float32)

    Xs = []
    ys = []

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

            Z = convert_species_to_atomic_nums(S)
            minimum = np.amin(E)

            offset = 0

            for z in Z:
                offset += atomizationEnergies[z]

            for k in range(len(E)):
                if energy_cutoff is not None and E[k] - minimum > energy_cutoff:
                    continue

                y = E[k] - offset
                ys.append(y)

                # print(np.expand_dims(Z, 1), R[k])

                X = np.concatenate([np.expand_dims(Z, 1), R[k]], axis=1)
                Xs.append(X)

    return Xs, ys

if __name__ == "__main__":

    Xs, ys = load_hdf5_files([
        "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s01.h5",
        # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s02.h5",
        # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s03.h5",
    ])

    # rd = RawDataset(Xs, ys)

    # fd = rd.featurize(batch_size=64, data_dir="/tmp")


    fd = FeaturizedDataset("/tmp")
    for af, gi, mi in fd.iterate():
        print(af, gi, mi)