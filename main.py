import numpy as np
import tempfile
import time
import tensorflow as tf

from khan.model.nn import MoleculeNN
from khan.training.trainer import Trainer
from khan.data import pyanitools as pya
from khan.data.dataset import RawDataset, FeaturizedDataset

from concurrent.futures import ThreadPoolExecutor

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

            # debug
            if len(Xs) > 16000:
                break

    return Xs, ys

if __name__ == "__main__":

    Xs, ys = load_hdf5_files([
        # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s01.h5",
        # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s02.h5",
        # "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s03.h5",
        "/home/yutong/roitberg_data/ANI-1_release/ani_gdb_s08.h5",
    ])

    # Xs, ys = Xs[:12000], ys[:12000]

    rd = RawDataset(Xs, ys)

    batch_size = 1024

    # data_dir = tempfile.mkdtemp()

    data_dir = "/media/yutong/fast_datablob/v3"

    # print("featurizing...")
    # fd = rd.featurize(batch_size=batch_size, data_dir=data_dir)
    
    fd = FeaturizedDataset(data_dir)
    print("done...")
    # batch_size = 1024

    f0_enq = tf.placeholder(dtype=tf.float32)
    f1_enq = tf.placeholder(dtype=tf.float32)
    f2_enq = tf.placeholder(dtype=tf.float32)
    f3_enq = tf.placeholder(dtype=tf.float32)
    gi_enq = tf.placeholder(dtype=tf.int32)
    mi_enq = tf.placeholder(dtype=tf.int32)
    yt_enq = tf.placeholder(dtype=tf.float32)

    staging = tf.contrib.staging.StagingArea(
        capacity=10, dtypes=[
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.int32,
            tf.int32,
            tf.float32])

    put_op = staging.put([f0_enq, f1_enq, f2_enq, f3_enq, gi_enq, mi_enq, yt_enq])
    get_op = staging.get()

    # feat_size = 768

    f0, f1, f2, f3, gi, mi, yt = get_op[0], get_op[1], get_op[2], get_op[3], get_op[4], get_op[5], get_op[6]

    mnn = MoleculeNN(
        type_map=["H", "C", "N", "O"],
        atom_type_features=[f0, f1, f2, f3],
        gather_idxs=gi,
        mol_idxs=mi,
        layer_sizes=(384, 256, 128, 64, 1))

    trainer = Trainer(mnn, yt)
    results_all = trainer.get_train_op()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_epochs = 32

    def submitter():
        for _ in range(num_epochs):
            for b_idx, (f0, f1, f2, f3, gi, mi, yt) in enumerate(fd.iterate()):
                try:
                    sess.run(put_op, feed_dict={
                        f0_enq: f0,
                        f1_enq: f1,
                        f2_enq: f2,
                        f3_enq: f3,
                        gi_enq: gi,
                        mi_enq: mi,
                        yt_enq: yt,
                    })
                except Exception as e:
                    print("OMG WTF BBQ", e)

    executor = ThreadPoolExecutor(4)

    executor.submit(submitter)

    tot_time = 0
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    
    print("num batches", fd.num_batches())

    st = time.time()
    for e in range(num_epochs):
        print("epoch:", e)
        for i in range(fd.num_batches()):
            # print("running", i)


            sess.run(results_all)

    tot_time = time.time() - st # this logic is a little messed up

    tpm = tot_time/(fd.num_batches()*batch_size*num_epochs)
    print("Time Per Mol:", tpm, "seconds")
    print("Samples per minute:", 60/tpm)