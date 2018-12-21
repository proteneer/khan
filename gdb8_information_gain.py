import functools
import os
import numpy as np
import time
import tensorflow as tf
import sklearn.model_selection
import scipy.optimize

from khan.data.dataset import RawDataset
from khan.training.trainer_multi_tower import TrainerMultiTower, flatten_results, initialize_module
import data_utils
from khan.utils.constants import ANGSTROM_IN_BOHR
from data_loaders import DataLoader
from concurrent.futures import ThreadPoolExecutor

from multiprocessing.dummy import Pool as ThreadPool 

import multiprocessing
import argparse

kT = 0.001 # in Hartree
NN_LAYERS = tuple([256]*4 + [1])


def expected_information_gain(trainers, min_Es):
    model_energies = []
    model_ms = []
    model_xs = []
    model_ys = []
    model_zs = []

    for t, min_Es_per_point in zip(trainers, min_Es):
        model_energies.append(t.tower_preds[0] - min_Es_per_point)
        x, y, z, m = t.tower_xs[0], t.tower_ys[0], t.tower_zs[0], t.tower_ms[0]
        model_xs.append(x)
        model_ys.append(y)
        model_zs.append(z)
        model_ms.append(m)

    expected_info_gain = 0.0

    # (ytz): implement custom information KL/Prob stuff here
    # iterate over M models
    M_models = len(trainers)
    N_points = len(min_Es[0])
    for i in range(N_points):
        P_i = tf.reduce_mean( [ tf.exp(-model_energies[j][i]/kT) for j in range(M_models)] )
        expected_info_gain += P_i

    grad_xs = []
    grad_ys = []
    grad_zs = []

    for t_idx, t in enumerate(trainers):
        grad_xs.append(tf.gradients(expected_info_gain, model_xs[t_idx]))
        grad_ys.append(tf.gradients(expected_info_gain, model_ys[t_idx]))
        grad_zs.append(tf.gradients(expected_info_gain, model_zs[t_idx]))


    return expected_info_gain, grad_xs, grad_ys, grad_zs, model_ms


def generate_feed_dict(trainer, mol_xs, mol_idxs, mol_yts, mol_grads, b_idx):

    atom_types = (mol_xs[:, 0]).astype(np.int32)

    feed_dict = {
        trainer.x_enq: mol_xs[:, 1],
        trainer.y_enq: mol_xs[:, 2],
        trainer.z_enq: mol_xs[:, 3],
        trainer.a_enq: atom_types,
        trainer.m_enq: mol_idxs,
        trainer.yt_enq: mol_yts,
        trainer.bi_enq: b_idx
    }

    if mol_grads is not None:
        feed_dict[trainer.force_enq_x] = mol_grads[:, 0]
        feed_dict[trainer.force_enq_y] = mol_grads[:, 1]
        feed_dict[trainer.force_enq_z] = mol_grads[:, 2]
    else:
        num_mols = mol_xs.shape[0]
        feed_dict[trainer.force_enq_x] = np.zeros((num_mols, 0), dtype=trainer.precision.as_numpy_dtype)
        feed_dict[trainer.force_enq_y] = np.zeros((num_mols, 0), dtype=trainer.precision.as_numpy_dtype)
        feed_dict[trainer.force_enq_z] = np.zeros((num_mols, 0), dtype=trainer.precision.as_numpy_dtype)

    return feed_dict


def submit_dataset(session, trainers, dataset, batch_size, shuffle, fuzz=None):
    try:
        executor = ThreadPoolExecutor(len(trainers))
        futures = []
        n_batches = dataset.num_batches(batch_size)
        for b_idx, (mol_xs, mol_idxs, mol_yts, mol_grads) in enumerate(dataset.iterate(batch_size=batch_size, shuffle=shuffle, fuzz=fuzz)):
            atom_types = (mol_xs[:, 0]).astype(np.int32)
            for t_idx, trainer in enumerate(trainers):
                assert(trainer.num_towers == 1)
                fd = generate_feed_dict(trainer, mol_xs, mol_idxs, mol_yts, mol_grads, b_idx)
                future = executor.submit(functools.partial(session.run, fetches=trainer.put_op, feed_dict=fd))
                futures.append(future)

            # important in conjunction with underlying FIFO queue to ensure order
            for f in futures:
                f.result()

    except Exception as e:
        print("QueueError:", e)
        exit()


def dataset_to_flat(xs):
    mol_idxs_coords = []
    mol_idxs_types = []
    atom_types = []
    flattened_xs = []
    for m_idx, x in enumerate(xs):
        atypes = x[:, 0]
        coords = x[:, 1:]
        mol_idxs_coords.extend([m_idx]*coords.size)
        mol_idxs_types.extend([m_idx]*atypes.size)
        flattened_xs.append(coords.flatten())
        atom_types.append(atypes)
    return np.array(mol_idxs_coords), np.array(mol_idxs_types), np.concatenate(flattened_xs), np.concatenate(atom_types)


def flat_to_dataset(mol_idxs_coords, mol_idxs_types, flattened_xs, atom_types):
    # print("flattened_xs", flattened_xs[:64])
    # print("mol_idxs_coords", mol_idxs_coords[:64])

    # print("LAST", mol_idxs_coords[-1])
    # print(type(mol_idxs_coords[0]))
    # print(flattened_xs[mol_idxs_coords==0])
    print(len(mol_idxs_coords), len(atom_types))
    split_xs = [flattened_xs[mol_idxs_coords==i] for i in range(mol_idxs_coords[-1]+1)]
    split_types = [atom_types[mol_idxs_types==i] for i in range(mol_idxs_types[-1]+1)]
    data = []
    for x, atypes in zip(split_xs, split_types):
        coords = x.reshape((-1, 3))
        types = atypes.reshape((-1, 1))
        acoords = np.hstack([types, coords])
        data.append(acoords)
    return data



def model_E_and_grad(xyz, nn_atom_types, model):
    # xyz and elements must be merged into [element, x, y, z]
    # model must be a Trainer object
    xyz = [[i]+list(xx) for i, xx in zip(nn_atom_types, xyz)]
    #print('in model_E_and_grad, xyz =', xyz)
    rd = RawDataset([xyz], [0.0])
    energy = float(model.predict(rd)[0])
    self_interaction = np.sum(data_utils.selfIxnNrgWB97X_631gdp[t] for t in nn_atom_types)
    energy += self_interaction
    gradient = list(model.coordinate_gradients(rd))[0]
    gradient = gradient.reshape(gradient.size)
    gradient *= ANGSTROM_IN_BOHR 
    return energy, gradient



def main():

    parser = argparse.ArgumentParser(description="Run ANI1 neural net training.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ani_lib', required=True, help="Location of the shared object for GPU featurization")
    parser.add_argument('--fitted', default=False, action='store_true', help="Whether or use fitted or self-ixn")
    parser.add_argument('--add_ffdata', default=False, action='store_true', help="Whether or not to add the forcefield data")
    parser.add_argument('--gpus', default=1, help="Number of gpus we use")

    parser.add_argument('--work-dir', default='~/work', help="location where work data is dumped")
    parser.add_argument('--train-dir', default='~/ANI-1_release', help="location where work data is dumped")

    args = parser.parse_args()

    print("Arguments", args)

    lib_path = os.path.abspath(args.ani_lib)
    print("Loading custom kernel from", lib_path)
    initialize_module(lib_path)

    ANI_TRAIN_DIR = args.train_dir
    ANI_WORK_DIR = args.work_dir

    save_dir = os.path.join(ANI_WORK_DIR, "save")

    use_fitted = args.fitted
    add_ffdata = args.add_ffdata

    data_loader = DataLoader(False)

    xyz = [ [3.0,  0.0, 0.0, 0.0], [0.0,  1.0, 0.0, 0.0], [0.0,  0.0, 1.0, 0.0] ]
    dataset = RawDataset([xyz], 0.0)

    batch_size = 256

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as sess:

        M = 3  # number of models
        N = dataset.num_mols()

        trainers = []
        min_Es = np.zeros((M, N))

        def min_E_func(x_flat, elements, model):
            # basic energy function to optimize, for finding min_E
            xyz = np.reshape(x_flat, (-1, 3))
            #print('in opt_E_func, xyz =', xyz, 'elements =', elements)
            E, grad = model_E_and_grad(xyz, elements, model)
            return E, grad

        for m in range(M):
            with tf.variable_scope("model_"+str(m)):
                trainer = TrainerMultiTower(
                    sess,
                    towers=["/cpu:"+str(m)],
                    layer_sizes=NN_LAYERS,
                    precision=tf.float64
                )
            trainers.append(trainer)

        expected_gain, dx, dy, dz, ms = expected_information_gain(trainers, min_Es)

        sess.run(tf.global_variables_initializer())

        model_filenames = ['/nfs/working/scidev/stevenso/learning/khan_internal/train/sep28_%d/save/best.npz' % (i+1) for i in range(M)]
        for m, trainer in enumerate(trainers):
            with tf.variable_scope("model_"+str(m)):
                trainer.load_numpy(model_filenames[m], strict=False)
            for n in range(N):
                xs = dataset.all_Xs[n]
                elements = [row[0] for row in xs]
                x0 = [row[1:] for row in xyz]
                x0 = np.reshape(x0, len(x0) * 3 )
                result = scipy.optimize.fmin_l_bfgs_b(
                     min_E_func,
                     x0,
                     args=(elements, trainer),
                     maxiter=1000, iprint=0, factr=1e1, approx_grad=False, epsilon=1e-6, pgtol=1e-6)
                min_E = result.fun
                print('min_E =', min_E)
                min_Es[m][n] = min_E

        def min_func(x0s, atom_types, mol_idxs_types, mol_idxs_coords):
            raw_data = flat_to_dataset(mol_idxs_types, mol_idxs_coords, x0s, atom_types)
            dataset.all_Xs = raw_data
        
            executor = ThreadPoolExecutor(1)

            future = executor.submit(
                submit_dataset,
                session=sess,
                trainers=trainers,
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                fuzz=None)

            n_batches = dataset.num_batches(batch_size)

            print("num_batches..", n_batches)

            dxs = []

            total_gain = 0

            for bid in range(n_batches):
                gain, nx, ny, nz, nm = sess.run([expected_gain, dx, dy, dz, ms])
                total_gain += gain
                indices = nm[0]

                # average the gradients across all models
                sx = np.sum(nx, axis=0)
                sy = np.sum(ny, axis=0)
                sz = np.sum(nz, axis=0)
                dxdydz = np.squeeze(np.stack([sx, sy, sz], axis=-1))
                split_dxdydz = [dxdydz[indices==i] for i in range(indices[-1]+1)]
                dxs.extend(split_dxdydz)

            future.result()

            grads = np.concatenate([x.reshape(-1) for x in dxs])

            print("Grads shape", grads.shape)

            print("DTYPE", type(total_gain), grads.dtype)

            return -total_gain, -grads  # negative because we're maximizing

        a_idxs, m_idxs, data, atypes = dataset_to_flat(dataset.all_Xs)

        print("starting minimization...")
        result = scipy.optimize.fmin_l_bfgs_b(
            min_func,
            data,
            args=(atypes, a_idxs, m_idxs),
            maxiter=50, iprint=1, factr=1e1, approx_grad=False, epsilon=1e-6, pgtol=1e-6)

        # scipy.optimize.minimize(
        #     fun=min_func,
        #     method='L-BFGS-B',
        #     x0=data,
        #     args=(atypes, a_idxs, m_idxs),
        #     jac=True
        # )

if __name__ == "__main__":
    main()

'''
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/gridengine/lib/lx-amd64:/opt/openmpi/lib:/nfs/utils/stow/Python-3.5.3/lib/:/nfs/utils/stow/cuda-9.0/lib64/:/home/yzhao/libs/cuda/lib64
source /home/yzhao/venv/bin/activate
python -u ../gdb8_information_gain.py --work-dir info_gain_0 --ani_lib ../gpu_featurizer/ani.so
'''




