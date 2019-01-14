import os
import numpy as np
import tensorflow as tf
import json
import argparse

from khan.model import activations
from khan.data.dataset import RawDataset
from khan.training.trainer_multi_tower import initialize_module
from khan.training.committee import Committee
from khan.utils.constants import KCAL_MOL_IN_HARTREE

from khan.lad import lad
from khan.lad import lad_coulomb

VARIANCE = 'committee variance'
RANDOM = 'random selection'
BOLTZMANN = 'boltzmann'
VARIANCE_WEIGHTED_BOLTZMANN = 'variance weighted boltzmann'
SELECTION_METHODS = [VARIANCE, RANDOM, BOLTZMANN, VARIANCE_WEIGHTED_BOLTZMANN]

kT = 0.001 # in Hartree
NN_LAYERS = (256, 256, 64, 1)

def parse_args():
    """
    Parse commandline arguments from sys.argv
    returns a namespace with arguments
    """

    parser = argparse.ArgumentParser(
        description="Optimize expected information gain for all molecules in input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'infile',
        default=None,
        type=str,
        help="Input json file holding geometries to optimize"
    )

    parser.add_argument(
        '--ani-lib',
        required=True,
        help="Location of the shared object for GPU featurization"
    )

    parser.add_argument(
        '--fitted',
        default=False,
        action='store_true',
        help="Whether or use fitted or self-ixn")

    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help="Number of gpus we use"
    )

    parser.add_argument('--saved-network',
        default=None,
        type=str,
        help="scratch directory holding saved networks to build committee")

    parser.add_argument(
        '--deep-network',
        action='store_true',
        help='Use James super deep network (256, 256, 256, 256, 256, 256, 256, 128, 64, 8, 1)'
    )

    parser.add_argument(
        '--activation-function',
        type=str,
        choices=activations.get_all_fn_names(),
        help='choice of activation function',
        default="celu"
    )

    parser.add_argument(
        '--numb-samples',
        type=int,
        default=0,
        help='If set to greater than 0 split data set into a number of samples and the rest'
    )

    parser.add_argument(
        '--sampling-method',
        type=str,
        default=VARIANCE,
        choices=SELECTION_METHODS,
        help='method used to choose samples'
    )
    parser.add_argument(
        '--lad-data',
        default=None,
        help='Location of reference LAD data (json).  If given long range coulomb energy based on LAD charges will be added to NN predictions') 

    parser.add_argument(
        '--jobname',
        default="analyze",
        type=str,
        help="name prepended to output files"
    )


    return parser.parse_args()

def read_molecules(fname, energy_cutoff=100.0/KCAL_MOL_IN_HARTREE):
    """
    read molecules from a json file with field "X"
    which holds geometry specification in khan format
    returns a list of numpy arrays 
    """

    assert fname.endswith(".json")

    with open(fname) as fin:
        data = json.load(fin)

    X = data.get("X", [])
    ys = data.get("Y", [])

    ndata = len(X)

#    for i, mol in enumerate(X):
#        atom_energies = [data_utils.selfIxnNrgWB97X[at] for at, x, y, z in mol]
#        self_interaction = sum(atom_energies)
#        ys[i] -= self_interaction

    ymin = min(ys)
    yfiltered = []
    Xfiltered = []
    for xi, yi in zip(X, ys):
        if yi - ymin < energy_cutoff:
            Xfiltered.append(np.array(xi))
            yfiltered.append(yi)

    print("Read %d of %d structures under energy cutoff" % (len(yfiltered), len(ys)))

    return Xfiltered, yfiltered 

def write_molecules(fname, X, y):
    """
    write X,y data to json
    """

    # make the data lists, not numpy arrays
    y_write = list(y)

    X_write = []
    for mol in X:
        molecule = []
        for at in mol:
            molecule.append(
                [int(at[0]), float(at[1]), float(at[2]), float(at[3])]
            )
        X_write.append(molecule)
        
    data = {"X": X_write, "Y": y_write}
    with open(fname, "w") as fout:
        fout.write(json.dumps(data))
        
def main():

    args = parse_args()
    print("Arguments", args)

    lib_path = os.path.abspath(args.ani_lib)
    print("Loading custom kernel from", lib_path)
    initialize_module(lib_path)

    # read molecules
    X, y = read_molecules(args.infile)
    rd = RawDataset(X)

    # setup lad  
    coulomb_correction = np.zeros(len(y))
    if args.lad_data:
        lad_params = dict(lad.LAD_PARAMS)
        reference_lads = lad.read_reference_lads(
            args.lad_data, lad_params)
        lad_coulomb.remove_coulomb(
            X, coulomb_correction, lad_params, reference_lads)
        coulomb_correction = -coulomb_correction


    config = tf.ConfigProto(allow_soft_placement=True)
    activation_fn = activations.get_fn_by_name(args.activation_function)

    with tf.Session(config=config) as sess:

        n_models = len(args.saved_network)  # number of models
        if n_models < 1:
            raise ValueError("Must have at least one saved network to build a committee")

        layers = NN_LAYERS 
        if args.deep_network:
            layers = (256, 256, 256, 256, 256, 256, 256, 128, 64, 8, 1)

        sess.run(tf.global_variables_initializer())
        print("build committee")
        committee = Committee(
            args.saved_network,
            None, # nmembers (read them)
            sess,
            tf.float64,
            True, # add atomization
            None, # EWC
            activation_fn=activation_fn,
            layer_sizes=layers
        )

        committee_data = committee.compute_data(rd)
        print("predicting energy and variance of datapoints ...")
        prediction_file = open(args.jobname + "_predictions.dat", "w")
        statistics_file = open(args.jobname + "_statistics.dat", "w")
        for i, data in enumerate(committee_data):
            model_energies = ["%.6f" % (e + coulomb_correction[i]) for e in data.predictions]
            prediction_file.write(("%d %.4f " + " ".join(model_energies) + "\n") % (i, y[i]))
            statistics_file.write("%d %.6f %.6f %.6f\n" % (i, y[i], data.energy + coulomb_correction[i], data.variance))

        # select data points
        if args.numb_samples > 0:
            # make list of tuples with all the data
            all_data = []
            ndata = len(committee_data)
            # could be slow...
            variances = [data.variance for data in committee_data]
            idx = np.argsort(variances)

            Xsamples = []
            ysamples = []
            Xrest = []
            yrest = []
            for cnt, i in enumerate(idx):
                # last nsamples go in the selected dataset
                if cnt < ndata - args.numb_samples:
                    print("(low var) sorted variances", variances[i])
                    Xrest.append(X[i])
                    yrest.append(y[i])
                else:
                    print("(high var) sorted variances", variances[i])
                    Xsamples.append(X[i])
                    ysamples.append(y[i])

            write_molecules(args.jobname + "_variance_samples.json", Xsamples, ysamples)
            write_molecules(args.jobname + "_variance_remainder.json",  Xrest, yrest)
        
        print("done")

if __name__ == "__main__":
    main()

