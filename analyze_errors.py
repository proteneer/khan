from khan.training.trainer_multi_tower import TrainerMultiTower, initialize_module
from khan.data.dataset import RawDataset
from khan.model import activations
from khan.utils.constants import KCAL_MOL_IN_HARTREE
from data_utils import load_reactivity_data, read_all_reactions 
import tensorflow as tf
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json


def histogram(errors, label):
    mu = errors.mean()
    sigma = errors.std()
    n, bins, patches =  plt.hist(
        errors,
        bins=40,
        normed=True,
        label=label,
        histtype='step'
    )
    return mu, sigma

def fdiff_grad(X, trainer, dx=0.001):
    """
    finite diff gradient
    """
    natoms = len(X)
    g = np.zeros((natoms, 3))
    X0 = X.copy()
    X1 = X0.copy()

    Xplus, Xminus = ([], [])
    y = []
    for i in range(natoms):
        for j in range(3):
            X1[:, :] = X0[:, :]
            X1[i, j + 1] += dx
            Xplus.append(X1.copy())
            X1[:, :] = X0[:, :]
            X1[i, j + 1] -= dx
            Xminus.append(X1.copy())
            y.append(0.0)

    plus = RawDataset(Xplus, y)
    minus = RawDataset(Xminus, y)

    eplus = np.array(trainer.predict(plus))
    eminus = np.array(trainer.predict(minus))

    g = (eplus - eminus) / (2.0*dx)

    return g
    

def write_reaction_data(name, X, E, trainer):
    """
    Write some results about this reaction to a file
    """

    rd = RawDataset(X, E)
    grad = list(trainer.coordinate_gradients(rd))
    predictions = trainer.predict(rd)

    # print a table of 
    # dft energy, ani1 energy, gradient norm, g.dx

    ngeoms = len(X)

    out_str = ["# dft energy, ani1 energy, grms, g.dx"] 
    dist = 0
    for i in range(ngeoms):
        carts = X[i][:, 1:]
        natoms, ncarts = grad[i].shape
        gxyz = grad[i].reshape(natoms*ncarts)

#       fdiff gradient tests
#        gfd = fdiff_grad(X[i], trainer)
#        gerr = gfd - gxyz
#        grms = np.sqrt(sum(gerr[:]**2.0)/(natoms*ncarts))
#        gdot = np.dot(gfd, gxyz) / (np.sqrt(np.dot(gxyz, gxyz)) * np.sqrt(np.dot(gfd, gfd)))
#        print("rms gradient error %.8f" % grms)
#        print("gradient dot prod %.4f" % gdot) 

        if i == 0:
            dX = X[i+1][:, 1:] - X[i][:, 1:]
        elif i == ngeoms - 1:
            dX = X[i][:, 1:] - X[i-1][:, 1:]
        else:
            dX = X[i+1][:, 1:] - X[i-1][:, 1:]
        dX = dX.reshape(natoms*ncarts)

        gdotx = np.dot(gxyz, dX)
        gdotg = np.dot(gxyz, gxyz)
        xdotx = np.dot(dX, dX)

        #print("%.4f %.4f %.4f" % (gdotx, gdotg, xdotx))

        grms = np.sqrt(gdotg / (natoms*ncarts))
        gdot = -gdotx / (np.sqrt(gdotg) * np.sqrt(xdotx))

        dE = (E[i] - E[0])*KCAL_MOL_IN_HARTREE
        dP = (predictions[i] - predictions[0])*KCAL_MOL_IN_HARTREE
        out_str.append("%.2f %.2f %.2f %.8f %.4f" % (dist, dE, dP, grms, gdot))

        if i < ngeoms - 1:
            dX = X[i+1][:, 1:] - X[i][:, 1:]
            dX = dX.reshape(natoms*ncarts)
            dist += np.sqrt(sum(dX[:]**2.0))
    

    with open(name + "_compare.dat", "w") as fin:
        fin.write("\n".join(out_str)) 
        

def main():

    args = parse_args(sys.argv)
    lib_path = os.path.abspath(args.ani_lib)
    initialize_module(lib_path)

    save_file = os.path.join(args.save_dir, "save_file.npz")
    if not os.path.exists(save_file):
        raise IOError("Saved NN numpy file does not exist")

    _, _, X_test, y_test, X_big, y_big = load_reactivity_data(args.reactivity_dir, 1.0)
    small_reactions, big_reactions = read_all_reactions(args.reactivity_dir)

    rd_test = RawDataset(X_test, y_test)
    rd_big = RawDataset(X_big, y_big)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        towers = ["/cpu:0"]
        layers = (128, 128, 64, 1)
        if args.deep_network:
            layers = (256, 256, 256, 256, 256, 256, 256, 128, 64, 8, 1)
        activation_fn = activations.get_fn_by_name(args.activation_function)

        trainer = TrainerMultiTower(
            sess,
            towers=towers,
            precision=tf.float64,
            layer_sizes=layers,
            activation_fn=activation_fn,
            fit_charges=args.fit_charges,
        )

        trainer.load_numpy(save_file)

        if args.analyze_reaction_errors:

            if not os.path.exists("small_reactions_comparison"):
                os.mkdir("small_reactions_comparison")
            if not os.path.exists("big_reactions_comparison"):
                os.mkdir("big_reactions_comparison")

            for dataname, data in (("small_reactions", small_reactions),
                ("big_reactions", big_reactions)):
            
                # get reactant, TS product
                Xr, Er = [], []
                Xts, Ets = [], []
                Xp, Ep = [], []

                for name in data:
                    Xs, Es = data[name]

                    if args.write_comparison_data:
                        # make a directory HERE
                        directory = dataname + "_comparison"
                        write_reaction_data(
                            os.path.join(directory, name),
                            Xs,
                            Es,
                            trainer
                        )

                    Xr.append(Xs[0])
                    Er.append(Es[0])
                    Xp.append(Xs[-1])
                    Ep.append(Es[-1])

                    # ts is highest energy point along path
                    emax = max(Es)
                    idx = Es.index(emax) 
                    Xts.append(Xs[idx])
                    Ets.append(Es[idx])

                # make datasets
                rd_r = RawDataset(Xr, Er)
                rd_p = RawDataset(Xp, Ep)
                rd_ts = RawDataset(Xts, Ets)

                Er = np.array(Er)
                Ep = np.array(Ep)
                Ets = np.array(Ets)

                # predict energies
                r_predictions = np.array(trainer.predict(rd_r))
                p_predictions = np.array(trainer.predict(rd_p))
                ts_predictions = np.array(trainer.predict(rd_ts))

                barriers = (Ets - Er)*KCAL_MOL_IN_HARTREE
                reverse_barriers = (Ets - Ep)*KCAL_MOL_IN_HARTREE
                predicted_barriers = (ts_predictions - r_predictions)*KCAL_MOL_IN_HARTREE
                predicted_reverse_barriers = (ts_predictions - p_predictions)*KCAL_MOL_IN_HARTREE
                rxn_e = (Ep - Er)*KCAL_MOL_IN_HARTREE
                predicted_rxn_e = (p_predictions - r_predictions)*KCAL_MOL_IN_HARTREE

                barrier_errors = barriers - predicted_barriers
                barrier_rmse = np.sqrt(sum(barrier_errors[:]**2.0)/len(barrier_errors))
                reverse_barrier_errors = reverse_barriers - predicted_reverse_barriers
                reverse_barrier_rmse = np.sqrt(sum(reverse_barrier_errors[:]**2.0)/len(reverse_barrier_errors))
                rxn_errors = rxn_e - predicted_rxn_e
                rxn_rmse = np.sqrt(sum(rxn_errors[:]**2.0)/len(rxn_errors))

                # barrier height plot
                bmu, bsigma = histogram(barrier_errors, "Barrier height errors")
                rbmu, rbsigma = histogram(
                    reverse_barrier_errors,
                    "Reverse Barrier height errors"
                )
                rmu, rsigma = histogram(rxn_errors, "Reaction energy errors")
                plt.xlabel("Error (kcal/mol)")
                plt.title("Reaction energetic errors for %s" % dataname)
                plt.legend()

                #plt.scatter(barriers, predicted_barriers)
                #plt.scatter(rxn_e, predicted_rxn_e)
                plt.savefig("%s_barrier_height_errors.pdf" % dataname)
                plt.clf()

                print("errors for %s" % dataname)
                print("Barrier RMSE %.2f rxn RMSE %.2f" % (barrier_rmse, rxn_rmse))
                print("Reverse Barrier RMSE %.2f" % reverse_barrier_rmse)
                print("rxn mu %f sigma %f" % (rmu, rsigma))
                print("barrier mu %f sigma %f" % (bmu, bsigma))
                print("reverse barrier mu %f sigma %f" % (rbmu, rbsigma))
                
        
        # plot distribution of raw errors
        if args.analyze_raw_errors:
            #evaluate errors in predictions
            rxn_predictions = trainer.predict(rd_test)
            big_predictions = trainer.predict(rd_big)
            rxn_errors = np.array(rxn_predictions) - np.array(y_test)
            big_errors = np.array(big_predictions) - np.array(y_big)
            rxn_rmse = np.sqrt(sum(rxn_errors[:]**2.0)/len(rxn_errors))
            big_rmse = np.sqrt(sum(big_errors[:]**2.0)/len(big_errors))
            rxn_errors = rxn_errors*KCAL_MOL_IN_HARTREE
            big_errors = big_errors*KCAL_MOL_IN_HARTREE

            print("small rmse %.4f big rmse %.4f" % (rxn_rmse*KCAL_MOL_IN_HARTREE, big_rmse*KCAL_MOL_IN_HARTREE))

            smu, ssigma = histogram(
                rxn_errors,
                "Atomization energy errors for small systems"
            )
            bmu, bsigma = histogram(
                big_errors,
                "Atomization energy errors for large systems"
            )
            plt.xlabel("Error (kcal/mol)")
            plt.title("Atomization energy errors")
            plt.legend()
            plt.savefig("atomization_errors.pdf")
            plt.clf()

            print("small atomization mu %f sigma %f" % (smu, ssigma))
            print("big atomization mu %f sigma %f" % (bmu, bsigma))

def parse_args(args):
    parser = argparse.ArgumentParser(description="test ANI1 NN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save-dir', default='~/work', help="Location where save data is dumped. If the folder does not exist then it will be created.")
    parser.add_argument("--reactivity-dir", default=None, help="location of reactivity data to test")
    parser.add_argument("--ani-lib", required=True, help="Location of shared object")
    parser.add_argument(
        "--analyze_raw_errors",
        action="store_true",
        default=False,
        help="analyze raw errors in test set"
    )
    parser.add_argument(
        "--analyze-reaction-errors",
        action="store_true",
        default=False,
        help="analyze error in barrier heights and reaction energies"
    )
    parser.add_argument(
        "--write-comparison-data",
        action="store_true",
        default=False,
        help="write data to visualize energy along path"
    )
    parser.add_argument(
        '--deep-network',
        action='store_true',
        help='Use James super deep network (256, 256, 256, 256, 256, 256, 256, 128, 64, 8, 1)'
    )
    parser.add_argument(
        '--fit-charges',
        action='store_true',
        help='fit charges'
    )
    parser.add_argument(
        '--activation-function',
        type=str,
        choices=activations.get_all_fn_names(),
        help='choice of activation function',
        default="celu"
    )
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
