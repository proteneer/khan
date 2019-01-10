import numpy as np
import os

import time
import data_utils
from khan.utils.constants import ANGSTROM_IN_BOHR, KCAL_MOL_IN_HARTREE

def load_calibration_file(calibration_file):
    with open(calibration_file, 'r') as fh:

        mapping = {}

        for line in fh.readlines():
            parts = line.split()
            path = parts[0].split(".")[0]
            energy = parts[-1]
            mapping[path] = float(energy)

        return mapping

mapping = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3
}

selfIxnNrgFitted = np.array([
    -374.85 / KCAL_MOL_IN_HARTREE, 
    -23898.1 / KCAL_MOL_IN_HARTREE, 
    -34337.6 / KCAL_MOL_IN_HARTREE, 
    -47188.0 / KCAL_MOL_IN_HARTREE
], dtype=np.float32)


selfIxnNrgWB97X = np.array([
    -0.499321232710,
    -37.8338334397,
    -54.5732824628,
    -75.0424519384], dtype=np.float32)

def extract_file(fname):

    with open(fname, "r") as f:
        force_lines = []
        geom_lines = []
        energy = None
        append_buffer = None
        energy = None

        for l in f.readlines():
            if append_buffer is not None:
                append_buffer.append(l)
            if "Total energy:" in l:
                energy = np.float64(l.split()[2])
            if "start of program der1b" in l:
                append_buffer = force_lines
            if "end of program der1b" in l:
                append_buffer = None
            if " Input geometry:" in l:
                append_buffer = geom_lines
            if " principal moments of inertia:" in l:
                append_buffer = None

        np_forces = []
        for f in force_lines[6:-6]:
            tmp = f.split()
            np_forces.append([np.float64(tmp[2]), np.float64(tmp[3]), np.float64(tmp[4])])

        np_geoms = []
        for g in geom_lines[2:-2]:
            tmp = g.split()
            np_geoms.append([
                            np.float64(mapping[tmp[0][0]]),
                            np.float64(tmp[1])*ANGSTROM_IN_BOHR,
                            np.float64(tmp[2])*ANGSTROM_IN_BOHR,
                            np.float64(tmp[3])*ANGSTROM_IN_BOHR
                            ])

        np_forces = np.array(np_forces) / ANGSTROM_IN_BOHR
        np_geoms = np.array(np_geoms)

        atom_types = np_geoms[:, 0]


        offset = 0
        for a in atom_types:
            offset += selfIxnNrgFitted[int(a)]

        energy -= offset

        assert np_forces.shape[0] == np_geoms.shape[0]

        return np_geoms, energy, np_forces

class DataLoader():

    def __init__(self, use_fitted, batch_size=1024):
        """
        Helper class used to load various datasets.
        """
        self.batch_size = batch_size
        self.use_fitted = use_fitted

        if self.use_fitted:
            raise Exception("using fitted parameters is not supported right now")

    def load_gdb8(self,
        data_dir,
        calibration_file=None,
        ff_train_dirs=None,
        gdb_min_n=1,
        gdb_max_n=8):
        """
        Load gdb datafiles
        we read files on the fully closed range [gdb_min_n, gdb_max_n]
        
        Params 
        ------
        data_dir: str
            location of data files
        calibration_file: str
            name of calibration file
        ff_train_dirs: list of str
            list of names of ff data files
        gdb_min_n: int
            minimum gdb file to add
        gdb_max_n: int
            maximum gdb file to add (inclusive)
                     
        """

        assert gdb_min_n >= 1
        assert gdb_max_n <= 8
        assert gdb_max_n >= gdb_min_n

        gdb_files = []
        for n in range(gdb_min_n, gdb_max_n + 1):
            gdb_files.append(os.path.join(data_dir, "ani_gdb_s0" + str(n) + ".h5"))

        if calibration_file:
            cal_map = load_calibration_file(calibration_file)
        else:
            cal_map = None

        Xs, ys = data_utils.load_hdf5_files(
            gdb_files,
            calibration_map=cal_map,
            use_fitted=self.use_fitted)

        if ff_train_dirs:
            for train_dir in ff_train_dirs:
                ff_train_Xs, ff_train_ys, _ = data_utils.load_ff_files(train_dir, use_fitted=self.use_fitted)
                Xs.extend(ff_train_Xs)
                ys.extend(ff_train_ys)

        return Xs, ys

    def load_gdb3_forces(self, data_dir):
        Xs = []
        ys = []
        Fs = []
        for f in os.listdir(data_dir):
            if os.path.splitext(f)[-1] == '.out':
                abspath = os.path.join(data_dir, f)
                x, e, f = extract_file(abspath)

                wb97offset = 0
                Z = x[:, 0].astype(np.int32)
                for z in Z:
                    wb97offset += selfIxnNrgWB97X[z]

                Xs.append(x)
                ys.append(e)
                Fs.append(f)

        return Xs, ys, Fs

    def load_gdb8_forces(self,
        data_dir,
        calibration_file=None,
        ff_train_dirs=None):

        gdb_files = [
            os.path.join(data_dir, "ani_gdb_s01.h5"),
            os.path.join(data_dir, "ani_gdb_s02.h5"),
            os.path.join(data_dir, "ani_gdb_s03.h5"),
            os.path.join(data_dir, "ani_gdb_s04.h5"),
            os.path.join(data_dir, "ani_gdb_s05.h5"),
            # os.path.join(data_dir, "ani_gdb_s06.h5"),
            #os.path.join(data_dir, "ani_gdb_s07.h5"),
            #os.path.join(data_dir, "ani_gdb_s08.h5"),
        ]

        if calibration_file:
            cal_map = load_calibration_file(calibration_file)
        else:
            cal_map = None

        Xs, ys, fs = data_utils.load_hdf5_minima_gradients(
            gdb_files,
            calibration_map=cal_map,
            use_fitted=self.use_fitted)

        return Xs, ys, fs


    def load_gdb11(self,
        data_dir,
        calibration_file=None,
        use_fitted=False):

        if calibration_file:
            cal_map = load_calibration_file(calibration_file)
        else:
            cal_map = None

        X_gdb11, y_gdb11 = data_utils.load_hdf5_files([
            os.path.join(data_dir, "ani1_gdb10_ts.h5")
        ],
        calibration_map=cal_map,
        use_fitted=self.use_fitted)

        return X_gdb11, y_gdb11

    def load_ff(self, data_dir):
        ff_test_Xs, ff_test_ys, ff_groups = data_utils.load_ff_files(data_dir, use_fitted=self.use_fitted)

        return ff_test_Xs, ff_test_ys, ff_groups

    def load_comp6(self, data_dir, max_atom_limit=data_utils.MAX_ATOM_LIMIT):
        """
        Load the COMP6 dataset
        Inputs
        --------
        data_dir: path name to directory
        Returns
        --------
        dictionary with filename: (Xs, ys) correspondence
        """

        comp6 = {}
        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                if fname.endswith(".h5"):
                    file_path = os.path.join(root, fname)
                    data_name = fname[:-3]
                    Xs, ys = data_utils.load_hdf5_files(
                        [file_path], max_atom_limit=max_atom_limit)
                    comp6[data_name] = (Xs, ys)

        return comp6

