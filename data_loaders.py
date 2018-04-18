import numpy as np
import os

import sklearn.model_selection

import time
from khan.data.dataset import RawDataset
import data_utils

def load_calibration_file(calibration_file):
    with open(calibration_file, 'r') as fh:

        mapping = {}

        for line in fh.readlines():
            parts = line.split()
            path = parts[0].split(".")[0]
            energy = parts[-1]
            mapping[path] = float(energy)

        return mapping

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
        ff_train_dir=None):

        gdb_files = [
            os.path.join(data_dir, "ani_gdb_s01.h5"),
            os.path.join(data_dir, "ani_gdb_s02.h5"),
            os.path.join(data_dir, "ani_gdb_s03.h5"),
            # os.path.join(data_dir, "ani_gdb_s04.h5"),
            # os.path.join(data_dir, "ani_gdb_s05.h5"),
            # os.path.join(data_dir, "ani_gdb_s06.h5"),
            # os.path.join(data_dir, "ani_gdb_s07.h5"),
            # os.path.join(data_dir, "ani_gdb_s08.h5"),
        ]

        if calibration_file:
            cal_map = load_calibration_file(calibration_file)
        else:
            cal_map = None

        Xs, ys = data_utils.load_hdf5_files(
            gdb_files,
            calibration_map=cal_map,
            use_fitted=self.use_fitted)

        if ff_train_dir is not None:
            ff_train_Xs, ff_train_ys, _ = data_utils.load_ff_files(ff_train_dir, use_fitted=self.use_fitted)
            Xs.extend(ff_train_Xs)
            ys.extend(ff_train_ys)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Xs, ys, test_size=0.25, random_state=0)

        return RawDataset(X_train, y_train), RawDataset(X_test,  y_test)

    def load_gdb11(self,
        data_dir,
        calibration_file=None,
        use_fitted=False):

        if calibration_file:
            cal_map = load_calibration_file(calibration_file)
        else:
            cal_map = None

        X_gdb11, y_gdb11 = data_utils.load_hdf5_files([
            os.path.join(data_dir, "ani1_gdb10_ts.h5"),
        ],
        calibration_map=cal_map,
        use_fitted=self.use_fitted)

        return RawDataset(X_gdb11, y_gdb11)

    def load_ff(self, data_dir):
        ff_test_Xs, ff_test_ys, ff_groups = data_utils.load_ff_files(data_dir, use_fitted=self.use_fitted)
        return RawDataset(ff_test_Xs, ff_test_ys), ff_groups
