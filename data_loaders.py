import numpy as np
import os

import sklearn

from khan.data.dataset import RawDataset, FeaturizedDataset
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

    def __init__(self,
        use_fitted,
        batch_size=1024,
        prod=False):

        self.batch_size = batch_size
        self.use_fitted = use_fitted
        self.prod = prod

    def load_gdb8(self,
        feat_dir,
        data_dir,
        calibration_file,
        ff_train_dir=None):

        feat_dir_train = os.path.join(feat_dir, "train")
        feat_dir_test = os.path.join(feat_dir, "test")

        if os.path.exists(feat_dir):
            fd_train = FeaturizedDataset(feat_dir_train)
            fd_test = FeaturizedDataset(feat_dir_test)
            return fd_train, fd_test

        if self.prod:
            gdb_files = [
                os.path.join(data_dir, "ani_gdb_s01.h5"),
                os.path.join(data_dir, "ani_gdb_s02.h5"),
                os.path.join(data_dir, "ani_gdb_s03.h5"),
                os.path.join(data_dir, "ani_gdb_s04.h5"),
                os.path.join(data_dir, "ani_gdb_s05.h5"),
                os.path.join(data_dir, "ani_gdb_s06.h5"),
                os.path.join(data_dir, "ani_gdb_s07.h5"),
                os.path.join(data_dir, "ani_gdb_s08.h5"),
            ]
        else:
            gdb_files = [os.path.join(data_dir, "ani_gdb_s01.h5")]

            Xs, ys = data_utils.load_hdf5_files(
                gdb_files,
                calibration_map=load_calibration_file(calibration_file),
                use_fitted=self.use_fitted)

        if ff_train_dir is not None:
            ff_train_Xs, ff_train_ys = data_utils.load_ff_files(ff_train_dir, use_fitted=self.use_fitted)
            Xs.extend(ff_train_Xs) # add training data here
            ys.extend(ff_train_ys)

        Xs, ys = sklearn.utils.shuffle(Xs, ys)    

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Xs, ys, test_size=0.25)

        rd_train = RawDataset(X_train, y_train)
        rd_test  = RawDataset(X_test,  y_test)
        print("--------------train length", len(X_train))
        fd_train = rd_train.featurize(self.batch_size, feat_dir_train)
        print("--------------test length", len(X_test))
        fd_test = rd_test.featurize(self.batch_size, feat_dir_test)

        return fd_train, fd_test

    def load_gdb11(self,
        feat_dir,
        data_dir,
        calibration_file,
        use_fitted=False):

        if os.path.exists(feat_dir):
            return FeaturizedDataset(feat_dir)

        X_gdb11, y_gdb11 = data_utils.load_hdf5_files([
            os.path.join(data_dir, "ani1_gdb10_ts.h5"),
        ],
        calibration_map=load_calibration_file(calibration_file),
        use_fitted=self.use_fitted)

        rd_gdb11  = RawDataset(X_gdb11, y_gdb11)
        return rd_gdb11.featurize(self.batch_size, feat_dir)

    def load_ff(self,
        feat_dir,
        data_dir):

        ff_groups = data_utils.load_ff_files_groups(data_dir, use_fitted=self.use_fitted)

        if os.path.exists(feat_dir):
            return FeaturizedDataset(feat_dir), ff_groups

        ff_test_Xs, ff_test_ys = data_utils.load_ff_files(data_dir, use_fitted=self.use_fitted)
        ff_db  = RawDataset(ff_test_Xs, ff_test_ys)
        return ff_db.featurize(self.batch_size, feat_dir), ff_groups

    # def load_ff_charged(feat_dir, data_dir):

    # def load_ff_ccsdt(feat_dir, data_dir):