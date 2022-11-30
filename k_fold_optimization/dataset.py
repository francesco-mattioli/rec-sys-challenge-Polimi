import os

import numpy as np
import pandas as pd
from scipy import sparse as sps

from Data_Handler.DataReader import DataReader


def load_dataset():
    def _load_dataset():
        dataReader = DataReader()
        ICM = dataReader.load_icm_df()
        targets = dataReader.load_target_df()
        URM = dataReader.load_powerful_binary_urm_df()
        return URM, ICM, targets

    URM, ICM, targets = _load_dataset()

    URM_coo = sps.coo_matrix((URM["Data"].values, (URM["UserID"].values, URM["ItemID"].values)))
    ICM_coo = sps.coo_matrix((ICM["data"].values, (ICM["item_id"].values, ICM["feature_id"].values)))
    targets = targets["user_id"].values

    return URM_coo, ICM_coo, targets


def split_train_test(URM_coo, percentage_validation):
    n_interactions = URM_coo.nnz

    train_mask = np.random.choice([True, False], n_interactions, p=[1 - percentage_validation, percentage_validation])
    URM_train_coo = sps.csr_matrix((URM_coo.data[train_mask], (URM_coo.row[train_mask], URM_coo.col[train_mask])),
                                   shape=URM_coo.shape)

    test_mask = np.logical_not(train_mask)
    URM_test_coo = sps.csr_matrix((URM_coo.data[test_mask], (URM_coo.row[test_mask], URM_coo.col[test_mask])),
                                  shape=URM_coo.shape)

    return URM_train_coo, URM_test_coo


def give_me_splitted_dataset(val_percentage):
    URM_coo, ICM_coo, targets = load_dataset()
    URM_train_csr, URM_test_csr = split_train_test(URM_coo, val_percentage)

    ICM_csr = ICM_coo.tocsr()

    return URM_train_csr, URM_test_csr, ICM_csr, targets


def give_me_k_folds(k):
    URM_coo, ICM_coo, targets = load_dataset()
    ICM_csr = ICM_coo.tocsr()

    n_interactions = URM_coo.nnz
    n_interactions_per_fold = int(URM_coo.nnz / k) + 1
    temp = np.arange(k).repeat(n_interactions_per_fold)
    np.random.seed(0)
    np.random.shuffle(temp)
    assignment_to_fold = temp[:n_interactions]

    URM_trains = []
    ICM_trains = []
    URM_tests = []
    for i in range(k):
        train_mask = assignment_to_fold != i
        test_mask = assignment_to_fold == i
        URM_train_csr = sps.csr_matrix((URM_coo.data[train_mask],
                                        (URM_coo.row[train_mask], URM_coo.col[train_mask])),
                                       shape=URM_coo.shape)
        URM_test_csr = sps.csr_matrix((URM_coo.data[test_mask],
                                       (URM_coo.row[test_mask], URM_coo.col[test_mask])),
                                      shape=URM_coo.shape)

        URM_trains.append(URM_train_csr)
        ICM_trains.append(ICM_csr)
        URM_tests.append(URM_test_csr)

    return URM_trains, URM_tests, ICM_trains


def give_me_randomized_k_folds_with_val_percentage(k, validation_percentage):
    URM_trains = []
    ICM_trains = []
    URM_tests = []

    for _ in range(k):
        URM_train_csr, URM_test_csr, ICM_csr, targets = give_me_splitted_dataset(validation_percentage)

        URM_trains.append(URM_train_csr)
        ICM_trains.append(ICM_csr)
        URM_tests.append(URM_test_csr)

    return URM_trains, URM_tests, ICM_trains
