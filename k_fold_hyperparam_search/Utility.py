import os

import numpy as np
import pandas as pd
from scipy import sparse as sps

from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample


class Utility():

    def give_me_randomized_k_folds_with_val_percentage(self,k,validation_percentage):

        dataReader = DataReader()

        URM = dataReader.load_augmented_binary_urm()
        URM_aug,ICM = dataReader.pad_with_zeros_ICMandURM(URM)
        UCM = dataReader.load_aug_ucm()

        URM_aug_trains = []
        URM_pow_trains = []
        ICM_train = ICM
        UCM_train = UCM
        URM_tests = []

        for _ in range(k):
            URM_train_aug, URM_validation = split_train_in_two_percentage_global_sample(URM_aug, train_percentage = 1-validation_percentage)
            URM_train_pow = dataReader.stackMatrixes(URM_train_aug)

            URM_aug_trains.append(URM_train_aug)
            URM_pow_trains.append(URM_train_pow)
            URM_tests.append(URM_validation)

        return URM_aug_trains,URM_pow_trains,ICM_train,UCM_train,URM_tests




def precision(self,recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(self,recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    return recall_score


def MAP(self,recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return map_score


def evaluate_algorithm(self,URM_test, recommender_object):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    for user_id in range(URM_test.shape[0]):

        relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(user_id)
            num_eval += 1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    return cumulative_precision, cumulative_recall, cumulative_MAP

