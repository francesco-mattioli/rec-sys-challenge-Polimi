#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender_utils import check_matrix
from Recommenders.Custom.CustomBaseSimilarityMatrixRecommender import CustomBaseUserSimilarityMatrixRecommender
from Data_Handler.DataReader import DataReader

from Recommenders.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np

from Recommenders.Similarity.Compute_Similarity import Compute_Similarity


class CustomUserKNNCFRecommender(CustomBaseUserSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "CustomUserKNNCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]


    def __init__(self, URM_train, verbose = True):
        super(CustomUserKNNCFRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, icm_weight_in_impressions=0.8, urm_weight=0.825,topK=1214, shrink=938.0611833211633, similarity='cosine', normalize=True, feature_weighting = "TF-IDF", URM_bias = False, **similarity_args):

        ########## START OF MODIFIED CODE @engpap #################
        URM_train_super_pow =  DataReader().load_URM_super_pow_and_ICM_stacked_with_weighted_impressions(self.URM_train, icm_weight_in_impressions, urm_weight)
        super(CustomUserKNNCFRecommender, self).post_init(URM_train_super_pow, verbose = True)
        ########## END OF MODIFIED CODE @engpap #################

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if URM_bias is not None:
            self.URM_train.data += URM_bias

        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')