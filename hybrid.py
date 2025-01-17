# Import recommenders
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.SLIM.SLIMElasticNetRecommender import *
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

# from Recommenders.FactorizationMachines.LightFMRecommender import LightFMItemHybridRecommender
from Data_Handler.DataReader import DataReader


# Import libraries
from tqdm import tqdm
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp
import pandas as pd


class HybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender"

    def __init__(self, URM_train):
        self.URM_train_aug = URM_train
        # self.ICM = ICM
        # self.URM_train_pow = self.stackMatrixes(dataReader, URM_train)
        # URM_train_aug_df=dataReader.csr_to_dataframe(URM_train)
        # self.URM_train_pow=dataReader.load_powerful_binary_urm_given_URM_train_df(URM_train_aug_df)

        super(HybridRecommender, self).__init__(URM_train)

    def fit(self):
        # Stack and normalize URM and ICM
        # URM_stacked = sps.vstack([self.URM_train, self.ICM.T])

        # Instantiate & fit the recommenders
        # self.ItemCF = ItemKNNCFRecommender(self.URM_train_pow)
        # self.ItemCF.fit(topK=1199, shrink=229.22107382005083,similarity='cosine', normalize=True, feature_weighting="TF-IDF")

        self.SLIM_ElasticNet = SLIMElasticNetRecommender(self.URM_train_aug)
        self.SLIM_ElasticNet.fit(l1_ratio=0.008213119901673099,alpha=0.0046000272149077145, positive_only=True, topK=498)

        self.RP3beta = RP3betaRecommender(self.URM_train_aug)
        self.RP3beta.fit(alpha=0.3648761546066018,
                         beta=0.5058870363874656, topK=480, normalize_similarity=True)

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # num_items = 24507
        num_items = 27968
        # num_items=19630
        # num_items = len(self.items) # num_items changes based on used urm
        item_weights = np.empty([len(user_id_array), num_items])

        for i in tqdm(range(len(user_id_array))):

            # w = self.ItemCF._compute_item_score(user_id_array[i], items_to_compute)
            w1 = self.RP3beta._compute_item_score(
                user_id_array[i], items_to_compute)
            w1 /= LA.norm(w1, 2)
            w2 = self.SLIM_ElasticNet._compute_item_score(user_id_array[i], items_to_compute)
            w2 /= LA.norm(w2, 2)
            
            w = w1 + w2
            item_weights[i, :] = w

        return item_weights


'''-----------------------------------------------------------------------------------------------------------------------------'''


class HybridRecommender_2(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender_2"

    N_CONFIG = 0

    def __init__(self, URM_train, ICM):
        """ Constructor of Hybrid_Recommender_2
        Args:
            URM_train (csr): augmented matrix
            ICM (csr): icm
        """
        self.URM_train_aug = URM_train
        self.ICM = ICM
        self.URM_train_pow = DataReader().stackMatrixes(URM_train)
        super(HybridRecommender_2, self).__init__(URM_train)

        self.normalize = None
        self.recommenders = {}
        self.weights = {}

    def fit(self, RP3betaRecommenderWeight=0.5, SLIMElasticNetRecommenderWeight=0.5, normalize=False):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.weights = {
            # ItemKNNCFRecommender: ItemKNNCFRecommenderWeight,
            SLIMElasticNetRecommender: SLIMElasticNetRecommenderWeight,
            RP3betaRecommender: RP3betaRecommenderWeight,
        }

        self.normalize = normalize

        for rec_class in self.weights.keys():
            if self.weights[rec_class] > 0.0:
                if rec_class not in self.recommenders:
                    # start = time.time()
                    if rec_class == SLIMElasticNetRecommender:
                        temp_rec = rec_class(self.URM_train_pow)
                    else:
                        temp_rec = rec_class(self.URM_train_aug)
                    temp_rec.fit()
                    self.recommenders[rec_class] = temp_rec
                    # end = time.time()
                    print("Fitted new instance of {}.".format(
                        rec_class.__name__))

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items_aug = 24507
        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        # w is predicted ratings array of a user
        for i in range(len(user_id_array)):
            for rec_class in self.recommenders.keys():
                if self.weights[rec_class] > 0.0:
                    w = self.recommenders[rec_class]._compute_item_score(
                        user_id_array[i], items_to_compute)
                    if self.normalize:
                        w *= 1.0 / w.max()
                    if rec_class != SLIMElasticNetRecommender:  # since we are using augmented matrix, items are different
                        w = np.pad(
                            w, ((0, 0), (0, num_items_pow-num_items_aug)))
                    w += np.multiply(w, self.weights[rec_class])
            item_weights[i, :] = w

        return item_weights


'''-----------------------------------------------------------------------------------------------------------------------------'''


class HybridRecommender_3(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender_3"

    def __init__(self, URM_train: sp.csr_matrix, ICM):
        self.URM_train_aug = URM_train
        self.ICM = ICM
        self.URM_train_pow = DataReader().stackMatrixes(URM_train)
        super(HybridRecommender_3, self).__init__(self.URM_train_pow)

    def fit(self):
        self.S_SLIM = SLIMElasticNetRecommender(self.URM_train_pow)
        self.RP3beta = RP3betaRecommender(self.URM_train_aug)
        self.ItemKNNCF = ItemKNNCFRecommender(self.URM_train_aug)
        self.UserKNN = UserKNNCFRecommender(self.URM_train_aug)
        self.S_SLIM.fit(l1_ratio=0.007467817120176792,
                        alpha=0.0016779515713674044, positive_only=True, topK=723)
        self.RP3beta.fit(alpha=0.2686781702308662,
                         beta=0.39113126168484014, topK=455, normalize_similarity=True)
        self.ItemKNNCF.fit(topK=1199, shrink=229.22107382005083,
                           similarity='cosine', normalize=True, feature_weighting="TF-IDF")
        self.UserKNN.fit(topK=1214, shrink=938.0611833211633, normalize=True,
                         feature_weighting='TF-IDF', similarity='cosine')

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items_aug = 24507
        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), 27968])

        for i in tqdm(range(len(user_id_array))):

            interactions_aug = len(
                self.URM_train_aug[user_id_array[i], :].indices)

            if interactions_aug < 15 or (interactions_aug >= 18 and interactions_aug <= 19):
                w = self.UserKNN._compute_item_score(
                    user_id_array[i], items_to_compute)
                w = np.pad(w, ((0, 0), (0, num_items_pow-num_items_aug)))
                item_weights[i, :] = w

            elif interactions_aug >= 15 and interactions_aug < 18:
                w = self.RP3beta._compute_item_score(
                    user_id_array[i], items_to_compute)
                w = np.pad(w, ((0, 0), (0, num_items_pow-num_items_aug)))
                item_weights[i, :] = w

            else:
                w = self.S_SLIM._compute_item_score(
                    user_id_array[i], items_to_compute)
                item_weights[i, :] = w

        return item_weights


class HybridRecommender_4(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender_4"

    def __init__(self, URM_train_aug, URM_train_pow, UserKNNCF, RP3beta_pow, S_SLIM):
        """ Constructor of Hybrid_Recommender_2
        Args:
            URM_train (csr): augmented matrix
            ICM (csr): icm
        """
        self.URM_train_aug = URM_train_aug
        self.URM_train_pow = URM_train_pow
        self.UserKNNCF = UserKNNCF
        self.RP3beta_pow = RP3beta_pow
        self.S_SLIM = S_SLIM
        super(HybridRecommender_4, self).__init__(self.URM_train_aug)

    def fit(self, UserKNNCF_tier1_weight=0.5, RP3beta_pow_tier1_weight=0.5, UserKNNCF_tier2_weight=0.5, RP3beta_pow_tier2_weight=0.5, RP3beta_pow_tier3_weight=0.5, S_SLIM_tier3_weight=0.5):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.UserKNNCF_tier1_weight = UserKNNCF_tier1_weight
        self.RP3beta_pow_tier1_weight = RP3beta_pow_tier1_weight

        self.UserKNNCF_tier2_weight = UserKNNCF_tier2_weight
        self.RP3beta_pow_tier2_weight = RP3beta_pow_tier2_weight

        self.RP3beta_pow_tier3_weight = RP3beta_pow_tier3_weight
        self.S_SLIM_tier3_weight = S_SLIM_tier3_weight

    '''
    self.UserKNNCF = UserKNNCFRecommender(self.URM_train_aug)
        self.UserKNNCF.fit()

        self.RP3beta_pow = RP3betaRecommender(self.URM_train_pow)
        self.RP3beta_pow.fit(
            alpha=0.3648761546066018,beta=0.5058870363874656, topK=480, normalize_similarity=True)

        self.S_SLIM = SLIMElasticNetRecommender(self.URM_train_pow)
        self.S_SLIM.fit()
    '''

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # num_items_aug = 24507
        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            interactions = len(self.URM_train_aug[user_id_array[i], :].indices)

            if interactions <= 15:  # TIER 1
                w1 = self.RP3beta_pow._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.RP3beta_pow_tier1_weight*w1 + self.UserKNNCF_tier1_weight*w2
                # w = np.pad(w, ((0, 0), (0, num_items_pow-num_items_aug)))
                item_weights[i, :] = w

            elif interactions > 15 and interactions <= 19:  # TIER 2
                w1 = self.RP3beta_pow._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.RP3beta_pow_tier2_weight*w1 + self.UserKNNCF_tier2_weight*w2
                # w = np.pad(w, ((0, 0), (0, num_items_pow-num_items_aug)))
                item_weights[i, :] = w

            elif interactions > 19 and interactions <= 28:  # TIER 3
                w1 = self.RP3beta_pow._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.S_SLIM._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.RP3beta_pow_tier3_weight*w1 + self.S_SLIM_tier3_weight*w2
                item_weights[i, :] = w

            else:  # TIER 4
                w = self.S_SLIM._compute_item_score(
                    user_id_array[i], items_to_compute)
                w /= LA.norm(w, 2)
                item_weights[i, :] = w

            item_weights[i, :] = w

        return item_weights


class HybridRecommender_5(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender_5"

    def __init__(self, URM_train_aug, URM_train_pow, UserKNNCF, RP3beta_pow, S_SLIM, EASE_R):
        """ Constructor of Hybrid_Recommender_2
        Args:
            URM_train (csr): augmented matrix
            ICM (csr): icm
        """
        self.URM_train_aug = URM_train_aug
        self.URM_train_pow = URM_train_pow
        self.UserKNNCF = UserKNNCF
        self.RP3beta_pow = RP3beta_pow
        self.S_SLIM = S_SLIM
        self.EASE_R = EASE_R
        super(HybridRecommender_5, self).__init__(self.URM_train_aug)

    def fit(self, UserKNNCF_tier1_weight=0.6345269660425519, RP3beta_pow_tier1_weight=0.33158219696928976, EASE_R_tier1_weight=0.09597531837611298, UserKNNCF_tier2_weight=0.9792376938449392, RP3beta_pow_tier2_weight=0.57023899912625, EASE_R_tier2_weight=0.38860109710692725, RP3beta_pow_tier3_weight=0.722924027983384, S_SLIM_tier3_weight=0.8268022628613843, EASE_R_tier3_weight=0.06536164657983635, S_SLIM_tier4_weight=0.9465915892992056, EASE_R_tier4_weight=0.4292689877661564):
       # 'UserKNNCF_tier1_weight': 0.6345269660425519, 'RP3beta_pow_tier1_weight': 0.33158219696928976, 'EASE_R_tier1_weight': 0.09597531837611298, 'UserKNNCF_tier2_weight': 0.9792376938449392, 'RP3beta_pow_tier2_weight': 0.570238999126259, 'EASE_R_tier2_weight': 0.38860109710692725, 'RP3beta_pow_tier3_weight': 0.722924027983384, 'S_SLIM_tier3_weight': 0.8268022628613843, 'EASE_R_tier3_weight': 0.06536164657983635, 'S_SLIM_tier4_weight': 0.9465915892992056, 'EASE_R_tier4_weight': 0.4292689877661564
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.UserKNNCF_tier1_weight = UserKNNCF_tier1_weight
        self.RP3beta_pow_tier1_weight = RP3beta_pow_tier1_weight
        self.EASE_R_tier1_weight = EASE_R_tier1_weight

        self.UserKNNCF_tier2_weight = UserKNNCF_tier2_weight
        self.RP3beta_pow_tier2_weight = RP3beta_pow_tier2_weight
        self.EASE_R_tier2_weight = EASE_R_tier2_weight

        self.RP3beta_pow_tier3_weight = RP3beta_pow_tier3_weight
        self.S_SLIM_tier3_weight = S_SLIM_tier3_weight
        self.EASE_R_tier3_weight = EASE_R_tier3_weight

        self.S_SLIM_tier4_weight = S_SLIM_tier4_weight
        self.EASE_R_tier4_weight = EASE_R_tier4_weight

    '''
    self.UserKNNCF = UserKNNCFRecommender(self.URM_train_aug)
        self.UserKNNCF.fit()
        self.RP3beta_pow = RP3betaRecommender(self.URM_train_pow)
        self.RP3beta_pow.fit(
            alpha=0.3648761546066018,beta=0.5058870363874656, topK=480, normalize_similarity=True)
        self.S_SLIM = SLIMElasticNetRecommender(self.URM_train_pow)
        self.S_SLIM.fit()
    '''

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # num_items_aug = 24507
        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            interactions = len(self.URM_train_aug[user_id_array[i], :].indices)

            if interactions <= 15:  # TIER 1
                w1 = self.RP3beta_pow._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w3 = self.EASE_R._compute_item_score(
                    user_id_array[i], items_to_compute)
                w3 /= LA.norm(w3, 2)

                w = self.RP3beta_pow_tier1_weight*w1 + \
                    self.UserKNNCF_tier1_weight*w2 + self.EASE_R_tier1_weight*w3
                # w = np.pad(w, ((0, 0), (0, num_items_pow-num_items_aug)))

            elif interactions > 15 and interactions <= 19:  # TIER 2
                w1 = self.RP3beta_pow._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w3 = self.EASE_R._compute_item_score(
                    user_id_array[i], items_to_compute)
                w3 /= LA.norm(w3, 2)

                w = self.RP3beta_pow_tier2_weight*w1 + \
                    self.UserKNNCF_tier2_weight*w2 + self.EASE_R_tier2_weight*w3
                # w = np.pad(w, ((0, 0), (0, num_items_pow-num_items_aug)))

            elif interactions > 19 and interactions <= 28:  # TIER 3
                w1 = self.RP3beta_pow._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.S_SLIM._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w3 = self.EASE_R._compute_item_score(
                    user_id_array[i], items_to_compute)
                w3 /= LA.norm(w3, 2)

                w = self.RP3beta_pow_tier3_weight*w1 + \
                    self.S_SLIM_tier3_weight*w2 + self.EASE_R_tier3_weight*w3

            else:  # TIER 4
                w1 = self.S_SLIM._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.EASE_R._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.S_SLIM_tier4_weight*w1 + self.EASE_R_tier4_weight*w2

            item_weights[i, :] = w

        return item_weights

    def _compute_item_score_per_user(self, user_id, items_to_compute=None):

        interactions = len(self.URM_train_aug[user_id, :].indices)

        if interactions <= 15:  # TIER 1

            w1 = self.RP3beta_pow._compute_item_score(
                user_id, items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.UserKNNCF._compute_item_score(
                user_id, items_to_compute)
            w2 /= LA.norm(w2, 2)

            w3 = self.EASE_R._compute_item_score(
                user_id, items_to_compute)
            w3 /= LA.norm(w3, 2)

            w = self.RP3beta_pow_tier1_weight*w1 + \
                self.UserKNNCF_tier1_weight*w2 + self.EASE_R_tier1_weight*w3
            # w = np.pad(w, ((0, 0), (0, num_items_pow-num_items_aug)))

        elif interactions > 15 and interactions <= 19:  # TIER 2

            w1 = self.RP3beta_pow._compute_item_score(
                user_id, items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.UserKNNCF._compute_item_score(
                user_id, items_to_compute)
            w2 /= LA.norm(w2, 2)

            w3 = self.EASE_R._compute_item_score(
                user_id, items_to_compute)
            w3 /= LA.norm(w3, 2)

            w = self.RP3beta_pow_tier2_weight*w1 + \
                self.UserKNNCF_tier2_weight*w2 + self.EASE_R_tier2_weight*w3
            # w = np.pad(w, ((0, 0), (0, num_items_pow-num_items_aug)))

        elif interactions > 19 and interactions <= 28:  # TIER 3

            w1 = self.RP3beta_pow._compute_item_score(
                user_id, items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.S_SLIM._compute_item_score(
                user_id, items_to_compute)
            w2 /= LA.norm(w2, 2)

            w3 = self.EASE_R._compute_item_score(
                user_id, items_to_compute)
            w3 /= LA.norm(w3, 2)

            w = self.RP3beta_pow_tier3_weight*w1 + \
                self.S_SLIM_tier3_weight*w2 + self.EASE_R_tier3_weight*w3

        else:  # TIER 4

            w1 = self.S_SLIM._compute_item_score(
                user_id, items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.EASE_R._compute_item_score(
                user_id, items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.S_SLIM_tier4_weight*w1 + self.EASE_R_tier4_weight * w2

        return w


class HybridRecommender_6(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender_6"

    def __init__(self, URM_train_aug, URM_train_pow, UserKNNCF, RP3beta_pow, S_SLIM, EASE_R):
        self.URM_train_aug = URM_train_aug
        self.URM_train_pow = URM_train_pow
        self.UserKNNCF = UserKNNCF
        self.RP3beta_pow = RP3beta_pow
        self.S_SLIM = S_SLIM
        self.EASE_R = EASE_R
        super(HybridRecommender_6, self).__init__(self.URM_train_aug)

    def fit(self, UserKNNCF_tier1_weight=0.9, RP3beta_pow_tier1_weight=0.6, UserKNNCF_tier2_weight=0.7, RP3beta_pow_tier2_weight=0.9, RP3beta_pow_tier3_weight=0.6, S_SLIM_tier3_weight=1.0, tiers_block_tail_weight=0.5, EASE_R_tail_weight=0.5):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.UserKNNCF_tier1_weight = UserKNNCF_tier1_weight
        self.RP3beta_pow_tier1_weight = RP3beta_pow_tier1_weight

        self.UserKNNCF_tier2_weight = UserKNNCF_tier2_weight
        self.RP3beta_pow_tier2_weight = RP3beta_pow_tier2_weight

        self.RP3beta_pow_tier3_weight = RP3beta_pow_tier3_weight
        self.S_SLIM_tier3_weight = S_SLIM_tier3_weight

        self.tiers_block_tail_weight = tiers_block_tail_weight
        self.EASE_R_tail_weight = EASE_R_tail_weight

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            interactions = len(self.URM_train_aug[user_id_array[i], :].indices)

            if interactions <= 17:  # TIER 1
                w1 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.EASE_R._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.UserKNNCF_tier1_weight*w1 + self.EASE_R_tier1_weight*w2

            elif interactions > 15 and interactions <= 19:  # TIER 2
                w1 = self.RP3beta_pow._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.RP3beta_pow_tier2_weight*w1 + self.UserKNNCF_tier2_weight*w2

            elif interactions > 19 and interactions <= 28:  # TIER 3
                w1 = self.RP3beta_pow._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.S_SLIM._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.RP3beta_pow_tier3_weight*w1 + self.S_SLIM_tier3_weight*w2

            else:  # TIER 4
                w = self.S_SLIM._compute_item_score(
                    user_id_array[i], items_to_compute)

            # At the end, add EASE_R
            w1 = w  # rename w
            w1 /= LA.norm(w1, 2)

            w2 = self.EASE_R._compute_item_score(
                user_id_array[i], items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.tiers_block_tail_weight*w1 + self.EASE_R_tail_weight*w2

            item_weights[i, :] = w

        return item_weights


class HybridRecommender_7(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender_7"

    def __init__(self, URM_train_aug, URM_train_pow, UCM, UserKNNCF, RP3beta_pow, S_SLIM, EASE_R, UserKNN_CFCBF_Hybrid):

        self.URM_train_aug = URM_train_aug
        self.URM_train_pow = URM_train_pow
        self.UCM = UCM

        if(UserKNNCF == None):
            self.UserKNNCF = UserKNNCFRecommender(self.URM_train_aug)
        else:
            self.UserKNNCF = UserKNNCF

        if(RP3beta_pow == None):
            self.RP3beta_pow = RP3betaRecommender(self.URM_train_pow)
        else:
            self.RP3beta_pow = RP3beta_pow

        if(S_SLIM == None):
            self.S_SLIM = S_SLIM(self.URM_train_pow)
        else:
            self.S_SLIM = S_SLIM

        if(EASE_R == None):
            self.EASE_R = EASE_R(self.URM_train_aug)
        else:
            self.EASE_R = EASE_R

        if(UserKNN_CFCBF_Hybrid == None):
            self.UserKNNCB_Hybrid = UserKNN_CFCBF_Hybrid_Recommender(
                self.URM_train_aug, self.UCM)
        else:
            self.UserKNNCB_Hybrid = UserKNN_CFCBF_Hybrid

        super(HybridRecommender_7, self).__init__(self.URM_train_aug)

    def fit(self, UserKNNCF_tier1_weight=0.6345269660425519, RP3beta_pow_tier1_weight=0.33158219696928976, EASE_R_tier1_weight=0.09597531837611298, UserKNNCB_Hybrid_tier2_weight=0.5, UserKNNCF_tier2_weight=0.9792376938449392, RP3beta_pow_tier2_weight=0.570238999126259, EASE_R_tier2_weight=0.38860109710692725, UserKNNCB_Hybrid_tier3_weight=0.5, RP3beta_pow_tier3_weight=0.722924027983384, S_SLIM_tier3_weight=0.8268022628613843, EASE_R_tier3_weight=0.06536164657983635, UserKNNCB_Hybrid_tier4_weight=0.5, S_SLIM_tier4_weight=0.9465915892992056, EASE_R_tier4_weight=0.4292689877661564):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.UserKNNCF_tier1_weight = UserKNNCF_tier1_weight
        self.RP3beta_pow_tier1_weight = RP3beta_pow_tier1_weight
        self.EASE_R_tier1_weight = EASE_R_tier1_weight

        self.UserKNNCB_Hybrid_tier2_weight = UserKNNCB_Hybrid_tier2_weight
        self.UserKNNCF_tier2_weight = UserKNNCF_tier2_weight
        self.RP3beta_pow_tier2_weight = RP3beta_pow_tier2_weight
        self.EASE_R_tier2_weight = EASE_R_tier2_weight

        self.UserKNNCB_Hybrid_tier3_weight = UserKNNCB_Hybrid_tier3_weight
        self.RP3beta_pow_tier3_weight = RP3beta_pow_tier3_weight
        self.S_SLIM_tier3_weight = S_SLIM_tier3_weight
        self.EASE_R_tier3_weight = EASE_R_tier3_weight

        self.UserKNNCB_Hybrid_tier4_weight = UserKNNCB_Hybrid_tier4_weight
        self.S_SLIM_tier4_weight = S_SLIM_tier4_weight
        self.EASE_R_tier4_weight = EASE_R_tier4_weight

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # num_items_aug = 24507
        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            interactions = len(self.URM_train_aug[user_id_array[i], :].indices)

            if interactions <= 15:  # TIER 1

                w1 = self.RP3beta_pow._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w3 = self.EASE_R._compute_item_score(
                    user_id_array[i], items_to_compute)
                w3 /= LA.norm(w3, 2)

                w = self.RP3beta_pow_tier1_weight*w1 + \
                    self.UserKNNCF_tier1_weight*w2 + self.EASE_R_tier1_weight*w3
                # w = np.pad(w, ((0, 0), (0, num_items_pow-num_items_aug)))

            elif interactions > 15 and interactions <= 19:  # TIER 2
                w4 = self.UserKNNCB_Hybrid._compute_item_score(
                    user_id_array[i], items_to_compute)
                w4 /= LA.norm(w4, 2)

                w1 = self.RP3beta_pow._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w3 = self.EASE_R._compute_item_score(
                    user_id_array[i], items_to_compute)
                w3 /= LA.norm(w3, 2)

                w = self.RP3beta_pow_tier2_weight*w1 + self.UserKNNCF_tier2_weight*w2 + \
                    self.EASE_R_tier2_weight*w3 + self.UserKNNCB_Hybrid_tier2_weight*w4
                # w = np.pad(w, ((0, 0), (0, num_items_pow-num_items_aug)))

            elif interactions > 19 and interactions <= 28:  # TIER 3

                w1 = self.RP3beta_pow._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.S_SLIM._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w3 = self.EASE_R._compute_item_score(
                    user_id_array[i], items_to_compute)
                w3 /= LA.norm(w3, 2)

                w4 = self.UserKNNCB_Hybrid._compute_item_score(
                    user_id_array[i], items_to_compute)
                w4 /= LA.norm(w4, 2)

                w = self.RP3beta_pow_tier3_weight*w1 + self.S_SLIM_tier3_weight*w2 + \
                    self.EASE_R_tier3_weight*w3 + self.UserKNNCB_Hybrid_tier3_weight*w4

            else:  # TIER 4

                w1 = self.S_SLIM._compute_item_score(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.EASE_R._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w3 = self.UserKNNCB_Hybrid._compute_item_score(
                    user_id_array[i], items_to_compute)
                w3 /= LA.norm(w3, 2)

                w = self.S_SLIM_tier4_weight*w1 + self.EASE_R_tier4_weight * \
                    w2 + self.UserKNNCB_Hybrid_tier4_weight*w3

            item_weights[i, :] = w

        return item_weights


############################################################# Hybrids per layer ###########################################################


class Hybrid_SSLIM_EASER(BaseRecommender):
    RECOMMENDER_NAME = "Hybrid_SSLIM_EASER"

    def __init__(self, URM_train_aug, URM_train_pow, SSLIM, EASE_R):
        self.URM_train_aug = URM_train_aug
        self.URM_train_pow = URM_train_pow
        self.SSLIM = SSLIM
        self.EASE_R = EASE_R
        super(Hybrid_SSLIM_EASER, self).__init__(self.URM_train_aug)

    def fit(self, SSLIM_weight=0.5, EASE_R_weight=0.5):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.SSLIM_weight = SSLIM_weight
        self.EASE_R_weight = EASE_R_weight

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            w1 = self.SSLIM._compute_item_score(
                user_id_array[i], items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.EASE_R._compute_item_score(
                user_id_array[i], items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.SSLIM_weight*w1 + self.EASE_R_weight*w2

            item_weights[i, :] = w

        return item_weights

    def _compute_item_score_per_user(self, user_id, items_to_compute=None):

        w1 = self.SSLIM._compute_item_score(
            user_id, items_to_compute)
        w1 /= LA.norm(w1, 2)

        w2 = self.EASE_R._compute_item_score(
            user_id, items_to_compute)
        w2 /= LA.norm(w2, 2)

        w = self.SSLIM_weight*w1 + self.EASE_R_weight*w2

        return w


class Hybrid_SSLIM_RP3B_aug(BaseRecommender):
    RECOMMENDER_NAME = "Hybrid_SSLIM_RP3B_aug"

    def __init__(self, URM_train_aug, SSLIM, RP3B):
        self.URM_train_aug = URM_train_aug
        self.SSLIM = SSLIM
        self.RP3B = RP3B
        super(Hybrid_SSLIM_RP3B_aug, self).__init__(self.URM_train_aug)

    def fit(self, alpha=0.5):
        """ Set the weights for every algorithm involved in the hybrid recommender """
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        #num_items_pow = 27286
        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            w1 = self.SSLIM._compute_item_score(
                user_id_array[i], items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.RP3B._compute_item_score(
                user_id_array[i], items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.alpha*w1 + (1-self.alpha)*w2

            item_weights[i, :] = w

        return item_weights

    def _compute_item_score_per_user(self, user_id, items_to_compute=None):

        w1 = self.SSLIM._compute_item_score(
            user_id, items_to_compute)
        w1 /= LA.norm(w1, 2)

        w2 = self.RP3B._compute_item_score(
            user_id, items_to_compute)
        w2 /= LA.norm(w2, 2)

        w = self.alpha*w1 + (1-self.alpha)*w2

        return w


class Hybrid_UserKNNCF_RP3B_aug(BaseRecommender):
    RECOMMENDER_NAME = "Hybrid_UserKNNCF_RP3B_aug"

    def __init__(self, URM_train_aug, URM_train_pow, UserKNNCF, RP3B):
        self.URM_train_aug = URM_train_aug
        self.URM_train_pow = URM_train_pow
        self.UserKNNCF = UserKNNCF
        self.RP3B = RP3B
        super(Hybrid_UserKNNCF_RP3B_aug, self).__init__(self.URM_train_aug)

    def fit(self, UserKNNCF_weight=0.2995420066475148, RP3B_weight=0.9911264072270123):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.UserKNNCF_weight = UserKNNCF_weight
        self.RP3B_weight = RP3B_weight

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            w1 = self.UserKNNCF._compute_item_score(
                user_id_array[i], items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.RP3B._compute_item_score(
                user_id_array[i], items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.UserKNNCF_weight*w1 + self.RP3B_weight*w2

            item_weights[i, :] = w

        return item_weights

    def _compute_item_score_per_user(self, user_id, items_to_compute=None):

        w1 = self.UserKNNCF._compute_item_score(
            user_id, items_to_compute)
        w1 /= LA.norm(w1, 2)

        w2 = self.RP3B._compute_item_score(
            user_id, items_to_compute)
        w2 /= LA.norm(w2, 2)

        w = self.UserKNNCF_weight*w1 + self.RP3B_weight*w2

        return w


class Hybrid_UserKNNCF_ItemKNNCF(BaseRecommender):
    RECOMMENDER_NAME = "Hybrid_UserKNNCF_ItemKNNCF"

    def __init__(self, URM_train_aug, URM_train_pow, UserKNNCF, ItemKNNCF):
        self.URM_train_aug = URM_train_aug
        self.URM_train_pow = URM_train_pow
        self.UserKNNCF = UserKNNCF
        self.ItemKNNCF = ItemKNNCF
        super(Hybrid_UserKNNCF_ItemKNNCF, self).__init__(self.URM_train_aug)

    def fit(self, alpha = 0.5):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            w1 = self.UserKNNCF._compute_item_score(
                user_id_array[i], items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.ItemKNNCF._compute_item_score(
                user_id_array[i], items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.alpha*w1 + (1-self.alpha)*w2

            item_weights[i, :] = w

        return item_weights

    def _compute_item_score_per_user(self, user_id, items_to_compute=None):

        w1 = self.UserKNNCF._compute_item_score(
            user_id, items_to_compute)
        w1 /= LA.norm(w1, 2)

        w2 = self.ItemKNNCF._compute_item_score(
            user_id, items_to_compute)
        w2 /= LA.norm(w2, 2)

        w = self.UserKNNCF_weight*w1 + self.ItemKNNCF_weight*w2

        return w

class Hybrid_HybridSSLIMRP3B_UserKNNCF(BaseRecommender):
    RECOMMENDER_NAME = "Hybrid_HybridSSLIMRP3B_UserKNNCF"

    def __init__(self, URM_train_aug, URM_train_pow, Hybrid_SSLIM_RP3B_aug, UserKNNCF):
        self.URM_train_aug = URM_train_aug
        self.URM_train_pow = URM_train_pow
        self.Hybrid_SSLIM_RP3B_aug = Hybrid_SSLIM_RP3B_aug
        self.UserKNNCF = UserKNNCF
        super(Hybrid_HybridSSLIMRP3B_UserKNNCF, self).__init__(self.URM_train_aug)

    def fit(self, UserKNNCF_weight=0.5, Hybrid_weight=0.5):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.UserKNNCF_weight = UserKNNCF_weight
        self.Hybrid_weight = Hybrid_weight

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            w1 = self.UserKNNCF._compute_item_score(
                user_id_array[i], items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.Hybrid_SSLIM_RP3B_aug._compute_item_score_per_user(
                user_id_array[i], items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.UserKNNCF_weight*w1 + self.Hybrid_weight*w2

            item_weights[i, :] = w

        return item_weights

    def _compute_item_score_per_user(self, user_id, items_to_compute=None):

        w1 = self.UserKNNCF._compute_item_score(
            user_id, items_to_compute)
        w1 /= LA.norm(w1, 2)

        w2 = self.Hybrid_SSLIM_RP3B_aug._compute_item_score_per_user(
            user_id, items_to_compute)
        w2 /= LA.norm(w2, 2)

        w = self.UserKNNCF_weight*w1 + self.Hybrid_weight*w2

        return w

class Hybrid_User_and_Item_KNN_CFCBF_Hybrid(BaseRecommender):
    RECOMMENDER_NAME = "Hybrid_User_and_Item_KNN_CFCBF_Hybrid"

    def __init__(self, URM_train_aug, URM_train_pow, ItemKNN_CFCBF_Hybrid_Recommender, UserKNN_CFCBF_Hybrid_Recommender):
        self.URM_train_aug = URM_train_aug
        self.URM_train_pow = URM_train_pow
        self.ItemKNN_CFCBF_Hybrid_Recommender = ItemKNN_CFCBF_Hybrid_Recommender
        self.UserKNN_CFCBF_Hybrid_Recommender = UserKNN_CFCBF_Hybrid_Recommender
        super(Hybrid_User_and_Item_KNN_CFCBF_Hybrid,
              self).__init__(self.URM_train_aug)

    def fit(self, ItemKNN_CFCBF_Hybrid_Recommender_weight=0.5009028244188916, UserKNN_CFCBF_Hybrid_Recommender_weight=0.5977245751359852):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.ItemKNN_CFCBF_Hybrid_Recommender_weight = ItemKNN_CFCBF_Hybrid_Recommender_weight
        self.UserKNN_CFCBF_Hybrid_Recommender_weight = UserKNN_CFCBF_Hybrid_Recommender_weight

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            w1 = self.ItemKNN_CFCBF_Hybrid_Recommender._compute_item_score(
                user_id_array[i], items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.UserKNN_CFCBF_Hybrid_Recommender._compute_item_score(
                user_id_array[i], items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.ItemKNN_CFCBF_Hybrid_Recommender_weight*w1 + \
                self.UserKNN_CFCBF_Hybrid_Recommender_weight*w2

            item_weights[i, :] = w

        return item_weights

    def _compute_item_score_per_user(self, user_id, items_to_compute=None):

        w1 = self.ItemKNN_CFCBF_Hybrid_Recommender._compute_item_score(
            user_id, items_to_compute)
        w1 /= LA.norm(w1, 2)

        w2 = self.UserKNN_CFCBF_Hybrid_Recommender._compute_item_score(
            user_id, items_to_compute)
        w2 /= LA.norm(w2, 2)

        w = self.ItemKNN_CFCBF_Hybrid_Recommender_weight*w1 + \
            self.UserKNN_CFCBF_Hybrid_Recommender_weight*w2

        return w


#######################################################################################################
##################### Hybrid of Hybrids ###############################################################
#######################################################################################################


class Hybrid_of_Hybrids(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_of_Hybrids"

    def __init__(self, URM_train_aug, Hybrid_SSLIM_RP3B_aug, UserKNNCF, S_SLIM):

        self.URM_train_aug = URM_train_aug

        self.Hybrid_SSLIM_RP3B_aug = Hybrid_SSLIM_RP3B_aug
        self.UserKNNCF = UserKNNCF
        self.S_SLIM = S_SLIM

        super(Hybrid_of_Hybrids, self).__init__(self.URM_train_aug)

    def fit(self, alpha=0.5, beta=0.5,gamma=0.5):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items_pow = 27286
        #num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            interactions = len(self.URM_train_aug[user_id_array[i], :].indices)

            if interactions <= 22:  # TIER 1

                w1 = self.Hybrid1_tier1._compute_item_score_per_user(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.alpha*w1 + (1-self.alpha)*w2

            elif interactions > 22 and interactions <= 24:  # TIER 2

                w1 = self.Hybrid1_tier1._compute_item_score_per_user(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.beta*w1 + (1-self.beta)*w2

            else:  # TIER 3

                w1 = self.Hybrid1_tier1._compute_item_score_per_user(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.S_SLIM._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.gamma*w1 +(1- self.gamma)*w2

            item_weights[i, :] = w

        return item_weights



# 0.06021

class Hybrid_Best(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Best"

    def __init__(self, URM_train_aug, URM_train_pow, ICM, UCM, Hybrid_SSLIM_RP3B_aug=None, Hybrid_UserKNNCF_ItemKNNCF=None, UserKNNCF=None, Hybrid_UserKNNCF_RP3B_aug=None, Hybrid_SSLIM_EASER=None):

        self.URM_train_aug = URM_train_aug
        self.URM_train_pow = URM_train_pow
        self.ICM = ICM
        self.UCM = UCM

        if(Hybrid_SSLIM_RP3B_aug == None):
            # TODO: based on hybrid
            self.Hybrid_SSLIM_RP3B_aug = Hybrid_SSLIM_RP3B_aug(
                self.URM_train_aug)
        else:
            self.Hybrid_SSLIM_RP3B_aug = Hybrid_SSLIM_RP3B_aug

        if(Hybrid_UserKNNCF_ItemKNNCF == None):
            # TODO: based on hybrid
            self.Hybrid_UserKNNCF_ItemKNNCF = Hybrid_UserKNNCF_ItemKNNCF(
                self.URM_train_aug)
        else:
            self.Hybrid_UserKNNCF_ItemKNNCF = Hybrid_UserKNNCF_ItemKNNCF

        if(Hybrid_UserKNNCF_RP3B_aug == None):
            # TODO: based on hybrid
            self.Hybrid_UserKNNCF_RP3B_aug = Hybrid_UserKNNCF_RP3B_aug(
                self.URM_train_pow)
        else:
            self.Hybrid_UserKNNCF_RP3B_aug = Hybrid_UserKNNCF_RP3B_aug

        if(Hybrid_SSLIM_EASER == None):
            # TODO: based on hybrid
            self.Hybrid_SSLIM_EASER = Hybrid_SSLIM_EASER(self.URM_train_pow)
        else:
            self.Hybrid_SSLIM_EASER = Hybrid_SSLIM_EASER

        self.UserKNNCF = UserKNNCF

        super(Hybrid_Best, self).__init__(self.URM_train_aug)

    def fit(self, Hybrid_1_tier1_weight=0.5, Hybrid_2_tier1_weight=0.5,
            Hybrid_1_tier2_weight=0.5, Hybrid_2_tier2_weight=0.5,
            Hybrid_1_tier3_weight=0.5, Hybrid_2_tier3_weight=0.5):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.Hybrid_1_tier1_weight = Hybrid_1_tier1_weight
        self.Hybrid_2_tier1_weight = Hybrid_2_tier1_weight

        self.Hybrid_1_tier2_weight = Hybrid_1_tier2_weight
        self.Hybrid_2_tier2_weight = Hybrid_2_tier2_weight

        self.Hybrid_1_tier3_weight = Hybrid_1_tier3_weight
        self.Hybrid_2_tier3_weight = Hybrid_2_tier3_weight

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            interactions = len(self.URM_train_aug[user_id_array[i], :].indices)

            if interactions <= 22:  # TIER 1

                w1 = self.Hybrid_SSLIM_RP3B_aug._compute_item_score_per_user(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                '''
                w3 = self.Hybrid_3_tier2._compute_item_score_per_user(
                    user_id_array[i], items_to_compute)
                w3 /= LA.norm(w3, 2)
                '''

                w = self.Hybrid_1_tier1_weight*w1 + self.Hybrid_2_tier1_weight*w2

            elif interactions > 22 and interactions <= 24:  # TIER 2

                w1 = self.Hybrid_SSLIM_RP3B_aug._compute_item_score_per_user(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.UserKNNCF._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.Hybrid_1_tier2_weight*w1 + self.Hybrid_2_tier2_weight*w2

            else:  # TIER 3

                w1 = self.Hybrid_SSLIM_RP3B_aug._compute_item_score_per_user(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.Hybrid_SSLIM_EASER._compute_item_score_per_user(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.Hybrid_1_tier3_weight*w1 + self.Hybrid_2_tier3_weight*w2

            item_weights[i, :] = w

        return item_weights

    def _compute_item_score_per_user(self, user_id, items_to_compute=None):

        interactions = len(self.URM_train_aug[user_id, :].indices)

        if interactions <= 22:  # TIER 1

            w1 = self.Hybrid_SSLIM_RP3B_aug._compute_item_score_per_user(
                user_id, items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.UserKNNCF._compute_item_score(
                user_id, items_to_compute)
            w2 /= LA.norm(w2, 2)

            '''
            w3 = self.Hybrid_3_tier2._compute_item_score_per_user(
                user_id_array[i], items_to_compute)
            w3 /= LA.norm(w3, 2)
            '''

            w = self.Hybrid_1_tier1_weight*w1 + self.Hybrid_2_tier1_weight*w2

        elif interactions > 22 and interactions <= 24:  # TIER 2

            w1 = self.Hybrid_SSLIM_RP3B_aug._compute_item_score_per_user(
                user_id, items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.UserKNNCF._compute_item_score(
                user_id, items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.Hybrid_1_tier2_weight*w1 + self.Hybrid_2_tier2_weight*w2

        else:  # TIER 3

            w1 = self.Hybrid_SSLIM_RP3B_aug._compute_item_score_per_user(
                user_id, items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.Hybrid_SSLIM_EASER._compute_item_score_per_user(
                user_id, items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.Hybrid_1_tier3_weight*w1 + self.Hybrid_2_tier3_weight*w2

        return w



class Hybrid_006022(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_006022"

    def __init__(self, URM_train_aug, URM_train_pow, ICM, UCM, Hybrid1_tier1=None, Hybrid2_tier1=None):

        self.URM_train_aug = URM_train_aug
        self.URM_train_pow = URM_train_pow
        self.ICM = ICM
        self.UCM = UCM
        self.Hybrid1_tier1 = Hybrid1_tier1
        self.Hybrid2_tier1 = Hybrid2_tier1

        super(Hybrid_006022, self).__init__(self.URM_train_aug)

    def fit(self, Hybrid_1_tier1_weight=0.5, Hybrid_2_tier1_weight=0.5,
            Hybrid_1_tier2_weight=0.5, Hybrid_2_tier2_weight=0.5,
            Hybrid_1_tier3_weight=0.5):
        """ Set the weights for every algorithm involved in the hybrid recommender """

        self.Hybrid_1_tier1_weight = Hybrid_1_tier1_weight
        self.Hybrid_2_tier1_weight = Hybrid_2_tier1_weight

        self.Hybrid_1_tier2_weight = Hybrid_1_tier2_weight
        self.Hybrid_2_tier2_weight = Hybrid_2_tier2_weight

        self.Hybrid_1_tier3_weight = Hybrid_1_tier3_weight

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items_pow = 27968
        item_weights = np.empty([len(user_id_array), num_items_pow])

        for i in range(len(user_id_array)):

            interactions = len(self.URM_train_aug[user_id_array[i], :].indices)

            if interactions <= 22:  # TIER 1

                w1 = self.Hybrid1_tier1._compute_item_score_per_user(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.Hybrid2_tier1._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.Hybrid_1_tier1_weight*w1 + self.Hybrid_2_tier1_weight*w2

            elif interactions > 22 and interactions <= 24:  # TIER 2

                w1 = self.Hybrid1_tier1._compute_item_score_per_user(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w2 = self.Hybrid2_tier1._compute_item_score(
                    user_id_array[i], items_to_compute)
                w2 /= LA.norm(w2, 2)

                w = self.Hybrid_1_tier2_weight*w1 + self.Hybrid_2_tier2_weight*w2

            else:  # TIER 3

                w1 = self.Hybrid1_tier1._compute_item_score_per_user(
                    user_id_array[i], items_to_compute)
                w1 /= LA.norm(w1, 2)

                w = self.Hybrid_1_tier3_weight*w1 

            item_weights[i, :] = w

        return item_weights

    def _compute_item_score_per_user(self, user_id, items_to_compute=None):

        interactions = len(self.URM_train_aug[user_id, :].indices)

        if interactions <= 22:  # TIER 1

            w1 = self.Hybrid1_tier1._compute_item_score_per_user(
                user_id, items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.Hybrid2_tier1._compute_item_score(
                user_id, items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.Hybrid_1_tier1_weight*w1 + self.Hybrid_2_tier1_weight*w2

        elif interactions > 22 and interactions <= 24:  # TIER 2

            w1 = self.Hybrid1_tier1._compute_item_score_per_user(
                user_id, items_to_compute)
            w1 /= LA.norm(w1, 2)

            w2 = self.Hybrid2_tier1._compute_item_score(
                user_id, items_to_compute)
            w2 /= LA.norm(w2, 2)

            w = self.Hybrid_1_tier2_weight*w1 + self.Hybrid_2_tier2_weight*w2

        else:  # TIER 3

            w1 = self.Hybrid1_tier1._compute_item_score_per_user(
                user_id, items_to_compute)
            w1 /= LA.norm(w1, 2)

            w = self.Hybrid_1_tier3_weight*w1
        return w


###################################################################
########################## LINEAR HYBRID ##########################

class Linear_Hybrid(BaseRecommender):
    RECOMMENDER_NAME = "Linear_Hybrid"

    def __init__(self, URM_train, recommender_1, recommender_2, recommender_3, recommender_4, recommender_5, recommender_6, recommender_7):
        super(Linear_Hybrid, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        self.recommender_4 = recommender_4
        self.recommender_5 = recommender_5
        self.recommender_6 = recommender_6
        self.recommender_7 = recommender_7

        
    def fit(self, alpha = 0.5, beta =0.5, teta = 0.5, gamma = 0.5, delta = 0.5, sigma = 0.5, rho = 0.5):
        
        self.alpha = alpha
        self.beta = beta
        self.teta = teta
        self.gamma = gamma
        self.delta = delta
        self.sigma = sigma
        self.rho = rho


    def _compute_item_score(self, user_id_array, items_to_compute):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array,items_to_compute)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array,items_to_compute)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array,items_to_compute)
        item_weights_4 = self.recommender_4._compute_item_score(user_id_array,items_to_compute)
        item_weights_5 = self.recommender_5._compute_item_score(user_id_array,items_to_compute)
        item_weights_6 = self.recommender_6._compute_item_score(user_id_array,items_to_compute)
        item_weights_7 = self.recommender_7._compute_item_score(user_id_array,items_to_compute)

        norm_item_weights_1 = LA.norm(item_weights_1, 2)
        norm_item_weights_2 = LA.norm(item_weights_2, 2)
        norm_item_weights_3 = LA.norm(item_weights_3, 2)
        norm_item_weights_4 = LA.norm(item_weights_4, 2)
        norm_item_weights_5 = LA.norm(item_weights_5, 2)
        norm_item_weights_6 = LA.norm(item_weights_4, 2)
        norm_item_weights_7 = LA.norm(item_weights_5, 2)

        #item_weights = item_weights_1 * self.alpha + item_weights_2 * self.beta + item_weights_3 * self.teta + item_weights_4 * self.gamma + self.teta + item_weights_5 * self.delta + item_weights_6 * self.sigma + item_weights_7 * self.rho
        item_weights = (item_weights_1 / norm_item_weights_1) * self.alpha + (item_weights_2 / norm_item_weights_2) * self.beta + (item_weights_3 / norm_item_weights_3) * self.teta + (item_weights_4 / norm_item_weights_4) * self.gamma + (item_weights_5 / norm_item_weights_5) * self.delta + (item_weights_6 / norm_item_weights_6) * self.sigma + (item_weights_7 / norm_item_weights_7) * self.rho


        return item_weights

    
    def _compute_item_score_per_user(self, user_id_array, items_to_compute):
        
        item_weights_1 = self.recommender_1._compute_item_score_per_user(user_id_array,items_to_compute)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array,items_to_compute)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
        
        '''
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
        '''
        if norm_item_weights_2 == 0 and self.recommender_2==EASE_R_Recommender:
            item_weights = (item_weights_1 / norm_item_weights_1)
        else:
            item_weights = (item_weights_1 / norm_item_weights_1) * self.alpha + (item_weights_2 / norm_item_weights_2) * (1-self.alpha)
        return item_weights




###################################################################
###################### LIST COMBINATION HYBRID ####################

class ListCombinationHybrid(BaseRecommender):

    RECOMMENDER_NAME = "ListCombinationHybrid"

    def __init__(self, URM_train_aug, recommender_1, recommender_2):
        super(ListCombinationHybrid, self).__init__(URM_train_aug)

        self.URM_train_aug = URM_train_aug
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

        
    def recommend(self, user_id, cutoff = None, remove_seen_flag=True):

        list1 = self.recommender_1.recommend(user_id_array=user_id, cutoff = cutoff,remove_seen_flag=remove_seen_flag)
        list2 = self.recommender_2.recommend(user_id_array=user_id, cutoff = cutoff,remove_seen_flag=remove_seen_flag)
        
        result = []
        i = 0
            
        while len(result) < cutoff:
            if list1[i] not in result:
                result.append(list1[i])
            if (list2[i] != list1[i]):
                if list2[i] not in result:
                    if len(result) < cutoff:
                        result.append(list2[i])
            i = i + 1

