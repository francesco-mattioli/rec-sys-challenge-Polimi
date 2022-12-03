# Import recommenders
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import *
#from Recommenders.FactorizationMachines.LightFMRecommender import LightFMItemHybridRecommender


# Import libraries
from tqdm import tqdm
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp


class HybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender"

    def __init__(self, URM_train, ICM):
        self.URM_train = URM_train
        self.ICM = ICM
        super(HybridRecommender, self).__init__(URM_train)

    def fit(self):
        # Stack and normalize URM and ICM
        # URM_stacked = sps.vstack([self.URM_train, self.ICM.T])

        # Instantiate & fit the recommenders
        # self.ItemCF = ItemKNNCFRecommender(URM_stacked)
        # self.ItemCF.fit(10, 2000)

        self.SLIM_ElasticNet = SLIMElasticNetRecommender(self.URM_train)
        self.SLIM_ElasticNet.fit(l1_ratio=0.008213119901673099,
                                 alpha=0.0046000272149077145, positive_only=True, topK=498)

        # self.LightFMItemHybridRecommender = LightFMItemHybridRecommender(self.URM_train, self.ICM)
        # self.LightFMItemHybridRecommender.fit(epochs = 10)

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items = 24507
        # num_items=19630
        # num_items = len(self.items) # num_items changes based on used urm
        item_weights = np.empty([len(user_id_array), num_items])

        for i in tqdm(range(len(user_id_array))):

            '''
            w1 = self.ItemCF._compute_item_score(
                user_id_array[i], items_to_compute)
            w1 /= LA.norm(w1, 2)
            w2 = self.SLIM_ElasticNet._compute_item_score(
                user_id_array[i], items_to_compute)
            w2 /= LA.norm(w2, 2)
            w = w1 + w2
            '''

            # w = self.ItemCF._compute_item_score(user_id_array[i], items_to_compute)
            w = self.SLIM_ElasticNet._compute_item_score(
                user_id_array[i], items_to_compute)
            # w = self.SLIM_BPR_Cython._compute_item_score(user_id_array[i], items_to_compute)
            # w = self.LightFMItemHybridRecommender._compute_item_score(user_id_array[i], items_to_compute)

            # In the i-th array of item_weights we assign the w array
            item_weights[i, :] = w

        return item_weights


'''-----------------------------------------------------------------------------------------------------------------------------'''


class HybridRecommender_2(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender_2"

    N_CONFIG = 0

    def __init__(self, URM_train: sp.csr_matrix, ICM):
        self.URM_train = URM_train
        self.ICM = ICM
        super(HybridRecommender_2, self).__init__(URM_train)

        self.normalize = None
        self.recommenders = {}
        self.weights = {}

    def fit(self, ItemKNNCFRecommenderWeight=0.5, UserKNNCFRecommenderWeight=0.5, SLIMElasticNetRecommenderWeight=0.5, normalize=False):
        """ Sets the weights for every algorithm involved in the hybrid recommender """

        self.weights = {
            ItemKNNCFRecommender: ItemKNNCFRecommenderWeight,
            SLIMElasticNetRecommender: SLIMElasticNetRecommenderWeight,
            #UserKNNCFRecommender: UserKNNCFRecommenderWeight,
        }

        self.normalize = normalize

        for rec_class in self.weights.keys():
            if self.weights[rec_class] > 0.0:
                if rec_class not in self.recommenders:
                    start = time.time()
                    temp_rec = rec_class(self.URM_train)
                    temp_rec.fit()
                    self.recommenders[rec_class] = temp_rec
                    end = time.time()
                    print(
                        "Fitted new instance of {}. Employed time: {} seconds".format(rec_class.__name__, end - start))

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items = 24507
        item_weights = np.empty([len(user_id_array), num_items])

        # w is predicted ratings array of a user
        for i in tqdm(range(len(user_id_array))):
            for rec_class in self.recommenders.keys():
                if self.weights[rec_class] > 0.0:
                    w = self.recommenders[rec_class]._compute_item_score(user_id_array[i], items_to_compute)
                    if self.normalize:
                        w *= 1.0 / w.max()
                    w += np.multiply(w, self.weights[rec_class])
            item_weights[i, :] = w

        return item_weights
