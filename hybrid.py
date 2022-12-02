# Import recommenders
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import *
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMItemHybridRecommender

# Import libraries
from tqdm import tqdm
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp

class HybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender"

    def __init__(self, URM_train,ICM):
        self.URM_train=URM_train
        self.ICM = ICM
        super(HybridRecommender, self).__init__(URM_train)
        

    def fit(self):
        # Stack and normalize URM and ICM
        #URM_stacked = sps.vstack([self.URM_train, self.ICM.T])
        
        # Instantiate & fit the recommenders
        #self.ItemCF = ItemKNNCFRecommender(URM_stacked)
        #self.ItemCF.fit(10, 2000)

        #self.SLIM_ElasticNet = SLIMElasticNetRecommender(self.URM_train)
        #self.SLIM_ElasticNet.fit(l1_ratio=0.008213119901673099, alpha =  0.0046000272149077145, positive_only=True, topK = 498)

        self.LightFMItemHybridRecommender = LightFMItemHybridRecommender(self.URM_train, self.ICM)
        self.LightFMItemHybridRecommender.fit(epochs = 10)


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        
        num_items = 24507
        #num_items=19630
        #num_items = len(self.items) # num_items changes based on used urm
        item_weights = np.empty([len(user_id_array), num_items])

        for i in tqdm(range(len(user_id_array))):
            
            '''
            w1 = self.ItemCF._compute_item_score(user_id_array[i], items_to_compute)
            w1 /= LA.norm(w1, 2)
            w2 = self.SLIM_ElasticNet._compute_item_score(user_id_array[i], items_to_compute)
            w2 /= LA.norm(w2, 2)
            w = w1 + w2 
            '''

            #w = self.ItemCF._compute_item_score(user_id_array[i], items_to_compute)
            #w = self.SLIM_ElasticNet._compute_item_score(user_id_array[i], items_to_compute)
            #w = self.SLIM_BPR_Cython._compute_item_score(user_id_array[i], items_to_compute)
            w = self.LightFMItemHybridRecommender._compute_item_score(user_id_array[i], items_to_compute)

            item_weights[i, :] = w # In the i-th array of item_weights we assign the w array

        return item_weights

    '''
    def _compute_item_score(self, user_id_array, items_to_compute):
    
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
        
        
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (1-self.alpha)

        return item_weights
    '''

class HybridRecommender_2(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender_2"

    N_CONFIG = 0

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True):

        super().__init__(URM, ICM, exclude_seen)
        self.normalize = None
        self.recommenders = {}
        self.weights = {}

    def fit(self, TopPopWeight=0.001, SLIMElasticNetRecommenderWeight=0.0, normalize=False):

        """ Sets the weights for every algorithm involved in the hybrid recommender """

        self.weights = {
            TopPop: TopPopWeight,
            SLIMElasticNetRecommender: SLIMElasticNetRecommenderWeight,
        }

        self.normalize = normalize

        for rec_class in self.weights.keys():
            if self.weights[rec_class] > 0.0:
                if rec_class not in self.recommenders:
                    start = time.time()
                    temp_rec = rec_class(self.URM, self.ICM)
                    temp_rec.fit()
                    self.recommenders[rec_class] = temp_rec
                    end = time.time()
                    print(
                        "Fitted new instance of {}. Employed time: {} seconds".format(rec_class.__name__, end - start))

    def compute_predicted_ratings(self, user_id):

        """ Computes predicted ratings across all different recommender algorithms """

        predicted_ratings = np.zeros(shape=self.URM.shape[1], dtype=np.float32)

        relevant_items = self.URM.indices[self.URM.indptr[user_id]:self.URM.indptr[user_id + 1]]
        if len(relevant_items) > 0:
            for rec_class in self.recommenders.keys():
                if self.weights[rec_class] > 0.0:
                    ratings = self.recommenders[rec_class].compute_predicted_ratings(user_id)
                    if self.normalize:
                        ratings *= 1.0 / ratings.max()
                    predicted_ratings += np.multiply(ratings, self.weights[rec_class])
        else:
            predicted_ratings = self.recommenders[TopPop].compute_predicted_ratings(user_id)

        return predicted_ratings