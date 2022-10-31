# Import recommenders
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.BaseRecommender import BaseRecommender

# Import libraries
from tqdm import tqdm
import numpy as np


class HybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Recommender"

    def __init__(self, URM_train):
        super(HybridRecommender, self).__init__(URM_train)

    def fit(self):
        self.ItemCF = ItemKNNCFRecommender(self.URM_train)
        # TODO: to improve passing specific parameters
        self.ItemCF.fit(5000, 2000)

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        num_items = 24507
        item_weights = np.empty([len(user_id_array), num_items])

        for i in tqdm(range(len(user_id_array))):
   
            w = self.ItemCF._compute_item_score(
                user_id_array[i], items_to_compute)

            # In the i-th array of item_weights we assign the w array
            item_weights[i, :] = w

        return item_weights
