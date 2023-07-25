from Data_Handler.DataReader import DataReader
import numpy as np
from tqdm import tqdm
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from hybrid import HybridRecommender
from impressions import Impressions
from evaluator import evaluate


dataReader = DataReader()
urm_df = dataReader.load_augmented_binary_urm_df()
urm = dataReader.load_augmented_binary_urm()
target = dataReader.load_target()


URM_train, URM_validation = split_train_in_two_percentage_global_sample(urm, train_percentage=0.9)
# Instantiate and fit hybrid recommender
recommender = HybridRecommender(URM_train,dataReader.get_unique_items_based_on_urm(urm_df))
recommender.fit()

all_items=dataReader.get_unique_items_based_on_urm(urm_df)
impressions=Impressions(target,all_items)

# Create CSV for submission
recommended_items_for_each_user = {}
recommended_items_for_each_user_updated = {}
for user_id in tqdm(target):
    recommended_items = recommender.recommend(user_id, cutoff=10, remove_seen_flag=True)
    recommended_items_for_each_user[int(user_id)] = recommended_items
    recommended_items_for_each_user_updated[int(user_id)] = impressions.update_ranking(user_id=int(user_id),recommended_items=recommended_items)


map = evaluate(recommended_items_for_each_user, URM_validation, target)
print('MAP score: {}'.format(map))

map_updated = evaluate(recommended_items_for_each_user_updated, URM_validation, target)
print('MAP score updated: {}'.format(map_updated))
