from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from impressions import Impressions
from hybrid import *
from tqdm import tqdm
from evaluator import evaluate
import pandas as pd
import numpy as np
# Read & split data
dataReader = DataReader()
URM = dataReader.load_augmented_binary_urm()
#urm = dataReader.load_powerful_binary_urm()
ICM= dataReader.load_icm()

'''
urm = dataReader.load_augmented_binary_urm_less_items()
icm = dataReader.load_augmented_binary_icm_less_items()
'''

target = dataReader.load_target()

# Instantiate Impressions object to update ranking at the end of recommendations
#item_ids = dataReader.get_unique_items_based_on_urm(dataReader.load_augmented_binary_urm_df())
#impressions = Impressions(target,item_ids)

#URM_train, URM_test = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.90)
#URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.90)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM, train_percentage = 1.0)

# Instantiate and fit hybrid recommender
recommender = HybridRecommender_3(URM_train,ICM)
#recommender = HybridRecommender(URM_train)
recommender.fit()

# Create CSV for submission
f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
recommended_items_for_each_user = {}
for user_id in tqdm(target):
    recommended_items = recommender.recommend(user_id, cutoff=10, remove_seen_flag=True)
    #recommended_items=impressions.update_ranking(user_id,recommended_items)
    recommended_items_for_each_user[int(user_id)]=recommended_items
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{user_id}, {well_formatted}\n")

# Evaluare recommended items
map=evaluate(recommended_items_for_each_user,URM_validation,target)
print('MAP score: {}'.format(map))