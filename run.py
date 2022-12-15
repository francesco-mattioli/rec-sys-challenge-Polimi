from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from impressions import Impressions
from hybrid import *
from tqdm import tqdm
from evaluator import evaluate
import pandas as pd
import numpy as np
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

# Read & split data
dataReader = DataReader()
#URM = dataReader.load_augmented_binary_urm()
#URM = dataReader.load_powerful_binary_urm()
#ICM= dataReader.load_icm()

target = dataReader.load_target()

# Instantiate Impressions object to update ranking at the end of recommendations
#item_ids = dataReader.get_unique_items_based_on_urm(dataReader.load_augmented_binary_urm_df())
#impressions = Impressions(target,item_ids)
#dataReader.save_impressions()

#URM_train, URM_test = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.90)
#URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.90)
#URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM, train_percentage = 0.9)

#URM_train_aug,icm = dataReader.pad_with_zeros_ICMandURM(URM_train)
#URM_train_pow = dataReader.stackMatrixes(URM_train)

URM = dataReader.load_augmented_binary_urm()
URM_aug,icm = dataReader.pad_with_zeros_ICMandURM(URM)

URM_train_aug, URM_validation = split_train_in_two_percentage_global_sample(URM_aug, train_percentage = 0.9)
URM_train_pow = dataReader.stackMatrixes(URM_train_aug)

ItemKNNCF = ItemKNNCFRecommender(URM_train_pow)
ItemKNNCF.fit()

UserKNNCF = UserKNNCFRecommender(URM_train_aug)
UserKNNCF.fit()

RP3beta_pow = RP3betaRecommender(URM_train_pow)
RP3beta_pow.fit(alpha=0.3648761546066018,beta=0.5058870363874656, topK=480, normalize_similarity=True)

S_SLIM = SLIMElasticNetRecommender(URM_train_pow)
S_SLIM.fit()

EASE_R = EASE_R_Recommender(URM_train_aug)
EASE_R.fit()


# Instantiate and fit hybrid recommender
recommender = HybridRecommender_7(URM_train_aug, URM_train_pow, UserKNNCF, RP3beta_pow, S_SLIM, EASE_R, ItemKNNCF)
#recommender = HybridRecommender(URM_train)
recommender.fit( ItemKNNCF_tier1_weight=0.0,UserKNNCF_tier1_weight=1.0, RP3beta_pow_tier1_weight=0.0, EASE_R_tier1_weight=0.0, ItemKNNCF_tier2_weight=0.12092091454953455,UserKNNCF_tier2_weight=1.0, RP3beta_pow_tier2_weight=1.0, EASE_R_tier2_weight=1.0, RP3beta_pow_tier3_weight=1.0, S_SLIM_tier3_weight=1.0, EASE_R_tier3_weight=0.8905976575459513, S_SLIM_tier4_weight= 0.8191971656205302, EASE_R_tier4_weight=0.0)
#ItemKNNCF_tier1_weight': 0.0, 'UserKNNCF_tier1_weight': 1.0, 'RP3beta_pow_tier1_weight': 0.0, 'EASE_R_tier1_weight': 0.0, 'ItemKNNCF_tier2_weight': 0.12092091454953455, 'UserKNNCF_tier2_weight': 1.0, 'RP3beta_pow_tier2_weight': 1.0, 'EASE_R_tier2_weight': 1.0, 'RP3beta_pow_tier3_weight': 1.0, 'S_SLIM_tier3_weight': 1.0, 'EASE_R_tier3_weight': 0.8905976575459513, 'S_SLIM_tier4_weight': 0.8191971656205302, 'EASE_R_tier4_weight': 0.0} - results: PRECISION: 0.0406584, PRECISION_RECALL_MIN_DEN: 0.0929893, RECALL: 0.0910246, MAP: 0.0191628, MAP_MIN_DEN: 0.0431771, MRR: 0.1414813, NDCG: 0.0795034, F1: 0.0562095, HIT_RATE: 0.2968239, ARHR_ALL_HITS: 0.1637821, NOVELTY: 0.0037414, AVERAGE_POPULARITY: 0.2030096, DIVERSITY_MEAN_INTER_LIST: 0.9409305, DIVERSITY_HERFINDAHL: 0.9940906, COVERAGE_ITEM: 0.1383009, COVERAGE_ITEM_HIT: 0.0485912, ITEMS_IN_GT: 0.8483624, COVERAGE_USER: 0.9325470, COVERAGE_USER_HIT: 0.2768022, USERS_IN_GT: 0.9325470, DIVERSITY_GINI: 0.0157205, SHANNON_ENTROPY: 8.8357298, RATIO_DIVERSITY_HERFINDAHL: 0.9944187, RATIO_DIVERSITY_GINI: 0.0352224, RATIO_SHANNON_ENTROPY: 0.6547374, RATIO_AVERAGE_POPULARITY: 3.9995146, RATIO_NOVELTY: 0.2306994, 
#SearchBayesianSkopt: New best config found. Config 132: {'ItemKNNCF_tier1_weight': 0.0, 'UserKNNCF_tier1_weight': 1.0, 'RP3beta_pow_tier1_weight': 1.0, 'EASE_R_tier1_weight': 0.0, 'ItemKNNCF_tier2_weight': 0.0, 'UserKNNCF_tier2_weight': 1.0, 'RP3beta_pow_tier2_weight': 0.6223036954997497, 'EASE_R_tier2_weight': 0.8876201410978622, 'RP3beta_pow_tier3_weight': 0.47317532322178224, 'S_SLIM_tier3_weight': 0.3029969887759448, 'EASE_R_tier3_weight': 0.2050864706067584, 'S_SLIM_tier4_weight': 1.0, 'EASE_R_tier4_weight': 0.0
# Create CSV for submission
f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
recommended_items_for_each_user = {}
for user_id in tqdm(target):
    recommended_items = recommender.recommend(user_id, cutoff=10, remove_seen_flag=True)
    #recommended_items=impressions.update_ranking(user_id,recommended_items,dataReader)
    recommended_items_for_each_user[int(user_id)]=recommended_items
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{user_id}, {well_formatted}\n")

# Evaluare recommended items
map=evaluate(recommended_items_for_each_user,URM_validation,target)
print('MAP score: {}'.format(map))