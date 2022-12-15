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

#ItemKNNCF = ItemKNNCFRecommender(URM_train_pow)
#ItemKNNCF.fit()

UserKNNCF = UserKNNCFRecommender(URM_train_aug)
UserKNNCF.fit()

RP3beta_pow = RP3betaRecommender(URM_train_pow)
RP3beta_pow.fit(alpha=0.3648761546066018,beta=0.5058870363874656, topK=480, normalize_similarity=True)

S_SLIM = SLIMElasticNetRecommender(URM_train_pow)
S_SLIM.fit()

EASE_R = EASE_R_Recommender(URM_train_aug)
EASE_R.fit()


# Instantiate and fit hybrid recommender
recommender = HybridRecommender_5(URM_train_aug, URM_train_pow, UserKNNCF, RP3beta_pow, S_SLIM, EASE_R)
#recommender = HybridRecommender(URM_train)
recommender.fit(UserKNNCF_tier1_weight=0.5729283925580592, RP3beta_pow_tier1_weight= 0.5126995921400189, EASE_R_tier1_weight=0.35135124299835835, UserKNNCF_tier2_weight=0.9573738859277221, RP3beta_pow_tier2_weight=0.8130000344620287, EASE_R_tier2_weight=0.8407574723860113, RP3beta_pow_tier3_weight=0.8153126010122613, S_SLIM_tier3_weight=0.39864407118121825, EASE_R_tier3_weight=0.8181461648227059, S_SLIM_tier4_weight= 0.3513706238839294, EASE_R_tier4_weight= 0.009916836090967942)
#'UserKNNCF_tier1_weight': 0.5729283925580592, 'RP3beta_pow_tier1_weight': 0.5126995921400189, 'EASE_R_tier1_weight': 0.35135124299835835, 'UserKNNCF_tier2_weight': 0.9573738859277221, 'RP3beta_pow_tier2_weight': 0.8130000344620287, 'EASE_R_tier2_weight': 0.8407574723860113, 'RP3beta_pow_tier3_weight': 0.8153126010122613, 'S_SLIM_tier3_weight': 0.39864407118121825, 'EASE_R_tier3_weight': 0.8181461648227059, 'S_SLIM_tier4_weight': 0.3513706238839294, 'EASE_R_tier4_weight': 0.009916836090967942
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