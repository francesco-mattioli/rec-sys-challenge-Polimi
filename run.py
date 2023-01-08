from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from impressions import Impressions
from hybrid import *
from tqdm import tqdm
from evaluator import evaluate
import pandas as pd
import numpy as np
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender


########################## READ & SPLIT DATA ##########################
dataReader = DataReader()

target = dataReader.load_target()

UCM = dataReader.load_aug_ucm()
URM = dataReader.load_augmented_binary_urm()
URM_aug, ICM = dataReader.pad_with_zeros_ICMandURM(URM)


URM_train_aug, URM_validation = split_train_in_two_percentage_global_sample( URM_aug, train_percentage=1.0)
#URM_train_super_pow = dataReader.stackMatrixes_with_impressions(URM_train_aug)
URM_train_pow = dataReader.stackMatrixes(URM_train_aug)

# Instantiate Impressions object to update ranking at the end of recommendations
#item_ids = dataReader.get_unique_items_based_on_urm(dataReader.load_augmented_binary_urm_df())
#impressions = Impressions(target,item_ids)
# dataReader.save_impressions()

########################## iNSTANTIATE & FIT SINGLE MODELS ##########################



RP3beta_aug = RP3betaRecommender(URM_train_aug)
RP3beta_aug.fit()

ItemKNNCF = ItemKNNCFRecommender(URM_train_pow)
ItemKNNCF.fit()

#RP3beta_pow = RP3betaRecommender(URM_train_pow)
#RP3beta_pow.fit(alpha=0.3648761546066018,beta=0.5058870363874656, topK=480, normalize_similarity=True)

S_SLIM = SLIMElasticNetRecommender(URM_train_pow)
S_SLIM.fit()

EASE_R = EASE_R_Recommender(URM_train_aug)
EASE_R.fit()

UserKNNCF = UserKNNCFRecommender(URM_train_aug)
UserKNNCF.fit()
'''
UserKNN_CFCBF_Hybrid_Recommender = UserKNN_CFCBF_Hybrid_Recommender(
    URM_train_aug, UCM)
UserKNN_CFCBF_Hybrid_Recommender.fit()

ItemKNN_CFCBF_Hybrid_Recommender = ItemKNN_CFCBF_Hybrid_Recommender(
    URM_train_aug, ICM)
ItemKNN_CFCBF_Hybrid_Recommender.fit()

'''


##########################################################################################################



'''
Hybrid_UserKNNCF_RP3B_aug = Hybrid_UserKNNCF_RP3B_aug(
    URM_train_aug, URM_train_pow, UserKNNCF, RP3beta_aug)
Hybrid_UserKNNCF_RP3B_aug.fit(
    UserKNNCF_weight=0.4348857237366932, RP3B_weight=0.027648314372221712)

Hybrid_SSLIM_EASER = Hybrid_SSLIM_EASER(
    URM_train_aug, URM_train_pow, S_SLIM, EASE_R)
Hybrid_SSLIM_EASER.fit(SSLIM_weight=0.5495139584252299, EASE_R_weight=0.0)

Hybrid_SSLIM_RP3B_aug = Hybrid_SSLIM_RP3B_aug(
    URM_train_aug, URM_train_pow, S_SLIM, RP3beta_aug)
Hybrid_SSLIM_RP3B_aug.fit(
    SSLIM_weight=0.8157521052599057, RP3B_weight=0.22946157569349823)


Hybrid_UserKNNCF_ItemKNNCF = Hybrid_UserKNNCF_ItemKNNCF(
    URM_train_aug, URM_train_pow, UserKNNCF, ItemKNNCF)
Hybrid_UserKNNCF_ItemKNNCF.fit(
    UserKNNCF_weight=1.0, ItemKNNCF_weight=0.8072073132929845)

Hybrid_Best = Hybrid_Best(URM_train_aug, URM_train_pow, ICM, UCM, Hybrid_SSLIM_RP3B_aug, Hybrid_UserKNNCF_ItemKNNCF, UserKNNCF, Hybrid_UserKNNCF_RP3B_aug, Hybrid_SSLIM_EASER)
Hybrid_Best.fit(Hybrid_1_tier1_weight= 0.5960289190957877, Hybrid_2_tier1_weight= 1.0, Hybrid_1_tier2_weight= 1.0, Hybrid_2_tier2_weight= 0.0, Hybrid_1_tier3_weight=0.4001445272204769, Hybrid_2_tier3_weight= 0.6909775763230392)

'''


'''
Hybrid_User_and_Item_KNN_CFCBF_Hybrid = Hybrid_User_and_Item_KNN_CFCBF_Hybrid(
    URM_train_aug, URM_train_pow, ItemKNN_CFCBF_Hybrid_Recommender, UserKNN_CFCBF_Hybrid_Recommender)
Hybrid_User_and_Item_KNN_CFCBF_Hybrid.fit()
'''

########################## INSTANTIATE & FIT FINAL HYBIRD MODEL ##########################

recommender = Linear_Hybrid(URM_train_aug, S_SLIM, RP3beta_aug, UserKNNCF, EASE_R)
recommender.fit(norm = 2, alpha = 0.4593383290865466, beta = 0.3392318459607915, gamma = 0.0)
########################## CREATE CSV FOR SUBMISISON ##########################
f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
recommended_items_for_each_user = {}
for user_id in tqdm(target):
    recommended_items = recommender.recommend(
        user_id, cutoff=10, remove_seen_flag=True)
    # recommended_items=impressions.update_ranking(user_id,recommended_items,dataReader)
    recommended_items_for_each_user[int(user_id)] = recommended_items
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{user_id}, {well_formatted}\n")

# Evaluate recommended items
map = evaluate(recommended_items_for_each_user, URM_validation, target)
print('MAP score: {}'.format(map))