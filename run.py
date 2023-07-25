from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from impressions import Impressions
from hybrid import *
from tqdm import tqdm
from evaluator import evaluate
import pandas as pd
import numpy as np
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Evaluation.Evaluator import EvaluatorHoldout

############################# READ & SPLIT DATA ##############################
dataReader = DataReader()

target = dataReader.load_target()

UCM = dataReader.load_aug_ucm()
URM = dataReader.load_augmented_binary_urm()
URM_aug, ICM = dataReader.pad_with_zeros_ICMandURM(URM)
URM_train_aug, URM_validation = split_train_in_two_percentage_global_sample(URM_aug, train_percentage=0.9)

URM_train_pow = dataReader.stackMatrixes(URM_train_aug)

ICM_stacked_with_weighted_impressions = dataReader.load_ICM_stacked_with_weighted_impressions(0)

URM_train_pow_padded, ICM_stacked_with_weighted_impressions_padded = dataReader.pad_with_zeros_given_ICMandURM(ICM_stacked_with_weighted_impressions, URM_train_pow)

URM_train_super_pow = dataReader.load_super_powerful_URM(URM_train_pow_padded, ICM_stacked_with_weighted_impressions_padded, 0.8)

evaluator_validation = EvaluatorHoldout(URM_validation, [10])

########################## iNSTANTIATE & FIT SINGLE MODELS ##########################

EASE_R = EASE_R_Recommender(URM_train_aug)
EASE_R.fit()

UserKNNCF = UserKNNCFRecommender(URM_train_aug)
UserKNNCF.fit()

ItemKNNCF = ItemKNNCFRecommender(URM_train_aug)
ItemKNNCF.fit()

RP3beta_aug = RP3betaRecommender(URM_train_aug)
RP3beta_aug.fit()

S_SLIM = SLIMElasticNetRecommender(URM_train_pow)
S_SLIM.fit()

S_SLIM_only_weighted_impressions = SLIMElasticNetRecommender(URM_train_super_pow)
S_SLIM_only_weighted_impressions.fit(l1_ratio= 0.02655220236250845, alpha= 0.0009855880367693063, topK=603)

########################## INSTANTIATE & FIT FIRST STEP HYBRID MODELS ##########################

Hybrid_SSLIM_RP3B_aug = Hybrid_SSLIM_RP3B_aug(URM_train_aug, S_SLIM, RP3beta_aug)
Hybrid_SSLIM_RP3B_aug.fit(alpha = 0.7447123958484749)

Hybrid_006022 = Hybrid_006022(URM_train_aug, URM_train_pow, ICM, UCM, Hybrid_SSLIM_RP3B_aug, UserKNNCF)
Hybrid_006022.fit(Hybrid_1_tier1_weight= 0.4730071105820606, Hybrid_2_tier1_weight= 1.0, Hybrid_1_tier2_weight= 1.0, Hybrid_2_tier2_weight= 1.0, Hybrid_1_tier3_weight=1.0)

Linear_Hybrid_1 = Linear_Hybrid(URM_train_aug,Hybrid_006022,EASE_R)
Linear_Hybrid_1.fit(norm= 2, alpha= 0.8845750718247858)

########################## INSTANTIATE & FIT FINAL HYBIRD MODEL ##########################

recommender = Linear_Hybrid(URM_train_aug,Linear_Hybrid_1,S_SLIM_only_weighted_impressions)
recommender.fit(norm= 2, alpha= 1.0)
            
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
