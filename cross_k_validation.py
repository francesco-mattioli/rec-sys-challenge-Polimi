import numpy as np

from hybrid import *
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender

from k_fold_hyperparam_search.Utility import Utility
from k_fold_hyperparam_search.evaluate import evaluate_algorithm
from k_fold_hyperparam_search.hyperparam_def import names, spaces


# Set parameters
n_calls = 25
k = 10
validation_percentage = 0.1
recommender_class = Hybrid_User_and_Item_KNN_CFCBF_Hybrid

utility = Utility()


print("Using randomized datasets. k={}, val_percentage={}".format(k, validation_percentage))

URM_aug_trains, URM_pow_trains, ICM, UCM, URM_tests = utility.give_me_randomized_k_folds_with_val_percentage(k, validation_percentage)

assert len(URM_aug_trains) == len(URM_tests)

print("Initalizing all k recommenders of type {}...".format(recommender_class.__name__))

recommenders = []
if(recommender_class == HybridRecommender_7):
    for URM_train_aug, URM_train_pow in zip(URM_aug_trains, URM_pow_trains):
        recommenders.append(recommender_class(
            URM_train_aug, URM_train_pow, UCM))

elif(recommender_class == UserKNN_CFCBF_Hybrid_Recommender):
    for URM_train_aug in URM_aug_trains:
        recommenders.append(recommender_class(URM_train_aug, UCM))

elif(recommender_class == ItemKNN_CFCBF_Hybrid_Recommender):
    for URM_train_pow in URM_pow_trains:
        recommenders.append(recommender_class(URM_train_pow,ICM))

elif(recommender_class == UserKNNCFRecommender):
    for URM_train_aug in URM_aug_trains:
        recommenders.append(recommender_class(URM_train_aug))

elif(recommender_class == SLIMElasticNetRecommender or recommender_class == RP3betaRecommender):
    for URM_train_pow in URM_pow_trains:
        recommenders.append(recommender_class(URM_train_pow))

elif(recommender_class==Hybrid_User_and_Item_KNN_CFCBF_Hybrid):
    for URM_train_aug, URM_train_pow in zip(URM_aug_trains, URM_pow_trains):
        _ItemKNN_CFCBF_Hybrid_Recommender = ItemKNN_CFCBF_Hybrid_Recommender(URM_train_aug,ICM)
        _ItemKNN_CFCBF_Hybrid_Recommender.fit(0.011278462705558101,topK=661,shrink=36)

        _UserKNN_CFCBF_Hybrid_Recommender = UserKNN_CFCBF_Hybrid_Recommender(URM_train_aug,UCM)
        _UserKNN_CFCBF_Hybrid_Recommender.fit(0.01,topK=669,shrink=50)

        recommenders.append(recommender_class(URM_train_aug,URM_train_pow,_ItemKNN_CFCBF_Hybrid_Recommender,_UserKNN_CFCBF_Hybrid_Recommender))

else:
    for URM_train_aug, URM_train_pow in zip(URM_aug_trains, URM_pow_trains):
        recommenders.append(recommender_class(
            URM_train_aug, URM_train_pow))

print("Finished initialization!\nStarting fitting and evaluating...")

scores = []
count=1
for recommender, test in zip(recommenders, URM_tests):
    recommender.fit(ItemKNN_CFCBF_Hybrid_Recommender_weight=0.1643020180322596, UserKNN_CFCBF_Hybrid_Recommender_weight= 0.1181324258536363)
    _, _, MAP = evaluate_algorithm(test, recommender)
    scores.append(MAP)
    print("Evaluated fold number: {}".format(count))
    count=count+1

print("Finished fit & evaluation!")

print(">>> Average MAP: {}, DIFF (= max_map - min_map): {}".format(sum(scores) / len(scores), max(scores) - min(scores)))
