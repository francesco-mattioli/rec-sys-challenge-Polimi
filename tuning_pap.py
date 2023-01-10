from hybrid import *
from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from tqdm import tqdm
from evaluator import evaluate
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.DataIO import DataIO
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
import os


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


############################## FITTING ##########################################################

#UserKNNCB_Hybrid = UserKNN_CFCBF_Hybrid_Recommender(URM_train_aug,UCM)
#UserKNNCB_Hybrid.fit(UCM_weight = 0.030666039949562303, topK = 374, shrink = 44, normalize = True)

EASE_R = EASE_R_Recommender(URM_train_aug)
EASE_R.fit()

UserKNNCF = UserKNNCFRecommender(URM_train_aug)
UserKNNCF.fit()

ItemKNNCF = ItemKNNCFRecommender(URM_train_aug)
ItemKNNCF.fit()

RP3beta_aug = RP3betaRecommender(URM_train_aug)
RP3beta_aug.fit()

#RP3beta_pow = RP3betaRecommender(URM_train_pow)
#RP3beta_pow.fit(alpha=0.3648761546066018,beta=0.5058870363874656, topK=480, normalize_similarity=True)

S_SLIM = SLIMElasticNetRecommender(URM_train_pow)
S_SLIM.fit()

S_SLIM_only_weighted_impressions = SLIMElasticNetRecommender(URM_train_super_pow)
S_SLIM_only_weighted_impressions.fit(l1_ratio= 0.02655220236250845, alpha= 0.0009855880367693063, topK=603)




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
    URM_train_aug, S_SLIM, RP3beta_aug)
Hybrid_SSLIM_RP3B_aug.fit(
    SSLIM_weight=0.8157521052599057, RP3B_weight=0.22946157569349823)
'''
Hybrid_SSLIM_RP3B_aug = Hybrid_SSLIM_RP3B_aug(
    URM_train_aug, S_SLIM, RP3beta_aug)
Hybrid_SSLIM_RP3B_aug.fit(alpha = 0.7447123958484749)

'''
Hybrid_UserKNNCF_ItemKNNCF = Hybrid_UserKNNCF_ItemKNNCF(
    URM_train_aug, URM_train_pow, UserKNNCF, ItemKNNCF)
Hybrid_UserKNNCF_ItemKNNCF.fit(
    UserKNNCF_weight=1.0, ItemKNNCF_weight=0.8072073132929845)

Hybrid_User_and_Item_KNN_CFCBF_Hybrid = Hybrid_User_and_Item_KNN_CFCBF_Hybrid(
    URM_train_aug, URM_train_pow, ItemKNN_CFCBF_Hybrid_Recommender, UserKNN_CFCBF_Hybrid_Recommender)
Hybrid_User_and_Item_KNN_CFCBF_Hybrid.fit()

Hybrid_Best = Hybrid_Best(URM_train_aug, URM_train_pow, ICM, UCM, Hybrid_SSLIM_RP3B_aug,
                                 Hybrid_UserKNNCF_ItemKNNCF, UserKNNCF, Hybrid_UserKNNCF_RP3B_aug, Hybrid_SSLIM_EASER)
Hybrid_Best.fit(Hybrid_1_tier1_weight= 0.5960289190957877, Hybrid_2_tier1_weight= 1.0, Hybrid_1_tier2_weight= 1.0, Hybrid_2_tier2_weight= 0.0, Hybrid_1_tier3_weight=0.4001445272204769, Hybrid_2_tier3_weight= 0.6909775763230392)
#HybridRecommender_5 = HybridRecommender_5(URM_train_aug, URM_train_pow, UserKNNCF, RP3beta_pow, S_SLIM, EASE_R)
# HybridRecommender_5.fit()
'''

Hybrid_006022 = Hybrid_006022(URM_train_aug, URM_train_pow, ICM, UCM, Hybrid_SSLIM_RP3B_aug, UserKNNCF)
Hybrid_006022.fit(Hybrid_1_tier1_weight= 0.4730071105820606, Hybrid_2_tier1_weight= 1.0, Hybrid_1_tier2_weight= 1.0, Hybrid_2_tier2_weight= 1.0, Hybrid_1_tier3_weight=1.0)

Linear_Hybrid_1 = Linear_Hybrid(URM_train_aug,Hybrid_006022,EASE_R)
Linear_Hybrid_1.fit(norm= 2, alpha= 0.8845750718247858)

############################ TUNING ######################################################

recommender_class = Linear_Hybrid
output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 150
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

# hybrid 5
'''
hyperparameters_range_dictionary = {
    #"ItemKNNCF_tier1_weight": Real(0,1),
    "UserKNNCF_tier1_weight": Real(0,1),
    "RP3beta_pow_tier1_weight": Real(0,1),
    "EASE_R_tier1_weight": Real(0,1),
    
    #"UserKNNCB_Hybrid_tier2_weight": Real(0,0.3),
    "UserKNNCF_tier2_weight": Real(0,1),
    "RP3beta_pow_tier2_weight": Real(0,1),
    "EASE_R_tier2_weight": Real(0,1),


    #"UserKNNCB_Hybrid_tier3_weight": Real(0,0.3),
    "RP3beta_pow_tier3_weight": Real(0,1),
    "S_SLIM_tier3_weight": Real(0,1),
    "EASE_R_tier3_weight": Real(0,1),

    #"UserKNNCB_Hybrid_tier4_weight": Real(0,0.3),
    "S_SLIM_tier4_weight": Real(0,1),
    "EASE_R_tier4_weight": Real(0,1),

}

hyperparameters_range_dictionary = {
    "Hybrid_1_tier1_weight": Real(0, 1),
    "Hybrid_2_tier1_weight": Real(0, 1),

    "Hybrid_1_tier2_weight": Real(0, 1),
    "Hybrid_2_tier2_weight": Real(0, 1),

    "Hybrid_1_tier3_weight": Real(0, 1),
    "Hybrid_2_tier3_weight": Real(0, 1),
}
'''

hyperparameters_range_dictionary = {
    "norm": Categorical([1,2]),
    "alpha": Real(0, 1),
}


# create a bayesian optimizer object, we pass the recommender and the evaluator
hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                           evaluator_validation=evaluator_validation)

# provide data needed to create instance of model (one on URM_train, the other on URM_all)
recommender_input_args = SearchInputRecommenderArgs(
    # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug,Linear_Hybrid_1,S_SLIM_only_weighted_impressions],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug,Linear_Hybrid_1,S_SLIM_only_weighted_impressions],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)


# let's run the bayesian search
hyperparameterSearch.search(recommender_input_args,
                            recommender_input_args_last_test=recommender_input_args_last_test,
                            hyperparameter_search_space=hyperparameters_range_dictionary,
                            n_cases=n_cases,
                            n_random_starts=n_random_starts,
                            save_model="last",
                            output_folder_path=output_folder_path,  # Where to save the results
                            output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
                            metric_to_optimize=metric_to_optimize,
                            cutoff_to_optimize=cutoff_to_optimize,
                            )
