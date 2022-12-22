from hybrid import *
from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from tqdm import tqdm
from evaluator import evaluate
# from Evaluation import Evaluator
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender


from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.DataIO import DataIO
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
import os
# Read & split data
dataReader = DataReader()

target = dataReader.load_target()

URM = dataReader.load_augmented_binary_urm()
URM_aug,icm = dataReader.pad_with_zeros_ICMandURM(URM)

URM_train_aug, URM_validation = split_train_in_two_percentage_global_sample(URM_aug, train_percentage = 0.9)
URM_train_pow = dataReader.stackMatrixes(URM_train_aug)
UCM = dataReader.load_aug_ucm()


evaluator_validation = EvaluatorHoldout(URM_validation, [10])


# Fitting

#UserKNNCB_Hybrid = UserKNN_CFCBF_Hybrid_Recommender(URM_train_aug,UCM)
#UserKNNCB_Hybrid.fit(UCM_weight = 0.030666039949562303, topK = 374, shrink = 44, normalize = True)

UserKNNCF = UserKNNCFRecommender(URM_train_aug)
UserKNNCF.fit()

RP3beta_aug = RP3betaRecommender(URM_train_aug)
RP3beta_aug.fit(alpha=0.6951524535062256,beta=0.39985511876562174, topK=82, normalize_similarity=True)

#S_SLIM = SLIMElasticNetRecommender(URM_train_pow)
#S_SLIM.fit()

#EASE_R = EASE_R_Recommender(URM_train_aug)
#EASE_R.fit()

# End fitting


recommender_class = Hybrid_UserKNNCF_RP3B_aug

output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 200
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

#hybrid 5
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
'''


hyperparameters_range_dictionary = {
    "UserKNNCF_weight": Real(0,1),
    "RP3B_weight": Real(0,1),
}


# create a bayesian optimizer object, we pass the recommender and the evaluator
hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                           evaluator_validation=evaluator_validation)

# provide data needed to create instance of model (one on URM_train, the other on URM_all)
recommender_input_args = SearchInputRecommenderArgs(
    # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug,URM_train_pow,UserKNNCF,RP3beta_aug],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug,URM_train_pow,UserKNNCF,RP3beta_aug],
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
