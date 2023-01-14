from hybrid import *
from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from tqdm import tqdm
from evaluator import evaluate
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.Custom.CustomItemKNNCFRecommender import CustomItemKNNCFRecommender
from Recommenders.Custom.CustomSLIMElasticNetRecommender import CustomSLIMElasticNetRecommender
from Recommenders.Custom.CustomUserKNNCFRecommender import CustomUserKNNCFRecommender
from Recommenders.Custom.CustomRP3betaRecommender import CustomRP3betaRecommender

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.DataIO import DataIO
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
import os


############################# READ & SPLIT DATA ##############################
dataReader = DataReader()

target = dataReader.load_target()

URM = dataReader.load_augmented_binary_urm()
URM_aug, ICM = dataReader.pad_with_zeros_ICMandURM(URM)
URM_train_aug, URM_validation = split_train_in_two_percentage_global_sample(URM_aug, train_percentage=0.9)

evaluator_validation = EvaluatorHoldout(URM_validation, [10])


############################ TUNING ######################################################

recommender_class = CustomRP3betaRecommender
#recommender_class = CustomUserKNNCFRecommender
#recommender_class = CustomItemKNNCFRecommender
#recommender_class = CustomSLIMElasticNetRecommender
output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 1000
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10


# For CustomSLIMElasticNet
'''
hyperparameters_range_dictionary = {
    "l1_ratio" : Categorical([0.1,0.01]),
    "alpha" : Categorical([0.1,0.01]),
    "topK": Integer(500,700),
    "icm_weight_in_impressions": Real(0, 1),
    "urm_weight": Real(0, 1),
}
'''

# For CustomItemKNN
hyperparameters_range_dictionary = {
    "topK": Integer(1000,2000),
    "alpha": Real(0,1),
    "beta": Real(0,1),
    "icm_weight_in_impressions": Real(0, 1),
    "urm_weight": Real(0, 1),
}


# create a bayesian optimizer object, we pass the recommender and the evaluator
hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                           evaluator_validation=evaluator_validation)

# provide data needed to create instance of model (one on URM_train, the other on URM_all)
recommender_input_args = SearchInputRecommenderArgs(
    # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug],
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
