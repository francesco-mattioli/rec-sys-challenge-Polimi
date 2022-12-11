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

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.DataIO import DataIO
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
import os
# Read & split data
dataReader = DataReader()
#urm = dataReader.load_urm()
#urm = dataReader.load_binary_urm()
URM = dataReader.load_augmented_binary_urm()
#urm = dataReader.load_powerful_binary_urm(mult_param_urm=1,mult_param_icm=1)
target = dataReader.load_target()
# dataReader.print_statistics(target)

#URM_train_v0, URM_test = split_train_in_two_percentage_global_sample(urm, train_percentage=0.90)
#URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_v0, train_percentage=0.90)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM, train_percentage=0.9)

URM_validation_aug = dataReader.pad_with_zeros_ICMandURM(URM_validation)
URM_train_aug,icm = dataReader.pad_with_zeros_ICMandURM(URM_train)
URM_train_pow = dataReader.stackMatrixes(URM_train)

# Instantiate and fit hybrid recommender
#recommender = HybridRecommender_3(URM_train,ICM)
#recommender = HybridRecommender_4(URM_train_aug,URM_train_pow)

evaluator_validation = EvaluatorHoldout(URM_validation, [10])
#evaluator_test = EvaluatorHoldout(URM_test, [10])

recommender_class = HybridRecommender_4

output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 2000
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

hyperparameters_range_dictionary = {
    "UserKNNCF_tier1_weight": Real(0,1,prior='uniform'),
    "RP3beta_pow_tier1_weight": Real(0,1,prior='uniform'),
    
    "UserKNNCF_tier2_weight":Real(0,1,prior='uniform'),
    "RP3beta_pow_tier2_weight": Real(0,1,prior='uniform'),

    "RP3beta_pow_tier3_weight":Real(0,1,prior='uniform'),
    "S_SLIM_tier3_weight": Real(0,1,prior='uniform'),
}

'''
hyperparameters_range_dictionary = {
    "alpha": Real(0,1,prior='uniform'),
    "beta": Real(0,1,prior='uniform'),
    "topK": Integer(400,2000,prior='uniform'),
}
'''

# create a bayesian optimizer object, we pass the recommender and the evaluator
hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                           evaluator_validation=evaluator_validation)

# provide data needed to create instance of model (one on URM_train, the other on URM_all)
recommender_input_args = SearchInputRecommenderArgs(
    # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug,URM_train_pow],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_validation_aug,URM_train_pow],
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
