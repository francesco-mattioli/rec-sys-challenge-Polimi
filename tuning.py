from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from tqdm import tqdm
from evaluator import evaluate
# from Evaluation import Evaluator
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
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
urm = dataReader.load_augmented_binary_urm()
#urm = dataReader.load_powerful_binary_urm()
target = dataReader.load_target()
# dataReader.print_statistics(target)

#URM_train_v0, URM_test = split_train_in_two_percentage_global_sample(urm, train_percentage=0.90)
#URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_v0, train_percentage=0.90)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(urm, train_percentage=0.85)


evaluator_validation = EvaluatorHoldout(URM_validation, [10])
#evaluator_test = EvaluatorHoldout(URM_test, [10])

recommender_class = RP3betaRecommender

output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    
n_cases = 100
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"   
cutoff_to_optimize = 10

hyperparameters_range_dictionary = {
    "alpha": Real(low=0.1, high=0.9, prior='uniform'),
    "beta": Real(low=0.1, high=0.9, prior='uniform'),
    "topK": Integer(10, 500),
    "implicit": Categorical([True, False])
}


#create a bayesian optimizer object, we pass the recommender and the evaluator
hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation)

#provide data needed to create instance of model (one on URM_train, the other on URM_all)
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {},
    EARLYSTOPPING_KEYWORD_ARGS = {},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [urm],
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {},
    EARLYSTOPPING_KEYWORD_ARGS = {},
)


#let's run the bayesian search
hyperparameterSearch.search(recommender_input_args,
                       recommender_input_args_last_test = recommender_input_args_last_test,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       save_model = "last",
                       output_folder_path = output_folder_path, # Where to save the results
                       output_file_name_root = recommender_class.RECOMMENDER_NAME, # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )