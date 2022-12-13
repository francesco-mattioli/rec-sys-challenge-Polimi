from hybrid import *
from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from tqdm import tqdm
from evaluator import evaluate
# from Evaluation import Evaluator
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.DataIO import DataIO
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
import os
# Read & split data
dataReader = DataReader()

target = dataReader.load_target()

URM_aug= dataReader.load_augmented_binary_urm()
#URM_aug,ICM = dataReader.pad_with_zeros_ICMandURM(URM)

URM_train_aug, URM_validation = split_train_in_two_percentage_global_sample(URM_aug, train_percentage = 0.9)
#URM_train_pow = dataReader.stackMatrixes(URM_train_aug)

evaluator_validation = EvaluatorHoldout(URM_validation, [10])


recommender_class = EASE_R_Recommender

output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 2000
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

hyperparameters_range_dictionary = {
    "topK": Integer(200,1000),
    "l2_norm": Integer(50,1000),
    #"l2_norm": Categorical([70,80,90,100,200,300,1000,1500,2000,3000,10000]),
    #"topK": Integer(10,2000,prior='uniform'),
    #"shrink": Real(100,3000,prior='uniform'),
    #"similarity": Categorical(["euclidean",'dice', 'jaccard', 'tversky','cosine']),
    #"feature_weighting": Categorical(['BM25',"TF-IDF", "none"]),
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
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_validation],
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
