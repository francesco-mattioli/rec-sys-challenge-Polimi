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
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMItemHybridRecommender
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.DataIO import DataIO
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
import os

########################## READ & SPLIT DATA ##########################
dataReader = DataReader()
target = dataReader.load_target()

URM = dataReader.load_augmented_binary_urm()
URM_aug, ICM = dataReader.pad_with_zeros_ICMandURM(URM)
URM_train_aug, URM_validation = split_train_in_two_percentage_global_sample(URM_aug, train_percentage=0.9)

URM_train_pow = dataReader.stackMatrixes(URM_train_aug)

ICM_stacked_with_binary_impressions = dataReader.load_ICM_stacked_with_binary_impressions(0.8)

URM_train_pow_padded, ICM_stacked_with_binary_impressions_padded = dataReader.pad_with_zeros_given_ICMandURM(ICM_stacked_with_binary_impressions, URM_train_pow)

URM_train_super_pow = dataReader.load_super_powerful_URM(URM_train_pow_padded, ICM_stacked_with_binary_impressions_padded, 0.8)

evaluator_validation = EvaluatorHoldout(URM_validation, [10])

'''
RP3beta_aug = RP3betaRecommender(URM_train_aug)
RP3beta_aug.fit()

S_SLIM = SLIMElasticNetRecommender(URM_train_super_pow)
S_SLIM.fit(l1_ratio=0.006011021694075882,alpha=0.0013369897413235414, topK=459)


UserKNNCF = UserKNNCFRecommender(URM_train_aug)
UserKNNCF.fit()

Hybrid_SSLIM_RP3B_aug = Hybrid_SSLIM_RP3B_aug(URM_train_aug, S_SLIM, RP3beta_aug)
Hybrid_SSLIM_RP3B_aug.fit(alpha = 0.7447123958484749)
'''

recommender_class = LightFMItemHybridRecommender

output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 200
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10
'''
hyperparameters_range_dictionary = {
    "l1_ratio": Categorical([0.1,0.01,0.001,0.0001,0.00001]),
    "alpha": Categorical([0.1,0.01,0.001,0.0001,0.00001]),
    "topK": Integer(100, 1000),

}
'''
hyperparameters_range_dictionary = {
    "epochs":Integer(20,300),
    "loss": Categorical(['bpr', 'warp', 'warp-kos']), 
    "sgd_mode": Categorical(['adagrad', 'adadelta']),
    "n_components": Integer(5,150),
    "item_alpha": Categorical([0.1,0.01,0.001,0.0001,0.00001]),
    "user_alpha": Categorical([0.1,0.01,0.001,0.0001,0.00001]),
    "learning_rate": Categorical([0.1,0.01,0.001,0.0001,0.00001])

}


earlystopping_keywargs = {"validation_every_n": 5,
                          "stop_on_validation": True,
                          "evaluator_object": evaluator_validation,
                          "lower_validations_allowed": 5,
                          "validation_metric": metric_to_optimize,
                          }

# create a bayesian optimizer object, we pass the recommender and the evaluator
hyperparameterSearch = SearchBayesianSkopt(
    recommender_class, evaluator_validation=evaluator_validation)

# provide data needed to create instance of model (one on URM_train, the other on URM_all)
recommender_input_args = SearchInputRecommenderArgs(
    # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug, ICM_stacked_with_binary_impressions_padded],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS=earlystopping_keywargs,
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug,ICM_stacked_with_binary_impressions_padded],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS=earlystopping_keywargs,
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
