from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from tqdm import tqdm
from evaluator import evaluate
# from Evaluation import Evaluator
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.DataIO import DataIO

# Read & split data
dataReader = DataReader()
#urm = dataReader.load_urm()
#urm = dataReader.load_binary_urm()
#urm = dataReader.load_augmented_binary_urm()
urm = dataReader.load_powerful_binary_urm()
target = dataReader.load_target()
# dataReader.print_statistics(target)

#URM_train_v0, URM_test = split_train_in_two_percentage_global_sample(urm, train_percentage=0.90)
#URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_v0, train_percentage=0.90)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(urm, train_percentage=0.85)


evaluator_validation = EvaluatorHoldout(URM_validation, [10])
#evaluator_test = EvaluatorHoldout(URM_test, [10])


runHyperparameterSearch_Collaborative(recommender_class=SLIMElasticNetRecommender, URM_train=URM_train, URM_train_last_test=urm,n_cases=100,n_random_starts=15,
                                       evaluator_validation=evaluator_validation, evaluator_validation_earlystopping=evaluator_validation, metric_to_optimize="MAP",cutoff_to_optimize=10)

