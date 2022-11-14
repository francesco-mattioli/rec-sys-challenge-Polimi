from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from hybrid import HybridRecommender
from tqdm import tqdm
from evaluator import evaluate
#from Evaluation import Evaluator
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Evaluation.Evaluator import EvaluatorHoldout

# Read & split data
dataReader = DataReader()
#urm = dataReader.load_urm()
urm=dataReader.load_binary_urm()
target = dataReader.load_target()
#dataReader.print_statistics(target)

URM_train, URM_test = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)

#print("these are the values: "+runHyperparameterSearch_Collaborative(SLIMElasticNetRecommender,URM_train,evaluator_validation=))