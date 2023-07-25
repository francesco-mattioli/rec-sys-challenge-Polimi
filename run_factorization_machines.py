from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from hybrid import HybridRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMItemHybridRecommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
from tqdm import tqdm
from evaluator import evaluate
from Evaluation.Evaluator import EvaluatorHoldout
# Read & split data
dataReader = DataReader()
#urm = dataReader.load_urm()
#urm = dataReader.load_binary_urm()
urm = dataReader.load_augmented_binary_urm()
#icm = dataReader.load_icm()
urm_pad, icm_pad = dataReader.paddingICMandURM(urm)
#urm = dataReader.load_powerful_binary_urm()
target = dataReader.load_target()
#dataReader.print_statistics(target)

#URM_train, URM_test = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(urm_pad, train_percentage = 0.9)
# Instantiate and fit hybrid recommender
recommender = LightFMItemHybridRecommender(URM_train, icm_pad)
recommender.fit(epochs=50)
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
result_df, _ = evaluator_validation.evaluateRecommender(recommender)
print(result_df)
