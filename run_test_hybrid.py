from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from hybrid import HybridRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from tqdm import tqdm
from evaluator import evaluate
from Evaluation.Evaluator import EvaluatorHoldout
import pandas as pd
import numpy as np
# Read & split data
dataReader = DataReader()
#urm = dataReader.load_urm()
#urm = dataReader.load_binary_urm()
#urm = dataReader.load_augmented_binary_urm()
#urm = dataReader.load_powerful_binary_urm()
urm = dataReader.load_augmented_binary_urm()
#urm_df=dataReader.load_powerful_binary_urm_df()
#urm = dataReader.load_powerful_binary_urm()
'''
urm = dataReader.load_augmented_binary_urm_less_items()
icm = dataReader.load_augmented_binary_icm_less_items()
'''
target = dataReader.load_target()
#dataReader.print_statistics(target)
#URM_train, URM_test = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.90)
#URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.90)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.85)
evaluator_validation=EvaluatorHoldout(URM_validation, cutoff_list=[10])
# Instantiate and fit hybrid recommender
recommender_1 = RP3betaRecommender(URM_train)
recommender_1.fit(alpha=0.9188152746499686, beta =  0.3150796458750398, topK = 61, implicit=False)
recommender_2 = SLIMElasticNetRecommender(URM_train)
recommender_2.fit(l1_ratio=0.004255000410494794, alpha = 0.003186578950522464, positive_only=True, topK = 319)

recommender = HybridRecommender(URM_train, recommender_1, recommender_2)

for norm in [1, 2, np.inf, -np.inf]:

    recommender.fit(norm, alpha = 0.66)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender)
    print("Norm: {}, Result: {}".format(norm, result_df.loc[10]["MAP"]))

