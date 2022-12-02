from skopt.space import Real, Integer, Categorical

from Recommenders.SLIM.SLIMElasticNetRecommender import *
from Recommenders.GraphBased.RP3betaRecommender import *

import sys
sys.path.append("..")
from hybrid import HybridRecommender_2

names = {}
spaces = {}

names[SLIMElasticNetRecommender] = "SLIMElasticNetRecommender"
spaces[SLIMElasticNetRecommender] = [
    Real(low=0.001, high=0.01, prior='uniform', name='l1_ratio'),
    Real(low=0.001, high=0.01, prior='log-uniform',
         name='alpha'),  # old version: low=0.00001
    Integer(low=320, high=800, name='topK')
]

names[RP3betaRecommender] = "RP3betaRecommender_tree"
spaces[RP3betaRecommender] = [
    Integer(400, 1200, name="topK"),
    Real(0, 1, name='alpha'),
    Real(0, 1, name='beta'),
    Categorical([False], name="normalize_similarity"),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose", "BM25",
                "BM25-Transpose"], name='feature_weighting'),
    Real(1, 20, name='K'),
    Real(0.5, 1, name='B'),
]

'''
Remove comments form code below when we will implement HybridRecommender with weigths 
'''

names[HybridRecommender_2] = "HybridRecommender_2"
spaces[HybridRecommender_2] = [
    Real(0, 2, name="TopPopWeight"),
    Real(0, 5, name='SLIMElasticNetRecommenderWeight'),
    Categorical([True], name="normalize")
]
