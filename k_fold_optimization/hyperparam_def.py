from skopt.space import Real, Integer, Categorical

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import *
from Recommenders.GraphBased.RP3betaRecommender import *

import sys
sys.path.append("..")
from hybrid import HybridRecommender_2

names = {}
spaces = {}

names[ItemKNNCFRecommender] = "ItemKNNCFRecommender"
spaces[ItemKNNCFRecommender] = [
    Integer(0, 10000, name='topK'),
    Real(0, 1000, name='shrink'),
    Categorical([True], name='normalize'),
    Categorical([None, "TF-IDF", "BM25"], name='feature_weighting'),
    Categorical(["cosine", "tanimoto", "dice"], name='similarity'),
]


names[SLIMElasticNetRecommender] = "SLIMElasticNetRecommender"
spaces[SLIMElasticNetRecommender] = [
    Real(low=0.001, high=0.01, prior='uniform', name='l1_ratio'),
    Real(low=0.001, high=0.01, prior='log-uniform', name='alpha'),
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

names[HybridRecommender_2] = "HybridRecommender_2"
spaces[HybridRecommender_2] = [
    #Real(0, 2, name="UserKNNCFRecommenderWeight"),
    Real(0, 2, name="ItemKNNCFRecommenderWeight"),
    Real(0, 2, name='SLIMElasticNetRecommenderWeight'),
    Categorical([True], name="normalize")
]
