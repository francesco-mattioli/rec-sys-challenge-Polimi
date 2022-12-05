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
    Integer(1000, 10000, name='topK'),
    Real(10, 1000, name='shrink'),
    Categorical([True], name='normalize'),
    Categorical(["TF-IDF"], name='feature_weighting'),
    Categorical(["cosine"], name='similarity'),
]


names[SLIMElasticNetRecommender] = "SLIMElasticNetRecommender"
spaces[SLIMElasticNetRecommender] = [
    Real(low=0.001, high=0.01, prior='uniform', name='l1_ratio'),
    Real(low=0.001, high=0.01, prior='log-uniform', name='alpha'),
    Integer(low=320, high=800, name='topK')
]

# alpha=0.9188152746499686, beta=0.3150796458750398, min_rating=0, topK=61, implicit=False, normalize_similarity=True):
names[RP3betaRecommender] = "RP3betaRecommender_tree"
spaces[RP3betaRecommender] = [
    Real(0, 1, name='alpha'),
    Real(0, 1, name='beta'),
    Integer(400, 1200, name="topK"),
    Categorical([False,True], name="normalize_similarity"),
]

names[HybridRecommender_2] = "HybridRecommender_2"
spaces[HybridRecommender_2] = [
    #Real(0, 2, name="UserKNNCFRecommenderWeight"),
    Real(0, 2, name="ItemKNNCFRecommenderWeight"),
    Real(0, 2, name='SLIMElasticNetRecommenderWeight'),
    Categorical([True], name="normalize")
]
