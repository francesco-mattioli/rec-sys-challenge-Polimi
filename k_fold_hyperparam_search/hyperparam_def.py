from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from Recommenders.SLIM.SLIMElasticNetRecommender import * 
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.GraphBased.RP3betaRecommender import *

from skopt.space import Real, Integer, Categorical


import sys
sys.path.append("..")
from hybrid import *

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

names[UserKNNCFRecommender] = "UserKNNCFRecommender"
spaces[UserKNNCFRecommender] = [
    Integer(1000, 10000, name='topK'),
    Real(10, 1000, name='shrink'),
    Categorical([True], name='normalize'),
    Categorical(["TF-IDF","none"], name='feature_weighting'),
    Categorical(["cosine"], name='similarity'),
]



names[SLIMElasticNetRecommender] = "SLIMElasticNetRecommender"
spaces[SLIMElasticNetRecommender] = [
    Real(low=0.001, high=0.01, prior='uniform', name='l1_ratio'),
    Real(low=0.001, high=0.01, prior='log-uniform', name='alpha'),
    Integer(low=320, high=800, name='topK')
]

names[SLIM_BPR_Cython] = "SLIM_BPR_Recommender"
spaces[SLIM_BPR_Cython] = [
    Integer(1, 400, name='topK'),
    Categorical([150], name='epochs'),  # Integer(1, 200, name="epochs")
    Real(0, 2, name='lambda_i'),
    Real(0, 2, name='lambda_j'),
    Categorical([1e-4, 1e-3, 1e-2], name="learning_rate")
]


# alpha=0.9188152746499686, beta=0.3150796458750398, min_rating=0, topK=61, implicit=False, normalize_similarity=True):
names[RP3betaRecommender] = "RP3betaRecommender"
spaces[RP3betaRecommender] = [
    Categorical([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.08,0.008,0.0008,0.1,0.01,0.001,0.0001,0.00001], name="alpha"),
    Categorical([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.08,0.008,0.0008,0.1,0.01,0.001,0.0001,0.00001], name='beta'),
    Integer(400, 1200, name="topK"),
    Categorical([True], name="normalize_similarity"),
]

names[HybridRecommender_2] = "HybridRecommender_2"
spaces[HybridRecommender_2] = [
    Real(0,1,name="RP3betaRecommenderWeight"),
    #Real(0, 1, name="ItemKNNCFRecommenderWeight"),
    Real(0, 1, name='SLIMElasticNetRecommenderWeight'),
    Categorical([True], name="normalize")
]


names[HybridRecommender_4] = "HybridRecommender_4"
spaces[HybridRecommender_4] = [
    Real(0,1,name="UserKNNCF_tier1_weight"),
    Real(0,1,name="RP3beta_aug_tier1_weight"),
    
    Real(0,1,name="UserKNNCF_tier2_weight"),
    Real(0,1,name="RP3beta_aug_tier2_weight"),

    Real(0,1,name="RP3beta_pow_tier3_weight"),
    Real(0,1,name="S_SLIM_tier3_weight"),
]



names[UserKNN_CFCBF_Hybrid_Recommender] = "UserKNN_CFCBF_Hybrid_Recommender"
spaces[UserKNN_CFCBF_Hybrid_Recommender] = [
    Real(low = 1e-3, high = 0.05,name="UCM_weight"),
    Integer(500,1000,name="topK"),
    Integer(30,200,name="shrink"),
]
