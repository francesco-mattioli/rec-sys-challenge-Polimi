from skopt.space import Real, Integer, Categorical

from Recommenders.SLIM.SLIMElasticNetRecommender import *

names = {}
spaces = {}

names[SLIMElasticNetRecommender] = "SLIMElasticNetRecommender"
spaces[SLIMElasticNetRecommender] = [
    Real(low=0.0001, high=0.1, prior='uniform', name='l1_ratio'),
    Real(low=0.00001, high=0.1, prior='log-uniform', name='alpha'),
    Integer(low=200, high=800, name='topK')
]
