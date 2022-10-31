from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Evaluation.Evaluator import EvaluatorHoldout

import matplotlib.pyplot as pyplot

class Evaluator:

    def evaluate_topK(self,URM_train, evaluator_validation):
        x_tick = [10, 50, 100, 200, 500]
        MAP_per_k = []

        for topK in x_tick:

            recommender = UserKNNCFRecommender(URM_train)
            recommender.fit(shrink=0.0, topK=topK)

            result_df, _ = evaluator_validation.evaluateRecommender(
                recommender)

            MAP_per_k.append(result_df.loc[10]["MAP"])

        pyplot.plot(x_tick, MAP_per_k)
        pyplot.ylabel('MAP')
        pyplot.xlabel('TopK')
        pyplot.show()


    def evaluator_shrinkage(self,URM_train,evaluator_validation):
        x_tick = [0, 10, 50, 100, 200, 500]
        MAP_per_shrinkage = []

        for shrink in x_tick:
            
            recommender = ItemKNNCFRecommender(URM_train)
            recommender.fit(shrink=shrink, topK=100)
            
            result_df, _ = evaluator_validation.evaluateRecommender(recommender)
            
            MAP_per_shrinkage.append(result_df.loc[10]["MAP"])
        
        pyplot.plot(x_tick, MAP_per_shrinkage)
        pyplot.ylabel('MAP')
        pyplot.xlabel('Shrinkage')
        pyplot.show()


    def evaluate(self,URM_train,URM_test,URM_validation):
        evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
        #evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
        self.evaluate_topK(URM_train,evaluator_validation)
        self.evaluator_shrinkage(URM_train,evaluator_validation)

