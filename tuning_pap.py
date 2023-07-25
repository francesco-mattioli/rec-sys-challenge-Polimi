from hybrid import *
from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from tqdm import tqdm
from evaluator import evaluate
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from Recommenders.Custom.CustomSLIMElasticNetRecommender import CustomSLIMElasticNetRecommender
from Recommenders.Custom.CustomItemKNNCFRecommender import CustomItemKNNCFRecommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.HybridRecommenders.BaseHybridSimilarity import BaseHybridSimilarity
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.DataIO import DataIO
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
import os


############################# READ & SPLIT DATA ##############################
dataReader = DataReader()

target = dataReader.load_target()

URM = dataReader.load_augmented_binary_urm()
URM_aug, ICM = dataReader.pad_with_zeros_ICMandURM(URM)
URM_train_aug, URM_validation = split_train_in_two_percentage_global_sample(URM_aug, train_percentage=0.9)
URM_train_pow = dataReader.stackMatrixes(URM_train_aug)




evaluator_validation = EvaluatorHoldout(URM_validation, [10])


############################## FITTING ##########################################################

#UserKNNCB_Hybrid = UserKNN_CFCBF_Hybrid_Recommender(URM_train_aug,UCM)
#UserKNNCB_Hybrid.fit(UCM_weight = 0.030666039949562303, topK = 374, shrink = 44, normalize = True)

EASE_R = EASE_R_Recommender(URM_train_aug)
EASE_R.fit()

IALS = IALSRecommender(URM_train_aug)
IALS.fit(epochs = 15, num_factors = 50, confidence_scaling = 'log', alpha = 0.6349270178060594, epsilon = 0.16759381713132152, reg = 2.0856600118526794e-05)

UserKNNCF = UserKNNCFRecommender(URM_train_aug)
UserKNNCF.fit()

#ItemKNNCF = ItemKNNCFRecommender(URM_train_aug)
#ItemKNNCF.fit()

RP3beta_aug = RP3betaRecommender(URM_train_aug)
RP3beta_aug.fit()

P3alpha = P3alphaRecommender(URM_train_aug)
P3alpha.fit(topK=150, alpha=1.2040177868858861)

S_SLIM = SLIMElasticNetRecommender(URM_train_pow)
S_SLIM.fit()



#CustomSlim = CustomSLIMElasticNetRecommender(URM_train_aug)
#CustomSlim.fit(l1_ratio = 0.0001, alpha = 0.001, topK = 750, icm_weight_in_impressions = 1.0, urm_weight = 0.8555768222937054)

CustomItemKNNCF = CustomItemKNNCFRecommender(URM_train_aug)
CustomItemKNNCF.fit(topK = 66, shrink = 165.4742394926627, icm_weight_in_impressions = 0.02543754033616919, urm_weight = 0.38639914658270913)

#Basesimilarity = BaseHybridSimilarity(URM_train_aug,S_SLIM,RP3beta_aug)
#Basesimilarity.fit(topK = 803, alpha = 0.6232393364076014)




############################ TUNING ######################################################

recommender_class = Linear_Hybrid
output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 300
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

# hybrid 5
'''
hyperparameters_range_dictionary = {
    #"ItemKNNCF_tier1_weight": Real(0,1),
    "UserKNNCF_tier1_weight": Real(0,1),
    "RP3beta_pow_tier1_weight": Real(0,1),
    "EASE_R_tier1_weight": Real(0,1),
    
    #"UserKNNCB_Hybrid_tier2_weight": Real(0,0.3),
    "UserKNNCF_tier2_weight": Real(0,1),
    "RP3beta_pow_tier2_weight": Real(0,1),
    "EASE_R_tier2_weight": Real(0,1),


    #"UserKNNCB_Hybrid_tier3_weight": Real(0,0.3),
    "RP3beta_pow_tier3_weight": Real(0,1),
    "S_SLIM_tier3_weight": Real(0,1),
    "EASE_R_tier3_weight": Real(0,1),

    #"UserKNNCB_Hybrid_tier4_weight": Real(0,0.3),
    "S_SLIM_tier4_weight": Real(0,1),
    "EASE_R_tier4_weight": Real(0,1),

}

hyperparameters_range_dictionary = {
    "Hybrid_1_tier1_weight": Real(0, 1),
    "Hybrid_2_tier1_weight": Real(0, 1),

    "Hybrid_1_tier2_weight": Real(0, 1),
    "Hybrid_2_tier2_weight": Real(0, 1),

    "Hybrid_1_tier3_weight": Real(0, 1),
    "Hybrid_2_tier3_weight": Real(0, 1),
}




}
hyperparameters_range_dictionary = {
    "alpha": Real(0, 1),
}

'''
hyperparameters_range_dictionary = {
    "alpha": Categorical(np.arange(0.3,1.05,0.05).round(2).tolist()),
    "beta": Categorical(np.arange(0,1.05,0.05).round(2).tolist()),
    "teta": Categorical(np.arange(0.3,1.05,0.05).round(2).tolist()),
    "gamma": Categorical(np.arange(0,1.05,0.05).round(2).tolist()),
    "delta": Categorical(np.arange(0,1.05,0.05).round(2).tolist()),
    "sigma": Categorical(np.arange(0,1.05,0.05).round(2).tolist()),
    "rho": Categorical(np.arange(0,1.05,0.05).round(2).tolist()),


}

# create a bayesian optimizer object, we pass the recommender and the evaluator
hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                           evaluator_validation=evaluator_validation)

# provide data needed to create instance of model (one on URM_train, the other on URM_all)
recommender_input_args = SearchInputRecommenderArgs(
    # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug,S_SLIM, UserKNNCF, RP3beta_aug, CustomItemKNNCF, EASE_R, IALS, P3alpha],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_aug,S_SLIM, UserKNNCF, RP3beta_aug, CustomItemKNNCF, EASE_R, IALS, P3alpha],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)


# let's run the bayesian search
hyperparameterSearch.search(recommender_input_args,
                            recommender_input_args_last_test=recommender_input_args_last_test,
                            hyperparameter_search_space=hyperparameters_range_dictionary,
                            n_cases=n_cases,
                            n_random_starts=n_random_starts,
                            save_model="last",
                            output_folder_path=output_folder_path,  # Where to save the results
                            output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
                            metric_to_optimize=metric_to_optimize,
                            cutoff_to_optimize=cutoff_to_optimize,
                            )
