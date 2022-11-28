#if __name__ == '__main__':
import tqdm
from Recommenders.DataIO import DataIO
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Data_Handler.DataReader import DataReader

import os
from skopt.space import Real, Integer, Categorical

# Read data
dataReader = DataReader()
#urm = dataReader.load_powerful_binary_urm()
urm = dataReader.load_augmented_binary_urm()
target = dataReader.load_target()

# Split data into train and validation data 80/20
URM_train, URM_validation = split_train_in_two_percentage_global_sample(
    urm, train_percentage=0.85)

# Create an evaluator object to evaluate validation set and use it for hyperparameter tuning
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])


# Create folder for tuning results
output_folder_path = "result_experiments/"

recommender_class = SLIMElasticNetRecommender
# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


# Configure tuning parameters
n_cases = 50
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

hyperparameters_range_dictionary = {
    "l1_ratio": Real(low=0.0001, high=0.1, prior='log-uniform'),
    "alpha": Real(low=0.001, high=0.1, prior='log-uniform'),
    "topK": Integer(450, 1000)
}
    

earlystopping_keywargs = {
                        "validation_every_n":5,
                        "stop_on_validation": True,
                        "evaluator_object": evaluator_validation,
                        "lower_validations_allowed":5,
                        "validation_metric": metric_to_optimize,
                        }


# create a bayesian optimizer object, we pass the recommender and the evaluator
hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                            evaluator_validation=evaluator_validation)


# provide data needed to create instance of model (one on URM_train, the other on URM_all)
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS = {}
    )


recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[urm],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS =  {}
)

# let's run the bayesian search
hyperparameterSearch.search(recommender_input_args=recommender_input_args,
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


# explore the results of the search
data_loader = DataIO(folder_path=output_folder_path)
search_metadata = data_loader.load_data(
    recommender_class.RECOMMENDER_NAME + "_metadata.zip")

search_metadata.keys()
hyperparameters_df = search_metadata["hyperparameters_df"]
print(hyperparameters_df)

result_on_validation_df = search_metadata["result_on_validation_df"]
print(result_on_validation_df)

best_hyperparameters = search_metadata["hyperparameters_best"]
print(best_hyperparameters)


# Run Recommender and get the best recommendations
recommender = SLIMElasticNetRecommender(urm)
recommender.fit()
recommender.save_model(output_folder_path, file_name = recommender.RECOMMENDER_NAME + "_my_own_save.zip" )

# Create CSV for submission
f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
recommended_items_for_each_user = {}
for user_id in tqdm(target):
    recommended_items = recommender.recommend(user_id, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{user_id}, {well_formatted}\n")

print('MAP score: {}'.format(map))