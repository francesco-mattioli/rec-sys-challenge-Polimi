
if __name__ == '__main__':
    ################################## IMPORTS ##################################

    from skopt.space import Real, Integer, Categorical
    from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
    from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Data_Handler.DataReader import DataReader

    import os

    # Model to be tuned
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender



    ################################# READ DATA #################################

    reader = DataReader()
    urm = reader.load_powerful_binary_urm()
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(urm, train_percentage=0.90)

    ################################ EVALUATORS ##################################

    evaluator_validation = EvaluatorHoldout(URM_validation, [10])

    ############################### OPTIMIZER SETUP ###############################

    recommender_class = MultiThreadSLIM_SLIMElasticNetRecommender
    parameterSearch = SearchBayesianSkopt(recommender_class,
                                        evaluator_validation=evaluator_validation)
    hyperparameters_range_dictionary = {
        "l1_ratio": Real(low=0.0001, high=0.1, prior='log-uniform'),
        "alpha": Real(low=0.00001, high=0.1, prior='log-uniform'),
        "topK": Integer(200,800)
    }

    '''
    Insert here the hyperparameters to be tuned.
    These hyperparameters should correspond to the parameters of the fit function
    of the model to be tuned
    '''

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {}
    )

    output_folder_path = "result_optimizer/"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    parameterSearch.search(recommender_input_args,
                        hyperparameter_search_space = hyperparameters_range_dictionary,
                        n_cases = 200,
                        n_random_starts = 20,
                        save_model="no",
                        output_folder_path = output_folder_path,
                        output_file_name_root = recommender_class.RECOMMENDER_NAME,
                        metric_to_optimize = "MAP")