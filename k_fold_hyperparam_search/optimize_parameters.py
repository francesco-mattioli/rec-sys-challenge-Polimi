import os
import pandas as pd
import skopt
from skopt.utils import use_named_args


# Import utilities for k_fold_hyperparam_search
from k_fold_hyperparam_search.Utility import Utility
from k_fold_hyperparam_search.evaluate import evaluate_algorithm
from k_fold_hyperparam_search.hyperparam_def import names, spaces


# Import some recommenders for instantiating them properly
from hybrid import *
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender


#########################################################################################################
#########################################################################################################
#########################################################################################################
# Code to store results

output_root_path = "kfold_tuning_results/"

# If directory does not exist, create
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)


def load_df(name):
    filename = output_root_path + name + ".res"
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        return df
    else:
        return None


def read_df(name, param_names, metric="MAP"):
    df = load_df(name)

    if df is not None:
        y = df[metric].tolist()
        x_series = [df[param_name].tolist() for param_name in param_names]
        x = [t for t in zip(*x_series)]

        return x, y
    else:
        return None, None


def store_df(name, df: pd.DataFrame):
    filename = output_root_path + name + ".res"
    df.to_pickle(filename)


def append_new_data_to_df(name, new_df):
    df = load_df(name)
    df = df.append(new_df, ignore_index=True)
    store_df(name, df)


def create_df(param_tuples, param_names, value_list, metric="MAP"):
    df = pd.DataFrame(data=param_tuples, columns=param_names)
    df[metric] = value_list
    return df


#########################################################################################################
#########################################################################################################
#########################################################################################################
# Tuning - Optimization function

def optimize_parameters(recommender_class: type, n_calls=100, k=5, validation_percentage=0.05, n_random_starts=None,
                        seed=None, limit_at=1000, forest=False, xi=0.01):
    if n_random_starts is None:
        n_random_starts = int(0.5 * n_calls)

    name = names[recommender_class]
    space = spaces[recommender_class]

    utility = Utility()

    if validation_percentage > 0:
        print("Using randomized datasets. k={}, val_percentage={}".format(
            k, validation_percentage))
        URM_aug_trains, URM_pow_trains, ICM, UCM, URM_tests = utility.give_me_randomized_k_folds_with_val_percentage(
            k, validation_percentage)

    else:
        raise Exception("Validation set percentage must be grater than 0")

    if len(URM_aug_trains) > limit_at:
        URM_aug_trains = URM_aug_trains[:limit_at]
        URM_pow_trains = URM_pow_trains[:limit_at]
        URM_tests = URM_tests[:limit_at]

    assert len(URM_aug_trains) == len(URM_tests)

    print("Starting optimization: N_folds={}, Recommender={}".format(
        len(URM_aug_trains), names[recommender_class]))

    recommenders = []
    if(recommender_class == HybridRecommender_7):
        for URM_train_aug, URM_train_pow in zip(URM_aug_trains, URM_pow_trains):
            recommenders.append(recommender_class(
                URM_train_aug, URM_train_pow, UCM))

    elif(recommender_class == UserKNN_CFCBF_Hybrid_Recommender):
        for URM_train_aug in URM_aug_trains:
            recommenders.append(recommender_class(URM_train_aug, UCM))

    elif(recommender_class == ItemKNN_CFCBF_Hybrid_Recommender):
        for URM_train_pow in URM_pow_trains:
            recommenders.append(recommender_class(URM_train_pow,ICM))

    elif(recommender_class == UserKNNCFRecommender):
        for URM_train_aug in URM_aug_trains:
            recommenders.append(recommender_class(URM_train_aug))

    elif(recommender_class == SLIMElasticNetRecommender or recommender_class == RP3betaRecommender):
        for URM_train_pow in URM_pow_trains:
            recommenders.append(recommender_class(URM_train_pow))

    else:
        for URM_train_aug, URM_train_pow in zip(URM_aug_trains, URM_pow_trains):
            recommenders.append(recommender_class(
                URM_train_aug, URM_train_pow))

    @use_named_args(space)
    def objective(**params):
        scores = []
        for recommender, test in zip(recommenders, URM_tests):
            recommender.fit(**params)
            _, _, MAP = evaluate_algorithm(test, recommender)
            scores.append(MAP)
            #print("current MAP: {}".format(MAP))
            #print("current parameters: {}".format(params))

        print(">>> Just Evaluated this: {}".format(params))
        print(">>> MAP: {}, diff (= max_map - min_map): {}".format(sum(scores) /
              len(scores), max(scores) - min(scores)))
        print("\n")

        return sum(scores) / len(scores)

    param_names = [v.name for v in spaces[recommender_class]]
    xs, ys = read_df(name, param_names)

    if not forest:
        res_gp = skopt.gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            n_points=10000,
            n_jobs=1,
            # noise = 'gaussian',
            noise=1e-5,
            acq_func='gp_hedge',
            acq_optimizer='auto',
            random_state=None,
            verbose=True,
            n_restarts_optimizer=10,
            xi=xi,
            kappa=1.96,
            x0=xs,
            y0=ys,
        )
    else:
        res_gp = skopt.forest_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            verbose=True,
            x0=xs,
            y0=ys,
            acq_func="EI",
            xi=xi
        )

    print("Writing a total of {} points for {}. Newly added records: {}".format(len(res_gp.x_iters), name,
                                                                                n_calls))

    df = create_df(res_gp.x_iters, param_names, res_gp.func_vals, "MAP")
    store_df(names[recommender_class], df)

    print(name + " reached best performance = ", -res_gp.fun, " at: ", res_gp.x)
