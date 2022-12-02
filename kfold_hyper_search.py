from k_fold_optimization.optimize_parameters import optimize_parameters

from Recommenders.SLIM.SLIMElasticNetRecommender import *
from hybrid import HybridRecommender_2


if __name__ == '__main__':
    val_percentage = 0.1
    k = 5 # old version: k=10
    limit_at = 5
    n_calls=50

    '''
    rec_class = SLIMElasticNetRecommender
    optimize_parameters(
        URMrecommender_class=rec_class,
        validation_percentage=val_percentage,
        k=k,
        n_calls=50,
        limit_at=limit_at,
        forest=True,
    )
    '''

    n_random_starts=1

    rec_class = HybridRecommender_2
    optimize_parameters(
        URMrecommender_class=rec_class,
        validation_percentage=val_percentage,
        n_random_starts=n_random_starts,
        k=k,
        n_calls=n_calls,
        limit_at=limit_at,
        forest=True,
        xi=0.001
    )