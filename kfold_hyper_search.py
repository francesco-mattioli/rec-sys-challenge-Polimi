from Recommenders.SLIM.SLIMElasticNetRecommender import *
from k_fold_optimization.optimize_parameters import optimize_parameters

if __name__ == '__main__':
    val_percentage = 0.1
    k = 10
    limit_at = 10

    rec_class = SLIMElasticNetRecommender
    optimize_parameters(
        URMrecommender_class=rec_class,
        validation_percentage=val_percentage,
        k=10,
        n_calls=50,
        limit_at=10,
        forest=True,
    )