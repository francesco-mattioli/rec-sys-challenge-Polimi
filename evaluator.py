from tqdm import tqdm
import numpy as np
import scipy.sparse as sp


def mean_average_precision(recommendations: np.array, relevant_items: np.array) -> float:
        is_relevant = np.in1d(recommendations, relevant_items, assume_unique=True)

        precision_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

        map_score = np.sum(precision_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

        return map_score


def evaluate(recommended_items_for_each_user, urm_test: sp.csr_matrix, target: sp.csr_matrix):       
    accum_map = 0
    num_users_evaluated = 0

    for user_id in tqdm(target):
        user_profile_start = urm_test.indptr[user_id]
        user_profile_end = urm_test.indptr[user_id+1]
    
        relevant_items = urm_test.indices[user_profile_start:user_profile_end]
    
        if relevant_items.size == 0:
            relevant_items = np.zeros(urm_test.indices[0])

        accum_map += mean_average_precision(recommended_items_for_each_user[int(user_id)], relevant_items)
        num_users_evaluated += 1

    accum_map /=  max(num_users_evaluated, 1)

    return accum_map

