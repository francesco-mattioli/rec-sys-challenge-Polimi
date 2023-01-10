from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from impressions import Impressions
from hybrid import *
from tqdm import tqdm
from evaluator import evaluate
import pandas as pd
import numpy as np
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender


########################## READ & SPLIT DATA ##########################
dataReader = DataReader()

target = dataReader.load_target()
'''
UCM = dataReader.load_aug_ucm()
URM = dataReader.load_augmented_binary_urm()
URM_aug, ICM = dataReader.pad_with_zeros_ICMandURM(URM)


URM_train_aug, URM_validation = split_train_in_two_percentage_global_sample( URM_aug, train_percentage=1.0)
#URM_train_super_pow = dataReader.stackMatrixes_with_impressions(URM_train_aug)
URM_train_pow = dataReader.stackMatrixes(URM_train_aug)

# Instantiate Impressions object to update ranking at the end of recommendations
#item_ids = dataReader.get_unique_items_based_on_urm(dataReader.load_augmented_binary_urm_df())
#impressions = Impressions(target,item_ids)
# dataReader.save_impressions()
'''

URM = dataReader.load_augmented_binary_urm() 
#print(URM.shape) #(41629, 24507)

URM_aug, ICM = dataReader.pad_with_zeros_ICMandURM(URM) 
#print(URM_aug.shape) #(41629, 27968)
#print(ICM.shape) #(27968, 5)

ICM_stacked_with_binary_impressions = dataReader.load_ICM_stacked_with_binary_impressions(0.8)
#print(ICM_stacked_with_binary_impressions.shape) #(27286, 7005)

URM_aug, ICM = dataReader.pad_with_zeros_given_ICMandURM(ICM_stacked_with_binary_impressions, URM) 

#print(URM_aug.shape) #(41629, 27286)
#print(ICM.shape) #(27286, 7005)

URM_train_aug, URM_validation = split_train_in_two_percentage_global_sample(URM_aug, train_percentage=0.9)

#URM_train_pow = dataReader.stackMatrixes(URM_train_aug)

#URM_train_pow_df = dataReader.csr_to_dataframe(URM_train_pow,'UserID','ItemID','Data')

URM_train_super_pow = dataReader.load_super_powerful_URM(URM_train_aug, ICM_stacked_with_binary_impressions, 0.8)
'''
URM_train_super_pow_df  = dataReader.csr_to_dataframe(URM_train_super_pow,'UserID','ItemID','Data')

print(len(np.setdiff1d(
            URM_train_super_pow_df['ItemID'].unique(), URM_train_pow_df['ItemID'].unique())))

print(len(np.setdiff1d(
            URM_train_pow_df['ItemID'].unique(), URM_train_super_pow_df['ItemID'].unique())))


print(URM_train_pow.shape)
print(URM_train_super_pow.shape)
'''
#URM_train_super_pow, ICM = dataReader.pad_with_zeros_given_ICMandURM(ICM_stacked_with_binary_impressions, URM)

########################## iNSTANTIATE & FIT SINGLE MODELS ##########################


#ItemKNNCF = ItemKNNCFRecommender(URM_train_pow)
#ItemKNNCF.fit()

#RP3beta_pow = RP3betaRecommender(URM_train_pow)
#RP3beta_pow.fit(alpha=0.3648761546066018,beta=0.5058870363874656, topK=480, normalize_similarity=True)

#EASE_R = EASE_R_Recommender(URM_train_aug)
#EASE_R.fit()

UserKNNCF = UserKNNCFRecommender(URM_train_aug)
UserKNNCF.fit()

RP3beta_aug = RP3betaRecommender(URM_train_aug)
RP3beta_aug.fit()

S_SLIM = SLIMElasticNetRecommender(URM_train_super_pow)
S_SLIM.fit(l1_ratio=0.006011021694075882,
           alpha=0.0013369897413235414, topK=459)

'''
UserKNN_CFCBF_Hybrid_Recommender = UserKNN_CFCBF_Hybrid_Recommender(
    URM_train_aug, UCM)
UserKNN_CFCBF_Hybrid_Recommender.fit()

ItemKNN_CFCBF_Hybrid_Recommender = ItemKNN_CFCBF_Hybrid_Recommender(
    URM_train_aug, ICM)
ItemKNN_CFCBF_Hybrid_Recommender.fit()

'''


##########################################################################################################


'''
Hybrid_UserKNNCF_RP3B_aug = Hybrid_UserKNNCF_RP3B_aug(
    URM_train_aug, URM_train_pow, UserKNNCF, RP3beta_aug)
Hybrid_UserKNNCF_RP3B_aug.fit(
    UserKNNCF_weight=0.4348857237366932, RP3B_weight=0.027648314372221712)

'''

#Hybrid_SSLIM_EASER = Hybrid_SSLIM_EASER(
#    URM_train_aug, URM_train_pow, S_SLIM, EASE_R)
#Hybrid_SSLIM_EASER.fit(SSLIM_weight=0.5495139584252299, EASE_R_weight=0.0)

Hybrid_SSLIM_RP3B_aug = Hybrid_SSLIM_RP3B_aug(
    URM_train_aug, S_SLIM, RP3beta_aug)
Hybrid_SSLIM_RP3B_aug.fit(alpha = 0.7447123958484749)

'''
Hybrid_UserKNNCF_ItemKNNCF = Hybrid_UserKNNCF_ItemKNNCF(
    URM_train_aug, URM_train_pow, UserKNNCF, ItemKNNCF)
Hybrid_UserKNNCF_ItemKNNCF.fit(
    UserKNNCF_weight=1.0, ItemKNNCF_weight=0.8072073132929845)
'''


'''
Hybrid_User_and_Item_KNN_CFCBF_Hybrid = Hybrid_User_and_Item_KNN_CFCBF_Hybrid(
    URM_train_aug, URM_train_pow, ItemKNN_CFCBF_Hybrid_Recommender, UserKNN_CFCBF_Hybrid_Recommender)
Hybrid_User_and_Item_KNN_CFCBF_Hybrid.fit()
'''

########################## INSTANTIATE & FIT FINAL HYBIRD MODEL ##########################

recommender = Hybrid_of_Hybrids(URM_train_aug, Hybrid_SSLIM_RP3B_aug, UserKNNCF, S_SLIM)

recommender.fit(alpha=0.24953067333115547,beta=0.5529848994377775,gamma=0.4543600764791589)
            
########################## CREATE CSV FOR SUBMISISON ##########################
f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
recommended_items_for_each_user = {}
for user_id in tqdm(target):
    recommended_items = recommender.recommend(
        user_id, cutoff=10, remove_seen_flag=True)
    # recommended_items=impressions.update_ranking(user_id,recommended_items,dataReader)
    recommended_items_for_each_user[int(user_id)] = recommended_items
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{user_id}, {well_formatted}\n")

# Evaluate recommended items
map = evaluate(recommended_items_for_each_user, URM_validation, target)
print('MAP score: {}'.format(map))
