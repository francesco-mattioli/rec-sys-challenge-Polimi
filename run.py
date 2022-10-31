from Data_Handler.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from hybrid import HybridRecommender
from tqdm import tqdm
from evaluator import Evaluator

# Read & split data
dataReader = DataReader()
urm = dataReader.load_urm()
target = dataReader.load_target()

URM_train, URM_test = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)

evaluator=Evaluator()
evaluator.evaluate(URM_train,URM_test,URM_validation)

# Instantiate and fit hybrid recommender
recommender = HybridRecommender(URM_train)
recommender.fit()

# Create CSV for submission
f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
for t in tqdm(target):
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{t}, {well_formatted}\n")
