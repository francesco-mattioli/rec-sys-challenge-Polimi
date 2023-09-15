# Recommender System Challenge

<p align="center">
  <img width="100%" src="https://www.symbola.net/wp-content/uploads/2020/07/Polimi.jpg" alt="header" />
</p>

Welcome to the code repository for the Recommender Systems Challenge hosted by Politecnico di Milano. This competition is about developing a recommendation system for TV shows. The key objective is to predict the TV shows a user would interact with, taking into account variables such as the duration and categories of the TV shows.

Please read the README till the end for a detailed understanding of the project!

## Project Structure

Here is a brief explanation of the main files and modules in the repository:

- `boost_submission.py`: This script enhances the ranking list for submissions, aimed at improving the Mean Average Precision (MAP).
- `cross_k_validation.py`: This script performs cross-validation for hyperparameter tuning.
- `evaluator.py`: This script measures the performance of the recommendation system, utilizing MAP as a metric.
- `hybrid_user_wise.ipynb`: This Python notebook is dedicated to the analysis of user interactions and categorizing users according to their interaction frequency. It provides visual representation to understand whihc recommeneder system yielsds the best results for each user category.
- `hybrid.py`: This script houses a collection of hybrid recommenders that leverage different models. As you navigate down the file, the most recent recommenders are found at the end. We initially employed a simple "HybridRecommender", then progressively transitioned to a more intricate "Linear Hybrid", which combines several hybrid recommenders with varying tuned weights.
- `impressions.py`: This script provides the "update_ranking" function used in `boost_submission.py`.
- `requirements.txt`: A file containing a list of items to be installed using pip install.
- `run_factorization_machines.py`: This script trains and evaluates a factorization machines model (using LightFM).
- `run_kfold_hyper_search.py`: This script performs k-fold hyperparameter search for improving the recommendation model.
- `run.py`: This script serves as the primary initiation point to setup and execute the recommendation system, as well as compute the MAP score.
- `tuning.py`: This script provides functionalities for hyperparameter tuning.

Here is a brief explanation of the two matrixes used:

- `URM_aug`: This is a user-item interaction matrix that represents user interactions with items in the system. Each entry (user, item) in the matrix is set to '1' if the user has either watched the item or opened the item's details page. The matrix is then padded with zeros in the missing items that are present in the Item-Content Matrix (ICM) but not in the User-Content Matrix (UCM).
- `URM_aug_pow`: This is similar to URM_aug, but it includes an additional step where the ICM is stacked vertically below the URM_aug matrix. The vertical stacking ensures that the cardinality (number of columns) of both matrices coincides, enabling proper concatenation. For the concatenation we used an hyperparameter that is fundamental because it balances the importance of the collaborative part with respect to the context-based part. It can take a value between 0 and 1 (if we want to build a hybrid recommender system).

Below, you will find a schematic representation of our top-performing hybrid recommender system, where numeric values within rounded boxes indicate the respective weights.

<img src="https://github.com/engpap/rec-sys-challenge/blob/master/Docs/schema_best_hybrid.png" alt="Project Logo" width="700"/>




## Competition Details

The datasets includes around 1.8M interactions, 41k users, 27k items (TV shows) and two features: the TV shows length (number of episodes or movies) and 4 categories. For some user interactions the data also includes the impressions, representing which items were available on the screen when the user clicked on that TV shows.

The training-test split was accomplished via random holdout, with an 85% training and 15% test ratio. The goal was to recommend a list of potentially relevant items (10 per user). Mean Average Precision at 10 (MAP@10) was used for evaluation. Various types of recommender algorithms were utilized, such as collaborative-filtering, content-based, hybrid, etc. Impressions were utilized to enhance the recommendation quality.

## Competition Results
Each team received a final score based on the quality of recommendations.

Public Leaderboard: Secured 1st position out of 93 teams in the initial deadline.
Private Leaderboard: Achieved 3rd position out of 93 teams in the initial deadline.

## Getting Started

### Requirements

To install the requirements, run the following command:

```bash
pip install -r requirements.txt
```

### Perform a recommendation

To run our final and best recommender system, run the following command:

```bash
python3 run.py
```
