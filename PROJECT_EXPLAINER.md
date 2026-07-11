# Pustaka Recommender System - Project Explainer

This document explains what the Pustaka Recommender System does, what we built, what it recommends, how recommendations are generated, and which factors are used in the process.

Note: the `documents` plugin was not callable in this workspace, so this explainer was generated directly in the repository from the project code and artifacts.

## 1. What This Project Does

Pustaka is a book recommendation system built on top of user-book interaction data.

Its goal is:

- recommend books a user has not already interacted with
- personalize recommendations using multiple recommendation strategies
- expose the results through a FastAPI backend and a React demo frontend

In the current project, the demo app shows:

- a user selector
- recent reading activity for the selected user
- favorite genres inferred from transaction history
- top recommended books for the selected user
- model performance cards and a recommendation workflow UI

## 2. What We Have Built

The repository contains a full recommendation pipeline, not just a UI.

### Data pipeline

The project processes raw data in phases:

1. Merge raw datasets
2. Clean users, books, and transactions
3. Perform EDA
4. Engineer features
5. Create train/test split
6. Train multiple recommenders
7. Combine models into a hybrid recommender
8. Evaluate models
9. Serve recommendations through FastAPI
10. Display them in a React demo

### Main application layers

- `src/` contains the end-to-end data science pipeline
- `models/` stores generated recommendation outputs and model artifacts
- `backend/app/` serves recommendation data through FastAPI
- `demo/` is the React frontend

## 3. What The System Recommends

The system recommends books.

More specifically, for each user it returns:

- top 10 unseen books
- ranked from highest to lowest recommendation score
- enriched with metadata such as title, author, genre, language, format, and rating

The live backend currently serves precomputed hybrid recommendations from:

- `models/hybrid/hybrid_recommendations.csv`

That means the app is not retraining models in real time. It loads already-generated recommendation outputs and serves them quickly through the API.

## 4. Dataset Size Used In The Project

Based on the checked-in cleaned and processed data:

- users: `1000`
- books: `600`
- transactions: `3728`
- train interactions: `2982`
- test interactions: `746`
- hybrid recommendation rows: `8950`
- users covered by hybrid recommendations: `895`
- unique books appearing in hybrid recommendations: `552`

## 5. How Recommendation Happens

The final recommendation engine is hybrid. It combines three separate recommenders:

- SVD collaborative filtering
- KNN collaborative filtering
- content-based filtering

The hybrid score is built from normalized outputs of those models.

### Final hybrid formula

The checked-in hybrid weights are:

- SVD: `0.20`
- KNN: `0.60`
- Content-Based: `0.20`

So the final score is:

`hybrid_score = svd_score + knn_score + cb_score`

Where each of those is a weighted, normalized contribution.

## 6. How Each Model Works

### 6.1 SVD collaborative filtering

SVD learns hidden preference patterns from user-item interactions.

It uses:

- `user_id`
- `book_id`
- `interaction_weight`

Implementation details from the project:

- algorithm: Surprise `SVD`
- `n_factors = 100`
- `n_epochs = 30`
- `lr_all = 0.005`
- `reg_all = 0.02`
- `random_state = 42`

What it is doing conceptually:

- learns latent factors for users and books
- predicts how strongly a user may like an unseen book
- generates top candidate books per user

### 6.2 KNN collaborative filtering

KNN finds users with similar reading behavior and recommends books liked by similar users.

It uses:

- the weighted user-item interaction matrix
- cosine similarity between users

Implementation details from the project:

- user-based KNN for recommendation generation
- item-based KNN is also fitted, but user-based KNN is the main recommendation function used
- cosine distance
- brute-force neighbor search
- `20` nearest neighbors

What it is doing conceptually:

- find users whose reading patterns are close to the target user
- collect books those neighbors interacted with
- exclude books already seen by the target user
- aggregate weighted neighbor scores

### 6.3 Content-based filtering

Content-based filtering recommends books that are similar to books the user has already read.

It uses a text profile built from:

- genre
- author
- language
- format

Implementation details from the project:

- TF-IDF vectorization
- cosine similarity between books

What it is doing conceptually:

- convert book metadata into a content profile
- find similar books for each source book
- for each user, collect content-based candidates from books already seen in training data
- invert rank into score using `1 / rank`
- aggregate content similarity signals per user

## 7. What Factors Are Used To Recommend

This is the most important part for explaining the system clearly.

### 7.1 Factors directly used in live recommendation scoring

These factors directly affect the final recommendations:

#### A. User-book interaction strength

The project turns transaction type into an interaction weight:

- `paperback_purchase = 5`
- `ebook_rental = 3`
- `audiobook_rental = 2`
- fallback default = `1`

This matters because both SVD and KNN learn from these weighted interactions.

#### B. Similarity to other users

Used in KNN.

If similar users interacted strongly with a book and the current user has not seen it, that book gets a stronger score.

#### C. Similarity to previously read books

Used in content-based filtering.

If a candidate book is similar in genre, author, language, or format to books the user already interacted with, it gets a stronger content score.

#### D. Latent preference patterns

Used in SVD.

This is the hidden structure learned from many users and books together, beyond explicit metadata.

#### E. Exclusion of seen books

A user is not recommended books already present in their training interactions.

This is enforced in the hybrid pipeline as a safety filter as well.

#### F. Final weighted combination

The final ranking depends on:

- `20%` SVD contribution
- `60%` KNN contribution
- `20%` content-based contribution

### 7.2 Metadata factors used in content similarity

These fields are used to build book content profiles:

- genre
- author
- language
- format

### 7.3 Interaction history factors used in profile display

These do not directly change the hybrid ranking in the live backend, but they are used in the dashboard profile experience:

- recent transaction history
- recent books read
- favorite genres derived from transaction history
- city
- state
- preferred language
- subscription type

### 7.4 Engineered features created in the pipeline

The project also generates a wider set of engineered features:

- `user_total_transactions`
- `user_unique_books`
- `user_active_days`
- `user_activity_score`
- `membership_days`
- `book_popularity`
- `book_unique_users`
- `book_age`
- `genre_popularity`
- `author_popularity`
- `days_since_interaction`

Important clarification:

These features are created in `src/phase4_feature_engineering.py`, but the checked-in final recommendation-serving path does not directly use all of them in the hybrid scoring formula.

The most directly used engineered value in the actual recommenders is `interaction_weight`.

So if you explain this project in a viva, report, or presentation, the safest wording is:

"The project engineers a broad feature set for users, books, and interactions, but the current production recommendation flow mainly uses weighted interactions for collaborative models and metadata similarity for the content-based model."

## 8. Full Recommendation Flow

Here is the end-to-end logic:

1. Raw users, books, and transactions are merged and cleaned.
2. Transaction dates are standardized and invalid records are removed.
3. Transaction types are converted into interaction weights.
4. Data is split chronologically into train and test sets using an `80/20` time-based split.
5. SVD learns latent user-book preferences from training interactions.
6. KNN learns user similarity from the weighted interaction matrix.
7. Content-based filtering builds similarity between books from metadata.
8. Each model generates candidate recommendations for unseen books.
9. Scores are normalized per user.
10. Weighted model outputs are combined into a hybrid score.
11. Top `10` books per user are kept.
12. FastAPI serves those precomputed recommendations to the demo frontend.

## 9. Why Hybrid Recommendation Was Used

No single recommendation method is perfect.

This project uses a hybrid approach because:

- collaborative filtering captures behavior patterns across users
- content-based filtering helps when metadata similarity matters
- a hybrid model reduces dependence on a single signal
- combining methods usually improves robustness and coverage

In practical terms:

- KNN contributes strong neighborhood-based personalization
- SVD contributes latent preference learning
- content-based filtering helps connect similar books through metadata

## 10. Evaluation Results In This Project

The project evaluates models using:

- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- Coverage@K
- Diversity@K

### Checked-in results at K = 10

| Model | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | Diversity@10 | Coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVD | 0.0086 | 0.0500 | 0.0261 | 0.0356 | 0.5500 | 0.0417 |
| KNN | 0.0411 | 0.2459 | 0.0914 | 0.1413 | 0.5950 | 0.4333 |
| Hybrid | 0.0291 | 0.1654 | 0.0730 | 0.1059 | 0.5586 | 0.5317 |

### Interpretation

From the checked-in evaluation artifacts:

- KNN performs best on precision, recall, MAP, and NDCG
- Hybrid performs best on coverage
- SVD performs worst among the three in the current saved results

That means:

- if the goal is strongest ranking quality in this snapshot, KNN is best
- if the goal is broader catalog spread, Hybrid is best
- the demo still focuses on the hybrid engine because it combines multiple signals and gives a more balanced recommendation story

This is an important point to explain honestly in project documentation.

## 11. What The Backend Actually Serves

The FastAPI backend currently serves:

- `/users` -> available user IDs from hybrid recommendations
- `/recommend` -> top-N hybrid recommendations for a selected user
- `/users/{user_id}/profile` -> user profile, recent books, and favorite genres
- `/books` -> basic book catalog metadata

The service reads from:

- `models/hybrid/hybrid_recommendations.csv`
- `data/cleaned/books_clean.csv`
- `data/cleaned/users_clean.csv`
- `data/cleaned/transactions_clean.csv`

This means the app is currently inference-light at runtime:

- it does not train models in the API
- it loads prepared recommendation outputs and joins them with metadata

## 12. What The Frontend Shows

The React demo:

- loads user IDs from the backend
- fetches profile information when the user changes
- shows recent books read and favorite genres
- calls the recommendation API
- displays recommended books with score-derived explanations

The explanation text in the UI is generated from available model signals such as:

- genre match
- rating
- KNN contribution
- content contribution
- SVD contribution

## 13. Strengths Of The Current Project

- full pipeline from raw data to deployable app
- multiple recommendation strategies implemented
- hybrid recommender with explicit weight control
- time-based train/test split instead of random leakage-prone splitting
- ranking-based evaluation metrics included
- real UI and backend integration

## 14. Current Limitations

- recommendations are precomputed, not generated live per request
- many engineered features are not yet fully exploited in the final hybrid score
- hybrid is not the top performer on all ranking metrics in the saved evaluation
- some metadata values can be missing and need null-safe handling
- the demo analytics cards are partly hardcoded rather than fully data-driven

## 15. Short Explanation You Can Reuse

If you need a short oral explanation for a presentation or viva, use this:

"We built a hybrid book recommendation system using user-book interaction history and book metadata. First, we cleaned and processed users, books, and transactions. Then we converted transaction types into interaction weights and split the data chronologically into train and test sets. We trained three recommenders: SVD for latent collaborative filtering, KNN for neighborhood-based collaborative filtering, and a content-based model using TF-IDF similarity over genre, author, language, and format. After normalizing model outputs per user, we combined them into a hybrid score using weights 0.2 for SVD, 0.6 for KNN, and 0.2 for content-based filtering. The system returns top 10 unseen books per user and serves them through a FastAPI backend to a React frontend. In our saved evaluation, KNN gives the strongest ranking performance, while Hybrid gives the best coverage across the catalog."

## 16. File References

Core recommendation logic lives in:

- `src/phase6b_content_based.py`
- `src/phase6c_svd.py`
- `src/phase6d_knn.py`
- `src/phase6e_hybrid.py`

Feature engineering and evaluation live in:

- `src/phase4_feature_engineering.py`
- `src/phase5_train_test_split.py`
- `src/phase7_evaluation.py`

Serving and UI live in:

- `backend/app/services/recommendation_service.py`
- `backend/app/api/recommend.py`
- `demo/src/App.jsx`
- `demo/src/api/api.js`

