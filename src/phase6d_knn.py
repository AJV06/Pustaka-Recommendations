import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# =====================================================
# PATHS
# =====================================================

PROCESSED_DIR = Path("data/processed")
FEATURE_DIR   = Path("data/features")
MODEL_DIR     = Path("models/collaborative")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# LOAD DATA
# =====================================================

train_df = pd.read_csv(
    PROCESSED_DIR / "train_interactions.csv"
)

test_df = pd.read_csv(
    PROCESSED_DIR / "test_interactions.csv"
)

print("Train Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

# =====================================================
# BUILD INTERACTION MATRIX
# =====================================================

# Encode user and book IDs to integer indices
all_users = sorted(train_df["user_id"].unique())
all_books = sorted(train_df["book_id"].unique())

user2idx = {u: i for i, u in enumerate(all_users)}
book2idx = {b: i for i, b in enumerate(all_books)}
idx2user = {i: u for u, i in user2idx.items()}
idx2book = {i: b for b, i in book2idx.items()}

n_users = len(all_users)
n_books = len(all_books)

# Aggregate duplicate (user, book) interactions by summing weights
agg = (
    train_df
    .groupby(["user_id", "book_id"])["interaction_weight"]
    .sum()
    .reset_index()
)

rows = agg["user_id"].map(user2idx).values
cols = agg["book_id"].map(book2idx).values
vals = agg["interaction_weight"].values.astype(np.float32)

# User-Item matrix (users × books)
user_item_matrix = csr_matrix(
    (vals, (rows, cols)),
    shape=(n_users, n_books)
)

# Item-User matrix (books × users) – transpose
item_user_matrix = user_item_matrix.T.tocsr()

print(f"\nUser-Item matrix: {user_item_matrix.shape}")
print(f"Item-User matrix: {item_user_matrix.shape}")

# =====================================================
# FIT ITEM-BASED KNN
# =====================================================

N_NEIGHBORS_ITEM = 20

knn_item = NearestNeighbors(
    n_neighbors=N_NEIGHBORS_ITEM + 1,   # +1 to exclude self
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

print("\nFitting Item-KNN...")
knn_item.fit(item_user_matrix)

# =====================================================
# FIT USER-BASED KNN
# =====================================================

N_NEIGHBORS_USER = 20

knn_user = NearestNeighbors(
    n_neighbors=N_NEIGHBORS_USER + 1,   # +1 to exclude self
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

print("Fitting User-KNN...")
knn_user.fit(user_item_matrix)

# =====================================================
# USER-BASED RECOMMENDATION FUNCTION
# =====================================================

# Pre-compute books seen by every user (for filtering)
user_seen = (
    train_df
    .groupby("user_id")["book_id"]
    .apply(set)
    .to_dict()
)


def recommend_user_knn(user_id, top_n=10):
    """
    User-based KNN recommendation.
    Finds k nearest neighbour users and aggregates
    their interaction scores for unseen books.
    """

    if user_id not in user2idx:
        return []

    uid = user2idx[user_id]
    user_vec = user_item_matrix[uid]

    # Find similar users
    distances, indices = knn_user.kneighbors(
        user_vec,
        n_neighbors=N_NEIGHBORS_USER + 1
    )

    # distances[0][0] is self – skip it
    neighbor_indices = indices[0][1:]
    neighbor_sims    = 1 - distances[0][1:]   # cosine distance → similarity

    seen_books = user_seen.get(user_id, set())

    # Aggregate weighted scores across neighbours
    scores = {}
    for sim, nidx in zip(neighbor_sims, neighbor_indices):
        if sim <= 0:
            continue
        neighbour_vec = user_item_matrix[nidx].toarray().flatten()
        for bidx, weight in enumerate(neighbour_vec):
            if weight == 0:
                continue
            book_id = idx2book[bidx]
            if book_id in seen_books:
                continue
            scores[book_id] = scores.get(book_id, 0.0) + sim * weight

    if not scores:
        return []

    sorted_books = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return sorted_books


# =====================================================
# GENERATE RECOMMENDATIONS FOR ALL TRAIN USERS
# =====================================================

print("\nGenerating KNN Recommendations...")

recommendation_rows = []
users = train_df["user_id"].unique()

for idx, user in enumerate(users):

    recs = recommend_user_knn(user, top_n=10)

    for rank, (book, score) in enumerate(recs, start=1):
        recommendation_rows.append({
            "user_id":         user,
            "book_id":         book,
            "predicted_score": round(score, 4),
            "rank":            rank
        })

    if idx % 100 == 0:
        print(f"  Processed {idx}/{len(users)} users")

recommendations = pd.DataFrame(recommendation_rows)

# =====================================================
# SIMPLE EVALUATION ON TEST SET
# =====================================================

# RMSE / MAE on test interactions using user-KNN predictions
print("\nEvaluating on test set...")

preds, actuals = [], []

for _, row in test_df.iterrows():
    uid  = row["user_id"]
    bid  = row["book_id"]
    true = row["interaction_weight"]

    if uid not in user2idx or bid not in book2idx:
        continue

    uidx = user2idx[uid]
    bidx = book2idx[bid]

    user_vec         = user_item_matrix[uidx]
    distances, indices = knn_user.kneighbors(
        user_vec,
        n_neighbors=N_NEIGHBORS_USER + 1
    )

    neighbor_indices = indices[0][1:]
    neighbor_sims    = 1 - distances[0][1:]

    numerator   = 0.0
    denominator = 0.0

    for sim, nidx in zip(neighbor_sims, neighbor_indices):
        if sim <= 0:
            continue
        val = user_item_matrix[nidx, bidx]
        if hasattr(val, "toarray"):
            val = val.toarray().flatten()[0]
        numerator   += sim * val
        denominator += abs(sim)

    predicted = (numerator / denominator) if denominator > 0 else 0.0
    preds.append(predicted)
    actuals.append(true)

preds   = np.array(preds)
actuals = np.array(actuals)

if len(preds) > 0:
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    mae  = np.mean(np.abs(preds - actuals))
else:
    rmse = mae = float("nan")

print(f"\nRMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

# =====================================================
# SAVE
# =====================================================

recommendations.to_csv(
    MODEL_DIR / "knn_recommendations.csv",
    index=False
)

metrics = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "Users", "Recommendations"],
    "Value":  [rmse, mae, len(users), len(recommendations)]
})

metrics.to_csv(
    MODEL_DIR / "knn_metrics.csv",
    index=False
)

joblib.dump(knn_item, MODEL_DIR / "knn_item_model.pkl")
joblib.dump(knn_user, MODEL_DIR / "knn_user_model.pkl")

# Save index mappings so other phases can reuse them
joblib.dump(
    {
        "user2idx": user2idx,
        "book2idx": book2idx,
        "idx2user": idx2user,
        "idx2book": idx2book,
    },
    MODEL_DIR / "knn_index_maps.pkl"
)

# =====================================================
# SUMMARY
# =====================================================

print("\nKNN COMPLETE")
print(f"Users         : {len(users)}")
print(f"Recommendations: {len(recommendations)}")
print(f"RMSE          : {rmse:.4f}")
print(f"MAE           : {mae:.4f}")

print("\nSaved Files:")
print("  models/collaborative/knn_recommendations.csv")
print("  models/collaborative/knn_metrics.csv")
print("  models/collaborative/knn_item_model.pkl")
print("  models/collaborative/knn_user_model.pkl")
print("  models/collaborative/knn_index_maps.pkl")
