import pandas as pd
import numpy as np

from pathlib import Path

# =====================================================
# PATHS
# =====================================================

PROCESSED_DIR = Path("data/processed")
FEATURE_DIR   = Path("data/features")
SVD_DIR       = Path("models/svd")
CB_DIR        = Path("models/content_based")
KNN_DIR       = Path("models/collaborative")
HYBRID_DIR    = Path("models/hybrid")

HYBRID_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# WEIGHTS
# =====================================================

W_SVD     = 0.50
W_KNN     = 0.30
W_CONTENT = 0.20

TOP_N = 10

# =====================================================
# LOAD TRAINING DATA (for user→books mapping)
# =====================================================

train_df = pd.read_csv(
    PROCESSED_DIR / "train_interactions.csv"
)

book_features = pd.read_csv(
    FEATURE_DIR / "book_features.csv"
)

all_users = train_df["user_id"].unique()

# Books each user has already seen
user_seen = (
    train_df
    .groupby("user_id")["book_id"]
    .apply(set)
    .to_dict()
)

# =====================================================
# LOAD SVD RECOMMENDATIONS
# =====================================================

svd_recs = pd.read_csv(
    SVD_DIR / "user_recommendations.csv"
)

# Columns: user_id, book_id, predicted_score, rank
print("SVD recs:", svd_recs.shape)

# =====================================================
# LOAD KNN RECOMMENDATIONS
# =====================================================

knn_path = KNN_DIR / "knn_recommendations.csv"

if knn_path.exists():
    knn_recs = pd.read_csv(knn_path)
    # Columns: user_id, book_id, predicted_score, rank
    print("KNN recs:", knn_recs.shape)
    use_knn = True
else:
    print("WARNING: KNN recommendations not found – skipping KNN.")
    knn_recs = pd.DataFrame(columns=["user_id", "book_id", "predicted_score"])
    use_knn = False

# =====================================================
# LOAD CONTENT-BASED RECOMMENDATIONS
# =====================================================

cb_recs = pd.read_csv(
    CB_DIR / "content_recommendations.csv"
)

# Columns: source_book, recommended_book, rank, title, author, genre
# This is book→book.  We pivot to user→book via the user's training history.
print("CB recs (raw):", cb_recs.shape)

# For each user build content score = average similarity rank
# across all their training books.

# Lower rank = better, so we invert: score = 1 / rank
cb_recs["cb_score"] = 1.0 / cb_recs["rank"]

# Build user-level content scores
user_book_cb = []

for user in all_users:
    seen = user_seen.get(user, set())
    # source books this user has read
    user_sources = [b for b in seen if b in cb_recs["source_book"].values]

    if not user_sources:
        continue

    # filter CB recs for this user's source books
    user_cb = cb_recs[
        cb_recs["source_book"].isin(user_sources)
    ].copy()

    # exclude books user has already seen
    user_cb = user_cb[~user_cb["recommended_book"].isin(seen)]

    if user_cb.empty:
        continue

    # aggregate across source books: sum of inverse-rank scores
    agg = (
        user_cb
        .groupby("recommended_book")["cb_score"]
        .sum()
        .reset_index()
        .rename(columns={"recommended_book": "book_id", "cb_score": "predicted_score"})
    )
    agg["user_id"] = user

    user_book_cb.append(agg)

if user_book_cb:
    content_user_recs = pd.concat(user_book_cb, ignore_index=True)
else:
    content_user_recs = pd.DataFrame(
        columns=["user_id", "book_id", "predicted_score"]
    )

print("Content user-level recs:", content_user_recs.shape)


# =====================================================
# NORMALISE SCORES PER USER (min-max within each model)
# =====================================================

def minmax_per_user(df, score_col="predicted_score"):
    """Normalise scores to [0, 1] per user."""
    out = df.copy()
    grp = out.groupby("user_id")[score_col]
    mn  = grp.transform("min")
    mx  = grp.transform("max")
    denom = (mx - mn).replace(0, 1)         # avoid div-by-zero
    out[score_col] = (out[score_col] - mn) / denom
    return out


svd_norm     = minmax_per_user(svd_recs[["user_id", "book_id", "predicted_score"]])
knn_norm     = minmax_per_user(knn_recs[["user_id", "book_id", "predicted_score"]]) if use_knn else knn_recs
content_norm = minmax_per_user(content_user_recs[["user_id", "book_id", "predicted_score"]])

# Tag each frame with its model weight
svd_norm["svd_score"]     = svd_norm["predicted_score"]     * W_SVD
knn_norm["knn_score"]     = knn_norm["predicted_score"]     * W_KNN     if use_knn else 0
content_norm["cb_score"]  = content_norm["predicted_score"] * W_CONTENT

# =====================================================
# MERGE ALL SCORES
# =====================================================

# Start with SVD as base
merged = svd_norm[["user_id", "book_id", "svd_score"]].copy()

# Merge KNN
if use_knn and not knn_norm.empty:
    merged = merged.merge(
        knn_norm[["user_id", "book_id", "knn_score"]],
        on=["user_id", "book_id"],
        how="outer"
    )
else:
    merged["knn_score"] = 0.0

# Merge Content
if not content_norm.empty:
    merged = merged.merge(
        content_norm[["user_id", "book_id", "cb_score"]],
        on=["user_id", "book_id"],
        how="outer"
    )
else:
    merged["cb_score"] = 0.0

# Fill NaN scores with 0 (model did not produce a rec for this pair)
merged["svd_score"]  = merged["svd_score"].fillna(0.0)
merged["knn_score"]  = merged["knn_score"].fillna(0.0)
merged["cb_score"]   = merged["cb_score"].fillna(0.0)

# Compute hybrid score
merged["hybrid_score"] = (
    merged["svd_score"] +
    merged["knn_score"] +
    merged["cb_score"]
)

# =====================================================
# RE-RANK: TOP-N PER USER
# =====================================================

# Remove books user already saw (safety net)
def filter_seen(row):
    return row["book_id"] not in user_seen.get(row["user_id"], set())

merged = merged[merged.apply(filter_seen, axis=1)]

# Sort by hybrid score, keep top-N per user
merged = merged.sort_values(
    ["user_id", "hybrid_score"],
    ascending=[True, False]
)

merged["rank"] = (
    merged
    .groupby("user_id")
    .cumcount() + 1
)

hybrid_recs = merged[merged["rank"] <= TOP_N].copy()

hybrid_recs = hybrid_recs[[
    "user_id", "book_id", "hybrid_score",
    "svd_score", "knn_score", "cb_score", "rank"
]]

print("\nHybrid recommendations:", hybrid_recs.shape)
print(f"Unique users with recs: {hybrid_recs['user_id'].nunique()}")

# =====================================================
# SAVE
# =====================================================

hybrid_recs.to_csv(
    HYBRID_DIR / "hybrid_recommendations.csv",
    index=False
)

# Model-weight metadata
weights = pd.DataFrame({
    "Model":  ["SVD", "KNN", "Content-Based"],
    "Weight": [W_SVD, W_KNN, W_CONTENT]
})

weights.to_csv(
    HYBRID_DIR / "hybrid_weights.csv",
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\nHYBRID COMPLETE")
print(f"Users covered   : {hybrid_recs['user_id'].nunique()}")
print(f"Recommendations : {len(hybrid_recs)}")
print(f"Weights — SVD: {W_SVD}, KNN: {W_KNN}, Content: {W_CONTENT}")

print("\nSaved Files:")
print("  models/hybrid/hybrid_recommendations.csv")
print("  models/hybrid/hybrid_weights.csv")
