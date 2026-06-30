"""
Phase 7 – Evaluation
====================
Computes ranking metrics for every recommendation model:
  - SVD                  (models/svd/user_recommendations.csv)
  - KNN                  (models/collaborative/knn_recommendations.csv)
  - Hybrid               (models/hybrid/hybrid_recommendations.csv)

Metrics (at K = 5 and K = 10):
  - Precision@K
  - Recall@K
  - MAP@K  (Mean Average Precision)
  - NDCG@K (Normalised Discounted Cumulative Gain)
  - Coverage  (fraction of catalog recommended by at least one user)
  - Diversity (mean intra-list genre diversity)

Outputs:
  reports/metrics/evaluation_results.csv   – summary table
  reports/metrics/per_user_metrics.csv     – per-user detail
"""

import pandas as pd
import numpy as np

from pathlib import Path

# =====================================================
# PATHS
# =====================================================

PROCESSED_DIR = Path("data/processed")
FEATURE_DIR   = Path("data/features")
REPORTS_DIR   = Path("reports/metrics")

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# GROUND TRUTH
# =====================================================

test_df = pd.read_csv(
    PROCESSED_DIR / "test_interactions.csv"
)

# Relevant items per user = books they interacted with in the test set
ground_truth = (
    test_df
    .groupby("user_id")["book_id"]
    .apply(set)
    .to_dict()
)

# =====================================================
# LOAD BOOK METADATA (for diversity)
# =====================================================

book_features = pd.read_csv(
    FEATURE_DIR / "book_features.csv"
)

book_genre = (
    book_features
    .set_index("book_id")["genre"]
    .to_dict()
)

total_catalog = len(book_features)

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def precision_at_k(recommended, relevant, k):
    """Fraction of top-K recs that are relevant."""
    topk = recommended[:k]
    hits = sum(1 for b in topk if b in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    """Fraction of relevant items captured in top-K."""
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for b in topk if b in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended, relevant, k):
    """Average Precision for a single user at K."""
    if not relevant:
        return 0.0
    topk = recommended[:k]
    score = 0.0
    hits  = 0
    for i, book in enumerate(topk, start=1):
        if book in relevant:
            hits   += 1
            score  += hits / i
    return score / min(len(relevant), k)


def ndcg_at_k(recommended, relevant, k):
    """Normalised DCG at K."""
    topk = recommended[:k]

    # Ideal DCG: first min(|relevant|, k) positions are hits
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_n))
    if idcg == 0:
        return 0.0

    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, book in enumerate(topk)
        if book in relevant
    )
    return dcg / idcg


def genre_diversity(recommended, book_genre_map):
    """
    Intra-list diversity: fraction of unique genres in the recommendation list.
    1.0 = all different genres, 0.0 = all same genre.
    """
    if len(recommended) <= 1:
        return 0.0
    genres = [book_genre_map.get(b, "unknown") for b in recommended]
    unique = len(set(genres))
    return unique / len(genres)


# =====================================================
# LOAD ALL MODEL RECOMMENDATIONS
# =====================================================

def load_recs(path, user_col="user_id", book_col="book_id",
              rank_col="rank", score_col=None):
    """
    Load a recommendations CSV and return a dict:
      user_id -> [book_id, ...] ordered by rank (ascending).
    """
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {path} not found - skipping.")
        return None

    df = pd.read_csv(p)

    if rank_col in df.columns:
        df = df.sort_values([user_col, rank_col])
    elif score_col and score_col in df.columns:
        df = df.sort_values([user_col, score_col], ascending=[True, False])

    recs = (
        df
        .groupby(user_col)[book_col]
        .apply(list)
        .to_dict()
    )
    return recs


model_configs = {
    "SVD": {
        "path": "models/svd/user_recommendations.csv",
        "rank_col": "rank"
    },
    "KNN": {
        "path": "models/collaborative/knn_recommendations.csv",
        "rank_col": "rank"
    },
    "Hybrid": {
        "path": "models/hybrid/hybrid_recommendations.csv",
        "rank_col": "rank"
    },
}

models = {}
for name, cfg in model_configs.items():
    recs = load_recs(cfg["path"], rank_col=cfg["rank_col"])
    if recs is not None:
        models[name] = recs
        print(f"Loaded {name}: {len(recs)} users")

# =====================================================
# EVALUATE
# =====================================================

K_VALUES = [5, 10]

summary_rows   = []
per_user_rows  = []

for model_name, recs_dict in models.items():
    print(f"\nEvaluating {model_name}...")

    # Users that appear in BOTH recs and ground truth
    eval_users = [
        u for u in ground_truth
        if u in recs_dict and ground_truth[u]
    ]

    if not eval_users:
        print(f"  No overlap between recs and ground truth for {model_name}")
        continue

    recommended_books_all = set()

    for k in K_VALUES:

        p_list, r_list, ap_list, ndcg_list, div_list = [], [], [], [], []

        for user in eval_users:

            recommended = recs_dict.get(user, [])
            relevant    = ground_truth.get(user, set())

            p      = precision_at_k(recommended, relevant, k)
            r      = recall_at_k(recommended, relevant, k)
            ap     = average_precision_at_k(recommended, relevant, k)
            ndcg   = ndcg_at_k(recommended, relevant, k)
            div    = genre_diversity(recommended[:k], book_genre)

            p_list.append(p)
            r_list.append(r)
            ap_list.append(ap)
            ndcg_list.append(ndcg)
            div_list.append(div)

            # Collect recommended books for coverage computation
            if k == max(K_VALUES):
                recommended_books_all.update(recommended[:k])

            if k == 10:
                per_user_rows.append({
                    "model":        model_name,
                    "user_id":      user,
                    "precision@10": round(p, 4),
                    "recall@10":    round(r, 4),
                    "map@10":       round(ap, 4),
                    "ndcg@10":      round(ndcg, 4),
                    "diversity@10": round(div, 4),
                })

        coverage = len(recommended_books_all) / total_catalog

        summary_rows.append({
            "Model":       model_name,
            "K":           k,
            "Precision@K": round(np.mean(p_list), 4),
            "Recall@K":    round(np.mean(r_list), 4),
            "MAP@K":       round(np.mean(ap_list), 4),
            "NDCG@K":      round(np.mean(ndcg_list), 4),
            "Diversity@K": round(np.mean(div_list), 4),
            "Coverage@K":  round(coverage, 4),
            "Eval_Users":  len(eval_users),
        })

        print(
            f"  K={k} | "
            f"P={np.mean(p_list):.4f} | "
            f"R={np.mean(r_list):.4f} | "
            f"MAP={np.mean(ap_list):.4f} | "
            f"NDCG={np.mean(ndcg_list):.4f} | "
            f"Cov={coverage:.4f}"
        )

# =====================================================
# SAVE
# =====================================================

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    REPORTS_DIR / "evaluation_results.csv",
    index=False
)

if per_user_rows:
    per_user_df = pd.DataFrame(per_user_rows)
    per_user_df.to_csv(
        REPORTS_DIR / "per_user_metrics.csv",
        index=False
    )

# =====================================================
# PRINT COMPARISON TABLE
# =====================================================

print("\n" + "=" * 75)
print("EVALUATION SUMMARY")
print("=" * 75)

if not summary_df.empty:
    for k_val in K_VALUES:
        subset = summary_df[summary_df["K"] == k_val]
        print(f"\n-- At K = {k_val} --")
        print(
            subset[[
                "Model", "Precision@K", "Recall@K",
                "MAP@K", "NDCG@K", "Diversity@K", "Coverage@K"
            ]]
            .to_string(index=False)
        )

print("\nSaved Files:")
print("  reports/metrics/evaluation_results.csv")
print("  reports/metrics/per_user_metrics.csv")
print("\nPhase 7 COMPLETE")
