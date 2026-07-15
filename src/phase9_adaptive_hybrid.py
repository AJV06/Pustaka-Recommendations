import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# PATHS
# =====================================================
PROCESSED_DIR = Path("data/processed")
SVD_DIR       = Path("models/svd")
CB_DIR        = Path("models/content_based")
KNN_DIR       = Path("models/collaborative")
HYBRID_DIR    = Path("models/hybrid")
REPORTS_DIR   = Path("reports/metrics")

HYBRID_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 10

# =====================================================
# HELPER: MIN-MAX NORMALISATION
# =====================================================
def minmax_per_user(df, score_col="predicted_score"):
    out = df.copy()
    grp = out.groupby("user_id")[score_col]
    mn  = grp.transform("min")
    mx  = grp.transform("max")
    denom = (mx - mn).replace(0, 1)
    out[score_col] = (out[score_col] - mn) / denom
    return out

# =====================================================
# PHASE 9: ADAPTIVE HYBRID
# =====================================================
def run_adaptive_hybrid():
    print("Loading data...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_interactions.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test_interactions.csv")
    
    # 1. Determine User Tiers
    user_counts = train_df.groupby("user_id").size().to_dict()
    all_users = train_df["user_id"].unique()
    user_seen = train_df.groupby("user_id")["book_id"].apply(set).to_dict()
    
    # Generate weight mapping per user
    user_weights = {}
    tier_counts = {"Cold": 0, "Warm": 0, "Heavy": 0}
    
    for u in all_users:
        count = user_counts.get(u, 0)
        if count < 5:
            user_weights[u] = {"w_cb": 0.8, "w_knn": 0.1, "w_svd": 0.1}
            tier_counts["Cold"] += 1
        elif count < 20:
            user_weights[u] = {"w_cb": 0.2, "w_knn": 0.6, "w_svd": 0.2}
            tier_counts["Warm"] += 1
        else:
            user_weights[u] = {"w_cb": 0.0, "w_knn": 0.8, "w_svd": 0.2}
            tier_counts["Heavy"] += 1
            
    print(f"\nUser Tiers Identified:\n - Cold (<5): {tier_counts['Cold']}\n - Warm (5-19): {tier_counts['Warm']}\n - Heavy (20+): {tier_counts['Heavy']}")
    
    # 2. Load Recommendations
    print("\nLoading Base Models...")
    svd_recs = pd.read_csv(SVD_DIR / "user_recommendations.csv")
    knn_recs = pd.read_csv(KNN_DIR / "knn_recommendations.csv")
    cb_recs  = pd.read_csv(CB_DIR / "content_recommendations.csv")
    
    # Precompute CB scores
    cb_recs["cb_score"] = 1.0 / cb_recs["rank"]
    user_book_cb = []
    for user in all_users:
        seen = user_seen.get(user, set())
        user_sources = [b for b in seen if b in cb_recs["source_book"].values]
        if not user_sources: continue
        user_cb = cb_recs[cb_recs["source_book"].isin(user_sources)].copy()
        user_cb = user_cb[~user_cb["recommended_book"].isin(seen)]
        if user_cb.empty: continue
        agg = user_cb.groupby("recommended_book")["cb_score"].sum().reset_index()
        agg = agg.rename(columns={"recommended_book": "book_id", "cb_score": "predicted_score"})
        agg["user_id"] = user
        user_book_cb.append(agg)
        
    content_user_recs = pd.concat(user_book_cb, ignore_index=True) if user_book_cb else pd.DataFrame(columns=["user_id", "book_id", "predicted_score"])
    
    # 3. Normalize
    print("Normalizing Scores...")
    svd_norm = minmax_per_user(svd_recs[["user_id", "book_id", "predicted_score"]])
    knn_norm = minmax_per_user(knn_recs[["user_id", "book_id", "predicted_score"]])
    content_norm = minmax_per_user(content_user_recs[["user_id", "book_id", "predicted_score"]])
    
    svd_norm = svd_norm.rename(columns={"predicted_score": "svd_score"})
    knn_norm = knn_norm.rename(columns={"predicted_score": "knn_score"})
    content_norm = content_norm.rename(columns={"predicted_score": "cb_score"})
    
    # Merge
    print("Applying Adaptive Weights...")
    merged = pd.merge(svd_norm, knn_norm, on=["user_id", "book_id"], how="outer")
    merged = pd.merge(merged, content_norm, on=["user_id", "book_id"], how="outer")
    merged.fillna(0.0, inplace=True)
    
    # Vectorized Weight Application
    w_svd_map = merged["user_id"].map(lambda u: user_weights.get(u, {"w_svd": 0.2})["w_svd"])
    w_knn_map = merged["user_id"].map(lambda u: user_weights.get(u, {"w_knn": 0.6})["w_knn"])
    w_cb_map  = merged["user_id"].map(lambda u: user_weights.get(u, {"w_cb": 0.2})["w_cb"])
    
    merged["hybrid_score"] = (
        merged["svd_score"] * w_svd_map +
        merged["knn_score"] * w_knn_map +
        merged["cb_score"] * w_cb_map
    )
    
    # 4. Extract Top N
    print("Extracting Top-10 Recommendations...")
    # Remove books user already saw
    merged = merged[~merged.set_index(["user_id", "book_id"]).index.isin([(u, b) for u, books in user_seen.items() for b in books])]
    merged = merged.sort_values(["user_id", "hybrid_score"], ascending=[True, False])
    
    merged["rank"] = merged.groupby("user_id").cumcount() + 1
    hybrid_recs = merged[merged["rank"] <= TOP_N].copy()
    
    out_path = HYBRID_DIR / "adaptive_hybrid_recommendations.csv"
    hybrid_recs.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    
    # 5. Evaluate using phase7 logic
    print("\nEvaluating Adaptive Hybrid...")
    from phase7_evaluation import ndcg_at_k, precision_at_k, recall_at_k, average_precision_at_k
    
    ground_truth = test_df.groupby("user_id")["book_id"].apply(set).to_dict()
    user_recs_dict = hybrid_recs.groupby("user_id")["book_id"].apply(list).to_dict()
    eval_users = [u for u in ground_truth if u in user_recs_dict and ground_truth[u]]
    
    ndcgs, precs, recs, maps = [], [], [], []
    for u in eval_users:
        relevant = ground_truth[u]
        recommended = user_recs_dict[u]
        ndcgs.append(ndcg_at_k(recommended, relevant, 10))
        precs.append(precision_at_k(recommended, relevant, 10))
        recs.append(recall_at_k(recommended, relevant, 10))
        maps.append(average_precision_at_k(recommended, relevant, 10))
        
    print("\n============= PHASE 9 EVALUATION =============")
    print(f"Model: Adaptive Hybrid (User-Specific Weights)")
    print(f"Evaluated Users: {len(eval_users)}")
    print(f"NDCG@10: {np.mean(ndcgs):.4f}")
    print(f"Precision@10: {np.mean(precs):.4f}")
    print(f"Recall@10: {np.mean(recs):.4f}")
    print(f"MAP@10: {np.mean(maps):.4f}")
    print("==============================================")

if __name__ == "__main__":
    run_adaptive_hybrid()
