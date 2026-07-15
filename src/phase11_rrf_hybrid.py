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

HYBRID_DIR.mkdir(parents=True, exist_ok=True)
TOP_N = 10
K_RRF = 60

def run_rrf_hybrid():
    print("Loading datasets...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_interactions.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test_interactions.csv")
    
    all_users = train_df["user_id"].unique()
    user_seen = train_df.groupby("user_id")["book_id"].apply(set).to_dict()
    user_counts = train_df.groupby("user_id").size().to_dict()
    
    # 1. Load Base Models
    print("Loading Base Models...")
    svd_recs = pd.read_csv(SVD_DIR / "user_recommendations.csv")
    knn_recs = pd.read_csv(KNN_DIR / "knn_recommendations.csv")
    cb_recs  = pd.read_csv(CB_DIR / "content_recommendations.csv")
    
    # Extract raw ranks
    svd_ranks = svd_recs[["user_id", "book_id", "rank"]].rename(columns={"rank": "svd_rank"})
    knn_ranks = knn_recs[["user_id", "book_id", "rank"]].rename(columns={"rank": "knn_rank"})
    
    # Precompute CB ranks
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
        agg = agg.sort_values("predicted_score", ascending=False)
        agg["cb_rank"] = np.arange(1, len(agg) + 1)
        agg["user_id"] = user
        user_book_cb.append(agg[["user_id", "book_id", "cb_rank"]])
        
    cb_ranks = pd.concat(user_book_cb, ignore_index=True) if user_book_cb else pd.DataFrame(columns=["user_id", "book_id", "cb_rank"])
    
    # 2. Merge Ranks
    print("Computing Reciprocal Rank Fusion (RRF)...")
    merged = pd.merge(svd_ranks, knn_ranks, on=["user_id", "book_id"], how="outer")
    merged = pd.merge(merged, cb_ranks, on=["user_id", "book_id"], how="outer")
    
    # Fill missing ranks with Infinity
    merged["svd_rank"] = merged["svd_rank"].fillna(np.inf)
    merged["knn_rank"] = merged["knn_rank"].fillna(np.inf)
    merged["cb_rank"]  = merged["cb_rank"].fillna(np.inf)
    
    # 3. Apply Adaptive RRF
    # Cold users use only CB. Warm use SVD+KNN+CB. Heavy use SVD+KNN.
    def compute_rrf(row):
        u = row["user_id"]
        count = user_counts.get(u, 0)
        
        score = 0.0
        
        if count < 5:
            # Cold User -> Only Content
            if row["cb_rank"] != np.inf:
                score += 1.0 / (K_RRF + row["cb_rank"])
        elif count < 20:
            # Warm User -> All three
            if row["svd_rank"] != np.inf: score += 1.0 / (K_RRF + row["svd_rank"])
            if row["knn_rank"] != np.inf: score += 1.0 / (K_RRF + row["knn_rank"])
            if row["cb_rank"] != np.inf: score += 1.0 / (K_RRF + row["cb_rank"])
        else:
            # Heavy User -> SVD + KNN
            if row["svd_rank"] != np.inf: score += 1.0 / (K_RRF + row["svd_rank"])
            if row["knn_rank"] != np.inf: score += 1.0 / (K_RRF + row["knn_rank"])
            
        return score

    merged["rrf_score"] = merged.apply(compute_rrf, axis=1)
    
    # Drop rows with 0 score
    merged = merged[merged["rrf_score"] > 0.0]
    
    # 4. Extract Top N
    print("Extracting Top-10 Recommendations...")
    merged = merged[~merged.set_index(["user_id", "book_id"]).index.isin([(u, b) for u, books in user_seen.items() for b in books])]
    merged = merged.sort_values(["user_id", "rrf_score"], ascending=[True, False])
    
    merged["rank"] = merged.groupby("user_id").cumcount() + 1
    hybrid_recs = merged[merged["rank"] <= TOP_N].copy()
    
    out_path = HYBRID_DIR / "rrf_recommendations.csv"
    hybrid_recs.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    
    # 5. Evaluate
    print("\nEvaluating RRF Hybrid...")
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
        
    print("\n============= PHASE 11 EVALUATION =============")
    print(f"Model: Reciprocal Rank Fusion (Adaptive)")
    print(f"K_RRF Constant: {K_RRF}")
    print(f"Evaluated Users: {len(eval_users)}")
    print(f"NDCG@10: {np.mean(ndcgs):.4f}")
    print(f"Precision@10: {np.mean(precs):.4f}")
    print(f"Recall@10: {np.mean(recs):.4f}")
    print(f"MAP@10: {np.mean(maps):.4f}")
    print("===============================================")

if __name__ == "__main__":
    run_rrf_hybrid()
