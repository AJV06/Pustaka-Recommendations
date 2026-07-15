import pandas as pd
import numpy as np
from pathlib import Path
import itertools

# PATHS
PROCESSED_DIR = Path("data/processed")
SVD_DIR       = Path("models/svd")
CB_DIR        = Path("models/content_based")
KNN_DIR       = Path("models/collaborative")
REPORTS_DIR   = Path("reports/metrics")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# NDCG Function from phase7
def ndcg_at_k(recommended, relevant, k=10):
    topk = recommended[:k]
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_n))
    if idcg == 0: return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, book in enumerate(topk) if book in relevant)
    return dcg / idcg

# Min-Max Normalisation from phase6e
def minmax_per_user(df, score_col="predicted_score"):
    out = df.copy()
    grp = out.groupby("user_id")[score_col]
    mn  = grp.transform("min")
    mx  = grp.transform("max")
    denom = (mx - mn).replace(0, 1)
    out[score_col] = (out[score_col] - mn) / denom
    return out

def run_grid_search():
    print("Loading datasets...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_interactions.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_interactions.csv")
    
    all_users = train_df["user_id"].unique()
    user_seen = train_df.groupby("user_id")["book_id"].apply(set).to_dict()
    ground_truth = test_df.groupby("user_id")["book_id"].apply(set).to_dict()
    
    # Load Recs
    svd_recs = pd.read_csv(SVD_DIR / "user_recommendations.csv")
    knn_recs = pd.read_csv(KNN_DIR / "knn_recommendations.csv")
    cb_recs = pd.read_csv(CB_DIR / "content_recommendations.csv")
    
    # Pre-compute CB scores
    print("Pre-computing Content-Based scores per user...")
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
    
    # MinMax Scale
    print("Normalizing scores...")
    svd_norm = minmax_per_user(svd_recs[["user_id", "book_id", "predicted_score"]])
    knn_norm = minmax_per_user(knn_recs[["user_id", "book_id", "predicted_score"]])
    content_norm = minmax_per_user(content_user_recs[["user_id", "book_id", "predicted_score"]])
    
    # Grid Search Params
    weight_steps = [round(x * 0.1, 1) for x in range(11)]
    valid_combos = [w for w in itertools.product(weight_steps, repeat=3) if round(sum(w), 1) == 1.0]
    
    print(f"Testing {len(valid_combos)} weight combinations...")
    
    best_ndcg = -1
    best_weights = None
    results = []
    
    for w_svd, w_knn, w_cb in valid_combos:
        # Apply weights
        sn = svd_norm.copy()
        sn["svd_score"] = sn["predicted_score"] * w_svd
        kn = knn_norm.copy()
        kn["knn_score"] = kn["predicted_score"] * w_knn
        cn = content_norm.copy()
        cn["cb_score"] = cn["predicted_score"] * w_cb
        
        # Merge
        merged = sn[["user_id", "book_id", "svd_score"]].merge(kn[["user_id", "book_id", "knn_score"]], on=["user_id", "book_id"], how="outer")
        merged = merged.merge(cn[["user_id", "book_id", "cb_score"]], on=["user_id", "book_id"], how="outer")
        merged.fillna(0.0, inplace=True)
        merged["hybrid_score"] = merged["svd_score"] + merged["knn_score"] + merged["cb_score"]
        
        # Filter seen and rank
        merged = merged[~merged.set_index(["user_id", "book_id"]).index.isin([(u, b) for u, books in user_seen.items() for b in books])]
        merged = merged.sort_values(["user_id", "hybrid_score"], ascending=[True, False])
        top10 = merged.groupby("user_id").head(10)
        
        # Evaluate
        user_recs_dict = top10.groupby("user_id")["book_id"].apply(list).to_dict()
        eval_users = [u for u in ground_truth if u in user_recs_dict and ground_truth[u]]
        
        ndcgs = [ndcg_at_k(user_recs_dict[u], ground_truth[u], 10) for u in eval_users]
        mean_ndcg = np.mean(ndcgs) if ndcgs else 0
        
        results.append({
            "W_SVD": w_svd,
            "W_KNN": w_knn,
            "W_CB": w_cb,
            "NDCG@10": round(mean_ndcg, 4)
        })
        
        if mean_ndcg > best_ndcg:
            best_ndcg = mean_ndcg
            best_weights = (w_svd, w_knn, w_cb)
            
    res_df = pd.DataFrame(results).sort_values("NDCG@10", ascending=False)
    print("\n========= GRID SEARCH RESULTS =========")
    print(res_df.head(10).to_string(index=False))
    print("\nBEST WEIGHTS:", best_weights, "-> NDCG:", round(best_ndcg, 4))
    res_df.to_csv(REPORTS_DIR / "grid_search_weights.csv", index=False)

if __name__ == "__main__":
    run_grid_search()
