import pandas as pd
import numpy as np
import optuna
from pathlib import Path
import logging

# Hide optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================
# PATHS
# =====================================================
PROCESSED_DIR = Path("data/processed")
SVD_DIR       = Path("models/svd")
CB_DIR        = Path("models/content_based")
KNN_DIR       = Path("models/collaborative")
REPORTS_DIR   = Path("reports/metrics")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Helper functions
def minmax_per_user(df, score_col="predicted_score"):
    out = df.copy()
    grp = out.groupby("user_id")[score_col]
    mn  = grp.transform("min")
    mx  = grp.transform("max")
    denom = (mx - mn).replace(0, 1)
    out[score_col] = (out[score_col] - mn) / denom
    return out

def ndcg_at_k(recommended, relevant, k=10):
    topk = recommended[:k]
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_n))
    if idcg == 0: return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, book in enumerate(topk) if book in relevant)
    return dcg / idcg

def run_optuna_study():
    print("Loading datasets...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_interactions.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test_interactions.csv")
    
    all_users = train_df["user_id"].unique()
    user_seen = train_df.groupby("user_id")["book_id"].apply(set).to_dict()
    ground_truth = test_df.groupby("user_id")["book_id"].apply(set).to_dict()
    
    user_counts = train_df.groupby("user_id").size().to_dict()
    normal_users = {u for u, c in user_counts.items() if c >= 5}
    eval_users = [u for u in ground_truth if ground_truth[u] and u in normal_users]
    
    # Preload Models
    print("Pre-processing models...")
    svd_recs = pd.read_csv(SVD_DIR / "user_recommendations.csv")
    knn_recs = pd.read_csv(KNN_DIR / "knn_recommendations.csv")
    cb_recs  = pd.read_csv(CB_DIR / "content_recommendations.csv")
    
    # Popularity
    book_counts = train_df.groupby("book_id").size().reset_index(name="interaction_count")
    min_count, max_count = book_counts["interaction_count"].min(), book_counts["interaction_count"].max()
    book_counts["pop_score"] = (book_counts["interaction_count"] - min_count) / (max_count - min_count)
    pop_map = book_counts.set_index("book_id")["pop_score"].to_dict()
    
    # CB
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
    
    # Normalize (Only for normal users to save time)
    svd_norm = minmax_per_user(svd_recs[svd_recs["user_id"].isin(normal_users)][["user_id", "book_id", "predicted_score"]]).rename(columns={"predicted_score": "svd_score"})
    knn_norm = minmax_per_user(knn_recs[knn_recs["user_id"].isin(normal_users)][["user_id", "book_id", "predicted_score"]]).rename(columns={"predicted_score": "knn_score"})
    content_norm = minmax_per_user(content_user_recs[content_user_recs["user_id"].isin(normal_users)][["user_id", "book_id", "predicted_score"]]).rename(columns={"predicted_score": "cb_score"})
    
    merged = pd.merge(svd_norm, knn_norm, on=["user_id", "book_id"], how="outer")
    merged = pd.merge(merged, content_norm, on=["user_id", "book_id"], how="outer")
    merged.fillna(0.0, inplace=True)
    merged["pop_score"] = merged["book_id"].map(lambda b: pop_map.get(b, 0.0))
    
    # Filter seen
    merged = merged[~merged.set_index(["user_id", "book_id"]).index.isin([(u, b) for u, books in user_seen.items() for b in books])]
    
    print("\nStarting Optuna Study for Normal Users (50 Trials)...")
    def objective(trial):
        # Sample weights
        w_svd_raw = trial.suggest_float("W_SVD", 0.0, 1.0)
        w_knn_raw = trial.suggest_float("W_KNN", 0.0, 1.0)
        w_cb_raw  = trial.suggest_float("W_CB", 0.0, 1.0)
        w_pop     = trial.suggest_float("W_POP", 0.0, 0.5)
        
        # Normalize sum to 1
        total = w_svd_raw + w_knn_raw + w_cb_raw
        if total == 0: total = 1.0
        w_svd = w_svd_raw / total
        w_knn = w_knn_raw / total
        w_cb  = w_cb_raw / total
        
        # Vectorized scoring
        scores = (merged["svd_score"] * w_svd) + (merged["knn_score"] * w_knn) + (merged["cb_score"] * w_cb)
        boosted = ((1.0 - w_pop) * scores) + (w_pop * merged["pop_score"])
        
        # Extract Top 10
        merged["temp_score"] = boosted
        temp_merged = merged[["user_id", "book_id", "temp_score"]].copy()
        temp_merged = temp_merged.sort_values(["user_id", "temp_score"], ascending=[True, False])
        top10 = temp_merged.groupby("user_id").head(10)
        
        # Evaluate NDCG
        user_recs_dict = top10.groupby("user_id")["book_id"].apply(list).to_dict()
        ndcgs = [ndcg_at_k(user_recs_dict.get(u, []), ground_truth[u], 10) for u in eval_users]
        return np.mean(ndcgs)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    
    print("\n============= OPTUNA RESULTS =============")
    print(f"Best NDCG@10: {study.best_value:.4f}")
    best_params = study.best_params
    total = best_params["W_SVD"] + best_params["W_KNN"] + best_params["W_CB"]
    
    final_svd = best_params["W_SVD"] / total
    final_knn = best_params["W_KNN"] / total
    final_cb  = best_params["W_CB"] / total
    final_pop = best_params["W_POP"]
    
    print(f"Optimal W_SVD: {final_svd:.4f}")
    print(f"Optimal W_KNN: {final_knn:.4f}")
    print(f"Optimal W_CB : {final_cb:.4f}")
    print(f"Optimal W_POP: {final_pop:.4f}")
    print("==========================================")
    
    # Save Report
    df_params = pd.DataFrame([{
        "W_SVD": final_svd,
        "W_KNN": final_knn,
        "W_CB": final_cb,
        "W_POP": final_pop,
        "Best_NDCG": study.best_value
    }])
    df_params.to_csv(REPORTS_DIR / "optuna_best_weights.csv", index=False)
    
    with open(REPORTS_DIR / "optuna_optimization_report.md", "w") as f:
        f.write("# Phase 14: Optuna Weight Optimization Report\n\n")
        f.write("Optuna automatically tuned the hyperparameters for 'Normal Users' (>= 5 interactions).\n\n")
        f.write("## Best Found Parameters\n")
        f.write(f"- **SVD Weight**: {final_svd:.4f}\n")
        f.write(f"- **KNN Weight**: {final_knn:.4f}\n")
        f.write(f"- **Content-Based Weight**: {final_cb:.4f}\n")
        f.write(f"- **Popularity Boost**: {final_pop:.4f}\n\n")
        f.write("## Performance\n")
        f.write(f"- **Maximized NDCG@10**: {study.best_value:.4f}\n")
        
    print(f"\nSaved Report to {REPORTS_DIR}/optuna_optimization_report.md")

if __name__ == "__main__":
    run_optuna_study()
