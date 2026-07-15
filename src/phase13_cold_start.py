import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# PATHS & CONSTANTS
# =====================================================
PROCESSED_DIR = Path("data/processed")
FEATURE_DIR   = Path("data/features")
SVD_DIR       = Path("models/svd")
CB_DIR        = Path("models/content_based")
KNN_DIR       = Path("models/collaborative")
HYBRID_DIR    = Path("models/hybrid")

TOP_CANDIDATES = 30
TOP_N = 10
W_POP = 0.15
LAMBDA_DIV = 0.5

def minmax_per_user(df, score_col="predicted_score"):
    out = df.copy()
    grp = out.groupby("user_id")[score_col]
    mn  = grp.transform("min")
    mx  = grp.transform("max")
    denom = (mx - mn).replace(0, 1)
    out[score_col] = (out[score_col] - mn) / denom
    return out

def run_cold_start_pipeline():
    print("Loading datasets...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_interactions.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test_interactions.csv")
    book_features = pd.read_csv(FEATURE_DIR / "book_features.csv")
    
    book_genre = book_features.set_index("book_id")["genre"].to_dict()
    
    all_users = train_df["user_id"].unique()
    user_seen = train_df.groupby("user_id")["book_id"].apply(set).to_dict()
    user_counts = train_df.groupby("user_id").size().to_dict()
    
    cold_users = {u for u, c in user_counts.items() if c < 5}
    normal_users = {u for u, c in user_counts.items() if c >= 5}
    
    # 1. Compute Popularity Scores
    print("Computing Popularity Scores...")
    book_counts = train_df.groupby("book_id").size().reset_index(name="interaction_count")
    min_count, max_count = book_counts["interaction_count"].min(), book_counts["interaction_count"].max()
    book_counts["pop_score"] = (book_counts["interaction_count"] - min_count) / (max_count - min_count)
    pop_map = book_counts.set_index("book_id")["pop_score"].to_dict()
    
    # 2. Adaptive Weights for Normal Users
    user_weights = {}
    for u in normal_users:
        c = user_counts.get(u, 0)
        if c < 20: user_weights[u] = {"w_cb": 0.2, "w_knn": 0.6, "w_svd": 0.2}
        else: user_weights[u] = {"w_cb": 0.0, "w_knn": 0.8, "w_svd": 0.2}
            
    # 3. Load Base Models
    print("Loading Base Models...")
    svd_recs = pd.read_csv(SVD_DIR / "user_recommendations.csv")
    knn_recs = pd.read_csv(KNN_DIR / "knn_recommendations.csv")
    cb_recs  = pd.read_csv(CB_DIR / "content_recommendations.csv")
    
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
    
    print("Normalizing Scores...")
    svd_norm = minmax_per_user(svd_recs[["user_id", "book_id", "predicted_score"]]).rename(columns={"predicted_score": "svd_score"})
    knn_norm = minmax_per_user(knn_recs[["user_id", "book_id", "predicted_score"]]).rename(columns={"predicted_score": "knn_score"})
    content_norm = minmax_per_user(content_user_recs[["user_id", "book_id", "predicted_score"]]).rename(columns={"predicted_score": "cb_score"})
    
    # ---------------------------------------------------------
    # PIPELINE A: NORMAL USERS (Phase 12: Adaptive + Pop + MMR)
    # ---------------------------------------------------------
    print("Running Normal User Pipeline (MMR Diversity)...")
    merged_normal = pd.merge(svd_norm[svd_norm["user_id"].isin(normal_users)], 
                             knn_norm[knn_norm["user_id"].isin(normal_users)], on=["user_id", "book_id"], how="outer")
    merged_normal = pd.merge(merged_normal, 
                             content_norm[content_norm["user_id"].isin(normal_users)], on=["user_id", "book_id"], how="outer")
    merged_normal.fillna(0.0, inplace=True)
    
    w_svd_map = merged_normal["user_id"].map(lambda u: user_weights.get(u, {"w_svd": 0.2})["w_svd"])
    w_knn_map = merged_normal["user_id"].map(lambda u: user_weights.get(u, {"w_knn": 0.6})["w_knn"])
    w_cb_map  = merged_normal["user_id"].map(lambda u: user_weights.get(u, {"w_cb": 0.2})["w_cb"])
    
    merged_normal["adaptive_score"] = merged_normal["svd_score"] * w_svd_map + merged_normal["knn_score"] * w_knn_map + merged_normal["cb_score"] * w_cb_map
    merged_normal["pop_score"] = merged_normal["book_id"].map(lambda b: pop_map.get(b, 0.0))
    merged_normal["boosted_score"] = ((1.0 - W_POP) * merged_normal["adaptive_score"]) + (W_POP * merged_normal["pop_score"])
    
    # Extract Top 30
    merged_normal = merged_normal[~merged_normal.set_index(["user_id", "book_id"]).index.isin([(u, b) for u, books in user_seen.items() for b in books])]
    merged_normal = merged_normal.sort_values(["user_id", "boosted_score"], ascending=[True, False])
    merged_normal["rank"] = merged_normal.groupby("user_id").cumcount() + 1
    candidates_normal = merged_normal[merged_normal["rank"] <= TOP_CANDIDATES].copy()
    
    # MMR
    candidates_normal = minmax_per_user(candidates_normal, score_col="boosted_score")
    normal_final_rows = []
    for user, group in candidates_normal.groupby("user_id"):
        pool = group.to_dict("records")
        selected = []
        selected_genres = set()
        while len(selected) < TOP_N and len(pool) > 0:
            best_score = -np.inf
            best_idx = -1
            for i, cand in enumerate(pool):
                cand_genre = book_genre.get(cand["book_id"], "unknown")
                penalty = 1.0 if cand_genre in selected_genres else 0.0
                mmr_score = (LAMBDA_DIV * cand["boosted_score"]) - ((1 - LAMBDA_DIV) * penalty)
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            winner = pool.pop(best_idx)
            selected.append(winner)
            selected_genres.add(book_genre.get(winner["book_id"], "unknown"))
        for i, rec in enumerate(selected):
            rec["rank"] = i + 1
            normal_final_rows.append(rec)
            
    normal_recs_df = pd.DataFrame(normal_final_rows)
    
    # ---------------------------------------------------------
    # PIPELINE B: COLD USERS (Content + Popularity Only)
    # ---------------------------------------------------------
    print("Running Cold User Pipeline (CB + Popularity)...")
    merged_cold = content_norm[content_norm["user_id"].isin(cold_users)].copy()
    merged_cold["pop_score"] = merged_cold["book_id"].map(lambda b: pop_map.get(b, 0.0))
    # 85% CB, 15% POP
    merged_cold["boosted_score"] = ((1.0 - W_POP) * merged_cold["cb_score"]) + (W_POP * merged_cold["pop_score"])
    
    merged_cold = merged_cold[~merged_cold.set_index(["user_id", "book_id"]).index.isin([(u, b) for u, books in user_seen.items() for b in books])]
    merged_cold = merged_cold.sort_values(["user_id", "boosted_score"], ascending=[True, False])
    merged_cold["rank"] = merged_cold.groupby("user_id").cumcount() + 1
    cold_recs_df = merged_cold[merged_cold["rank"] <= TOP_N].copy()
    
    # ---------------------------------------------------------
    # COMBINE & EVALUATE
    # ---------------------------------------------------------
    final_recs = pd.concat([normal_recs_df, cold_recs_df], ignore_index=True)
    out_path = HYBRID_DIR / "cold_start_recommendations.csv"
    final_recs.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    
    print("\nEvaluating Separately...")
    from phase7_evaluation import ndcg_at_k, precision_at_k, recall_at_k, average_precision_at_k, genre_diversity
    
    ground_truth = test_df.groupby("user_id")["book_id"].apply(set).to_dict()
    user_recs_dict = final_recs.groupby("user_id")["book_id"].apply(list).to_dict()
    eval_users = [u for u in ground_truth if u in user_recs_dict and ground_truth[u]]
    
    cold_eval_users = [u for u in eval_users if u in cold_users]
    normal_eval_users = [u for u in eval_users if u in normal_users]
    
    def evaluate_group(users, group_name):
        ndcgs, precs, recs, maps, divs = [], [], [], [], []
        for u in users:
            relevant = ground_truth[u]
            recommended = user_recs_dict[u]
            ndcgs.append(ndcg_at_k(recommended, relevant, 10))
            precs.append(precision_at_k(recommended, relevant, 10))
            recs.append(recall_at_k(recommended, relevant, 10))
            maps.append(average_precision_at_k(recommended, relevant, 10))
            divs.append(genre_diversity(recommended[:10], book_genre))
        
        print(f"\n============= {group_name.upper()} EVALUATION =============")
        print(f"Evaluated Users: {len(users)}")
        print(f"Diversity@10: {np.mean(divs) if divs else 0:.4f}")
        print(f"NDCG@10: {np.mean(ndcgs) if ndcgs else 0:.4f}")
        print(f"Precision@10: {np.mean(precs) if precs else 0:.4f}")
        print(f"Recall@10: {np.mean(recs) if recs else 0:.4f}")
        print(f"MAP@10: {np.mean(maps) if maps else 0:.4f}")
        print("===============================================")
        
    evaluate_group(eval_users, "Overall")
    evaluate_group(normal_eval_users, "Normal Users (Phase 12 Logic)")
    evaluate_group(cold_eval_users, "Cold Users (CB + Pop Only)")

if __name__ == "__main__":
    run_cold_start_pipeline()
