import pandas as pd
import numpy as np
from pathlib import Path
import logging
from app.config import settings
from typing import List, Dict, Any

logger = logging.getLogger("api_logger")

class RecommenderEngine:
    def __init__(self):
        self.user_recommendations = {}
        self.similar_books = {}
        self.popular_books = []
        self.book_metadata = {}
        self.user_profiles = {}
        self.all_users = []
        
        self.model_path = Path(settings.MODEL_PATH)
        self.data_path = Path(settings.DATA_PATH)
        self.is_loaded = False

    def load_all(self):
        logger.info("Loading pre-trained recommendations into memory...")
        try:
            self._load_metadata()
            self._load_popular()
            self._load_personalized()
            self._load_similar()
            self._load_user_profiles()
            self.is_loaded = True
            logger.info("Recommendation engine successfully loaded!")
        except Exception as e:
            logger.error(f"Failed to load recommendation data: {e}")
            raise e

    def _load_metadata(self):
        logger.info("Loading book metadata...")
        features_path = self.data_path / "features" / "book_features.csv"
        if features_path.exists():
            df = pd.read_csv(features_path)
            for _, row in df.iterrows():
                book_id = str(row["book_id"])
                # Extract all necessary metadata for the frontend
                self.book_metadata[book_id] = {
                    "book_id": book_id,
                    "title": str(row.get("title", f"Book {book_id}")),
                    "author": str(row.get("author", "Unknown Author")),
                    "genre": str(row.get("genre", "General")),
                    "rating": float(row.get("rating", 0.0)) if pd.notna(row.get("rating")) else 0.0
                }
        else:
            logger.warning(f"Metadata file not found at {features_path}")

    def _load_popular(self):
        logger.info("Computing popular books...")
        interactions_path = self.data_path / "processed" / "train_interactions.csv"
        if interactions_path.exists():
            df = pd.read_csv(interactions_path)
            book_counts = df.groupby("book_id").size().reset_index(name="count")
            book_counts = book_counts.sort_values("count", ascending=False).head(50)
            
            max_count = book_counts["count"].max()
            book_counts["score"] = book_counts["count"] / max_count
            
            for _, row in book_counts.iterrows():
                book_id = str(row["book_id"])
                meta = self.book_metadata.get(book_id, {"title": f"Book {book_id}", "author": "Unknown", "genre": "General", "rating": 0.0})
                self.popular_books.append({
                    "book_id": book_id,
                    "title": meta["title"],
                    "author": meta["author"],
                    "genre": meta["genre"],
                    "rating": meta["rating"],
                    "score": round(float(row["score"]), 4),
                    "hybrid_score": round(float(row["score"]), 4),
                    "knn_score": 0.0,
                    "cb_score": 0.0,
                    "svd_score": 0.0
                })
        else:
            logger.warning(f"Interactions file not found at {interactions_path}")

    def _load_personalized(self):
        logger.info("Loading personalized recommendations...")
        recs_path = self.model_path / "hybrid" / "cold_start_recommendations.csv"
        if not recs_path.exists():
            recs_path = self.model_path / "hybrid" / "diverse_recommendations.csv"
            
        if recs_path.exists():
            df = pd.read_csv(recs_path)
            for user, group in df.groupby("user_id"):
                recs = []
                for _, row in group.iterrows():
                    book_id = str(row["book_id"])
                    meta = self.book_metadata.get(book_id, {"title": f"Book {book_id}", "author": "Unknown", "genre": "General", "rating": 0.0})
                    
                    score = float(row.get("boosted_score", row.get("predicted_score", 0.0)))
                    
                    recs.append({
                        "book_id": book_id,
                        "title": meta["title"],
                        "author": meta["author"],
                        "genre": meta["genre"],
                        "rating": meta["rating"],
                        "score": round(score, 4),
                        "hybrid_score": round(score, 4),
                        "knn_score": round(float(row.get("knn_score", 0.0)), 4),
                        "cb_score": round(float(row.get("cb_score", 0.0)), 4),
                        "svd_score": round(float(row.get("svd_score", 0.0)), 4)
                    })
                self.user_recommendations[str(user)] = recs
        else:
            logger.warning(f"Personalized recommendations not found at {recs_path}")

    def _load_similar(self):
        logger.info("Loading content similarities...")
        recs_path = self.model_path / "content_based" / "content_recommendations.csv"
        if recs_path.exists():
            df = pd.read_csv(recs_path)
            for source, group in df.groupby("source_book"):
                recs = []
                for _, row in group.iterrows():
                    book_id = str(row["recommended_book"])
                    meta = self.book_metadata.get(book_id, {"title": f"Book {book_id}", "author": "Unknown", "genre": "General", "rating": 0.0})
                    score = 1.0 / (float(row["rank"]) + 1.0)
                    recs.append({
                        "book_id": book_id,
                        "title": meta["title"],
                        "author": meta["author"],
                        "genre": meta["genre"],
                        "rating": meta["rating"],
                        "score": round(score, 4)
                    })
                self.similar_books[str(source)] = recs

    def _load_user_profiles(self):
        logger.info("Building user profiles from interactions...")
        interactions_path = self.data_path / "processed" / "train_interactions.csv"
        if interactions_path.exists():
            df = pd.read_csv(interactions_path)
            
            # Extract unique users
            self.all_users = sorted([str(u) for u in df["user_id"].unique().tolist()])
            
            # Build profiles
            for user, group in df.groupby("user_id"):
                user_id = str(user)
                recent_books = []
                genres_seen = {}
                
                # Sort by most recent interaction (proxy via index if no real timestamp)
                # Group has 'transaction_date' if available, otherwise just use order
                if "transaction_date" in group.columns:
                    sorted_group = group.sort_values("transaction_date", ascending=False).head(50)
                else:
                    sorted_group = group.head(50)
                
                for idx, row in sorted_group.iterrows():
                    book_id = str(row["book_id"])
                    meta = self.book_metadata.get(book_id, {"title": f"Book {book_id}", "author": "Unknown", "genre": "General"})
                    
                    # Track genre frequency
                    genre = meta["genre"]
                    genres_seen[genre] = genres_seen.get(genre, 0) + 1
                    
                    # Store up to 5 recent books
                    if len(recent_books) < 5:
                        recent_books.append({
                            "transaction_id": str(row.get("transaction_id", f"txn_{idx}")),
                            "book_id": book_id,
                            "title": meta["title"],
                            "author": meta["author"],
                            "genre": genre,
                            "transaction_type": str(row.get("transaction_type", "checkout")),
                            "transaction_date": str(row.get("transaction_date", "2024-01-01"))
                        })
                        
                top_genres = sorted(genres_seen.keys(), key=lambda k: genres_seen[k], reverse=True)[:3]
                
                self.user_profiles[user_id] = {
                    "user_id": user_id,
                    "user_name": f"User {user_id}",
                    "city": "Bangalore", # Mocked
                    "state": "Karnataka", # Mocked
                    "preferred_language": "English", # Mocked
                    "subscription_type": "Premium", # Mocked
                    "favorite_genres": top_genres,
                    "recent_books": recent_books
                }
                
    # ========================================================
    # ACCESS METHODS
    # ========================================================

    def get_users(self) -> List[str]:
        return self.all_users[:100] # Return top 100 for the dropdown UI

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        return self.user_profiles.get(user_id)

    def get_recommendations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        recs = self.user_recommendations.get(user_id, self.popular_books)
        return recs[:limit]

    def get_similar_books(self, book_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.similar_books.get(book_id, [])[:limit]

    def get_popular_books(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.popular_books[:limit]

    def search_books(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = query.lower()
        results = []
        for book_id, meta in self.book_metadata.items():
            if query in meta["title"].lower() or query in meta["author"].lower():
                results.append({
                    "book_id": book_id,
                    "title": meta["title"],
                    "author": meta["author"],
                    "genre": meta["genre"],
                    "rating": meta["rating"],
                    "score": 1.0
                })
                if len(results) >= limit:
                    break
        return results

# Singleton instance
engine = RecommenderEngine()
