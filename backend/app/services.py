from app.recommender import engine
from fastapi import HTTPException
import logging

logger = logging.getLogger("api_logger")

class RecommendationService:
    @staticmethod
    def get_users():
        if not engine.is_loaded:
            raise HTTPException(status_code=503, detail="Engine not ready")
        return engine.get_users()

    @staticmethod
    def get_user_profile(user_id: str):
        if not engine.is_loaded:
            raise HTTPException(status_code=503, detail="Engine not ready")
            
        profile = engine.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
            
        return profile

    @staticmethod
    def get_recommendations(user_id: str, limit: int):
        if not engine.is_loaded:
            raise HTTPException(status_code=503, detail="Engine not ready")
        
        recs = engine.get_recommendations(user_id, limit)
        if not recs:
            raise HTTPException(status_code=404, detail="No recommendations found for this user")
            
        logger.info(f"Served {len(recs)} recommendations for user {user_id}")
        return recs # Frontend expects raw list of BookRecommendation, not wrapped object

    @staticmethod
    def get_similar_books(book_id: str, limit: int):
        if not engine.is_loaded:
            raise HTTPException(status_code=503, detail="Engine not ready")
            
        if book_id not in engine.book_metadata:
            raise HTTPException(status_code=404, detail="Book not found")
            
        recs = engine.get_similar_books(book_id, limit)
        return {"book_id": book_id, "similar_books": recs}

    @staticmethod
    def get_popular_books(limit: int):
        if not engine.is_loaded:
            raise HTTPException(status_code=503, detail="Engine not ready")
            
        recs = engine.get_popular_books(limit)
        return {"popular_books": recs}

    @staticmethod
    def search_books(query: str, limit: int):
        if not engine.is_loaded:
            raise HTTPException(status_code=503, detail="Engine not ready")
            
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        results = engine.search_books(query, limit)
        return {"results": results}
