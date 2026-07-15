from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from app.schemas import (
    BookRecommendation,
    SimilarBooksResponse, 
    PopularBooksResponse,
    SearchResponse,
    RecommendRequest,
    UserProfileResponse
)
from app.services import RecommendationService

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.get("/users", response_model=List[str])
def get_users():
    return RecommendationService.get_users()

@router.get("/users/{user_id}/profile", response_model=UserProfileResponse)
def get_user_profile(user_id: str):
    return RecommendationService.get_user_profile(user_id)

@router.get("/recommend/{user_id}", response_model=List[BookRecommendation])
def get_recommendations(user_id: str, limit: int = Query(10, ge=1, le=100)):
    return RecommendationService.get_recommendations(user_id, limit)

@router.get("/similar/{book_id}", response_model=SimilarBooksResponse)
def get_similar_books(book_id: str, limit: int = Query(10, ge=1, le=100)):
    return RecommendationService.get_similar_books(book_id, limit)

@router.get("/popular", response_model=PopularBooksResponse)
def get_popular_books(limit: int = Query(10, ge=1, le=100)):
    return RecommendationService.get_popular_books(limit)

@router.get("/search", response_model=SearchResponse)
def search_books(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=100)):
    return RecommendationService.search_books(q, limit)

@router.post("/recommend", response_model=List[BookRecommendation])
def post_recommendations(request: RecommendRequest):
    # Frontend passes top_n or limit
    limit = request.top_n if request.top_n is not None else request.limit
    return RecommendationService.get_recommendations(request.user_id, limit)
