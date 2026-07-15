from pydantic import BaseModel, Field
from typing import List, Optional

class BookRecommendation(BaseModel):
    book_id: str
    title: str
    author: str
    score: float
    genre: Optional[str] = None
    rating: Optional[float] = None
    hybrid_score: Optional[float] = None
    knn_score: Optional[float] = None
    cb_score: Optional[float] = None
    svd_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[BookRecommendation]

class SimilarBooksResponse(BaseModel):
    book_id: str
    similar_books: List[BookRecommendation]

class PopularBooksResponse(BaseModel):
    popular_books: List[BookRecommendation]

class SearchResponse(BaseModel):
    results: List[BookRecommendation]

class RecommendRequest(BaseModel):
    user_id: str
    top_n: Optional[int] = Field(default=10, ge=1, le=100)
    # Keeping limit for backward compatibility if needed, but top_n is used by frontend
    limit: Optional[int] = Field(default=10, ge=1, le=100)

class RecentBook(BaseModel):
    transaction_id: str
    book_id: str
    title: str
    author: str
    genre: str
    transaction_type: str
    transaction_date: str

class UserProfileResponse(BaseModel):
    user_id: str
    user_name: str
    city: str
    state: str
    preferred_language: str
    subscription_type: str
    favorite_genres: List[str]
    recent_books: List[RecentBook]