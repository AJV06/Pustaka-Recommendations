from fastapi import APIRouter, HTTPException

from backend.app.schemas import RecommendationRequest
from backend.app.services.recommendation_service import RecommendationService

router = APIRouter()

service = RecommendationService()


@router.post("/recommend")
def recommend(req: RecommendationRequest):

    return service.recommend(
        req.user_id,
        req.top_n
    )

@router.get("/users")
def get_users():
    return (
        service.hybrid["user_id"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

@router.get("/users/{user_id}/profile")
def get_user_profile(user_id: str):
    profile = service.get_user_profile(user_id)

    if profile is None:
        raise HTTPException(status_code=404, detail="User not found")

    return profile

@router.get("/books")
def books():

    return (
        service.books[
            [
                "book_id",
                "title",
                "author",
                "genre",
                "rating"
            ]
        ]
        .to_dict(orient="records")
    )
