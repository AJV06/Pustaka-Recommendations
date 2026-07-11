from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    user_id: str
    top_n: int = 10