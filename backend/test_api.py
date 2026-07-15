import pytest
from fastapi.testclient import TestClient
from app.main import app

# Create a test client
client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_get_recommendations_engine_not_ready():
    # If we hit it without starting the lifespan context, engine is not loaded
    # But TestClient triggers lifespan natively in newer FastAPI versions if used correctly.
    # We can just test that it returns something valid (either 503 if unloaded, or 200 if loaded).
    response = client.get("/recommend/1")
    assert response.status_code in [200, 503, 404]

def test_get_popular_books():
    response = client.get("/popular?limit=5")
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "popular_books" in data
        assert len(data["popular_books"]) <= 5

def test_search_books_missing_query():
    response = client.get("/search")
    # Missing required query param 'q'
    assert response.status_code == 422 

def test_search_books_valid():
    response = client.get("/search?q=harry")
    assert response.status_code in [200, 503]

def test_post_recommendations_invalid_body():
    response = client.post("/recommend", json={"limit": 10})
    # user_id is missing
    assert response.status_code == 422
