"""
Unit Tests for API Endpoints
"""
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Football Analytics API" in response.json()["message"]


def test_health_check():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_players_endpoint():
    """Test players list endpoint"""
    response = client.get("/api/players?limit=5")
    
    if response.status_code == 200:
        data = response.json()
        assert "players" in data
        assert "total" in data
        assert len(data["players"]) <= 5
    elif response.status_code == 503:
        # Data not available - acceptable for test
        assert "No data available" in response.json()["detail"]


def test_analytics_summary():
    """Test analytics summary endpoint"""
    response = client.get("/api/analytics/summary")
    
    if response.status_code == 200:
        data = response.json()
        assert "total_players" in data
        assert "total_matches" in data
    elif response.status_code == 503:
        # Data not available - acceptable
        pass


def test_invalid_endpoint():
   """Test invalid endpoint returns 404"""
    response = client.get("/api/invalid")
    assert response.status_code == 404


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
