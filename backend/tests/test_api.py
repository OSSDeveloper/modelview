# Backend Tests - TDD First
# These tests define the expected API behavior

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"


class TestModelsEndpoint:
    """Test models API endpoints"""

    def test_get_models_returns_list(self, client):
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_model_by_id(self, client):
        response = client.get("/api/v1/models/1")
        assert response.status_code in [200, 404]

    def test_create_model(self, client):
        payload = {
            "name": "Test Model",
            "type": "diagram",
            "data": {"nodes": [], "edges": []}
        }
        response = client.post("/api/v1/models", json=payload)
        assert response.status_code in [200, 201, 400]


class TestVisualizationEndpoint:
    """Test visualization data endpoint"""

    def test_get_visualization_data(self, client):
        response = client.get("/api/v1/visualization/1")
        assert response.status_code in [200, 404]

    def test_get_visualization_types(self, client):
        response = client.get("/api/v1/visualization/types")
        assert response.status_code == 200
        assert isinstance(response.json(), list)