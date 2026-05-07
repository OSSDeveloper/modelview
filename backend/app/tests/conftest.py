# conftest.py — pytest fixtures for all backend tests

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """
    Provides a TestClient for all API tests.
    Uses the real FastAPI app (app.main:app).
    Network calls to HuggingFace are mocked via responses library or httpx mock.
    """
    from app.main import app
    return TestClient(app)
