# Backend Test Fixtures

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def sample_model():
    """Sample model data for testing"""
    return {
        "name": "Sample Process Model",
        "type": "process_flow",
        "data": {
            "nodes": [
                {"id": "1", "label": "Start", "type": "start"},
                {"id": "2", "label": "Process", "type": "task"},
                {"id": "3", "label": "End", "type": "end"}
            ],
            "edges": [
                {"source": "1", "target": "2"},
                {"source": "2", "target": "3"}
            ]
        }
    }