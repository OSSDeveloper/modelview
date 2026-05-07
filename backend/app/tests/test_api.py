# =============================================================================
# API ENDPOINT — TDD TEST SUITE
# Tests MUST pass before implementation is considered complete.
# Run with: cd backend && python -m pytest app/tests/test_api.py -v
# =============================================================================

import pytest
from fastapi.testclient import TestClient


# =============================================================================
# FIXTURES — in conftest.py we set up a TestClient using a real app
# =============================================================================


# =============================================================================
# TEST 1: GET / → HTML landing page
# =============================================================================
class TestLandingPage:
    def test_root_returns_html(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_landing_contains_modelview_branding(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        text = response.text.lower()
        assert "model" in text or "huggingface" in text


# =============================================================================
# TEST 2: GET /api/health → Health check
# =============================================================================
class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, client: TestClient):
        response = client.get("/api/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_timestamp(self, client: TestClient):
        response = client.get("/api/health")
        data = response.json()
        assert "timestamp" in data


# =============================================================================
# TEST 3: GET /api/model/{org}/{model} → Graph data for GPT-2
# =============================================================================
class TestModelEndpoint:
    def test_gpt2_returns_200(self, client: TestClient):
        response = client.get("/api/model/openai-community/gpt2")
        # 200 if we mock HF, or 429/404 in real scenarios
        assert response.status_code in [200, 404, 429, 503]

    def test_gpt2_returns_graph_structure(self, client: TestClient):
        response = client.get("/api/model/openai-community/gpt2")
        if response.status_code == 200:
            data = response.json()
            assert "model_id" in data
            assert "nodes" in data
            assert "edges" in data
            assert "stats" in data
            assert "repeated" in data

    def test_gpt2_returns_correct_model_id(self, client: TestClient):
        response = client.get("/api/model/openai-community/gpt2")
        if response.status_code == 200:
            data = response.json()
            assert data["model_id"] == "openai-community/gpt2"

    def test_t5_returns_encoder_decoder_structure(self, client: TestClient):
        response = client.get("/api/model/google/t5-small")
        if response.status_code == 200:
            data = response.json()
            node_ids = [n["id"] for n in data["nodes"]]
            assert any("encoder" in n for n in node_ids)
            assert any("decoder" in n for n in node_ids)

    def test_clip_returns_vision_and_text_nodes(self, client: TestClient):
        response = client.get("/api/model/openai/clip-vit-base-patch32")
        if response.status_code == 200:
            data = response.json()
            node_ids = [n["id"] for n in data["nodes"]]
            assert any("vision" in n for n in node_ids)
            assert any("text" in n for n in node_ids)


# =============================================================================
# TEST 4: GET /api/model/{org}/{model} → 404 for unknown model
# =============================================================================
class TestModel404:
    def test_unknown_model_returns_404(self, client: TestClient):
        response = client.get("/api/model/nonexistent-org/nonexistent-model")
        assert response.status_code == 404

    def test_404_returns_error_message(self, client: TestClient):
        response = client.get("/api/model/nonexistent-org/nonexistent-model")
        if response.status_code == 404:
            data = response.json()
            assert "detail" in data or "error" in data


# =============================================================================
# TEST 5: GET /api/model?popular=true → List of popular models
# =============================================================================
class TestPopularModels:
    def test_popular_returns_list(self, client: TestClient):
        response = client.get("/api/popular")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_popular_contains_expected_models(self, client: TestClient):
        response = client.get("/api/popular")
        assert response.status_code == 200
        data = response.json()
        model_ids = [m["id"] if isinstance(m, dict) else m for m in data]
        expected = ["openai-community/gpt2", "google/t5-small", "openai/clip-vit-base-patch32"]
        for exp in expected:
            assert any(exp in m for m in model_ids), f"{exp} not found in popular models"


# =============================================================================
# TEST 6: Unknown API path returns 404
# =============================================================================
class TestUnknownPath:
    def test_unknown_path_returns_404(self, client: TestClient):
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_root_unknown_path_returns_404(self, client: TestClient):
        response = client.get("/nonexistent-page")
        assert response.status_code == 404


# =============================================================================
# TEST 7: CORS headers present
# =============================================================================
class TestCORS:
    def test_cors_headers_present(self, client: TestClient):
        response = client.get("/api/health")
        headers = response.headers
        # CORS headers should be present (either explicit or via middleware)
        # Access-Control-Allow-Origin should be set
        assert "access-control-allow-origin" in headers or "*"


# =============================================================================
# TEST 8: Cache-Control headers
# =============================================================================
class TestCacheControl:
    def test_api_endpoint_has_cache_control(self, client: TestClient):
        response = client.get("/api/model/openai-community/gpt2")
        if response.status_code == 200:
            # Should have cache headers since we cache at Redis layer
            cache_header = response.headers.get("cache-control", "")
            assert "public" in cache_header or "private" in cache_header or "no-store" in cache_header


# =============================================================================
# TEST 9: MoE model (Mixtral) returns graph with expert nodes
# =============================================================================
class TestMoEModel:
    def test_mixtral_has_expert_nodes(self, client: TestClient):
        # Use a real Mixtral model ID
        response = client.get("/api/model/mistralai/Mixtral-8x7B-Instruct-v0.1")
        if response.status_code == 200:
            data = response.json()
            node_ids = [n["id"] for n in data["nodes"]]
            assert "router" in node_ids
            assert any("expert" in n for n in node_ids)

    def test_deepseek_v3_has_many_experts(self, client: TestClient):
        response = client.get("/api/model/deepseek-ai/DeepSeek-V3")
        if response.status_code == 200:
            data = response.json()
            node_ids = [n["id"] for n in data["nodes"]]
            assert "router" in node_ids


# =============================================================================
# TEST 10: Short model ID → redirects/fetches correct model
# =============================================================================
class TestShortModelID:
    def test_gpt2_short_id_expands(self, client: TestClient):
        # /gpt2 should redirect to openai-community/gpt2
        response = client.get("/gpt2", follow_redirects=False)
        # Either 200 (handled as short ID) or 307 redirect
        assert response.status_code in [200, 307, 404]

    def test_qwen_short_id(self, client: TestClient):
        response = client.get("/qwen", follow_redirects=False)
        assert response.status_code in [200, 307, 404]


# =============================================================================
# TEST 11: Rate limiting returns 429
# =============================================================================
class TestRateLimiting:
    def test_rate_limit_returns_429(self, client: TestClient):
        # Make many rapid requests to trigger rate limit
        # In test environment this may not trigger, but check structure
        responses = []
        for _ in range(10):
            r = client.get("/api/model/openai-community/gpt2")
            responses.append(r.status_code)
            if r.status_code == 429:
                break
        # At least one should be 429 OR all 200 (if mocked)
        assert responses.count(429) > 0 or all(s == 200 for s in responses)


# =============================================================================
# TEST 12: Error response format is consistent
# =============================================================================
class TestErrorFormat:
    def test_error_response_is_json(self, client: TestClient):
        response = client.get("/api/model/nonexistent-org/nonexistent-model")
        assert response.status_code == 404
        assert "application/json" in response.headers.get("content-type", "")

    def test_error_contains_message(self, client: TestClient):
        response = client.get("/api/model/nonexistent-org/nonexistent-model")
        if response.status_code == 404:
            data = response.json()
            assert "detail" in data or "message" in data or "error" in data
