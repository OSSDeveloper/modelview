# =============================================================================
# HF CLIENT — TDD TEST SUITE
# Tests MUST pass before implementation is considered complete.
# Run with: cd backend && python -m pytest app/tests/test_hf_client.py -v
# =============================================================================

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.hf_client import HuggingFaceClient, HuggingFaceAPIError, ModelNotFoundError, RateLimitError


# =============================================================================
# FIXTURES — Mock responses
# =============================================================================

GPT2_CONFIG_JSON = """{
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "n_positions": 1024,
    "vocab_size": 50257,
    "intermediate_size": 3072
}"""

CLIP_CONFIG_JSON = """{
    "model_type": "clip",
    "architectures": ["CLIPModel"],
    "hidden_size": 512,
    "projection_dim": 512,
    "vision_config": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "patch_size": 32,
        "image_size": 224
    },
    "text_config": {
        "hidden_size": 512,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "vocab_size": 49408
    }
}"""


# =============================================================================
# TEST 1: Successful config fetch — GPT-2
# =============================================================================
class TestFetchGPT2:
    @pytest.mark.asyncio
    async def test_fetch_returns_config_dict(self):
        client = HuggingFaceClient()
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = GPT2_CONFIG_JSON
            mock_response.json.return_value = __import__("json").loads(GPT2_CONFIG_JSON)
            mock_get.return_value = mock_response

            config = await client.fetch_config("openai-community", "gpt2")
            assert config["model_type"] == "gpt2"
            assert config["n_layer"] == 12

    @pytest.mark.asyncio
    async def test_url_uses_raw_endpoint(self):
        client = HuggingFaceClient()
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = GPT2_CONFIG_JSON
            mock_response.json.return_value = __import__("json").loads(GPT2_CONFIG_JSON)
            mock_get.return_value = mock_response

            await client.fetch_config("openai-community", "gpt2")
            called_url = mock_get.call_args[0][0]
            assert "huggingface.co" in str(called_url)
            assert "raw/main/config.json" in str(called_url)


# =============================================================================
# TEST 2: CLIP with nested vision/text config
# =============================================================================
class TestFetchCLIP:
    @pytest.mark.asyncio
    async def test_nested_config_parsed(self):
        client = HuggingFaceClient()
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = CLIP_CONFIG_JSON
            mock_response.json.return_value = __import__("json").loads(CLIP_CONFIG_JSON)
            mock_get.return_value = mock_response

            config = await client.fetch_config("openai", "clip-vit-base-patch32")
            assert "vision_config" in config
            assert config["vision_config"]["hidden_size"] == 768
            assert "text_config" in config
            assert config["text_config"]["vocab_size"] == 49408


# =============================================================================
# TEST 3: Model not found — 404
# =============================================================================
class Test404:
    @pytest.mark.asyncio
    async def test_raises_model_not_found(self):
        client = HuggingFaceClient()
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            with pytest.raises(ModelNotFoundError):
                await client.fetch_config("nonexistent-org", "nonexistent-model")


# =============================================================================
# TEST 4: Rate limiting — 429
# =============================================================================
class TestRateLimit:
    @pytest.mark.asyncio
    async def test_raises_rate_limit_error(self):
        client = HuggingFaceClient()
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "60"}
            mock_get.return_value = mock_response

            with pytest.raises(RateLimitError):
                await client.fetch_config("some-org", "some-model")


# =============================================================================
# TEST 5: Timeout handling
# =============================================================================
class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_raises_api_error(self):
        client = HuggingFaceClient()
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Request timed out")

            with pytest.raises(HuggingFaceAPIError) as exc_info:
                await client.fetch_config("openai-community", "gpt2")
            assert "timeout" in str(exc_info.value).lower()


# =============================================================================
# TEST 6: Cache hit
# =============================================================================
class TestCaching:
    @pytest.mark.asyncio
    async def test_second_call_returns_cached(self):
        client = HuggingFaceClient()
        cache_key = "openai-community/gpt2"

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = GPT2_CONFIG_JSON
            mock_response.json.return_value = __import__("json").loads(GPT2_CONFIG_JSON)
            mock_get.return_value = mock_response

            # First call — should hit network
            config1 = await client.fetch_config("openai-community", "gpt2")
            # Second call — should return cached
            config2 = await client.fetch_config("openai-community", "gpt2")

            # Network should only be called once
            assert mock_get.call_count == 1
            assert config1 == config2


# =============================================================================
# TEST 7: Cache miss → fetch from HuggingFace
# =============================================================================
class TestCacheMiss:
    @pytest.mark.asyncio
    async def test_cache_miss_fetches_from_network(self):
        client = HuggingFaceClient()
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = GPT2_CONFIG_JSON
            mock_response.json.return_value = __import__("json").loads(GPT2_CONFIG_JSON)
            mock_get.return_value = mock_response

            config = await client.fetch_config("openai-community", "gpt2")
            assert mock_get.call_count == 1


# =============================================================================
# TEST 8: Gated model — Llama (minimal public config)
# =============================================================================
class TestGatedModel:
    @pytest.mark.asyncio
    async def test_gated_model_still_fetches(self):
        client = HuggingFaceClient()
        llama_public_config = '{"model_type": "llama", "num_hidden_layers": 32}'
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = llama_public_config
            mock_response.json.return_value = {"model_type": "llama", "num_hidden_layers": 32}
            mock_get.return_value = mock_response

            config = await client.fetch_config("meta-llama", "Llama-3.1-8B-Instruct")
            assert config["model_type"] == "llama"


# =============================================================================
# TEST 9: Model URL parsing — short ID to org/model
# =============================================================================
class TestURLParsing:
    def test_short_model_id_converted(self):
        client = HuggingFaceClient()
        url = client._build_config_url("gpt2")
        assert "openai-community/gpt2" in url

    def test_full_model_id_unchanged(self):
        client = HuggingFaceClient()
        url = client._build_config_url("openai-community/gpt2")
        assert "openai-community/gpt2" in url

    def test_org_model_unchanged(self):
        client = HuggingFaceClient()
        url = client._build_config_url("google/t5-small")
        assert "google/t5-small" in url


# =============================================================================
# TEST 10: Health check endpoint
# =============================================================================
class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_returns_ok(self):
        client = HuggingFaceClient()
        result = await client.health_check()
        assert result is True
