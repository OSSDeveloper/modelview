# backend/app/services/hf_client.py
# HuggingFace API client with Redis caching

import httpx
import redis.asyncio as redis
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime

HF_CONFIG_URL = "https://huggingface.co/{org}/{model}/raw/main/config.json"
HF_API_URL = "https://huggingface.co/api/models/{org}/{model}"
CACHE_TTL = 86400  # 24 hours

# Short name → org/model
SHORT_ID_MAP: Dict[str, str] = {
    "gpt2": "openai-community/gpt2",
    "t5": "google/t5-small",
    "clip": "openai/clip-vit-base-patch32",
    "vit": "google/vit-base-patch16-224",
    "resnet": "microsoft/resnet-50",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen": "Qwen/Qwen2.5-0.8B",
    "qwen-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "phi": "microsoft/Phi-3-mini-4k-instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "gemma": "google/gemma-2-2b-it",
    "falcon": "tiiuae/falcon-7b-instruct",
}


class HuggingFaceAPIError(Exception):
    """Base exception for HuggingFace API errors."""
    pass


class ModelNotFoundError(HuggingFaceAPIError):
    """Raised when model returns 404."""
    pass


class RateLimitError(HuggingFaceAPIError):
    """Raised when rate limited (429)."""
    pass


class HuggingFaceClient:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis: Optional[redis.Redis] = None
        self._http = httpx.AsyncClient(timeout=30.0)

    async def _get_redis(self) -> Optional[redis.Redis]:
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception:
                self._redis = None
        return self._redis

    def _expand_short_id(self, model_id: str) -> tuple[str, str]:
        """Expand short model ID to org/model. Returns (org, model)."""
        if "/" not in model_id:
            # Check if it's a known short name
            if model_id in SHORT_ID_MAP:
                expanded = SHORT_ID_MAP[model_id]
                parts = expanded.split("/")
                return parts[0], parts[1]
            # Treat as org/model with default org
            return "openai-community", model_id
        parts = model_id.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 3:
            # org/model-variant → org, model-variant
            return parts[0], "/".join(parts[1:])
        return model_id, "model"

    def _build_config_url(self, model_id: str) -> str:
        org, model = self._expand_short_id(model_id)
        return HF_CONFIG_URL.format(org=org, model=model)

    def _cache_key(self, model_id: str) -> str:
        return f"modelview:config:{model_id}"

    async def fetch_config(self, org: str, model: str) -> Dict[str, Any]:
        """
        Fetch config.json from HuggingFace for the given org/model.
        Uses Redis cache with 24h TTL.
        """
        model_id = f"{org}/{model}"
        cache_key = self._cache_key(model_id)

        # Try cache first
        r = await self._get_redis()
        if r:
            try:
                cached = await r.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        # Fetch from HuggingFace
        url = HF_CONFIG_URL.format(org=org, model=model)
        try:
            response = await self._http.get(url)
        except httpx.TimeoutException as e:
            raise HuggingFaceAPIError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise HuggingFaceAPIError(f"Request failed: {e}")

        if response.status_code == 404:
            raise ModelNotFoundError(f"Model not found: {model_id}")
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            raise RateLimitError(f"Rate limited. Retry after {retry_after}s")
        if response.status_code != 200:
            raise HuggingFaceAPIError(
                f"API returned {response.status_code}: {response.text[:200]}"
            )

        try:
            config = response.json()
        except json.JSONDecodeError:
            raise HuggingFaceAPIError(f"Invalid JSON response from {url}")

        # Store in cache
        if r:
            try:
                await r.setex(cache_key, CACHE_TTL, json.dumps(config))
            except Exception:
                pass

        return config

    async def health_check(self) -> bool:
        """Check if HuggingFace API is reachable."""
        try:
            r = await self._http.get("https://huggingface.co", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._http.aclose()
        if self._redis:
            await self._redis.aclose()
