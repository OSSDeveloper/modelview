# backend/app/main.py
# FastAPI application for modelview.processbricks.com

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Any

from app.services.hf_client import (
    HuggingFaceClient,
    HuggingFaceAPIError,
    ModelNotFoundError,
    RateLimitError,
)
from app.services.graph_builder import ModelGraphBuilder, SHORT_ID_MAP


# =============================================================================
# POPULAR MODELS — Hardcoded for /api/popular endpoint
# =============================================================================
POPULAR_MODELS: List[Dict[str, str]] = [
    {"id": "openai-community/gpt2", "name": "GPT-2", "type": "decoder", "description": "12-layer decoder-only transformer"},
    {"id": "google/t5-small", "name": "T5 Small", "type": "encoder-decoder", "description": "Small encoder-decoder T5"},
    {"id": "openai/clip-vit-base-patch32", "name": "CLIP ViT-B/32", "type": "multimodal", "description": "Vision-Language contrastive model"},
    {"id": "google/vit-base-patch16-224", "name": "ViT Base", "type": "vision", "description": "Vision Transformer for images"},
    {"id": "microsoft/resnet-50", "name": "ResNet-50", "type": "cnn", "description": "50-layer residual network"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.2", "name": "Mistral 7B", "type": "decoder", "description": "7B parameter MoE-optimized decoder"},
    {"id": "Qwen/Qwen2.5-0.8B", "name": "Qwen 0.8B", "type": "decoder", "description": "Compact Qwen2.5 model"},
    {"id": "deepseek-ai/DeepSeek-V3", "name": "DeepSeek V3", "type": "moe", "description": "Large MoE with 128 experts"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B", "type": "decoder", "description": "Meta's 8B instruction-tuned model"},
    {"id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi-3 Mini", "type": "decoder", "description": "Microsoft's compact SLM"},
    {"id": "google/gemma-2-2b-it", "name": "Gemma 2 2B", "type": "decoder", "description": "Google's 2B instruction-tuned model"},
    {"id": "tiiuae/falcon-7b-instruct", "name": "Falcon 7B", "type": "decoder", "description": "TII's 7B decoder model"},
]

# Short ID aliases
SHORT_ID_MAP.update({
    "gpt2": "openai-community/gpt2",
    "t5": "google/t5-small",
    "clip": "openai/clip-vit-base-patch32",
    "vit": "google/vit-base-patch16-224",
    "resnet": "microsoft/resnet-50",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen": "Qwen/Qwen2.5-0.8B",
    "phi": "microsoft/Phi-3-mini-4k-instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "gemma": "google/gemma-2-2b-it",
    "falcon": "tiiuae/falcon-7b-instruct",
})


# =============================================================================
# LIFESPAN
# =============================================================================
hf_client: HuggingFaceClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hf_client
    redis_url = "redis://modelview-redis:6379"  # Docker Compose service name
    hf_client = HuggingFaceClient(redis_url=redis_url)
    yield
    await hf_client.close()


# =============================================================================
# APP
# =============================================================================
app = FastAPI(
    title="ModelView API",
    description="HuggingFace model architecture visualizer — graph from config.json",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HELPERS
# =============================================================================
def expand_model_id(model_id: str) -> str:
    """Expand short model ID to full org/model form."""
    if model_id in SHORT_ID_MAP:
        return SHORT_ID_MAP[model_id]
    return model_id


def split_model_id(model_id: str) -> tuple[str, str]:
    """Split 'org/model' into (org, model)."""
    if "/" not in model_id:
        expanded = expand_model_id(model_id)
        if "/" in expanded:
            return expanded.split("/", 1)
        return "openai-community", expanded
    parts = model_id.split("/", 1)
    return parts[0], parts[1]


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/api/health")
async def health():
    """Health check endpoint."""
    hf_ok = await hf_client.health_check()
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "hf_reachable": hf_ok,
    }


@app.get("/api/popular")
async def popular_models() -> List[Dict[str, str]]:
    """Return list of popular models to visualize."""
    return POPULAR_MODELS


@app.get("/api/model/{org}/{model}")
async def get_model_graph(org: str, model: str) -> Dict[str, Any]:
    """
    Fetch HuggingFace model config and return graph visualization data.
    Cache at Redis layer in hf_client.
    """
    model_id = f"{org}/{model}"

    try:
        config = await hf_client.fetch_config(org, model)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except HuggingFaceAPIError as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace API error: {e}")

    builder = ModelGraphBuilder(config, model_id)
    return builder.build()


@app.get("/{short_id}")
async def short_model_redirect(short_id: str, response: Response):
    """
    Handle short model IDs like /gpt2 → /api/model/openai-community/gpt2
    Returns the graph directly for browser convenience.
    """
    if short_id in ("api", "api-docs", "docs", "health"):
        raise HTTPException(status_code=404)

    expanded = expand_model_id(short_id)
    if expanded == short_id and "/" not in short_id:
        raise HTTPException(status_code=404, detail=f"Unknown model: {short_id}")

    org, model = split_model_id(expanded)
    try:
        config = await hf_client.fetch_config(org, model)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {expanded}")
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except HuggingFaceAPIError as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace API error: {e}")

    builder = ModelGraphBuilder(config, expanded)
    return builder.build()


@app.get("/")
async def root() -> HTMLResponse:
    """Serve the landing page HTML."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ModelView — Visualize HuggingFace Architectures</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e8; min-height: 100vh; }
    .hero { text-align: center; padding: 80px 20px 40px; }
    .hero h1 { font-size: 3rem; font-weight: 700; background: linear-gradient(135deg, #7c3aed, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 12px; }
    .hero p { font-size: 1.2rem; color: #8888aa; max-width: 560px; margin: 0 auto 40px; }
    .input-row { display: flex; max-width: 640px; margin: 0 auto; gap: 8px; }
    .input-row input { flex: 1; padding: 14px 18px; font-size: 1rem; border: 1px solid #2a2a3a; border-radius: 8px; background: #12121c; color: #e0e0e8; outline: none; }
    .input-row input:focus { border-color: #7c3aed; }
    .input-row button { padding: 14px 28px; font-size: 1rem; font-weight: 600; background: #7c3aed; color: white; border: none; border-radius: 8px; cursor: pointer; }
    .input-row button:hover { background: #6d28d9; }
    .popular { max-width: 900px; margin: 60px auto; padding: 0 20px; }
    .popular h2 { font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6666888; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
    .card { background: #12121c; border: 1px solid #1e1e2e; border-radius: 10px; padding: 16px; cursor: pointer; transition: border-color 0.2s, transform 0.2s; text-decoration: none; color: inherit; display: block; }
    .card:hover { border-color: #7c3aed; transform: translateY(-2px); }
    .card-name { font-weight: 600; font-size: 0.95rem; margin-bottom: 4px; }
    .card-meta { font-size: 0.75rem; color: #666688; display: flex; gap: 8px; }
    .card-type { background: #1e1e2e; padding: 2px 8px; border-radius: 4px; }
    .footer { text-align: center; padding: 40px; color: #444466; font-size: 0.85rem; }
  </style>
</head>
<body>
  <div class="hero">
    <h1>ModelView</h1>
    <p>Visualize the architecture of any HuggingFace model. Paste a model ID or URL.</p>
    <form class="input-row" action="/model" method="get" onsubmit="this.action='/'+this.model.value.replace('https://huggingface.co/','').split('/').filter(Boolean).join(':');">
      <input name="model" placeholder="e.g. openai-community/gpt2 or meta-llama/Llama-3.1-8B-Instruct" />
      <button type="submit">Visualize</button>
    </form>
  </div>
  <div class="popular">
    <h2>Popular Models</h2>
    <div class="grid">
      <a class="card" href="/gpt2">
        <div class="card-name">GPT-2</div>
        <div class="card-meta"><span class="card-type">decoder</span><span>12 layers, 768 hidden</span></div>
      </a>
      <a class="card" href="/mistral">
        <div class="card-name">Mistral 7B</div>
        <div class="card-meta"><span class="card-type">decoder</span><span>MoE, 8 experts</span></div>
      </a>
      <a class="card" href="/t5">
        <div class="card-name">T5 Small</div>
        <div class="card-meta"><span class="card-type">encoder-decoder</span><span>6 layers</span></div>
      </a>
      <a class="card" href="/clip">
        <div class="card-name">CLIP ViT-B/32</div>
        <div class="card-meta"><span class="card-type">multimodal</span><span>vision + text</span></div>
      </a>
      <a class="card" href="/vit">
        <div class="card-name">ViT Base</div>
        <div class="card-meta"><span class="card-type">vision</span><span>12 layers, 768 hidden</span></div>
      </a>
      <a class="card" href="/resnet">
        <div class="card-name">ResNet-50</div>
        <div class="card-meta"><span class="card-type">cnn</span><span>50 layers, 2048 dim</span></div>
      </a>
      <a class="card" href="/deepseek">
        <div class="card-name">DeepSeek V3</div>
        <div class="card-meta"><span class="card-type">moe</span><span>128 experts</span></div>
      </a>
      <a class="card" href="/llama">
        <div class="card-name">Llama 3.1 8B</div>
        <div class="card-meta"><span class="card-type">decoder</span><span>32 layers, sliding window</span></div>
      </a>
      <a class="card" href="/qwen">
        <div class="card-name">Qwen 0.8B</div>
        <div class="card-meta"><span class="card-type">decoder</span><span>compact</span></div>
      </a>
      <a class="card" href="/gemma">
        <div class="card-name">Gemma 2 2B</div>
        <div class="card-meta"><span class="card-type">decoder</span><span>Google SLM</span></div>
      </a>
      <a class="card" href="/phi">
        <div class="card-name">Phi-3 Mini</div>
        <div class="card-meta"><span class="card-type">decoder</span><span>Microsoft SLM</span></div>
      </a>
      <a class="card" href="/falcon">
        <div class="card-name">Falcon 7B</div>
        <div class="card-meta"><span class="card-type">decoder</span><span>7B parameters</span></div>
      </a>
    </div>
  </div>
  <div class="footer">
    Powered by <a href="https://huggingface.co" style="color:#7c3aed;">HuggingFace</a> ·
    Built with <a href="https://modelview.processbricks.com" style="color:#7c3aed;">ModelView</a>
  </div>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={"Access-Control-Allow-Origin": "*"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
