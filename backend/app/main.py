"""Main FastAPI Application - TDD Implemented"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# Pydantic Models
class ModelData(BaseModel):
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

class ModelCreate(BaseModel):
    name: str
    type: str
    data: ModelData

class ModelResponse(BaseModel):
    id: str
    name: str
    type: str
    data: ModelData
    created_at: datetime
    updated_at: datetime

class VisualizationType(BaseModel):
    type: str
    label: str
    description: str

# In-memory storage (replace with database in production)
models_db: Dict[str, ModelResponse] = {}

# FastAPI App
app = FastAPI(
    title="Model View API",
    description="API for model visualization",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/v1/models", response_model=List[ModelResponse])
async def get_models():
    """Get all models"""
    return list(models_db.values())

@app.get("/api/v1/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str):
    """Get a specific model by ID"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    return models_db[model_id]

@app.post("/api/v1/models", response_model=ModelResponse, status_code=201)
async def create_model(model: ModelCreate):
    """Create a new model"""
    model_id = str(uuid.uuid4())
    now = datetime.utcnow()
    model_response = ModelResponse(
        id=model_id,
        name=model.name,
        type=model.type,
        data=model.data,
        created_at=now,
        updated_at=now
    )
    models_db[model_id] = model_response
    return model_response

@app.delete("/api/v1/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    del models_db[model_id]
    return {"message": "Model deleted successfully"}

@app.get("/api/v1/visualization/types", response_model=List[VisualizationType])
async def get_visualization_types():
    """Get available visualization types"""
    return [
        VisualizationType(type="process_flow", label="Process Flow", description="Flow diagram for processes"),
        VisualizationType(type="network", label="Network Graph", description="Network topology visualization"),
        VisualizationType(type="tree", label="Tree View", description="Hierarchical tree structure"),
        VisualizationType(type="force_directed", label="Force Directed", description="Physics-based node graph"),
    ]

@app.get("/api/v1/visualization/{model_id}")
async def get_visualization_data(model_id: str):
    """Get visualization data for a model"""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    model = models_db[model_id]
    return {
        "id": model.id,
        "name": model.name,
        "type": model.type,
        "nodes": model.data.nodes,
        "edges": model.data.edges
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)