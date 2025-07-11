"""
Simple FastAPI application for Railway deployment test.
This is a minimal version to get the deployment working first.
"""

import os
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple response models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    message: str
    environment: str

# Create FastAPI app
app = FastAPI(
    title="IRC RAG API - Minimal",
    description="Minimal deployment test for IRC RAG system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "IRC RAG API - Minimal Version",
        "status": "running",
        "environment": os.getenv('RAILWAY_ENVIRONMENT', 'local')
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        message="Minimal IRC RAG API is running",
        environment=os.getenv('RAILWAY_ENVIRONMENT', 'local')
    )

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify deployment."""
    return {
        "status": "success",
        "message": "Deployment test successful!",
        "port": os.getenv('PORT', '8000'),
        "railway_env": os.getenv('RAILWAY_ENVIRONMENT'),
        "data_path": "/data" if os.getenv('RAILWAY_ENVIRONMENT') else "local"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
