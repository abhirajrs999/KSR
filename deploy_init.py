#!/usr/bin/env python3
"""
Railway deployment initialization script.
This script sets up the environment and initializes the vector database for Railway deployment.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from config.settings import Settings
from database.vector_store import ChromaVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

async def initialize_deployment():
    """Initialize the deployment environment."""
    logger.info("Starting Railway deployment initialization...")
    
    # Check environment
    if os.getenv('RAILWAY_ENVIRONMENT'):
        logger.info(f"Railway environment detected: {os.getenv('RAILWAY_ENVIRONMENT')}")
        logger.info(f"Service: {os.getenv('RAILWAY_SERVICE_NAME', 'unknown')}")
        logger.info(f"Deployment ID: {os.getenv('RAILWAY_DEPLOYMENT_ID', 'unknown')}")
    else:
        logger.info("Local environment detected")
    
    # Setup data directories
    if os.getenv('RAILWAY_ENVIRONMENT'):
        base_data_path = Path("/data")
        logger.info("Using Railway persistent storage at /data")
    else:
        base_data_path = Path(__file__).parent / "data"
        logger.info("Using local data directory")
    
    # Create all necessary directories
    directories = [
        base_data_path / "raw_pdfs",
        base_data_path / "processed" / "parsed_docs",
        base_data_path / "processed" / "chunks",
        base_data_path / "processed" / "metadata",
        base_data_path / "vector_db"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Set environment variables for data paths
    os.environ['RAW_PDFS_DIR'] = str(base_data_path / "raw_pdfs")
    os.environ['PROCESSED_DOCS_DIR'] = str(base_data_path / "processed" / "parsed_docs")
    os.environ['CHUNKS_DIR'] = str(base_data_path / "processed" / "chunks")
    os.environ['METADATA_DIR'] = str(base_data_path / "processed" / "metadata")
    os.environ['VECTOR_DB_DIR'] = str(base_data_path / "vector_db")
    
    # Initialize vector database
    try:
        logger.info("Initializing vector database...")
        vector_store = ChromaVectorStore()
        await vector_store.initialize()
        
        # Check if vector DB has any existing data
        collection = vector_store.get_collection()
        if collection:
            doc_count = collection.count()
            logger.info(f"Vector database initialized successfully with {doc_count} documents")
        else:
            logger.info("Vector database initialized successfully (empty)")
            
    except Exception as e:
        logger.error(f"Failed to initialize vector database: {e}")
        # Don't fail deployment if vector DB init fails - it can be initialized on first request
        logger.warning("Continuing without vector DB initialization...")
    
    # Check for existing data
    raw_pdfs = list((base_data_path / "raw_pdfs").glob("*.pdf"))
    if raw_pdfs:
        logger.info(f"Found {len(raw_pdfs)} existing PDF files in persistent storage")
    else:
        logger.info("No existing PDF files found - this is normal for new deployments")
    
    # Check environment variables
    required_env_vars = ['GEMINI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    else:
        logger.info("All required environment variables are set")
    
    logger.info("Railway deployment initialization completed successfully!")
    return True

async def check_system_health():
    """Check system health and dependencies."""
    logger.info("Running system health check...")
    
    try:
        # Test imports
        from config.settings import Settings
        from database.vector_store import ChromaVectorStore
        from api.gemini_chat import GeminiChatEngine
        logger.info("✓ All core modules imported successfully")
        
        # Test settings
        settings = Settings()
        logger.info("✓ Settings loaded successfully")
        
        # Test vector store
        vector_store = ChromaVectorStore()
        logger.info("✓ Vector store created successfully")
        
        # Test Gemini API (if key is available)
        if os.getenv('GEMINI_API_KEY'):
            chat_engine = GeminiChatEngine()
            logger.info("✓ Gemini chat engine initialized successfully")
        else:
            logger.warning("⚠ GEMINI_API_KEY not set - chat functionality will not work")
        
        logger.info("System health check completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return False

def main():
    """Main function for deployment initialization."""
    logger.info("Railway Deployment Initialization Script v2.0")
    
    # Run health check
    health_ok = asyncio.run(check_system_health())
    if not health_ok:
        logger.error("System health check failed!")
        sys.exit(1)
    
    # Run initialization
    init_ok = asyncio.run(initialize_deployment())
    if not init_ok:
        logger.error("Deployment initialization failed!")
        sys.exit(1)
    
    logger.info("Deployment initialization completed successfully!")
    logger.info("Ready to start the API server")

if __name__ == "__main__":
    main()
