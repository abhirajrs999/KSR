"""
Main entry point for the IRC RAG API deployment on Railway.
This file sets up the FastAPI application with proper configuration for deployment.
"""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from config.settings import Settings
from database.vector_store import ChromaVectorStore
from api.gemini_chat import GeminiChatEngine
from database.query_engine import IRCQueryEngine
from processing.document_parser import IRCDocumentParser
from processing.metadata_extractor import IRCMetadataExtractor
from processing.chunker import IRCChunker
from processing.embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only console logging for Railway
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to search for in IRC documents")
    irc_code: Optional[str] = Field(None, description="Optional IRC code to filter by")
    limit: Optional[int] = Field(5, description="Maximum number of results to return", ge=1, le=20)

class QueryResponse(BaseModel):
    response: str = Field(..., description="Generated response from the RAG system")
    citations: List[str] = Field(..., description="List of citations from source documents")
    relevant_chunks: List[Dict[str, Any]] = Field(..., description="Relevant document chunks used")
    processing_time: float = Field(..., description="Time taken to process the query")
    query: str = Field(..., description="Original query")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    vector_db_documents: int = Field(..., description="Number of documents in vector database")
    api_version: str = Field(..., description="API version")
    deployment_info: Dict[str, str] = Field(..., description="Deployment information")

class UploadResponse(BaseModel):
    message: str = Field(..., description="Upload status message")
    file_name: str = Field(..., description="Name of uploaded file")
    processing_status: str = Field(..., description="Processing status")

class IRCRAGAPI:
    """FastAPI application for IRC RAG system optimized for Railway deployment."""

    def __init__(self):
        """Initialize the API with all components."""
        # Override settings for Railway deployment with persistent storage
        self.setup_railway_paths()
        
        self.settings = Settings()
        self.app = FastAPI(
            title="IRC RAG API",
            description="Retrieval-Augmented Generation API for Indian Roads Congress (IRC) documents",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize components
        self.vector_store = ChromaVectorStore()
        self.chat_engine = GeminiChatEngine()
        self.query_engine = IRCQueryEngine(
            vector_store=self.vector_store,
            chat_engine=self.chat_engine
        )
        
        # Processing components for document upload
        self.parser = IRCDocumentParser()
        self.metadata_extractor = IRCMetadataExtractor()
        self.chunker = IRCChunker()
        self.embedding_generator = EmbeddingGenerator()
        
        self.setup_middleware()
        self.setup_routes()

    def setup_railway_paths(self):
        """Setup paths for Railway deployment with persistent storage."""
        # Use /data directory for persistent storage on Railway
        if os.getenv('RAILWAY_ENVIRONMENT'):
            base_data_path = Path("/data")
            logger.info("Railway environment detected, using persistent storage at /data")
        else:
            # Local development
            base_data_path = Path(__file__).parent / "data"
            logger.info("Local environment detected, using local data directory")
        
        # Override environment variables for data paths
        os.environ['RAW_PDFS_DIR'] = str(base_data_path / "raw_pdfs")
        os.environ['PROCESSED_DOCS_DIR'] = str(base_data_path / "processed" / "parsed_docs")
        os.environ['CHUNKS_DIR'] = str(base_data_path / "processed" / "chunks")
        os.environ['METADATA_DIR'] = str(base_data_path / "processed" / "metadata")
        os.environ['VECTOR_DB_DIR'] = str(base_data_path / "vector_db")
        
        # Ensure directories exist
        directories = [
            base_data_path / "raw_pdfs",
            base_data_path / "processed" / "parsed_docs",
            base_data_path / "processed" / "chunks",
            base_data_path / "processed" / "metadata",
            base_data_path / "vector_db"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")

    def setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "IRC RAG API - Railway Deployment",
                "version": "2.0.0",
                "description": "Retrieval-Augmented Generation API for IRC documents",
                "docs": "/docs",
                "health": "/health",
                "deployment": "Railway with persistent storage"
            }

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint with deployment info."""
            try:
                # Initialize vector store if not already done
                await self.vector_store.initialize()
                
                # Get document count
                collection = self.vector_store.get_collection()
                doc_count = collection.count() if collection else 0
                
                # Deployment info
                deployment_info = {
                    "environment": os.getenv('RAILWAY_ENVIRONMENT', 'local'),
                    "service_name": os.getenv('RAILWAY_SERVICE_NAME', 'irc-rag-api'),
                    "deployment_id": os.getenv('RAILWAY_DEPLOYMENT_ID', 'local'),
                    "git_commit": os.getenv('RAILWAY_GIT_COMMIT_SHA', 'unknown')[:8],
                    "data_path": os.getenv('VECTOR_DB_DIR', 'local'),
                    "port": os.getenv('PORT', '8000')
                }
                
                return HealthResponse(
                    status="healthy",
                    timestamp=datetime.now().isoformat(),
                    vector_db_documents=doc_count,
                    api_version="2.0.0",
                    deployment_info=deployment_info
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

        @self.app.post("/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            """Query IRC documents using the RAG system."""
            start_time = datetime.now()
            
            try:
                logger.info(f"Processing query: {request.query}")
                
                # Process query through the RAG system
                result = await self.query_engine.query(
                    query=request.query,
                    irc_code=request.irc_code,
                    limit=request.limit
                )
                
                if "error" in result:
                    raise HTTPException(status_code=500, detail=result["error"])
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return QueryResponse(
                    response=result["response"],
                    citations=result["citations"],
                    relevant_chunks=result["relevant_chunks"],
                    processing_time=processing_time,
                    query=request.query
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Query processing failed: {e}")
                raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

        @self.app.post("/upload", response_model=UploadResponse)
        async def upload_document(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            description: Optional[str] = Form(None)
        ):
            """Upload and process a new IRC document."""
            try:
                # Validate file type
                if not file.filename.lower().endswith('.pdf'):
                    raise HTTPException(status_code=400, detail="Only PDF files are supported")
                
                # Get the raw PDFs directory from settings
                raw_pdfs_dir = Path(os.getenv('RAW_PDFS_DIR'))
                file_path = raw_pdfs_dir / file.filename
                
                # Save file to raw_pdfs directory
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Add background task for processing
                background_tasks.add_task(
                    self.process_uploaded_document,
                    file_path,
                    description
                )
                
                return UploadResponse(
                    message="File uploaded successfully. Processing started in background.",
                    file_name=file.filename,
                    processing_status="processing"
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"File upload failed: {e}")
                raise HTTPException(status_code=500, detail=f"File upload failed: {e}")

        @self.app.get("/documents", response_model=List[Dict[str, Any]])
        async def list_documents():
            """List all processed documents."""
            try:
                documents = []
                metadata_dir = Path(os.getenv('METADATA_DIR'))
                chunks_dir = Path(os.getenv('CHUNKS_DIR'))
                
                # Get metadata files
                metadata_files = list(metadata_dir.glob("*_metadata.json"))
                
                for metadata_file in metadata_files:
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        # Get chunk count
                        chunk_file = chunks_dir / f"{metadata_file.stem.replace('_metadata', '')}_chunks.json"
                        chunk_count = 0
                        if chunk_file.exists():
                            with open(chunk_file, 'r', encoding='utf-8') as f:
                                chunks = json.load(f)
                                chunk_count = len(chunks)
                        
                        documents.append({
                            "source_file": metadata_file.stem.replace('_metadata', ''),
                            "title": metadata.get('title', 'Unknown'),
                            "irc_code": metadata.get('irc_code', 'Unknown'),
                            "revision_year": metadata.get('revision_year', 'Unknown'),
                            "chunk_count": chunk_count,
                            "metadata_file": str(metadata_file)
                        })
                    except Exception as e:
                        logger.warning(f"Error reading metadata file {metadata_file}: {e}")
                
                return documents
                
            except Exception as e:
                logger.error(f"Failed to list documents: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

        @self.app.get("/search")
        async def search_documents(
            query: str,
            irc_code: Optional[str] = None,
            limit: int = 10
        ):
            """Direct search in vector database without LLM response."""
            try:
                # Initialize vector store if needed
                await self.vector_store.initialize()
                
                # Search in vector database
                results = self.vector_store.search_by_text(
                    query=query,
                    n_results=limit,
                    filter_criteria={"irc_code": irc_code} if irc_code else None
                )
                
                return {
                    "query": query,
                    "results": results,
                    "total_results": len(results)
                }
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    async def process_uploaded_document(self, file_path: Path, description: Optional[str] = None):
        """Process an uploaded document in the background."""
        try:
            logger.info(f"Processing uploaded document: {file_path.name}")
            
            # Get directories from environment
            parsed_docs_dir = Path(os.getenv('PROCESSED_DOCS_DIR'))
            metadata_dir = Path(os.getenv('METADATA_DIR'))
            chunks_dir = Path(os.getenv('CHUNKS_DIR'))
            
            # Step 1: Parse PDF
            parsed_doc = await self.parser.parse_pdf(str(file_path))
            if not parsed_doc:
                raise Exception("Failed to parse PDF")
            
            # Save parsed document
            parsed_doc_path = parsed_docs_dir / f"{file_path.stem}_parsed.json"
            with open(parsed_doc_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_doc, f, indent=2, ensure_ascii=False)
            
            # Step 2: Extract metadata
            metadata = self.metadata_extractor.extract_metadata(parsed_doc)
            if description:
                metadata['description'] = description
            
            # Save metadata
            metadata_path = metadata_dir / f"{file_path.stem}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Step 3: Create chunks
            chunks = self.chunker.create_chunks(parsed_doc=parsed_doc, metadata=metadata)
            
            # Save chunks
            chunks_path = chunks_dir / f"{file_path.stem}_chunks.json"
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            # Step 4: Add to vector database
            await self.vector_store.initialize()
            
            # Generate embeddings and add to vector store
            texts = [chunk['content'] if 'content' in chunk else chunk['text'] for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            if embeddings:
                documents = []
                metadatas = []
                ids = []
                
                for i, chunk in enumerate(chunks):
                    doc_id = f"{file_path.stem}_chunk_{i}"
                    
                    # Prepare metadata
                    pages = chunk.get('pages', [])
                    chunk_metadata = {
                        'source_file': str(file_path.stem),
                        'chunk_index': i,
                        'pages': ','.join(map(str, pages)) if pages else '',
                        'title': str(metadata.get('title', '')),
                        'irc_code': str(metadata.get('irc_code', '')),
                        'revision_year': str(metadata.get('revision_year', '')),
                        'clause_numbers': str(chunk.get('clause_numbers', '')),
                        'chunk_type': str(chunk.get('chunk_type', 'text'))
                    }
                    
                    documents.append(chunk['content'] if 'content' in chunk else chunk['text'])
                    metadatas.append(chunk_metadata)
                    ids.append(doc_id)
                
                success_count = await self.vector_store.add_documents(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                
                logger.info(f"Added {success_count} chunks to vector database")
            
            logger.info(f"Successfully processed document: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path.name}: {e}")

# Create the FastAPI app instance
api = IRCRAGAPI()
app = api.app

def main():
    """Main function to run the API server."""
    try:
        port = int(os.getenv('PORT', 8000))
        logger.info(f"Starting IRC RAG API on port {port}")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            log_level="info",
            reload=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        print(f"Failed to start API server: {e}")

if __name__ == "__main__":
    main()
