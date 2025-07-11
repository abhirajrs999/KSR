#!/usr/bin/env python3
"""
Railway Setup Script for IRC RAG System

This script initializes the vector database and processes existing documents
for the IRC RAG system deployed on Railway with persistent storage.

Usage:
- Run after Railway deployment to set up the vector database
- Can be run manually to rebuild/update the vector database
- Processes all documents in the raw_pdfs directory
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config.settings import Settings
from database.vector_store import ChromaVectorStore
from processing.document_parser import IRCDocumentParser
from processing.metadata_extractor import IRCMetadataExtractor
from processing.chunker import IRCChunker
from processing.embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RailwaySetup:
    """Setup class for initializing IRC RAG system on Railway."""
    
    def __init__(self):
        """Initialize setup with proper paths for Railway deployment."""
        self.setup_railway_paths()
        self.settings = Settings()
        
        # Initialize components
        self.parser = IRCDocumentParser()
        self.metadata_extractor = IRCMetadataExtractor()
        self.chunker = IRCChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = ChromaVectorStore()
        
    def setup_railway_paths(self):
        """Setup paths for Railway deployment with persistent storage."""
        # Use /data directory for persistent storage on Railway
        if os.getenv('RAILWAY_ENVIRONMENT'):
            base_data_path = Path("/data")
            logger.info("Railway environment detected, using persistent storage at /data")
        else:
            # Local development
            base_data_path = Path(__file__).parent.parent / "data"
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
    
    async def check_existing_data(self) -> Dict[str, Any]:
        """Check what data already exists in the persistent storage."""
        raw_pdfs_dir = Path(os.getenv('RAW_PDFS_DIR', '/data/raw_pdfs'))
        processed_docs_dir = Path(os.getenv('PROCESSED_DOCS_DIR', '/data/processed/parsed_docs'))
        metadata_dir = Path(os.getenv('METADATA_DIR', '/data/processed/metadata'))
        chunks_dir = Path(os.getenv('CHUNKS_DIR', '/data/processed/chunks'))
        vector_db_dir = Path(os.getenv('VECTOR_DB_DIR', '/data/vector_db'))
        
        # Check existing files
        pdf_files = list(raw_pdfs_dir.glob("*.pdf"))
        parsed_files = list(processed_docs_dir.glob("*_parsed.json"))
        metadata_files = list(metadata_dir.glob("*_metadata.json"))
        chunk_files = list(chunks_dir.glob("*_chunks.json"))
        
        # Check vector database
        await self.vector_store.initialize()
        collection = self.vector_store.get_collection()
        vector_db_count = collection.count() if collection else 0
        
        status = {
            "raw_pdfs": len(pdf_files),
            "parsed_docs": len(parsed_files),
            "metadata_files": len(metadata_files),
            "chunk_files": len(chunk_files),
            "vector_db_documents": vector_db_count,
            "pdf_files": [f.name for f in pdf_files],
            "needs_processing": len(pdf_files) > vector_db_count
        }
        
        logger.info(f"Data status: {status}")
        return status
    
    async def copy_existing_pdfs(self) -> int:
        """Copy existing PDFs from the repository to persistent storage."""
        # This is for the case where PDFs are included in the repository
        # In production, you might upload PDFs separately
        repo_pdfs_dir = Path(__file__).parent.parent / "data" / "raw_pdfs"
        target_pdfs_dir = Path(os.getenv('RAW_PDFS_DIR', '/data/raw_pdfs'))
        
        copied_count = 0
        
        if repo_pdfs_dir.exists():
            for pdf_file in repo_pdfs_dir.glob("*.pdf"):
                target_file = target_pdfs_dir / pdf_file.name
                if not target_file.exists():
                    # Copy file to persistent storage
                    import shutil
                    shutil.copy2(pdf_file, target_file)
                    logger.info(f"Copied PDF: {pdf_file.name}")
                    copied_count += 1
                else:
                    logger.info(f"PDF already exists: {pdf_file.name}")
        
        return copied_count
    
    async def process_document(self, pdf_path: Path) -> bool:
        """Process a single PDF document."""
        try:
            logger.info(f"Processing document: {pdf_path.name}")
            start_time = time.time()
            
            # Get directories
            parsed_docs_dir = Path(os.getenv('PROCESSED_DOCS_DIR', '/data/processed/parsed_docs'))
            metadata_dir = Path(os.getenv('METADATA_DIR', '/data/processed/metadata'))
            chunks_dir = Path(os.getenv('CHUNKS_DIR', '/data/processed/chunks'))
            
            # Step 1: Parse PDF
            logger.info(f"Parsing PDF: {pdf_path.name}")
            parsed_doc = await self.parser.parse_pdf(str(pdf_path))
            if not parsed_doc:
                logger.error(f"Failed to parse PDF: {pdf_path.name}")
                return False
            
            # Save parsed document
            parsed_doc_path = parsed_docs_dir / f"{pdf_path.stem}_parsed.json"
            with open(parsed_doc_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_doc, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved parsed document: {parsed_doc_path.name}")
            
            # Step 2: Extract metadata
            logger.info(f"Extracting metadata for: {pdf_path.name}")
            metadata = self.metadata_extractor.extract_metadata(parsed_doc)
            
            # Save metadata
            metadata_path = metadata_dir / f"{pdf_path.stem}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata: {metadata_path.name}")
            
            # Step 3: Create chunks
            logger.info(f"Creating chunks for: {pdf_path.name}")
            chunks = self.chunker.create_chunks(parsed_doc=parsed_doc, metadata=metadata)
            
            # Save chunks
            chunks_path = chunks_dir / f"{pdf_path.stem}_chunks.json"
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(chunks)} chunks: {chunks_path.name}")
            
            # Step 4: Add to vector database
            logger.info(f"Adding chunks to vector database for: {pdf_path.name}")
            texts = [chunk['content'] if 'content' in chunk else chunk['text'] for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            if embeddings:
                documents = []
                metadatas = []
                ids = []
                
                for i, chunk in enumerate(chunks):
                    doc_id = f"{pdf_path.stem}_chunk_{i}"
                    
                    # Prepare metadata
                    pages = chunk.get('pages', [])
                    chunk_metadata = {
                        'source_file': str(pdf_path.stem),
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
                
                processing_time = time.time() - start_time
                logger.info(f"Successfully processed {pdf_path.name} in {processing_time:.2f}s - Added {success_count} chunks")
                return True
            else:
                logger.error(f"Failed to generate embeddings for: {pdf_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            return False
    
    async def setup_vector_database(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Setup and populate the vector database."""
        logger.info("Setting up vector database...")
        
        # Check existing data
        status = await self.check_existing_data()
        
        if not force_rebuild and not status["needs_processing"]:
            logger.info("Vector database already populated. Use --force to rebuild.")
            return status
        
        # Copy existing PDFs if needed (for first deployment)
        copied_pdfs = await self.copy_existing_pdfs()
        if copied_pdfs > 0:
            logger.info(f"Copied {copied_pdfs} PDFs to persistent storage")
        
        # Get PDFs to process
        raw_pdfs_dir = Path(os.getenv('RAW_PDFS_DIR', '/data/raw_pdfs'))
        pdf_files = list(raw_pdfs_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("No PDF files found to process")
            return {"error": "No PDF files found"}
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Initialize vector store
        await self.vector_store.initialize()
        
        # Clear existing data if force rebuild
        if force_rebuild:
            logger.info("Force rebuild requested - clearing existing vector database")
            collection = self.vector_store.get_collection()
            if collection and collection.count() > 0:
                # Clear the collection
                self.vector_store.client.delete_collection(collection.name)
                # Reinitialize
                await self.vector_store.initialize()
        
        # Process each PDF
        processed_count = 0
        failed_count = 0
        
        for pdf_file in pdf_files:
            try:
                success = await self.process_document(pdf_file)
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                failed_count += 1
        
        # Final status
        final_status = await self.check_existing_data()
        final_status.update({
            "setup_completed": True,
            "processed_documents": processed_count,
            "failed_documents": failed_count,
            "total_attempted": len(pdf_files)
        })
        
        logger.info(f"Setup completed: {processed_count} processed, {failed_count} failed")
        return final_status
    
    async def create_summary_report(self) -> Dict[str, Any]:
        """Create a summary report of the setup."""
        status = await self.check_existing_data()
        
        # Get metadata for all documents
        metadata_dir = Path(os.getenv('METADATA_DIR', '/data/processed/metadata'))
        documents = []
        
        for metadata_file in metadata_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                documents.append({
                    "file": metadata_file.stem.replace('_metadata', ''),
                    "title": metadata.get('title', 'Unknown'),
                    "irc_code": metadata.get('irc_code', 'Unknown'),
                    "revision_year": metadata.get('revision_year', 'Unknown')
                })
            except Exception as e:
                logger.warning(f"Error reading metadata file {metadata_file}: {e}")
        
        summary = {
            "status": status,
            "documents": documents,
            "environment": {
                "railway_environment": os.getenv('RAILWAY_ENVIRONMENT', 'local'),
                "service_name": os.getenv('RAILWAY_SERVICE_NAME', 'unknown'),
                "deployment_id": os.getenv('RAILWAY_DEPLOYMENT_ID', 'unknown'),
                "data_paths": {
                    "raw_pdfs": os.getenv('RAW_PDFS_DIR'),
                    "processed_docs": os.getenv('PROCESSED_DOCS_DIR'),
                    "metadata": os.getenv('METADATA_DIR'),
                    "chunks": os.getenv('CHUNKS_DIR'),
                    "vector_db": os.getenv('VECTOR_DB_DIR')
                }
            }
        }
        
        return summary


async def main():
    """Main function for Railway setup."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Railway Setup for IRC RAG System")
    parser.add_argument('--force', action='store_true', help='Force rebuild of vector database')
    parser.add_argument('--check-only', action='store_true', help='Only check status, do not process')
    parser.add_argument('--summary', action='store_true', help='Create and display summary report')
    
    args = parser.parse_args()
    
    try:
        setup = RailwaySetup()
        
        logger.info("Starting Railway setup for IRC RAG system...")
        logger.info(f"Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'local')}")
        
        if args.check_only:
            status = await setup.check_existing_data()
            print("\n=== Data Status ===")
            print(json.dumps(status, indent=2))
            return
        
        if args.summary:
            summary = await setup.create_summary_report()
            print("\n=== Setup Summary ===")
            print(json.dumps(summary, indent=2))
            return
        
        # Run the setup
        result = await setup.setup_vector_database(force_rebuild=args.force)
        
        print("\n=== Setup Results ===")
        print(json.dumps(result, indent=2))
        
        if result.get("setup_completed"):
            logger.info("✅ Railway setup completed successfully!")
            
            # Create final summary
            summary = await setup.create_summary_report()
            print("\n=== Final Summary ===")
            print(json.dumps(summary, indent=2))
        else:
            logger.error("❌ Railway setup failed or incomplete")
            return 1
            
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"Setup failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
