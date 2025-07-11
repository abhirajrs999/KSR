import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config.settings import Settings
from processing.embeddings import EmbeddingGenerator
from database.vector_store import ChromaVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_db_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VectorDatabaseBuilder:
    """Handles the creation of the vector database from processed chunks."""
    
    def __init__(self):
        """Initialize the vector database builder with components."""
        self.settings = Settings()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = ChromaVectorStore()
        
        # Setup paths
        self.chunks_dir = self.settings.chunks_dir
        self.vector_db_dir = self.settings.vector_db_dir  
        self.metadata_dir = self.settings.metadata_dir
        
        # Track processing mode
        self.incremental_mode = True  # Enable incremental processing by default

    def get_chunk_files(self) -> List[Path]:
        """Get all chunk files from the processed chunks directory."""
        if not self.chunks_dir.exists():
            logger.error(f"Chunks directory does not exist: {self.chunks_dir}")
            return []
        
        chunk_files = list(self.chunks_dir.glob("*_chunks.json"))
        logger.info(f"Found {len(chunk_files)} chunk files to process")
        return chunk_files

    def load_chunks_from_file(self, chunk_file: Path) -> List[Dict[str, Any]]:
        """
        Load chunks from a JSON file.
        
        Args:
            chunk_file: Path to the chunk file
            
        Returns:
            List of chunk dictionaries
        """
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Add source file information
            source_file = chunk_file.stem.replace('_chunks', '')
            for chunk in chunks:
                chunk['source_file'] = source_file
                chunk['chunk_file'] = str(chunk_file)
            
            return chunks
        except Exception as e:
            logger.error(f"Error loading chunks from {chunk_file}: {e}")
            return []

    def load_metadata_for_file(self, source_file: str) -> Dict[str, Any]:
        """
        Load metadata for a specific source file.
        
        Args:
            source_file: Name of the source file (without extension)
            
        Returns:
            Metadata dictionary
        """
        metadata_file = self.metadata_dir / f"{source_file}_metadata.json"
        try:
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading metadata for {source_file}: {e}")
        
        return {}

    def get_chunk_file_modification_time(self, chunk_file: Path) -> str:
        """Get the modification time of a chunk file."""
        try:
            timestamp = chunk_file.stat().st_mtime
            return str(timestamp)
        except Exception as e:
            logger.warning(f"Could not get modification time for {chunk_file}: {e}")
            return ""

    def should_process_chunk_file(self, chunk_file: Path) -> bool:
        """
        Determine if a chunk file should be processed based on incremental mode.
        
        Args:
            chunk_file: Path to the chunk file
            
        Returns:
            True if the file should be processed, False otherwise
        """
        if not self.incremental_mode:
            return True  # Process all files in full rebuild mode
        
        source_file = chunk_file.stem.replace('_chunks', '')
        
        # Check if source file is already indexed
        if not self.vector_store.is_source_file_indexed(source_file):
            logger.info(f"Source file '{source_file}' not found in vector DB - will process")
            return True
        
        # Check modification time if available
        current_mod_time = self.get_chunk_file_modification_time(chunk_file)
        stored_mod_time = self.vector_store.get_source_file_modification_time(source_file)
        
        if stored_mod_time is None:
            logger.info(f"No modification time stored for '{source_file}' - will process")
            return True
        
        if current_mod_time != stored_mod_time:
            logger.info(f"Source file '{source_file}' has been modified - will reprocess")
            return True
        
        logger.info(f"Source file '{source_file}' is up to date - skipping")
        return False

    def get_files_to_process(self) -> tuple[List[Path], List[Path]]:
        """
        Get chunk files categorized by whether they need processing.
        
        Returns:
            Tuple of (files_to_process, files_to_skip)
        """
        all_chunk_files = self.get_chunk_files()
        files_to_process = []
        files_to_skip = []
        
        for chunk_file in all_chunk_files:
            if self.should_process_chunk_file(chunk_file):
                files_to_process.append(chunk_file)
            else:
                files_to_skip.append(chunk_file)
        
        return files_to_process, files_to_skip

    async def process_chunk_file(self, chunk_file: Path) -> Dict[str, Any]:
        """
        Process a single chunk file and add to vector database.
        
        Args:
            chunk_file: Path to the chunk file
            
        Returns:
            Dictionary containing processing results
        """
        result = {
            "file_path": str(chunk_file),
            "status": "success",
            "errors": [],
            "processing_time": None,
            "chunks_processed": 0,
            "embeddings_generated": 0,
            "vectors_stored": 0
        }
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing chunk file: {chunk_file.name}")
            
            # Load chunks
            chunks = self.load_chunks_from_file(chunk_file)
            if not chunks:
                raise Exception("No chunks found in file")
            
            result["chunks_processed"] = len(chunks)
            logger.info(f"Loaded {len(chunks)} chunks from {chunk_file.name}")
            
            # Load metadata
            source_file = chunk_file.stem.replace('_chunks', '')
            metadata = self.load_metadata_for_file(source_file)
            
            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            # Embeddings are already lists from our updated generator
            if not embeddings or len(embeddings) == 0:
                raise Exception("Failed to generate embeddings")
            
            result["embeddings_generated"] = len(embeddings)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Prepare documents for vector store
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                # Create unique ID
                doc_id = f"{source_file}_chunk_{i}"
                
                # Prepare metadata - convert all list values to strings
                pages = chunk.get('pages', [])
                
                chunk_metadata = {
                    'source_file': str(source_file),
                    'chunk_index': i,
                    'pages': ','.join(map(str, pages)) if pages else '',
                    'title': str(metadata.get('title', '')),
                    'irc_code': str(metadata.get('irc_code', '')),
                    'revision_year': str(metadata.get('revision_year', '')),
                    'clause_numbers': str(chunk.get('clause_numbers', '')),
                    'chunk_type': str(chunk.get('chunk_type', 'text')),
                    'file_modified_time': self.get_chunk_file_modification_time(chunk_file)
                }
                
                documents.append(chunk['text'])
                metadatas.append(chunk_metadata)
                ids.append(doc_id)
            
            # Add to vector store using the new method for better handling
            logger.info(f"Adding {len(documents)} documents to vector store")
            success_count = await self.vector_store.add_documents_for_source(
                source_file=source_file,
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings,
                replace_existing=True  # Replace existing docs for this source file
            )
            
            result["vectors_stored"] = success_count
            logger.info(f"Successfully stored {success_count} vectors in database")
            
            # Calculate processing time
            result["processing_time"] = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed processing {chunk_file.name} in {result['processing_time']:.2f}s")
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error(f"Error processing {chunk_file.name}: {e}")
            result["processing_time"] = (datetime.now() - start_time).total_seconds()
        
        return result

    async def build_vector_database(self) -> Dict[str, Any]:
        """
        Build the complete vector database from all chunk files.
        
        Returns:
            Dictionary containing overall processing results
        """
        # Get files categorized by processing need
        files_to_process, files_to_skip = self.get_files_to_process()
        
        # Get collection stats before processing
        collection_stats = self.vector_store.get_collection_stats()
        
        logger.info(f"Vector database status:")
        logger.info(f"  - Existing documents: {collection_stats['total_documents']}")
        logger.info(f"  - Existing source files: {collection_stats['unique_source_files']}")
        logger.info(f"  - Files to process: {len(files_to_process)}")
        logger.info(f"  - Files to skip (already up-to-date): {len(files_to_skip)}")
        
        if not files_to_process and not files_to_skip:
            logger.error("No chunk files found to process")
            return {
                "status": "error",
                "message": "No chunk files found",
                "processed_files": [],
                "skipped_files": [],
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "total_chunks": 0,
                "total_embeddings": 0,
                "total_vectors": 0
            }
        
        logger.info(f"Starting vector database creation from {len(files_to_process)} chunk files")
        
        # Initialize vector store
        await self.vector_store.initialize()
        
        # Process only the files that need processing
        results = []
        skipped_results = []
        
        for chunk_file in files_to_process:
            result = await self.process_chunk_file(chunk_file)
            results.append(result)
            
            # Small delay between files to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        # Create results for skipped files
        for chunk_file in files_to_skip:
            source_file = chunk_file.stem.replace('_chunks', '')
            skipped_results.append({
                "file_path": str(chunk_file),
                "source_file": source_file,
                "status": "skipped",
                "reason": "already_up_to_date"
            })
        
        # Calculate summary statistics
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        total_chunks = sum(r.get("chunks_processed", 0) for r in results)
        total_embeddings = sum(r.get("embeddings_generated", 0) for r in results)
        total_vectors = sum(r.get("vectors_stored", 0) for r in results)
        total_time = sum(r.get("processing_time", 0) for r in results)
        
        summary = {
            "status": "completed",
            "total_files": len(files_to_process) + len(files_to_skip),
            "processed_files_count": len(files_to_process),
            "skipped_files_count": len(files_to_skip),
            "successful": successful,
            "failed": failed,
            "total_chunks": total_chunks,
            "total_embeddings": total_embeddings,
            "total_vectors": total_vectors,
            "total_processing_time": total_time,
            "average_time_per_file": total_time / len(files_to_process) if files_to_process else 0,
            "processed_files": results,
            "skipped_files": skipped_results,
            "incremental_mode": self.incremental_mode,
            "collection_stats_before": collection_stats,
            "collection_stats_after": self.vector_store.get_collection_stats()
        }
        
        # Save processing summary
        summary_path = self.vector_db_dir / "vector_db_creation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Vector database creation completed. Summary saved to: {summary_path}")
        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary of the vector database creation results."""
        print("\n" + "="*60)
        print("VECTOR DATABASE CREATION SUMMARY")
        print("="*60)
        print(f"Incremental mode: {'Enabled' if summary.get('incremental_mode', False) else 'Disabled'}")
        print(f"Total files found: {summary['total_files']}")
        print(f"Files processed: {summary['processed_files_count']}")
        print(f"Files skipped (up-to-date): {summary['skipped_files_count']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total chunks processed: {summary['total_chunks']}")
        print(f"Total embeddings generated: {summary['total_embeddings']}")
        print(f"Total vectors stored: {summary['total_vectors']}")
        print(f"Total processing time: {summary['total_processing_time']:.2f}s")
        
        if summary['processed_files_count'] > 0:
            print(f"Average time per processed file: {summary['average_time_per_file']:.2f}s")
        
        # Show collection stats
        if 'collection_stats_after' in summary:
            stats_after = summary['collection_stats_after']
            print(f"\nVector Database Status:")
            print(f"  - Total documents in DB: {stats_after['total_documents']}")
            print(f"  - Unique source files in DB: {stats_after['unique_source_files']}")
        
        if summary['failed'] > 0:
            print("\nFailed files:")
            for result in summary['processed_files']:
                if result['status'] == 'error':
                    print(f"  - {Path(result['file_path']).name}: {', '.join(result['errors'])}")
        
        if summary['skipped_files_count'] > 0:
            print(f"\nSkipped files (already up-to-date): {summary['skipped_files_count']}")
            for result in summary['skipped_files'][:5]:  # Show first 5
                print(f"  - {Path(result['file_path']).name}")
            if summary['skipped_files_count'] > 5:
                print(f"  ... and {summary['skipped_files_count'] - 5} more")
        
        print("\nVector database location:")
        print(f"  - Database location: {self.vector_db_dir}")
        print(f"  - Summary file: {self.vector_db_dir}/vector_db_creation_summary.json")
        print("="*60 + "\n")

    def test_vector_database(self):
        """Test the created vector database with a sample query - now sync."""
        try:
            logger.info("Testing vector database with sample query...")
            
            # Use the sync test method
            success = self.vector_store.test_search(query="IRC specifications", n_results=3)
            
            if success:
                print(f"\n✅ Test query successful! Vector database is working properly.")
                return True
            else:
                print(f"\n❌ Test query failed. Please check the logs.")
                return False
                
        except Exception as e:
            logger.error(f"Vector database test failed: {e}")
            print(f"\n❌ Test query failed with error: {e}")
            return False

async def main():
    """Main function to run the vector database creation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create or update vector database from processed chunks')
    parser.add_argument('--full-rebuild', action='store_true', 
                       help='Force full rebuild (process all files regardless of timestamps)')
    parser.add_argument('--force', action='store_true',
                       help='Clear existing vector database before processing')
    
    args = parser.parse_args()
    
    try:
        builder = VectorDatabaseBuilder()
        
        # Set processing mode based on arguments
        if args.full_rebuild:
            builder.incremental_mode = False
            logger.info("Running in FULL REBUILD mode - will process all files")
        else:
            builder.incremental_mode = True
            logger.info("Running in INCREMENTAL mode - will skip up-to-date files")
        
        # Check if chunks directory exists and has files
        if not builder.chunks_dir.exists():
            logger.error(f"Chunks directory not found: {builder.chunks_dir}")
            print(f"\nPlease run the document processing script first to create chunks in: {builder.chunks_dir}")
            return
        
        chunk_files = builder.get_chunk_files()
        if not chunk_files:
            logger.error("No chunk files found in chunks directory")
            print(f"\nPlease run the document processing script first to create chunks in: {builder.chunks_dir}")
            return
        
        # Force clear database if requested
        if args.force:
            logger.info("Clearing existing vector database...")
            builder.vector_store.clear_entire_database()
            print("✅ Existing vector database cleared completely")
        
        # Create vector database
        summary = await builder.build_vector_database()
        
        # Print summary
        builder.print_summary(summary)
        
        # Test the database
        if summary['status'] == 'completed' and summary['total_vectors'] > 0:
            builder.test_vector_database()  # Now sync method
            
    except Exception as e:
        logger.error(f"Fatal error in vector database creation: {e}")
        print(f"\nFatal error: {e}")
        print("Check the log file for more details.")

if __name__ == "__main__":
    asyncio.run(main())