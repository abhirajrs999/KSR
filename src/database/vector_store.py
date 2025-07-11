import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

# Fixed import - remove 'src.' prefix
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """
    Vector store implementation using ChromaDB for IRC documents.
    Handles storage, retrieval, and searching of document chunks with
    their embeddings and metadata.
    """
    
    def __init__(
        self,
        collection_name: str = "irc_documents"
    ):
        """
        Initialize the vector store with ChromaDB.
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self.db_path = settings.vector_db_dir / "chroma"
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Connected to ChromaDB at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
        
        # Create or get collection (without embedding function since we provide embeddings)
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Using collection '{collection_name}' without embedding function (embeddings provided externally)")
        except Exception as e:
            logger.error(f"Failed to create/get collection: {e}")
            raise
            
        # Initialize embedding generator for searches
        from processing.embeddings import EmbeddingGenerator
        self.embedding_generator = EmbeddingGenerator()

    async def initialize(self):
        """Initialize the vector store (for compatibility with scripts)."""
        logger.info("Vector store initialized")

    def get_collection(self):
        """Get the ChromaDB collection."""
        return self.collection

    def clear_collection(self):
        """Clear all data from the collection."""
        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
                logger.info(f"Cleared {len(all_data['ids'])} documents from collection")
            else:
                logger.info("Collection is already empty")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def clear_entire_database(self):
        """Clear the entire ChromaDB database directory (more thorough than just collection)."""
        try:
            import shutil
            
            # Get stats before clearing
            stats = self.get_collection_stats()
            logger.info(f"Clearing database with {stats['total_documents']} documents from {stats['unique_source_files']} source files")
            
            # Close the client connection
            del self.client
            del self.collection
            
            # Remove the entire database directory
            if self.db_path.exists():
                shutil.rmtree(self.db_path)
                logger.info(f"Removed entire database directory: {self.db_path}")
            
            # Recreate directory and reinitialize
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Reinitialize the client and collection
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("Database cleared and reinitialized successfully")
            
        except Exception as e:
            logger.error(f"Error clearing entire database: {e}")
            raise

    async def add_documents(
        self, 
        documents: List[str], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str], 
        embeddings: List[List[float]]
    ) -> int:
        """
        Add documents to the collection.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            embeddings: List of embeddings
            
        Returns:
            Number of successfully added documents
        """
        try:
            # Convert all metadata values to ChromaDB-compatible types
            clean_metadatas = []
            for metadata in metadatas:
                clean_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, list):
                        clean_metadata[key] = ','.join(map(str, value))
                    elif value is None:
                        clean_metadata[key] = ''
                    else:
                        clean_metadata[key] = str(value)
                clean_metadatas.append(clean_metadata)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=clean_metadatas
            )
            logger.info(f"Added {len(documents)} documents to collection")
            return len(documents)
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return 0

    def search_by_text(
        self,
        query: str,
        n_results: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using text query.
        Uses the same embedding model as storage to ensure dimension compatibility.
        
        Args:
            query: Text query to search for
            n_results: Number of results to return
            filter_criteria: Optional metadata filters
        Returns:
            List of matching documents with scores and metadata
        """
        try:
            # Generate query embedding using the same model as storage
            query_embedding = self.embedding_generator.generate_embeddings([query])[0]
            
            # Use query_embeddings instead of query_texts to avoid dimension mismatch
            results = self.collection.query(
                query_embeddings=[query_embedding],  # This ensures same dimension
                n_results=n_results,
                where=filter_criteria
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            return []

    def search_by_irc_code(
        self,
        irc_code: str,
        query: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for documents by IRC code with optional semantic search.
        Args:
            irc_code: IRC code to filter by
            query: Optional semantic search query
            n_results: Number of results to return
        Returns:
            List of matching documents
        """
        filter_criteria = {"irc_code": irc_code}
        if query:
            return self.search_by_text(query, n_results, filter_criteria)
        
        try:
            results = self.collection.get(
                where=filter_criteria,
                limit=n_results
            )
            
            return [{
                'id': id,
                'text': doc,
                'metadata': meta
            } for id, doc, meta in zip(
                results['ids'],
                results['documents'],
                results['metadatas']
            )]
        except Exception as e:
            logger.error(f"Error searching by IRC code: {e}")
            return []

    def test_search(self, query: str = "IRC specifications", n_results: int = 3) -> bool:
        """
        Test search functionality - sync method to avoid async issues.
        
        Args:
            query: Test query string
            n_results: Number of results to return
            
        Returns:
            True if test successful, False otherwise
        """
        try:
            results = self.search_by_text(query=query, n_results=n_results)
            if results:
                logger.info(f"Vector database test successful. Found {len(results)} results for test query.")
                return True
            else:
                logger.warning("Vector database test returned no results.")
                return False
        except Exception as e:
            logger.error(f"Vector database test failed: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        try:
            # Get all data to count
            all_data = self.collection.get()
            total_docs = len(all_data['ids']) if all_data['ids'] else 0
            
            # Get unique source files
            source_files = set()
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    if 'source_file' in metadata:
                        source_files.add(metadata['source_file'])
            
            stats = {
                'total_documents': total_docs,
                'unique_source_files': len(source_files),
                'source_files': list(source_files),
                'collection_name': self.collection_name,
                'db_path': str(self.db_path)
            }
            
            logger.info(f"Collection stats: {total_docs} documents from {len(source_files)} source files")
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'total_documents': 0, 'unique_source_files': 0, 'source_files': []}

    def get_documents_by_source_file(self, source_file: str) -> List[str]:
        """Get all document IDs for a specific source file."""
        try:
            results = self.collection.get(
                where={"source_file": source_file}
            )
            return results['ids'] if results['ids'] else []
        except Exception as e:
            logger.error(f"Error getting documents for source file {source_file}: {e}")
            return []

    def is_source_file_indexed(self, source_file: str) -> bool:
        """Check if a source file is already indexed in the vector database."""
        try:
            results = self.collection.get(
                where={"source_file": source_file},
                limit=1
            )
            has_docs = bool(results['ids'])
            if has_docs:
                logger.debug(f"Source file '{source_file}' is already indexed")
            else:
                logger.debug(f"Source file '{source_file}' is not indexed")
            return has_docs
        except Exception as e:
            logger.error(f"Error checking if source file {source_file} is indexed: {e}")
            return False

    def remove_source_file_documents(self, source_file: str) -> int:
        """Remove all documents for a specific source file."""
        try:
            # Get all document IDs for this source file
            doc_ids = self.get_documents_by_source_file(source_file)
            if doc_ids:
                self.collection.delete(ids=doc_ids)
                logger.info(f"Removed {len(doc_ids)} documents for source file: {source_file}")
                return len(doc_ids)
            else:
                logger.debug(f"No documents found for source file: {source_file}")
                return 0
        except Exception as e:
            logger.error(f"Error removing documents for source file {source_file}: {e}")
            return 0

    def get_source_file_modification_time(self, source_file: str) -> Optional[str]:
        """Get the modification time stored in metadata for a source file."""
        try:
            results = self.collection.get(
                where={"source_file": source_file},
                limit=1
            )
            if results['metadatas'] and results['metadatas'][0]:
                return results['metadatas'][0].get('file_modified_time')
            return None
        except Exception as e:
            logger.error(f"Error getting modification time for {source_file}: {e}")
            return None

    async def add_documents_for_source(
        self, 
        source_file: str,
        documents: List[str], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str], 
        embeddings: List[List[float]],
        replace_existing: bool = True
    ) -> int:
        """
        Add documents for a specific source file with option to replace existing.
        
        Args:
            source_file: Name of the source file
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            embeddings: List of embeddings
            replace_existing: Whether to replace existing documents for this source file
            
        Returns:
            Number of successfully added documents
        """
        try:
            # If replacing existing, remove old documents first
            if replace_existing:
                removed_count = self.remove_source_file_documents(source_file)
                if removed_count > 0:
                    logger.info(f"Replaced {removed_count} existing documents for {source_file}")
            
            # Add new documents
            return await self.add_documents(documents, metadatas, ids, embeddings)
            
        except Exception as e:
            logger.error(f"Error adding documents for source file {source_file}: {e}")
            return 0