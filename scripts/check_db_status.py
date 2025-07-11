#!/usr/bin/env python3
"""
Vector Database Status Checker

This script provides information about your current vector database status,
including what documents are indexed and statistics.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from database.vector_store import ChromaVectorStore
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Check and display vector database status."""
    try:
        print("🔍 Vector Database Status Checker")
        print("=" * 50)
        
        # Initialize vector store
        vector_store = ChromaVectorStore()
        
        # Get collection statistics
        stats = vector_store.get_collection_stats()
        
        print(f"📊 Collection Statistics:")
        print(f"  - Total documents: {stats['total_documents']:,}")
        print(f"  - Unique source files: {stats['unique_source_files']}")
        print(f"  - Database path: {stats['db_path']}")
        print(f"  - Collection name: {stats['collection_name']}")
        
        if stats['source_files']:
            print(f"\n📋 Source Files in Database:")
            for i, source_file in enumerate(sorted(stats['source_files']), 1):
                doc_count = len(vector_store.get_documents_by_source_file(source_file))
                mod_time = vector_store.get_source_file_modification_time(source_file)
                mod_time_str = f" (modified: {mod_time})" if mod_time else " (no timestamp)"
                print(f"  {i:2d}. {source_file} - {doc_count} documents{mod_time_str}")
        
        # Test search functionality
        print(f"\n🔍 Testing Search Functionality:")
        test_results = vector_store.search_by_text("IRC specifications", n_results=3)
        if test_results:
            print(f"  ✅ Search working - found {len(test_results)} results")
            for i, result in enumerate(test_results[:2], 1):
                print(f"     {i}. {result['metadata'].get('source_file', 'Unknown')} (distance: {result.get('distance', 'N/A'):.4f})")
        else:
            print(f"  ⚠️  Search returned no results")
        
        print(f"\n✅ Vector database status check completed!")
        
    except Exception as e:
        logger.error(f"Error checking vector database status: {e}")
        print(f"\n❌ Error: {e}")
        print("Make sure the vector database has been created first.")

if __name__ == "__main__":
    main()
