#!/usr/bin/env python3
"""Debug script to check metadata integrity in the vector database."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from database.vector_store import ChromaVectorStore

def debug_search_results():
    """Debug search results to find metadata issues."""
    vs = ChromaVectorStore()
    
    # Test the problematic search
    query = "VDF values axle load spectrum"
    print(f"Searching for: '{query}'")
    print("=" * 60)
    
    results = vs.search_by_text(query, n_results=5)
    
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  ID: {result['id']}")
        print(f"  Source File: {result['metadata'].get('source_file', 'N/A')}")
        print(f"  Chunk Index: {result['metadata'].get('chunk_index', 'N/A')}")
        print(f"  IRC Code: {result['metadata'].get('irc_code', 'N/A')}")
        print(f"  Distance: {result.get('distance', 'N/A')}")
        print(f"  Text Preview: {result['text'][:150]}...")
        print("-" * 40)

def debug_specific_source():
    """Check specific source file documents."""
    vs = ChromaVectorStore()
    
    # Check all documents from IRC-37-2019
    print("\nChecking IRC-37-2019 documents:")
    print("=" * 60)
    
    results = vs.collection.get(where={"source_file": "IRC-37-2019"}, limit=10)
    
    for i in range(min(5, len(results['ids']))):
        print(f"Document {i+1}:")
        print(f"  ID: {results['ids'][i]}")
        print(f"  Source File: {results['metadatas'][i].get('source_file', 'N/A')}")
        print(f"  Text Preview: {results['documents'][i][:150]}...")
        print("-" * 40)

def check_all_collections():
    """Check all documents in the collection for metadata consistency."""
    vs = ChromaVectorStore()
    
    print("\nChecking all collection metadata:")
    print("=" * 60)
    
    # Get all documents
    all_docs = vs.collection.get()
    total_docs = len(all_docs['ids'])
    
    print(f"Total documents in collection: {total_docs}")
    
    # Check for metadata inconsistencies
    source_file_counts = {}
    id_source_mismatches = []
    
    for i, doc_id in enumerate(all_docs['ids']):
        metadata = all_docs['metadatas'][i]
        source_file = metadata.get('source_file', 'UNKNOWN')
        
        # Count documents per source file
        source_file_counts[source_file] = source_file_counts.get(source_file, 0) + 1
        
        # Check if ID prefix matches source file
        id_prefix = doc_id.split('_chunk_')[0] if '_chunk_' in doc_id else doc_id
        if id_prefix != source_file:
            id_source_mismatches.append({
                'id': doc_id,
                'id_prefix': id_prefix,
                'metadata_source': source_file
            })
    
    print("\nDocuments per source file:")
    for source, count in sorted(source_file_counts.items()):
        print(f"  {source}: {count}")
    
    if id_source_mismatches:
        print(f"\nFound {len(id_source_mismatches)} ID/metadata mismatches:")
        for mismatch in id_source_mismatches[:10]:  # Show first 10
            print(f"  ID: {mismatch['id']}")
            print(f"    ID prefix: {mismatch['id_prefix']}")
            print(f"    Metadata source: {mismatch['metadata_source']}")
            print()
    else:
        print("\nNo ID/metadata mismatches found.")

if __name__ == "__main__":
    debug_search_results()
    debug_specific_source()
    check_all_collections()
