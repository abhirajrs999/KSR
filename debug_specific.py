#!/usr/bin/env python3
"""Debug specific problematic result."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from database.vector_store import ChromaVectorStore

def debug_specific_result():
    """Debug the specific problematic result."""
    vs = ChromaVectorStore()
    
    # Get the specific document ID that's problematic
    doc_id = "IRC-115-2014_chunk_791"
    
    print(f"Investigating document: {doc_id}")
    print("=" * 60)
    
    # Get the specific document
    result = vs.collection.get(ids=[doc_id])
    
    if result['ids']:
        print(f"ID: {result['ids'][0]}")
        print(f"Metadata: {result['metadatas'][0]}")
        print(f"Full text:")
        print("-" * 40)
        print(result['documents'][0])
        print("-" * 40)
    
    # Also search for content that should be in IRC-37-2019
    print("\nSearching for IRC-37 specific content:")
    print("=" * 60)
    
    # Let's check if there are any documents with IRC:37 in the text
    all_docs = vs.collection.get()
    irc37_refs = []
    
    for i, doc_text in enumerate(all_docs['documents']):
        if 'IRC:37' in doc_text or 'IRC-37' in doc_text:
            irc37_refs.append({
                'id': all_docs['ids'][i],
                'metadata': all_docs['metadatas'][i],
                'text_preview': doc_text[:200]
            })
    
    print(f"Found {len(irc37_refs)} documents with IRC-37 references:")
    for ref in irc37_refs[:10]:  # Show first 10
        print(f"  ID: {ref['id']}")
        print(f"  Source: {ref['metadata'].get('source_file', 'N/A')}")
        print(f"  Text: {ref['text_preview']}...")
        print()

if __name__ == "__main__":
    debug_specific_result()
