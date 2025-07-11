#!/usr/bin/env python3
"""Test script to verify improved table formatting."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from database.query_engine import EnhancedIRCQueryEngine
from database.vector_store import ChromaVectorStore
from api.gemini_chat import GeminiChatEngine

def test_table_formatting():
    """Test the table formatting functionality."""
    
    # Sample markdown table from the search results
    sample_text = """
# Table 400-16

| is sieve size | percent passing |
| ------------- | --------------- |
| 9.52 mm       | 100             |
| 4.75 mm       | 95–100          |
| 2.36 mm       | 80–100          |
| 1.18 mm       | 50–95           |
| 600 micron    | 25–60           |
| 300 micron    | 10–30           |
| 150 micron    | 0–15            |
| 75 micron     | 0–10            |

The joints shall be filled with sand passing a 2.35 mm size with the grading as in Table 400-17.

# Table 400-17

| is sieve size | percent passing |
| ------------- | --------------- |
| 2.36 mm       | 100             |
| 1.18 mm       | 90–100          |
| 600 micron    | 60–90           |
| 300 micron    | 30–60           |
| 150 micron    | 15–30           |
| 75 micron     | 0–10            |
"""

    # Initialize the query engine
    vs = ChromaVectorStore()
    chat_engine = GeminiChatEngine()
    query_engine = EnhancedIRCQueryEngine(vs, chat_engine)
    
    # Test the table cleaning method
    print("ORIGINAL TEXT:")
    print("=" * 60)
    print(sample_text)
    print("\n" + "=" * 60)
    
    print("CLEANED TEXT:")
    print("=" * 60)
    cleaned_text = query_engine._clean_markdown_table(sample_text)
    print(cleaned_text)
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_table_formatting()
