#!/usr/bin/env python3
"""
Quick Update Script for RAG Vector Database

This script provides an easy way to update your vector database when you add new PDFs
or when existing PDFs are updated. It automatically detects what needs to be processed.

Usage Examples:
    python quick_update.py                    # Incremental update (recommended)
    python quick_update.py --full-rebuild    # Process all files from scratch
    python quick_update.py --force           # Clear DB and rebuild everything
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from scripts.import_and_run import run_script_sequence

async def main():
    """Main function to run the quick update."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick update for RAG vector database')
    parser.add_argument('--full-rebuild', action='store_true', 
                       help='Process all files (ignore timestamps)')
    parser.add_argument('--force', action='store_true',
                       help='Clear existing database and rebuild')
    parser.add_argument('--process-new-pdfs', action='store_true',
                       help='First process any new PDFs, then update vector DB')
    
    args = parser.parse_args()
    
    print("🚀 RAG Vector Database Quick Update")
    print("=" * 50)
    
    scripts_to_run = []
    
    # If requested, process new PDFs first
    if args.process_new_pdfs:
        print("📄 Step 1: Processing any new PDF documents...")
        scripts_to_run.append("02_process_documents.py")
    
    # Prepare vector DB update arguments
    vector_db_args = []
    if args.full_rebuild:
        vector_db_args.append("--full-rebuild")
        print("🔄 Mode: Full rebuild (processing all files)")
    else:
        print("⚡ Mode: Incremental update (only new/changed files)")
    
    if args.force:
        vector_db_args.append("--force")
        print("🗑️  Mode: Force clear and rebuild")
    
    print(f"🗂️  Step {'2' if args.process_new_pdfs else '1'}: Updating vector database...")
    scripts_to_run.append(("03_create_vector_db.py", vector_db_args))
    
    # Run the scripts
    try:
        await run_script_sequence(scripts_to_run)
        print("\n✅ Quick update completed successfully!")
        print("Your RAG system is ready to use with the latest documents.")
    except Exception as e:
        print(f"\n❌ Update failed: {e}")
        print("Check the log files for more details.")

if __name__ == "__main__":
    asyncio.run(main())
