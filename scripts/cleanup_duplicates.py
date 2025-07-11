#!/usr/bin/env python3
"""
Cleanup Duplicate Files Script

This script removes duplicate JSON files that were created by the previous version
of the document processing pipeline. It specifically removes files with the 
"_parsed.json" suffix, keeping only the original ".json" files.

Usage: python scripts/cleanup_duplicates.py
"""

import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_duplicate_parsed_files():
    """Remove duplicate _parsed.json files."""
    try:
        settings = Settings()
        parsed_docs_dir = settings.parsed_docs_dir
        
        if not parsed_docs_dir.exists():
            logger.info("Parsed docs directory doesn't exist yet - nothing to clean up")
            return
        
        # Find all _parsed.json files
        duplicate_files = list(parsed_docs_dir.glob("*_parsed.json"))
        
        if not duplicate_files:
            logger.info("No duplicate _parsed.json files found")
            return
        
        logger.info(f"Found {len(duplicate_files)} duplicate files to remove:")
        
        removed_count = 0
        for duplicate_file in duplicate_files:
            # Check if the original file exists
            original_name = duplicate_file.name.replace("_parsed.json", ".json")
            original_file = parsed_docs_dir / original_name
            
            if original_file.exists():
                logger.info(f"  Removing duplicate: {duplicate_file.name} (original exists: {original_name})")
                duplicate_file.unlink()
                removed_count += 1
            else:
                logger.warning(f"  Keeping {duplicate_file.name} - original {original_name} not found")
        
        logger.info(f"✅ Cleanup completed: removed {removed_count} duplicate files")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def main():
    """Main function."""
    print("🧹 Cleaning up duplicate parsed files...")
    print("=" * 50)
    
    cleanup_duplicate_parsed_files()
    
    print("Cleanup completed!")

if __name__ == "__main__":
    main()
