#!/usr/bin/env python3
"""
Clear Vector Database Script

This script safely removes the existing ChromaDB vector database folder
to allow for a fresh start. It will:
1. Remove the entire vector_db folder
2. Recreate the directory structure
3. Provide confirmation of the cleanup

Usage: python scripts/clear_vector_db.py
"""

import sys
import shutil
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_vector_database():
    """Clear the entire vector database folder."""
    try:
        settings = Settings()
        vector_db_dir = settings.vector_db_dir
        
        print("🗑️  Vector Database Cleanup")
        print("=" * 50)
        print(f"Target directory: {vector_db_dir}")
        
        if vector_db_dir.exists():
            # Get size info before deletion
            total_size = sum(f.stat().st_size for f in vector_db_dir.rglob('*') if f.is_file())
            file_count = len(list(vector_db_dir.rglob('*')))
            
            print(f"Found existing vector database:")
            print(f"  - Files: {file_count}")
            print(f"  - Size: {total_size / (1024*1024):.2f} MB")
            
            # Remove the entire directory
            logger.info(f"Removing vector database directory: {vector_db_dir}")
            shutil.rmtree(vector_db_dir)
            print(f"✅ Removed existing vector database")
        else:
            print(f"ℹ️  No existing vector database found")
        
        # Recreate the directory structure
        vector_db_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created clean vector database directory: {vector_db_dir}")
        print(f"✅ Created clean directory structure")
        
        # Also clear any summary files
        summary_files = [
            vector_db_dir / "vector_db_creation_summary.json",
            vector_db_dir.parent / "vector_db_creation_summary.json"
        ]
        
        for summary_file in summary_files:
            if summary_file.exists():
                summary_file.unlink()
                print(f"✅ Removed summary file: {summary_file.name}")
        
        print(f"\n🎉 Vector database cleanup completed!")
        print(f"Ready for fresh vector database creation.")
        print(f"\nNext steps:")
        print(f"1. Run: python scripts/03_create_vector_db.py")
        print(f"2. Or run: python scripts/quick_update.py --process-new-pdfs")
        
    except Exception as e:
        logger.error(f"Error during vector database cleanup: {e}")
        print(f"\n❌ Error: {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clear vector database for fresh start')
    parser.add_argument('--confirm', action='store_true',
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    if not args.confirm:
        print("⚠️  This will completely remove the existing vector database.")
        print("All indexed documents will be deleted and need to be recreated.")
        response = input("\nAre you sure you want to proceed? (yes/no): ").lower().strip()
        
        if response not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    clear_vector_database()

if __name__ == "__main__":
    main()
