#!/usr/bin/env python3
"""
Bulk upload script for IRC RAG API deployed on Railway.
This script uploads multiple PDF documents to the deployed API for processing.
"""

import asyncio
import aiohttp
import aiofiles
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bulk_upload.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BulkUploader:
    """Bulk uploader for IRC documents to Railway-deployed API."""
    
    def __init__(self, api_url: str, max_concurrent: int = 3, timeout: int = 300):
        """
        Initialize bulk uploader.
        
        Args:
            api_url: Base URL of the deployed API
            max_concurrent: Maximum concurrent uploads
            timeout: Timeout for each upload in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.session = None
        
        # Statistics
        self.total_files = 0
        self.uploaded_count = 0
        self.failed_count = 0
        self.start_time = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def check_api_health(self) -> bool:
        """Check if the API is healthy and accessible."""
        try:
            async with self.session.get(f"{self.api_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"API Health Check Passed:")
                    logger.info(f"  Status: {health_data.get('status')}")
                    logger.info(f"  Documents in DB: {health_data.get('vector_db_documents')}")
                    logger.info(f"  Deployment: {health_data.get('deployment_info', {}).get('environment')}")
                    return True
                else:
                    logger.error(f"API health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Failed to connect to API: {e}")
            return False
    
    async def upload_single_file(self, file_path: Path, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a single PDF file to the API.
        
        Args:
            file_path: Path to the PDF file
            description: Optional description for the document
            
        Returns:
            Dict with upload result information
        """
        result = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'success': False,
            'response': None,
            'error': None,
            'upload_time': None
        }
        
        start_time = time.time()
        
        try:
            # Prepare form data
            data = aiohttp.FormData()
            
            # Add file
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
                data.add_field('file', file_content, filename=file_path.name, content_type='application/pdf')
            
            # Add description if provided
            if description:
                data.add_field('description', description)
            else:
                data.add_field('description', f"IRC document: {file_path.stem}")
            
            # Upload file
            async with self.session.post(f"{self.api_url}/upload", data=data) as response:
                result['upload_time'] = time.time() - start_time
                
                if response.status == 200:
                    response_data = await response.json()
                    result['success'] = True
                    result['response'] = response_data
                    logger.info(f"✅ Uploaded: {file_path.name} ({result['upload_time']:.1f}s)")
                    self.uploaded_count += 1
                else:
                    error_text = await response.text()
                    result['error'] = f"HTTP {response.status}: {error_text}"
                    logger.error(f"❌ Failed to upload {file_path.name}: {result['error']}")
                    self.failed_count += 1
                    
        except Exception as e:
            result['upload_time'] = time.time() - start_time
            result['error'] = str(e)
            logger.error(f"❌ Error uploading {file_path.name}: {e}")
            self.failed_count += 1
        
        return result
    
    async def upload_files(self, pdf_files: List[Path], descriptions: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Upload multiple PDF files concurrently.
        
        Args:
            pdf_files: List of PDF file paths
            descriptions: Optional dictionary mapping file names to descriptions
            
        Returns:
            List of upload results
        """
        self.total_files = len(pdf_files)
        self.start_time = time.time()
        
        logger.info(f"Starting bulk upload of {self.total_files} files...")
        logger.info(f"Max concurrent uploads: {self.max_concurrent}")
        
        # Create semaphore to limit concurrent uploads
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def upload_with_semaphore(file_path: Path) -> Dict[str, Any]:
            async with semaphore:
                description = None
                if descriptions:
                    description = descriptions.get(file_path.name)
                return await self.upload_single_file(file_path, description)
        
        # Execute uploads concurrently
        tasks = [upload_with_semaphore(file_path) for file_path in pdf_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    'file_name': pdf_files[i].name,
                    'file_path': str(pdf_files[i]),
                    'success': False,
                    'error': str(result),
                    'upload_time': None
                })
                self.failed_count += 1
            else:
                final_results.append(result)
        
        return final_results
    
    def print_summary(self):
        """Print upload summary statistics."""
        if self.start_time:
            total_time = time.time() - self.start_time
            
            logger.info("\n" + "="*60)
            logger.info("BULK UPLOAD SUMMARY")
            logger.info("="*60)
            logger.info(f"Total files processed: {self.total_files}")
            logger.info(f"Successfully uploaded: {self.uploaded_count}")
            logger.info(f"Failed uploads: {self.failed_count}")
            logger.info(f"Success rate: {(self.uploaded_count/self.total_files)*100:.1f}%")
            logger.info(f"Total time: {total_time:.1f} seconds")
            logger.info(f"Average time per file: {total_time/self.total_files:.1f} seconds")
            logger.info("="*60)

def load_descriptions_from_file(descriptions_file: Path) -> Dict[str, str]:
    """
    Load file descriptions from a JSON file.
    
    Expected format:
    {
        "file1.pdf": "Description for file 1",
        "file2.pdf": "Description for file 2"
    }
    """
    try:
        with open(descriptions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load descriptions file: {e}")
        return {}

def discover_pdf_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Discover PDF files in a directory."""
    if recursive:
        return list(directory.rglob("*.pdf"))
    else:
        return list(directory.glob("*.pdf"))

async def main():
    """Main function for bulk upload."""
    parser = argparse.ArgumentParser(description="Bulk upload PDFs to IRC RAG API on Railway")
    parser.add_argument("api_url", help="Base URL of the deployed API (e.g., https://your-app.up.railway.app)")
    parser.add_argument("pdf_directory", type=Path, help="Directory containing PDF files to upload")
    parser.add_argument("--descriptions", type=Path, help="JSON file with file descriptions")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Maximum concurrent uploads (default: 3)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per upload in seconds (default: 300)")
    parser.add_argument("--recursive", action="store_true", help="Search for PDFs recursively in subdirectories")
    parser.add_argument("--output", type=Path, help="Output file for upload results (JSON)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.pdf_directory.exists():
        logger.error(f"PDF directory does not exist: {args.pdf_directory}")
        return 1
    
    if not args.pdf_directory.is_dir():
        logger.error(f"PDF directory is not a directory: {args.pdf_directory}")
        return 1
    
    # Discover PDF files
    pdf_files = discover_pdf_files(args.pdf_directory, args.recursive)
    if not pdf_files:
        logger.error(f"No PDF files found in: {args.pdf_directory}")
        return 1
    
    logger.info(f"Found {len(pdf_files)} PDF files to upload")
    
    # Load descriptions if provided
    descriptions = None
    if args.descriptions:
        descriptions = load_descriptions_from_file(args.descriptions)
        logger.info(f"Loaded descriptions for {len(descriptions)} files")
    
    # Perform bulk upload
    async with BulkUploader(args.api_url, args.max_concurrent, args.timeout) as uploader:
        # Check API health first
        if not await uploader.check_api_health():
            logger.error("API health check failed. Aborting upload.")
            return 1
        
        # Upload files
        results = await uploader.upload_files(pdf_files, descriptions)
        
        # Print summary
        uploader.print_summary()
        
        # Save results if output file specified
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to: {args.output}")
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
        
        # Return appropriate exit code
        return 0 if uploader.failed_count == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
