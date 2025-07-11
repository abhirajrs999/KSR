import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config.settings import Settings
from processing.document_parser import IRCDocumentParser
from processing.metadata_extractor import IRCMetadataExtractor
from processing.chunker import IRCChunker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles the complete document processing pipeline."""

    def __init__(self):
        """Initialize the document processor with settings and components."""
        self.settings = Settings()
        self.parser = IRCDocumentParser()
        self.metadata_extractor = IRCMetadataExtractor()
        self.chunker = IRCChunker()
        
        # Setup paths
        self.raw_pdfs_dir = self.settings.raw_pdfs_dir
        self.parsed_docs_dir = self.settings.parsed_docs_dir
        self.chunks_dir = self.settings.chunks_dir
        self.metadata_dir = self.settings.metadata_dir

    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files from the raw_pdfs directory."""
        if not self.raw_pdfs_dir.exists():
            logger.error(f"Raw PDFs directory does not exist: {self.raw_pdfs_dir}")
            return []

        pdf_files = list(self.raw_pdfs_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        return pdf_files

    async def process_single_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process a single PDF document through the complete pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing processing results and status
        """
        result = {
            "file_path": str(pdf_path),
            "status": "success",
            "errors": [],
            "processing_time": None,
            "parsed_doc_path": None,
            "metadata_path": None,
            "chunks_path": None
        }

        start_time = datetime.now()

        try:
            logger.info(f"Processing: {pdf_path.name}")

            # Step 1: Parse PDF
            logger.info(f"Parsing PDF: {pdf_path.name}")
            parsed_doc = await self.parser.parse_pdf(str(pdf_path))
            
            if not parsed_doc:
                raise Exception("Failed to parse PDF")

            # Parsed document is already saved by parser - just record the path
            parsed_doc_path = self.parsed_docs_dir / f"{pdf_path.stem}.json"
            result["parsed_doc_path"] = str(parsed_doc_path)
            logger.info(f"Parsed document saved: {parsed_doc_path.name}")

            # Step 2: Extract metadata
            logger.info(f"Extracting metadata: {pdf_path.name}")
            metadata = self.metadata_extractor.extract_metadata(parsed_doc)
            
            # Save metadata
            metadata_path = self.metadata_dir / f"{pdf_path.stem}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            result["metadata_path"] = str(metadata_path)
            logger.info(f"Saved metadata: {metadata_path.name}")

            # Step 3: Create chunks
            logger.info(f"Creating chunks: {pdf_path.name}")
            chunks = self.chunker.create_chunks(
                parsed_doc=parsed_doc,
                metadata=metadata
            )
            
            # Save chunks
            chunks_path = self.chunks_dir / f"{pdf_path.stem}_chunks.json"
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            result["chunks_path"] = str(chunks_path)
            logger.info(f"Saved {len(chunks)} chunks: {chunks_path.name}")

            # Calculate processing time
            result["processing_time"] = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed processing {pdf_path.name} in {result['processing_time']:.2f}s")

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error(f"Error processing {pdf_path.name}: {e}")
            result["processing_time"] = (datetime.now() - start_time).total_seconds()

        return result

    async def process_all_documents(self) -> Dict[str, Any]:
        """
        Process all PDF documents in the pipeline.
        
        Returns:
            Dictionary containing overall processing results
        """
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            logger.error("No PDF files found to process")
            return {
                "status": "error",
                "message": "No PDF files found",
                "processed_files": [],
                "total_files": 0,
                "successful": 0,
                "failed": 0
            }

        logger.info(f"Starting processing of {len(pdf_files)} documents")
        
        # Process documents concurrently (with limit to avoid overwhelming the API)
        semaphore = asyncio.Semaphore(3)  # Limit concurrent processing
        
        async def process_with_semaphore(pdf_path: Path):
            async with semaphore:
                return await self.process_single_document(pdf_path)

        # Process all documents
        tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions from gather
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "file_path": str(pdf_files[i]),
                    "status": "error",
                    "errors": [str(result)],
                    "processing_time": 0
                })
            else:
                processed_results.append(result)

        # Calculate summary statistics
        successful = sum(1 for r in processed_results if r["status"] == "success")
        failed = len(processed_results) - successful
        total_time = sum(r.get("processing_time", 0) for r in processed_results)

        summary = {
            "status": "completed",
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "total_processing_time": total_time,
            "average_time_per_file": total_time / len(pdf_files) if pdf_files else 0,
            "processed_files": processed_results
        }

        # Save processing summary
        summary_path = self.parsed_docs_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Processing completed. Summary saved to: {summary_path}")
        return summary

    async def process_all_documents_optimized(self) -> Dict[str, Any]:
        """
        Process all PDF documents using optimized batch parsing.
        
        Returns:
            Dictionary containing overall processing results
        """
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            logger.error("No PDF files found to process")
            return {
                "status": "error",
                "message": "No PDF files found",
                "processed_files": [],
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "total_processing_time": 0
            }

        logger.info(f"Starting optimized batch processing of {len(pdf_files)} documents")
        start_time = datetime.now()
        
        # Use batch parsing for better efficiency
        try:
            batch_results = await self.parser.batch_parse_pdfs_optimized(pdf_files)
            
            # Process each parsed document through the remaining pipeline
            processed_results = []
            for pdf_path, parsed_doc in batch_results:
                if parsed_doc is None:
                    processed_results.append({
                        "file_path": str(pdf_path),
                        "status": "error",
                        "errors": ["Failed to parse PDF"],
                        "processing_time": 0
                    })
                    continue
                
                # Continue with metadata extraction and chunking
                result = await self._process_parsed_document(pdf_path, parsed_doc)
                processed_results.append(result)
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to original processing
            return await self.process_all_documents()
        
        # Calculate summary statistics
        total_time = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in processed_results if r["status"] == "success")
        failed = len(processed_results) - successful

        summary = {
            "status": "completed",
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "total_processing_time": total_time,
            "average_time_per_file": total_time / len(pdf_files) if pdf_files else 0,
            "processed_files": processed_results,
            "optimization_used": "batch_processing"
        }

        # Save processing summary
        summary_path = self.parsed_docs_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Optimized processing completed in {total_time:.2f}s. Summary saved to: {summary_path}")
        return summary

    async def _process_parsed_document(self, pdf_path: Path, parsed_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single parsed document through metadata extraction and chunking."""
        result = {
            "file_path": str(pdf_path),
            "status": "success",
            "errors": [],
            "processing_time": None,
            "parsed_doc_path": None,
            "metadata_path": None,
            "chunks_path": None
        }

        start_time = datetime.now()

        try:
            # Parsed document is already saved by batch parser
            parsed_doc_path = self.parsed_docs_dir / f"{pdf_path.stem}.json"
            result["parsed_doc_path"] = str(parsed_doc_path)
            logger.info(f"Using batch-parsed document: {parsed_doc_path.name}")

            # Step 2: Extract metadata
            logger.info(f"Extracting metadata: {pdf_path.name}")
            metadata = self.metadata_extractor.extract_metadata(parsed_doc)
            
            # Save metadata
            metadata_path = self.metadata_dir / f"{pdf_path.stem}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            result["metadata_path"] = str(metadata_path)
            logger.info(f"Saved metadata: {metadata_path.name}")

            # Step 3: Create chunks
            logger.info(f"Creating chunks: {pdf_path.name}")
            chunks = self.chunker.create_chunks(
                parsed_doc=parsed_doc,
                metadata=metadata
            )
            
            # Save chunks
            chunks_path = self.chunks_dir / f"{pdf_path.stem}_chunks.json"
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            result["chunks_path"] = str(chunks_path)
            logger.info(f"Saved {len(chunks)} chunks: {chunks_path.name}")

            # Calculate processing time
            result["processing_time"] = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed processing {pdf_path.name} in {result['processing_time']:.2f}s")

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error(f"Error processing {pdf_path.name}: {e}")
            result["processing_time"] = (datetime.now() - start_time).total_seconds()

        return result

    def print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary of the processing results."""
        print("\n" + "="*60)
        print("DOCUMENT PROCESSING SUMMARY")
        print("="*60)
        print(f"Total files processed: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total processing time: {summary['total_processing_time']:.2f}s")
        print(f"Average time per file: {summary['average_time_per_file']:.2f}s")
        
        if summary['failed'] > 0:
            print("\nFailed files:")
            for result in summary['processed_files']:
                if result['status'] == 'error':
                    print(f"  - {Path(result['file_path']).name}: {', '.join(result['errors'])}")
        
        print("\nOutput files created:")
        print(f"  - Parsed documents: {self.parsed_docs_dir}")
        print(f"  - Metadata: {self.metadata_dir}")
        print(f"  - Chunks: {self.chunks_dir}")
        print(f"  - Processing summary: {self.parsed_docs_dir}/processing_summary.json")
        print("="*60 + "\n")

async def main():
    """Main function to run the document processing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PDF documents for RAG system')
    parser.add_argument('--use-batch', action='store_true', default=True,
                       help='Use optimized batch processing (default: True)')
    parser.add_argument('--no-batch', action='store_true',
                       help='Disable batch processing (use individual file processing)')
    
    args = parser.parse_args()
    
    # Determine processing mode
    use_batch_processing = args.use_batch and not args.no_batch
    
    try:
        processor = DocumentProcessor()
        
        # Check if raw PDFs directory exists and has files
        if not processor.raw_pdfs_dir.exists():
            logger.error(f"Raw PDFs directory not found: {processor.raw_pdfs_dir}")
            print(f"\nPlease place your PDF files in: {processor.raw_pdfs_dir}")
            return

        pdf_files = processor.get_pdf_files()
        if not pdf_files:
            logger.error("No PDF files found in raw_pdfs directory")
            print(f"\nPlease add PDF files to: {processor.raw_pdfs_dir}")
            return

        print(f"\n📄 Document Processing Pipeline")
        print("=" * 50)
        print(f"Processing mode: {'Optimized batch processing' if use_batch_processing else 'Individual file processing'}")
        print(f"Files found: {len(pdf_files)}")
        
        # Process documents
        if use_batch_processing:
            summary = await processor.process_all_documents_optimized()
        else:
            summary = await processor.process_all_documents()
        
        # Print summary
        processor.print_summary(summary)

    except Exception as e:
        logger.error(f"Fatal error in document processing: {e}")
        print(f"\nFatal error: {e}")
        print("Check the log file for more details.")

if __name__ == "__main__":
    asyncio.run(main())