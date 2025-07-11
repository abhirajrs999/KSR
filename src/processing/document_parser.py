import asyncio
import json
import logging
from pathlib import Path
from typing import List, Union, Dict, Any

import aiofiles
from llama_parse import LlamaParse
from tqdm.asyncio import tqdm

# Fixed import - remove the relative import
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IRCDocumentParser:
    """
    A document parser for the IRC RAG system using LlamaParse.

    This class parses PDF documents and extracts their content into a structured
    JSON format. The output for each document is a single JSON file containing
    the full text, page-by-page content, and metadata, which is ideal for
    subsequent chunking and indexing. Tables are preserved within the
    markdown content of each page.
    """

    def __init__(self):
        """Initializes the document parser with optimized batch processing."""
        self.parser = LlamaParse(
            api_key=settings.LLAMA_PARSE_API_KEY,
            result_type="markdown",  # Markdown preserves tables and structure
            num_workers=4,  # Enable parallel processing for batch requests
            verbose=True
        )
        self.parsed_docs_dir = settings.parsed_docs_dir
        self.parsed_docs_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(3)  # Reduced since we're batching

    async def _save_json_output(self, filename_stem: str, data: Dict[str, Any]):
        """Saves the structured data as a single JSON file."""
        json_path = self.parsed_docs_dir / f"{filename_stem}.json"
        try:
            async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=4))
            logger.info(f"Successfully saved structured JSON to {json_path}")
        except IOError as e:
            logger.error(f"Error saving JSON file for {filename_stem}: {e}")

    async def parse_pdf(self, pdf_path: Union[str, Path]):
        """
        Parses a single PDF file and saves the output as a structured JSON file.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            logger.error(f"PDF file not found: {pdf_path}")
            return None

        filename_stem = pdf_path.stem
        json_path = self.parsed_docs_dir / f"{filename_stem}.json"
        
        # If already parsed, load and return the existing data
        if json_path.exists():
            logger.info(f"Loading already parsed file: {pdf_path.name}")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                return existing_data  # ✅ Return the existing data
            except Exception as e:
                logger.warning(f"Error loading existing file {json_path}: {e}")
                # Continue to re-parse if we can't load the existing file

        logger.info(f"Parsing {pdf_path.name}...")
        async with self.semaphore:
            try:
                documents = await self.parser.aload_data(str(pdf_path))
                if not documents:
                    logger.warning(f"No content parsed from {pdf_path.name}")
                    return

                full_text = "\n\n".join([doc.get_content() for doc in documents])
                
                # Create a representative metadata object for the document
                doc_metadata = documents[0].metadata.copy() if documents[0].metadata else {}
                doc_metadata.pop('page_label', None)
                doc_metadata['source_file'] = pdf_path.name

                pages_data = [{
                    "page_number": doc.metadata.get('page_label', i + 1),
                    "text": doc.get_content(),
                    "metadata": doc.metadata
                } for i, doc in enumerate(documents)]

                # Consolidate all information into a single JSON structure
                json_output = {
                    "full_text": full_text,
                    "metadata": doc_metadata,
                    "page_mapping": pages_data,
                }
                
                await self._save_json_output(filename_stem, json_output)
                return json_output

            except Exception as e:
                logger.error(f"Failed to parse {pdf_path.name}: {e}", exc_info=True)
                return None
            
            await asyncio.sleep(1)  # Respect API rate limits

    async def batch_parse_pdfs(self, pdf_paths: List[Union[str, Path]]):
        """
        Parses a batch of PDF files asynchronously with progress tracking.
        """
        tasks = [self.parse_pdf(pdf_path) for pdf_path in pdf_paths]
        for future in tqdm.as_completed(tasks, desc="Parsing PDFs"):
            await future

    async def batch_parse_pdfs_optimized(self, pdf_paths: List[Union[str, Path]]):
        """
        Parse PDFs individually to avoid content mixing issues.
        This method prioritizes accuracy over speed by parsing each file separately.
        """
        pdf_paths = [Path(p) for p in pdf_paths]
        
        # Separate files that need parsing vs. already parsed
        files_to_parse = []
        already_parsed = []
        
        for pdf_path in pdf_paths:
            filename_stem = pdf_path.stem
            json_path = self.parsed_docs_dir / f"{filename_stem}.json"
            
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    already_parsed.append((pdf_path, existing_data))
                    logger.info(f"Skipping already parsed file: {pdf_path.name}")
                except Exception as e:
                    logger.warning(f"Error loading existing file {json_path}: {e}")
                    files_to_parse.append(pdf_path)
            else:
                files_to_parse.append(pdf_path)
        
        logger.info(f"Individual processing: {len(files_to_parse)} new files, {len(already_parsed)} already parsed")
        
        # Process files individually to avoid content mixing
        individual_results = []
        if files_to_parse:
            logger.info(f"Parsing {len(files_to_parse)} PDF files individually to ensure content integrity...")
            
            for i, pdf_path in enumerate(files_to_parse):
                logger.info(f"Parsing file {i+1}/{len(files_to_parse)}: {pdf_path.name}")
                
                try:
                    result = await self.parse_pdf(pdf_path)
                    individual_results.append((pdf_path, result))
                    logger.info(f"Successfully parsed {pdf_path.name}")
                    
                    # Rate limiting between files
                    await asyncio.sleep(2)  # Increased delay to respect API limits
                    
                except Exception as e:
                    logger.error(f"Failed to parse {pdf_path.name}: {e}")
                    individual_results.append((pdf_path, None))
                    continue
        
        # Combine results
        all_results = []
        for pdf_path, result in already_parsed:
            all_results.append((pdf_path, result))
        for pdf_path, result in individual_results:
            all_results.append((pdf_path, result))
        
        logger.info(f"Completed individual parsing of {len(files_to_parse)} files")
        return all_results

    async def _process_pdf_documents(self, pdf_path: Path, documents: List) -> Dict[str, Any]:
        """Process documents for a single PDF and save as JSON."""
        try:
            if not documents:
                logger.warning(f"No content parsed from {pdf_path.name}")
                return None

            full_text = "\n\n".join([doc.get_content() for doc in documents])
            
            # Create metadata
            doc_metadata = documents[0].metadata.copy() if documents[0].metadata else {}
            doc_metadata.pop('page_label', None)
            doc_metadata['source_file'] = pdf_path.name

            pages_data = [{
                "page_number": doc.metadata.get('page_label', i + 1),
                "text": doc.get_content(),
                "metadata": doc.metadata
            } for i, doc in enumerate(documents)]

            # Consolidate into JSON structure
            json_output = {
                "full_text": full_text,
                "metadata": doc_metadata,
                "page_mapping": pages_data,
            }
            
            # Save JSON file
            filename_stem = pdf_path.stem
            await self._save_json_output(filename_stem, json_output)
            return json_output

        except Exception as e:
            logger.error(f"Error processing documents for {pdf_path.name}: {e}")
            return None