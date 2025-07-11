import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TableAwareIRCChunker:
    """
    Enhanced chunker that handles tables as special entities in IRC documents.
    
    This chunker:
    1. Detects tables and creates dedicated table chunks
    2. Preserves table structure and context
    3. Creates searchable table representations
    4. Maintains hierarchical document structure
    """
    
    def __init__(self, 
                max_chunk_size: int = 3000,      # INCREASED from 2048
                min_chunk_size: int = 200,       # INCREASED from 100
                overlap_sentences: int = 3):     # INCREASED from 2
        """Initialize the table-aware chunker with better defaults."""
        self.processed_dir = settings.parsed_docs_dir
        self.chunks_dir = settings.chunks_dir if hasattr(settings, 'chunks_dir') else Path('data/chunks')
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences
        
        # Compile regex patterns
        self.patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for structure and table detection."""
        return {
            # Existing patterns
            'main_chapter': re.compile(r'^(\d+)\.\s+([^\n]+?)(?=\n|$)', re.MULTILINE | re.IGNORECASE),
            'sub_clause': re.compile(r'^(\d+\.\d+):\s*([^\n]+?)(?=\n|$)', re.MULTILINE | re.IGNORECASE),
            'sub_sub_clause': re.compile(r'^(\d+\.\d+\.\d+):\s*([^\n]+?)(?=\n|$)', re.MULTILINE | re.IGNORECASE),
            'special_section': re.compile(r'^(Annexure-[IVX]+|Appendix-[A-Z]+|ANNEXURE-[IVX]+|APPENDIX-[A-Z]+):\s*([^\n]+?)(?=\n|$)', re.MULTILINE | re.IGNORECASE),
            
            # IMPROVED table patterns - more flexible
            'table_header': re.compile(r'Table\s+(\d+(?:\.\d+)*)\s*[:\-]?\s*([^\n]+)', re.IGNORECASE),
            'table_markdown': re.compile(r'\|.+?\|(?:\n\|.+?\|)*', re.MULTILINE | re.DOTALL),
            'table_content': re.compile(r'(?:Table\s+\d+(?:\.\d+)*.*?\n)((?:(?:\|.+?\|.*?\n)+)|(?:(?:\w+.*?\s+\w+.*?\n){3,}))', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            
            # NEW: Survey/percentage content patterns
            'survey_content': re.compile(r'(?:commercial\s+traffic|cvpd|survey|sample|percentage|per\s+cent)', re.IGNORECASE),
            'percentage_data': re.compile(r'\d+\s*(?:per\s*cent|percent|%)', re.IGNORECASE),
            
            # Figure patterns
            'figure': re.compile(r'(Figure\s+\d+(?:\.\d+)*)', re.IGNORECASE)
        }
    
    def _detect_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        IMPROVED table detection that's more forgiving.
        """
        tables = []
        
        # Method 1: Look for "Table X.Y" anywhere in text (more flexible)
        table_pattern = re.compile(r'Table\s+(\d+(?:\.\d+)*)\s*[:\-]?\s*([^\n]*)', re.IGNORECASE)
        table_matches = list(table_pattern.finditer(text))
        
        for i, match in enumerate(table_matches):
            table_number = match.group(1)
            table_title = match.group(2).strip() or f"Table {table_number}"
            start_pos = match.start()
            
            # Look ahead for table content (more generous search)
            search_window = text[start_pos:start_pos + 1500]  # Look in next 1500 chars
            
            # Try to find structured content
            table_data = []
            lines = search_window.split('\n')
            
            for line in lines[1:20]:  # Check next 20 lines
                line = line.strip()
                if not line:
                    continue
                    
                # Look for table-like patterns
                if ('|' in line or 
                    re.search(r'\w+\s{3,}\w+', line) or  # Multiple spaces
                    re.search(r'\d+\s+to\s+\d+', line) or  # Number ranges
                    re.search(r'\d+\s*(?:per\s*cent|%)', line)):  # Percentages
                    
                    # Split by multiple spaces or tabs
                    cells = re.split(r'\s{2,}|\t', line)
                    if len(cells) >= 2:
                        table_data.append([cell.strip() for cell in cells])
            
            # If we found table-like data, include it
            if table_data:
                end_pos = min(start_pos + 1500, len(text))
                table_content = text[start_pos:end_pos]
                
                tables.append({
                    'number': table_number,
                    'title': table_title,
                    'content': table_content,
                    'structured_data': table_data,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'searchable_text': self._create_searchable_table_text(table_number, table_title, table_data)
                })
                
                logger.info(f"Detected table: {table_number} - {table_title} with {len(table_data)} rows")
        
        return tables
    
    def _extract_table_data(self, table_content: str) -> List[List[str]]:
        """
        Extract structured data from table content.
        
        Args:
            table_content: Raw table content text
            
        Returns:
            List of rows, each row is a list of cells
        """
        # First try markdown table format
        markdown_data = self._parse_markdown_table(table_content)
        if markdown_data:
            return markdown_data
        
        # Try to parse plain text tables
        lines = table_content.split('\n')
        table_data = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines that don't look like table data
            if not line or len(line) < 10:
                continue
                
            # Look for patterns that suggest table rows
            # Pattern 1: Multiple words/numbers separated by significant whitespace
            if re.search(r'\w+\s{2,}\w+', line):
                # Split by multiple spaces
                cells = re.split(r'\s{2,}', line)
                if len(cells) >= 2:
                    table_data.append([cell.strip() for cell in cells])
            
            # Pattern 2: Tab-separated values
            elif '\t' in line:
                cells = line.split('\t')
                if len(cells) >= 2:
                    table_data.append([cell.strip() for cell in cells])
        
        return table_data if len(table_data) >= 2 else []
    
    def _parse_markdown_table(self, table_text: str) -> List[List[str]]:
        """
        Parse markdown-style table.
        
        Args:
            table_text: Markdown table text
            
        Returns:
            List of rows, each row is a list of cells
        """
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        table_data = []
        
        for line in lines:
            if line.startswith('|') and line.endswith('|'):
                # Remove leading and trailing pipes
                line = line[1:-1]
                # Split by pipes and clean cells
                cells = [cell.strip() for cell in line.split('|')]
                
                # Skip separator lines (contain only dashes and spaces)
                if not all(re.match(r'^[-\s]*$', cell) for cell in cells):
                    table_data.append(cells)
        
        return table_data
    
    def _create_searchable_table_text(self, table_number: str, table_title: str, table_data: List[List[str]]) -> str:
        """
        Create a searchable text representation of the table.
        
        Args:
            table_number: Table number
            table_title: Table title
            table_data: Structured table data
            
        Returns:
            Searchable text representation
        """
        searchable_parts = [
            f"Table {table_number}: {table_title}",
            f"This table shows {table_title.lower()}",
        ]
        
        if table_data:
            # Add header row if available
            if len(table_data) > 0:
                headers = table_data[0]
                searchable_parts.append(f"Table headers: {', '.join(headers)}")
            
            # Add data rows with natural language description
            for i, row in enumerate(table_data[1:], 1):
                if len(row) >= 2:
                    # Create natural language descriptions
                    row_description = f"Row {i}: {row[0]} has values {', '.join(row[1:])}"
                    searchable_parts.append(row_description)
                    
                    # Add key-value pairs for better searching
                    if len(table_data) > 0 and len(table_data[0]) == len(row):
                        headers = table_data[0]
                        for header, value in zip(headers, row):
                            searchable_parts.append(f"{header}: {value}")
        
        return '\n'.join(searchable_parts)
    
    def _find_pages_for_chunk(self, chunk_text: str, page_mapping: List[Dict[str, Any]]) -> List[int]:
        """Find page numbers where this chunk content appears."""
        pages = set()
        
        # Split chunk into sentences for better matching
        sentences = [s.strip() for s in chunk_text.split('.') if s.strip()]
        
        for page in page_mapping:
            page_text = page.get('text', '')
            page_num = page.get('page_number', 1)
            
            # Check if any significant sentences from chunk appear in this page
            matches = 0
            for sentence in sentences[:5]:  # Check first 5 sentences
                if len(sentence) > 20 and sentence.lower() in page_text.lower():
                    matches += 1
                    
            # If we find multiple sentence matches, include this page
            if matches >= 2 or (len(sentences) <= 2 and matches >= 1):
                pages.add(page_num)
        
        return sorted(list(pages)) if pages else [1]
    
    def _create_table_chunk(self, table: Dict[str, Any], doc_metadata: Dict[str, Any], 
                           page_mapping: List[Dict[str, Any]], chunk_id_counter: int) -> Dict[str, Any]:
        """
        Create a specialized chunk for a table.
        
        Args:
            table: Table information dictionary
            doc_metadata: Document metadata
            page_mapping: Page mapping data
            chunk_id_counter: Chunk ID counter
            
        Returns:
            Table chunk dictionary
        """
        # Find pages for this table
        pages = self._find_pages_for_chunk(table['content'], page_mapping)
        
        # Create comprehensive table metadata
        table_metadata = doc_metadata.copy()
        table_metadata.update({
            'chunk_type': 'table',
            'table_number': table['number'],
            'table_title': table['title'],
            'element_number': table['number'],
            'element_title': table['title'],
            'chunk_id': f"{doc_metadata.get('irc_code', 'unknown').replace(':', '').replace('-', '')}_table{table['number'].replace('.', '_')}",
            'pages': pages,
            'hierarchy_path': f"Table > {table['number']}",
            'has_structured_data': bool(table['structured_data']),
            'table_rows': len(table['structured_data']) if table['structured_data'] else 0,
            'table_columns': len(table['structured_data'][0]) if table['structured_data'] and table['structured_data'] else 0,
            'searchable_keywords': self._extract_table_keywords(table)
        })
        
        return {
            'content': table['searchable_text'],  # Use searchable text as content
            'text': table['searchable_text'],
            'raw_table_content': table['content'],  # Keep original for reference
            'structured_data': table['structured_data'],  # Keep structured data
            'metadata': table_metadata
        }
    
    def _extract_table_keywords(self, table: Dict[str, Any]) -> str:
        """
        Extract searchable keywords from table data.
        
        Args:
            table: Table information
            
        Returns:
            Comma-separated keywords
        """
        keywords = set()
        
        # Add table number and title words
        keywords.add(f"table{table['number']}")
        title_words = re.findall(r'\w+', table['title'].lower())
        keywords.update(title_words)
        
        # Extract keywords from structured data
        if table['structured_data']:
            for row in table['structured_data']:
                for cell in row:
                    # Extract meaningful words (not just numbers)
                    words = re.findall(r'[a-zA-Z]+', str(cell).lower())
                    keywords.update(word for word in words if len(word) > 2)
        
        return ', '.join(sorted(list(keywords)))
    
    def create_chunks(self, parsed_doc: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        SAFER chunking that preserves content while adding table awareness.
        
        Strategy:
        1. Try to detect tables
        2. Create OVERLAPPING chunks to ensure nothing is lost
        3. Don't remove content, just add table-specific chunks
        """
        try:
            text = parsed_doc.get('full_text', '')
            page_mapping = parsed_doc.get('page_mapping', [])
            
            if not text:
                logger.warning("No text found in parsed document")
                return []
            
            # Merge metadata
            doc_metadata = parsed_doc.get('metadata', {})
            combined_metadata = {
                'irc_code': metadata.get('irc_code') or doc_metadata.get('irc_code'),
                'revision_year': metadata.get('revision_year') or doc_metadata.get('revision_year'),
                'title': metadata.get('title') or doc_metadata.get('title'),
                'source_file': metadata.get('source_file') or doc_metadata.get('source_file'),
            }
            
            chunks = []
            chunk_id_counter = 0
            
            # Step 1: Try to detect and create table chunks (but don't rely on it)
            tables = self._detect_tables(text)
            logger.info(f"Detected {len(tables)} tables in document")
            
            for table in tables:
                table_chunk = self._create_table_chunk(table, combined_metadata, page_mapping, chunk_id_counter)
                chunks.append(table_chunk)
                chunk_id_counter += 1
            
            # Step 2: ALWAYS create overlapping sliding window chunks from FULL text
            # This ensures we don't lose any content, even if table detection fails
            chunk_size = 1500  # Larger chunks
            overlap = 400      # More overlap
            start = 0
            
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]
                
                if len(chunk_text.strip()) < 100:  # Skip tiny chunks
                    start += chunk_size - overlap
                    continue
                
                # Create metadata for sliding window chunk
                window_metadata = combined_metadata.copy()
                window_metadata.update({
                    'chunk_type': 'sliding_window',
                    'chunk_id': f"{combined_metadata.get('irc_code', 'unknown').replace(':', '').replace('-', '')}_window{chunk_id_counter}",
                    'pages': self._find_pages_for_chunk(chunk_text, page_mapping),
                    'chunk_start': start,
                    'chunk_end': end
                })
                
                # Check if this chunk contains survey/percentage content
                if (self.patterns['survey_content'].search(chunk_text) and 
                    self.patterns['percentage_data'].search(chunk_text)):
                    window_metadata['contains_survey_data'] = True
                    window_metadata['chunk_type'] = 'survey_content'
                    logger.info(f"Found survey content in chunk {chunk_id_counter}")
                
                chunk = {
                    'content': chunk_text,
                    'text': chunk_text,
                    'metadata': window_metadata
                }
                chunks.append(chunk)
                chunk_id_counter += 1
                
                if end >= len(text):
                    break
                start += chunk_size - overlap
            
            # Step 3: Add some LARGER context chunks for better coverage
            large_chunk_size = 3000
            start = 0
            while start < len(text):
                end = min(start + large_chunk_size, len(text))
                chunk_text = text[start:end]
                
                if len(chunk_text.strip()) > 1000:  # Only if substantial
                    context_metadata = combined_metadata.copy()
                    context_metadata.update({
                        'chunk_type': 'large_context',
                        'chunk_id': f"{combined_metadata.get('irc_code', 'unknown').replace(':', '').replace('-', '')}_large{chunk_id_counter}",
                        'pages': self._find_pages_for_chunk(chunk_text, page_mapping),
                    })
                    
                    chunk = {
                        'content': chunk_text,
                        'text': chunk_text,
                        'metadata': context_metadata
                    }
                    chunks.append(chunk)
                    chunk_id_counter += 1
                
                if end >= len(text):
                    break
                start += large_chunk_size // 2  # 50% overlap
            
            logger.info(f"Created {len(chunks)} total chunks:")
            logger.info(f"  - Table chunks: {len(tables)}")
            logger.info(f"  - Sliding window chunks: {sum(1 for c in chunks if c['metadata']['chunk_type'] == 'sliding_window')}")
            logger.info(f"  - Survey content chunks: {sum(1 for c in chunks if c['metadata'].get('contains_survey_data', False))}")
            logger.info(f"  - Large context chunks: {sum(1 for c in chunks if c['metadata']['chunk_type'] == 'large_context')}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(parsed_doc.get('full_text', ''), 
                                        parsed_doc.get('page_mapping', []), 
                                        metadata)
    
    def _create_regular_chunks(self, text: str, metadata: Dict[str, Any], 
                              page_mapping: List[Dict[str, Any]], start_counter: int) -> List[Dict[str, Any]]:
        """
        Create regular hierarchical chunks from text (with tables removed).
        
        This is a simplified version of the hierarchical chunking for text content.
        """
        chunks = []
        
        # Extract structural elements
        main_chapters = list(self.patterns['main_chapter'].finditer(text))
        sub_clauses = list(self.patterns['sub_clause'].finditer(text))
        
        # Combine and sort all elements
        all_elements = []
        for match in main_chapters:
            all_elements.append(('main_chapter', {
                'number': match.group(1),
                'title': match.group(2).strip(),
                'start_pos': match.start(),
                'match': match
            }))
        
        for match in sub_clauses:
            all_elements.append(('sub_clause', {
                'number': match.group(1),
                'title': match.group(2).strip(),
                'start_pos': match.start(),
                'match': match
            }))
        
        # Sort by position
        all_elements.sort(key=lambda x: x[1]['start_pos'])
        
        # Create chunks
        for i, (element_type, element) in enumerate(all_elements):
            start_pos = element['start_pos']
            end_pos = all_elements[i + 1][1]['start_pos'] if i + 1 < len(all_elements) else len(text)
            
            content = text[start_pos:end_pos].strip()
            
            if len(content) < self.min_chunk_size:
                continue
            
            # Create metadata
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_type': element_type,
                'element_number': element['number'],
                'element_title': element['title'],
                'chunk_id': f"{metadata.get('irc_code', 'unknown').replace(':', '').replace('-', '')}_chunk{start_counter + len(chunks)}",
                'pages': self._find_pages_for_chunk(content, page_mapping)
            })
            
            if element_type == 'main_chapter':
                chunk_metadata.update({
                    'chapter_number': element['number'],
                    'chapter_title': element['title'],
                    'hierarchy_path': element['number']
                })
            elif element_type == 'sub_clause':
                parent_chapter = element['number'].split('.')[0]
                chunk_metadata.update({
                    'chapter_number': parent_chapter,
                    'clause_number': element['number'],
                    'clause_title': element['title'],
                    'hierarchy_path': f"{parent_chapter} > {element['number']}"
                })
            
            chunks.append({
                'content': content,
                'text': content,
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def _fallback_chunking(self, text: str, page_mapping: List[Dict[str, Any]], 
                          metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback to simple chunking if hierarchical extraction fails."""
        chunks = []
        chunk_size = 1024
        chunk_overlap = 200
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            if not chunk_text.strip():
                start += chunk_size - chunk_overlap
                continue
            
            pages = self._find_pages_for_chunk(chunk_text, page_mapping)
            
            chunk = {
                'content': chunk_text,
                'text': chunk_text,
                'pages': pages,
                'metadata': {
                    **metadata,
                    'chunk_type': 'simple',
                    'chunk_id': f"{metadata.get('irc_code', 'unknown').replace(':', '').replace('-', '')}_fallback{chunk_id}",
                    'chunk_start': start,
                    'chunk_end': end
                }
            }
            chunks.append(chunk)
            chunk_id += 1
            
            if end == len(text):
                break
            start += chunk_size - chunk_overlap
        
        return chunks

# For backward compatibility
HierarchicalIRCChunker = TableAwareIRCChunker
IRCChunker = TableAwareIRCChunker