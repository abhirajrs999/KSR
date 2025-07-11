import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from api.gemini_chat import GeminiChatEngine
from database.vector_store import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Enhanced search result with citation information and table support."""
    text: str
    irc_code: str
    pages: List[int]
    relevance_score: float
    source_file: str
    title: str
    chunk_type: str = "text"
    revision_year: Optional[str] = None
    clause_numbers: Optional[List[str]] = None
    table_number: Optional[str] = None
    table_title: Optional[str] = None
    structured_data: Optional[List[List[str]]] = None

    def get_citation(self) -> str:
        """Generate a formatted citation string."""
        # Use source_file for more accurate citation if available
        if self.source_file and self.source_file != self.irc_code:
            citation = f"IRC {self.source_file}"
        else:
            citation = f"IRC {self.irc_code}"
            
        if self.revision_year:
            citation += f" ({self.revision_year})"
        
        if self.chunk_type == "table" and self.table_number:
            citation += f", Table {self.table_number}"
            if self.table_title:
                citation += f": {self.table_title}"
        
        if self.pages:
            citation += f", Page(s): {', '.join(map(str, self.pages))}"
        if self.clause_numbers:
            citation += f", Clause(s): {', '.join(self.clause_numbers)}"
        return citation

    def is_table(self) -> bool:
        """Check if this result is from a table."""
        return self.chunk_type == "table"

class EnhancedIRCQueryEngine:
    """
    Enhanced query engine with special handling for table queries.
    
    Features:
    - Detects table-related queries
    - Uses hybrid search for better table retrieval
    - Formats table results appropriately
    - Maintains regular text search capabilities
    - Dynamic parameter extraction for any IRC table
    """
    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        chat_engine: GeminiChatEngine,
        default_search_limit: int = 5,
        relevance_threshold: float = 0.001,  # Very low for testing - same as yours
        context_window: int = 2,
        max_results: int = 5  # New parameter for enhanced functionality
    ):
        """
        Initialize the enhanced IRC query engine.
        
        Args:
            vector_store: ChromaDB vector store instance
            chat_engine: Gemini chat engine for response generation
            default_search_limit: Default number of results (for backward compatibility)
            relevance_threshold: Minimum relevance score for results
            context_window: Context window size (for backward compatibility)
            max_results: Maximum number of results for enhanced search
        """
        self.vector_store = vector_store
        self.chat_engine = chat_engine
        self.default_search_limit = default_search_limit
        self.relevance_threshold = relevance_threshold
        self.context_window = context_window
        self.max_results = max_results
        
        # Compile patterns for query analysis
        self.table_query_patterns = [
            r'\btable\s+\d+(?:\.\d+)*\b',  # "table 4.2"
            r'\bvalues?\s+(?:for|of|in)\b',  # "values for", "values in"
            r'\bindicative\s+\w+\s+values?\b',  # "indicative VDF values"
            r'\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:values?|data)\b',  # "what is the value"
            r'\bshow\s+(?:me\s+)?(?:the\s+)?(?:table|values?|data)\b',  # "show me the table"
            r'\b(?:traffic\s+volume|vdf|commercial\s+vehicles?)\b',  # domain-specific terms
            r'\b(?:terrain|rolling|plain|hilly)\b',  # table-specific terms
            r'\b\d+-\d+\b',  # ranges like "150-1500"
            r'\bper\s+day\b',  # "per day"
        ]
        
        self.compiled_table_patterns = [re.compile(pattern, re.IGNORECASE) 
                                       for pattern in self.table_query_patterns]
    
    def _is_table_query(self, query: str) -> bool:
        """
        Determine if a query is likely asking for table information.
        
        Args:
            query: User query string
            
        Returns:
            True if query appears to be table-related
        """
        query_lower = query.lower()
        
        # Check for explicit table mentions
        if 'table' in query_lower:
            return True
        
        # Count pattern matches
        pattern_matches = sum(1 for pattern in self.compiled_table_patterns 
                            if pattern.search(query))
        
        # If multiple patterns match, likely a table query
        return pattern_matches >= 2
    
    def _extract_query_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract specific parameters from table-related queries.
        Enhanced for CVPD, survey, and percentage queries.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with extracted parameters
        """
        query_lower = query.lower()
        parameters = {}
        
        # Extract traffic volume ranges - enhanced patterns
        volume_patterns = [
            r'(\d+)\s*to\s*(\d+)',  # "3000 to 6000"
            r'(\d+)-(\d+)',  # "3000-6000"
            r'between\s+(\d+)\s+(?:and\s+|to\s+)(\d+)',  # "between 3000 and 6000"
            r'more\s+than\s+(\d+)',  # "more than 6000"
            r'less\s+than\s+(\d+)',  # "less than 3000"
            r'exceeding\s+(\d+)',  # "exceeding 6000"
            r'above\s+(\d+)',  # "above 3000"
            r'below\s+(\d+)',  # "below 6000"
        ]
        
        for pattern in volume_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if 'more than' in query_lower or 'exceeding' in query_lower or 'above' in query_lower:
                    parameters['traffic_volume'] = f"more than {match.group(1)}"
                elif 'less than' in query_lower or 'below' in query_lower:
                    parameters['traffic_volume'] = f"less than {match.group(1)}"
                elif len(match.groups()) >= 2:
                    parameters['traffic_volume'] = f"{match.group(1)}-{match.group(2)}"
                else:
                    parameters['traffic_volume'] = match.group(1)
                break
        
        # Extract terrain types
        terrain_patterns = [
            r'\b(rolling[/\s]*plain|plain[/\s]*rolling)\b',
            r'\b(hilly)\b',
            r'\b(mountainous)\b',
            r'\b(flat)\b',
            r'\b(undulating)\b'
        ]
        
        for pattern in terrain_patterns:
            match = re.search(pattern, query_lower)
            if match:
                terrain = match.group(1).replace('/', '').replace(' ', '')
                if 'rolling' in terrain or 'plain' in terrain:
                    parameters['terrain'] = 'rolling/plain'
                else:
                    parameters['terrain'] = match.group(1)
                break
        
        # Extract parameter types - enhanced for survey queries
        if 'vdf' in query_lower:
            parameters['parameter_type'] = 'VDF'
        elif 'factor' in query_lower:
            parameters['parameter_type'] = 'factor'
        elif 'percentage' in query_lower or 'percent' in query_lower or '%' in query:
            parameters['parameter_type'] = 'percentage'
        elif 'survey' in query_lower:
            parameters['parameter_type'] = 'survey'
        elif 'sample' in query_lower:
            parameters['parameter_type'] = 'sample'
        elif 'value' in query_lower:
            parameters['parameter_type'] = 'value'
        
        # Extract CVPD-specific terms
        if 'cvpd' in query_lower or 'commercial vehicles per day' in query_lower:
            parameters['measurement_unit'] = 'CVPD'
        
        if 'commercial traffic volume' in query_lower:
            parameters['traffic_type'] = 'commercial traffic volume'
        
        # Extract what's being asked for
        if 'minimum' in query_lower:
            parameters['requirement_type'] = 'minimum'
        elif 'maximum' in query_lower:
            parameters['requirement_type'] = 'maximum'
        
        if 'surveyed' in query_lower or 'survey' in query_lower:
            parameters['action'] = 'survey'
        
        # Extract specific numbers or ranges that might be important
        all_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        if all_numbers:
            parameters['numbers_mentioned'] = all_numbers
        
        return parameters
    
    def _extract_table_search_terms(self, query: str) -> List[str]:
        """
        Extract specific terms that help identify relevant tables.
        Enhanced to handle CVPD, survey, and percentage queries.
        
        Args:
            query: User query string
            
        Returns:
            List of search terms
        """
        terms = []
        query_lower = query.lower()
        
        # Extract table numbers
        table_matches = re.findall(r'table\s+(\d+(?:\.\d+)*)', query_lower)
        terms.extend([f"table{num}" for num in table_matches])
        
        # Extract value ranges - enhanced for CVPD queries
        range_matches = re.findall(r'(\d+)\s*(?:to|-)\s*(\d+)', query)
        for start, end in range_matches:
            terms.extend([f"{start}-{end}", f"{start} to {end}", start, end])
        
        # Extract single numbers that might be important
        number_matches = re.findall(r'\b(\d+)\b', query)
        terms.extend(number_matches)
        
        # Domain-specific terms - enhanced for survey/percentage queries
        domain_terms = [
            'vdf', 'traffic', 'volume', 'commercial', 'vehicles',
            'terrain', 'rolling', 'plain', 'hilly', 'indicative',
            'cvpd', 'survey', 'percentage', 'percent', 'minimum',
            'sample', 'size', 'axle', 'load', 'surveyed'
        ]
        for term in domain_terms:
            if term in query_lower:
                terms.append(term)
        
        # Specific combinations for CVPD queries
        if 'cvpd' in query_lower or 'commercial traffic volume' in query_lower:
            terms.extend([
                'commercial traffic volume',
                'CVPD', 
                'commercial vehicles per day',
                'traffic volume'
            ])
        
        if 'survey' in query_lower:
            terms.extend([
                'axle load survey',
                'sample size',
                'minimum sample',
                'traffic survey'
            ])
        
        if 'percent' in query_lower or '%' in query:
            terms.extend([
                'percentage',
                'per cent', 
                'percent',
                'minimum percentage'
            ])
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in terms:
            if term.lower() not in seen:
                unique_terms.append(term)
                seen.add(term.lower())
        
        return unique_terms
    
    def _extract_clause_numbers(self, text: str) -> List[str]:
        """Extract clause numbers from text using regex."""
        pattern = r'\b\d{1,3}(?:\.\d{1,3}){1,4}\b'
        return sorted(set(re.findall(pattern, text)))
    
    def _rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rank results based on multiple factors.
        """
        for result in results:
            # Adjust score based on presence of clause numbers
            if result.clause_numbers:
                result.relevance_score *= 1.2
            # Adjust score based on page continuity with other results
            page_continuity = sum(
                1 for r in results
                if any(abs(p1 - p2) <= 1 
                    for p1 in result.pages 
                    for p2 in r.pages)
            )
            result.relevance_score *= (1 + 0.1 * page_continuity)
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def _convert_to_search_result(self, result: Dict[str, Any]) -> Optional[SearchResult]:
        """
        Convert raw search result to SearchResult object.
        
        Args:
            result: Raw search result from vector store
            
        Returns:
            SearchResult object or None if conversion fails
        """
        try:
            metadata = result.get('metadata', {})
            
            # Get text content - handle both 'document' and 'text' fields
            text_content = result.get('document', '') or result.get('text', '')
            
            # Parse pages - handle both string and list formats (same as your working code)
            pages_raw = metadata.get('pages', '')
            if isinstance(pages_raw, str):
                pages = [int(p.strip()) for p in pages_raw.split(',') if p.strip().isdigit()]
            elif isinstance(pages_raw, list):
                pages = [int(p) for p in pages_raw if str(p).isdigit()]
            else:
                pages = []
            
            # Calculate relevance score (same as your working code)
            distance = result.get('distance', 1.0)
            relevance_score = max(0.0, 1.0 - distance)
            
            # Extract clause numbers (same as your working code)
            clause_numbers = self._extract_clause_numbers(text_content)
            
            # Fix IRC code issue - use source_file if available, fallback to irc_code
            source_file = metadata.get('source_file', '')
            irc_code = metadata.get('irc_code', '')
            
            # If source_file looks like IRC code format, use it for IRC code
            if source_file and ('IRC' in source_file.upper() or '-' in source_file):
                # Convert "IRC-37-2019" to "IRC:37-2019" format
                display_irc_code = source_file.replace('-', ':') if 'IRC' in source_file.upper() else irc_code
            else:
                display_irc_code = irc_code
            
            return SearchResult(
                text=text_content,
                irc_code=display_irc_code,  # Use corrected IRC code
                pages=pages,
                relevance_score=relevance_score,
                source_file=source_file,
                title=metadata.get('title', ''),
                chunk_type=metadata.get('chunk_type', 'text'),  # New field
                revision_year=metadata.get('revision_year'),
                clause_numbers=clause_numbers,
                table_number=metadata.get('table_number'),  # New field
                table_title=metadata.get('table_title'),  # New field
                structured_data=result.get('structured_data')  # New field
            )
            
        except Exception as e:
            logger.error(f"Error converting search result: {e}")
            return None
    
    def _search_tables(self, query: str, filter_criteria: Dict[str, Any]) -> List[SearchResult]:
        """
        Specialized search for table-related queries.
        NOW INCLUDES: Enhanced keyword search for table content in regular chunks.
        
        Args:
            query: User query
            filter_criteria: Metadata filters
            
        Returns:
            List of search results
        """
        results = []
        
        try:
            # Strategy 1: Search specifically for table chunks (if they exist)
            table_filter = {}
            if filter_criteria:
                table_filter.update(filter_criteria)
            table_filter['chunk_type'] = 'table'
            
            logger.info(f"Searching for table chunks with filter: {table_filter}")
            
            table_results = self.vector_store.search_by_text(
                query=query,
                n_results=self.max_results,
                filter_criteria=table_filter
            )
            
            logger.info(f"Found {len(table_results)} table-specific results")
            
            # Convert to SearchResult objects
            for result in table_results:
                if result.get('distance', 1.0) <= (1.0 - self.relevance_threshold):
                    search_result = self._convert_to_search_result(result)
                    if search_result:
                        results.append(search_result)
            
        except Exception as e:
            logger.error(f"Table chunk search failed (expected if no table chunks): {e}")
        
        # Strategy 2: Enhanced keyword-based search for table content in regular chunks
        # This is crucial for finding table data that exists in regular text chunks
        try:
            logger.info("Searching for table content using enhanced keywords...")
            
            # Extract table-specific terms from the query
            table_terms = self._extract_table_search_terms(query)
            
            # Create multiple search variations to catch table content
            search_variations = [
                query,  # Original query
                f"{query} table",  # Add "table" keyword
                f"table {' '.join(table_terms)}",  # Table + extracted terms
                ' '.join(table_terms),  # Just the key terms
            ]
            
            # Also add specific search terms based on query content
            query_lower = query.lower()
            if 'cvpd' in query_lower or 'commercial traffic volume' in query_lower:
                search_variations.extend([
                    "commercial traffic volume CVPD survey",
                    "minimum sample size axle load survey", 
                    "percentage commercial traffic surveyed",
                    "3000 6000 CVPD percentage"
                ])
            
            # Search with each variation
            all_keyword_results = []
            for search_term in search_variations:
                try:
                    keyword_results = self.vector_store.search_by_text(
                        query=search_term,
                        n_results=self.max_results,
                        filter_criteria=filter_criteria if filter_criteria else None
                    )
                    all_keyword_results.extend(keyword_results)
                    logger.info(f"Search '{search_term}' returned {len(keyword_results)} results")
                except Exception as e:
                    logger.error(f"Keyword search failed for '{search_term}': {e}")
            
            # Remove duplicates and convert results
            seen_ids = set()
            for result in all_keyword_results:
                result_id = result.get('id', str(result.get('metadata', {}).get('chunk_index', '')))
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    if result.get('distance', 1.0) <= (1.0 - self.relevance_threshold):
                        search_result = self._convert_to_search_result(result)
                        if search_result and not any(r.text == search_result.text for r in results):
                            results.append(search_result)
            
        except Exception as e:
            logger.error(f"Enhanced keyword search failed: {e}")
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        logger.info(f"Table search strategy returned {len(results)} total results")
        return results[:self.max_results]
    
    def _search_regular(self, query: str, filter_criteria: Dict[str, Any]) -> List[SearchResult]:
        """
        Regular search for non-table queries.
        
        Args:
            query: User query
            filter_criteria: Metadata filters
            
        Returns:
            List of search results
        """
        results = []
        
        try:
            logger.info(f"Regular search with filter: {filter_criteria}")
            
            # Use EXACT same pattern as your working search
            search_results = self.vector_store.search_by_text(
                query=query,
                n_results=self.max_results * 2,  # Get more results to filter
                filter_criteria=filter_criteria if filter_criteria else None
            )
            
            logger.info(f"Regular search returned {len(search_results)} results")
            
            # Convert and filter results with debug logging
            for i, result in enumerate(search_results):
                logger.info(f"Result {i}: distance={result.get('distance', 'N/A')}, "
                           f"document_length={len(result.get('document', ''))}, "
                           f"text_length={len(result.get('text', ''))}, "
                           f"source_file={result.get('metadata', {}).get('source_file', 'N/A')}")
                
                if result.get('distance', 1.0) <= (1.0 - self.relevance_threshold):
                    search_result = self._convert_to_search_result(result)
                    if search_result:
                        logger.info(f"Converted result {i}: text_length={len(search_result.text)}, "
                                   f"irc_code={search_result.irc_code}, "
                                   f"source_file={search_result.source_file}")
                        results.append(search_result)
            
        except Exception as e:
            logger.error(f"Error in regular search: {e}")
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:self.max_results]
    
    async def search(self, query: str, irc_code: Optional[str] = None, 
                    include_context: bool = True) -> Tuple[List[SearchResult], bool]:
        """
        Enhanced search with improved table-aware capabilities.
        Now uses HYBRID approach instead of table-only bias.
        
        Args:
            query: Natural language query
            irc_code: Optional IRC code to filter by
            include_context: Whether to include surrounding context
            
        Returns:
            Tuple of (search results, is_table_query)
        """
        is_table_query = self._is_table_query(query)
        logger.info(f"Query classified as table query: {is_table_query}")
        
        # If it's a table query, extract parameters for debugging
        if is_table_query:
            query_params = self._extract_query_parameters(query)
            logger.info(f"Extracted query parameters: {query_params}")
        
        # Build filter criteria - use same pattern as working /search
        filter_criteria = {"irc_code": irc_code} if irc_code else None
        
        results = []
        
        # NEW STRATEGY: Always do both table and regular search for table queries
        # This ensures we find content whether it's in table chunks or regular chunks
        if is_table_query:
            logger.info("Using HYBRID search strategy for table query...")
            
            # Get results from table search (includes enhanced keyword search)
            table_results = self._search_tables(query, filter_criteria)
            logger.info(f"Table search returned {len(table_results)} results")
            
            # ALSO get results from regular search
            regular_results = self._search_regular(query, filter_criteria)
            logger.info(f"Regular search returned {len(regular_results)} results")
            
            # Combine and deduplicate results
            all_results = table_results + regular_results
            seen_texts = set()
            for result in all_results:
                # Use first 100 chars as deduplication key
                text_key = result.text[:100] if result.text else ""
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    results.append(result)
            
            # Sort by relevance
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            results = results[:self.max_results]
            
            logger.info(f"Hybrid search found {len(results)} unique results")
            
        else:
            # Regular search for non-table queries
            results = self._search_regular(query, filter_criteria)
        
        return results, is_table_query
    
    # ADD this method to your EnhancedIRCQueryEngine class:

    def _clean_markdown_table(self, text: str) -> str:
        """
        Convert markdown table format to properly formatted tables.
        """
        lines = text.split('\n')
        cleaned_lines = []
        in_table = False
        table_rows = []
        
        for line in lines:
            stripped_line = line.strip()
            
            # Check if this is a table separator line (skip it)
            if stripped_line.startswith('|---') or (stripped_line.startswith('|') and all(c in '|-' for c in stripped_line.replace(' ', ''))):
                continue
            
            # Check if this is a table line
            if '|' in stripped_line and stripped_line.count('|') >= 2:
                if not in_table:
                    # Start of table
                    in_table = True
                    table_rows = []
                
                # Extract cells
                cells = [cell.strip() for cell in stripped_line.split('|') if cell.strip()]
                if cells:
                    table_rows.append(cells)
            else:
                # Not a table line
                if in_table and table_rows:
                    # End of table - format it
                    formatted_table = self._format_table(table_rows)
                    cleaned_lines.extend(formatted_table)
                    in_table = False
                    table_rows = []
                
                # Add the non-table line
                if stripped_line:  # Only add non-empty lines
                    cleaned_lines.append(line)
        
        # Handle table at end of text
        if in_table and table_rows:
            formatted_table = self._format_table(table_rows)
            cleaned_lines.extend(formatted_table)
        
        return '\n'.join(cleaned_lines)
    
    def _format_table(self, table_rows: List[List[str]]) -> List[str]:
        """
        Format table rows into a clean, readable format optimized for web UI display.
        """
        if not table_rows:
            return []
        
        formatted_lines = []
        
        # Create a clean table format that displays well in web UI
        if table_rows:
            # Header row
            header_row = table_rows[0]
            
            # Format as a structured table with clear headers
            for i, row in enumerate(table_rows):
                if i == 0:
                    # Header row - create a visual separator
                    formatted_lines.append("")  # Empty line before table
                    header_line = " | ".join(f"**{cell}**" for cell in row)
                    formatted_lines.append(header_line)
                    formatted_lines.append("-" * 50)  # Separator line
                else:
                    # Data rows
                    row_line = " | ".join(f"{row[j] if j < len(row) else ''}" for j in range(len(header_row)))
                    formatted_lines.append(row_line)
            
            formatted_lines.append("")  # Empty line after table
        
        return formatted_lines

    # MODIFY your generate_response method - add this processing:

    async def generate_response(self, query: str, search_results: List[SearchResult], 
                            is_table_query: bool = False) -> Tuple[str, List[str]]:
        """Generate a response using search results with dynamic table-aware formatting."""
        if not search_results:
            return "No relevant information found.", []
        
        # Prepare context with special handling for tables
        context_parts = []
        citations = []
        
        for i, result in enumerate(search_results[:5]):
            citation = result.get_citation()
            citations.append(citation)
            
            # CLEAN MARKDOWN TABLES in result text
            cleaned_text = self._clean_markdown_table(result.text)
            
            if result.is_table():
                context_parts.append(f"From {citation} [TABLE]:\n{cleaned_text}\n")
            else:
                context_parts.append(f"From {citation}:\n{cleaned_text}\n")
        
        context = "\n".join(context_parts)
        
        # REST OF YOUR EXISTING generate_response CODE...
        # (Keep all the existing logic for enhanced prompts, etc.)
        
        # Get primary IRC code from most relevant result
        primary_irc_code = search_results[0].irc_code if search_results else "IRC"
        
        # Enhanced prompt for table queries with dynamic parameter extraction
        if is_table_query and any(result.is_table() or 'table' in result.text.lower() for result in search_results):
            query_params = self._extract_query_parameters(query)
            
            logger.info(f"Extracted query parameters: {query_params}")
            
            enhanced_instructions = [
                "You are analyzing IRC table data. Please extract the specific value requested.",
                "IMPORTANT: Always clearly state the source document name (e.g., 'according to IRC document mort.250.2013' or 'as per IRC-37-2019').",
                "Format tables properly with clear headers and aligned columns, not as markdown with | symbols.",
                "Replace any markdown table formatting with proper table formatting."
            ]
            
            if query_params.get('traffic_volume'):
                enhanced_instructions.append(f"- Look for traffic volume: '{query_params['traffic_volume']}'")
            
            if query_params.get('terrain'):
                enhanced_instructions.append(f"- Find the value for terrain type: '{query_params['terrain']}'")
            
            if query_params.get('parameter_type'):
                enhanced_instructions.append(f"- Extract the {query_params['parameter_type']} value")
            
            enhanced_instructions.extend([
                "- Provide the exact numerical value from the table",
                "- Cite the specific IRC document and table number WITH the source file name",
                "- Be precise and only return the value that matches ALL the specified criteria",
                "- Format any tables in the response as proper tables, not markdown"
            ])
            
            enhanced_context = f"""
    Query: {query}

    Extracted Parameters:
    {chr(10).join([f"- {key}: {value}" for key, value in query_params.items()])}

    Table Data from IRC Documents:
    {context}

    Instructions:
    {chr(10).join(enhanced_instructions)}

    IMPORTANT: Make sure you match the EXACT parameters requested in the query. Do not return values for different parameters.
    """
            
            try:
                response = await self.chat_engine.query_irc_technical(
                    query=f"Extract the specific value from the IRC table data based on the exact parameters provided. Ignore markdown table formatting. Query: {query}",
                    irc_code=primary_irc_code,
                    context=enhanced_context
                )
                if response is None:
                    return "Failed to generate response.", citations
                return response, citations
            except Exception as e:
                logger.error(f"Error generating enhanced response: {e}")
        
        # Regular response generation
        try:
            # Enhanced prompt to ensure source attribution and proper table formatting
            enhanced_prompt = f"""
Please provide a comprehensive answer to the following question about IRC specifications:

Query: {query}

Context from IRC Documents:
{context}

IMPORTANT INSTRUCTIONS:
1. Always clearly state the source document name (e.g., "according to IRC document mort.250.2013" or "as per {primary_irc_code}").
2. If you include any tables in your response, format them properly with clear headers and aligned columns.
3. Replace any markdown table formatting (with | symbols) with proper, readable table formatting.
4. Provide specific citations including document names, table numbers, and clause numbers where applicable.
5. Be accurate and comprehensive in your technical response.

Citations available: {', '.join(citations)}
"""
            
            response = await self.chat_engine.query_irc_technical(
                query=enhanced_prompt,
                irc_code=primary_irc_code,
                context=context
            )
            if response is None:
                return "Failed to generate response.", citations
            return response, citations
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}", citations
    
    async def query(self, query: str, irc_code: Optional[str] = None, 
                   limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete query answering pipeline with table awareness.
        Maintains backward compatibility with existing API endpoints.
        
        Args:
            query: User query
            irc_code: Optional IRC code filter
            limit: Optional limit (for compatibility, uses max_results instead)
            
        Returns:
            Dictionary with response, citations, and metadata
        """
        try:
            logger.info(f"🚀 Processing enhanced query: '{query}'")
            
            # Search for relevant documents
            search_results, is_table_query = await self.search(query, irc_code)
            
            if not search_results:
                logger.warning("❌ No search results found")
                return {
                    "response": "I couldn't find relevant information in the IRC documents for your query. Please try rephrasing your question or check if you're asking about a specific IRC code.",
                    "citations": [],
                    "sources_found": 0,
                    "query_type": "table" if is_table_query else "text",
                    "confidence": 0.0,
                    "relevant_chunks": []  # For backward compatibility
                }
            
            logger.info(f"📊 Found {len(search_results)} search results (table query: {is_table_query})")
            
            # Debug: Log search result details
            for i, result in enumerate(search_results):
                logger.info(f"Search result {i}: text_length={len(result.text)}, "
                           f"irc_code={result.irc_code}, source_file={result.source_file}, "
                           f"relevance={result.relevance_score:.3f}")
            
            # Generate response
            response, citations = await self.generate_response(query, search_results, is_table_query)
            
            # Calculate confidence based on relevance scores
            avg_relevance = sum(r.relevance_score for r in search_results) / len(search_results)
            
            # Create backward-compatible response format
            return {
                "response": response,
                "citations": citations,
                "sources_found": len(search_results),
                "query_type": "table" if is_table_query else "text",
                "confidence": avg_relevance,
                "relevant_chunks": [  # Backward compatibility
                    {
                        "text": result.text,
                        "citation": result.get_citation(),
                        "relevance_score": result.relevance_score
                    }
                    for result in search_results[:limit or self.max_results]
                ],
                # Enhanced fields
                "table_results": [r for r in search_results if r.is_table()],
                "text_results": [r for r in search_results if not r.is_table()]
            }
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced query processing: {e}")
            return {
                "response": f"An error occurred while processing your query: {str(e)}",
                "citations": [],
                "sources_found": 0,
                "query_type": "error", 
                "confidence": 0.0,
                "relevant_chunks": [],
                "error": str(e)
            }

    async def answer_query(self, query: str, irc_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Alternative method name for enhanced query processing.
        Simply calls the main query method.
        """
        return await self.query(query, irc_code)

# For backward compatibility
IRCQueryEngine = EnhancedIRCQueryEngine