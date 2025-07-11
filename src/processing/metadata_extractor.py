import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedIRCMetadataExtractor:
    """
    Enhanced metadata extractor that identifies hierarchical structure in IRC documents.
    
    This extractor goes beyond basic pattern matching to understand the actual
    document structure, including:
    - Main chapters and their hierarchy
    - Sub-clauses and nested sections
    - Cross-references between sections
    - Table and figure placements
    - Special sections (Annexures, Appendices)
    """
    
    def __init__(self):
        """Initialize the enhanced metadata extractor with comprehensive patterns."""
        self.patterns = self._compile_comprehensive_patterns()
        
    def _compile_comprehensive_patterns(self) -> Dict[str, re.Pattern]:
        """Compile comprehensive regex patterns for IRC document analysis."""
        return {
            # IRC code patterns - more flexible
            'irc_code': re.compile(
                r'\bIRC(?:[:\s-]?\d{1,3}){1,2}(?:-\d{4})?\b', 
                re.IGNORECASE
            ),
            
            # Main chapters - flexible formatting
            'main_chapter': re.compile(
                r'^(\d+)\.\s*([^\n]+?)(?=\n|$)', 
                re.MULTILINE
            ),
            
            # Sub-clauses with colons
            'sub_clause': re.compile(
                r'^(\d+\.\d+):\s*([^\n]+?)(?=\n|$)', 
                re.MULTILINE
            ),
            
            # Sub-sub-clauses
            'sub_sub_clause': re.compile(
                r'^(\d+\.\d+\.\d+):\s*([^\n]+?)(?=\n|$)', 
                re.MULTILINE
            ),
            
            # Alternative numbering without colons
            'numbered_section': re.compile(
                r'^(\d+(?:\.\d+)*)\s+([^\n]+?)(?=\n|$)', 
                re.MULTILINE
            ),
            
            # Special sections
            'annexure': re.compile(
                r'^(Annexure-[IVX]+|ANNEXURE-[IVX]+):\s*([^\n]+?)(?=\n|$)', 
                re.MULTILINE | re.IGNORECASE
            ),
            
            'appendix': re.compile(
                r'^(Appendix-[A-Z]+|APPENDIX-[A-Z]+):\s*([^\n]+?)(?=\n|$)', 
                re.MULTILINE | re.IGNORECASE
            ),
            
            # Table references - more comprehensive
            'table_ref': re.compile(
                r'\b(?:Table|TABLE)\s*[\d.]+(?:\s*[-–—]\s*[^\n]*)?', 
                re.IGNORECASE
            ),
            
            # Figure references
            'figure_ref': re.compile(
                r'\b(?:Figure|FIGURE|Fig\.?)\s*[\d.]+(?:\s*[-–—]\s*[^\n]*)?', 
                re.IGNORECASE
            ),
            
            # Cross-references
            'section_ref': re.compile(
                r'\b(?:Section|section|clause|Clause)\s+(\d+(?:\.\d+)*)', 
                re.IGNORECASE
            ),
            
            # Revision and year patterns
            'revision': re.compile(
                r'(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)\s+REVISION', 
                re.IGNORECASE
            ),
            
            'year': re.compile(r'\b(19|20)\d{2}\b'),
            
            # Title patterns - look for emphasized or capitalized titles
            'potential_title': re.compile(
                r'^[A-Z][A-Z\s]{10,}$', 
                re.MULTILINE
            ),
            
            # Page number patterns
            'page_number': re.compile(r'\bPage\s*(\d+)\b', re.IGNORECASE)
        }
    
    def _extract_references(self, text: str, pattern: re.Pattern) -> List[str]:
        """Extract unique, sorted references using a regex pattern."""
        matches = pattern.findall(text)
        if isinstance(matches[0], tuple) if matches else False:
            # Handle tuple results from groups
            matches = [match[0] if isinstance(match, tuple) else match for match in matches]
        return sorted(list(set(matches)))
    
    def _extract_hierarchical_structure(self, text: str) -> Dict[str, Any]:
        """
        Extract the complete hierarchical structure of the document.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary containing hierarchical structure information
        """
        structure = {
            'main_chapters': [],
            'sub_clauses': [],
            'sub_sub_clauses': [],
            'special_sections': [],
            'hierarchy_map': {},
            'cross_references': []
        }
        
        # Extract main chapters
        for match in self.patterns['main_chapter'].finditer(text):
            chapter = {
                'number': match.group(1),
                'title': match.group(2).strip(),
                'position': match.start(),
                'level': 1
            }
            structure['main_chapters'].append(chapter)
            structure['hierarchy_map'][chapter['number']] = chapter
        
        # Extract sub-clauses
        for match in self.patterns['sub_clause'].finditer(text):
            clause = {
                'number': match.group(1),
                'title': match.group(2).strip(),
                'position': match.start(),
                'level': 2,
                'parent': match.group(1).split('.')[0]
            }
            structure['sub_clauses'].append(clause)
            structure['hierarchy_map'][clause['number']] = clause
        
        # Extract sub-sub-clauses
        for match in self.patterns['sub_sub_clause'].finditer(text):
            sub_clause = {
                'number': match.group(1),
                'title': match.group(2).strip(),
                'position': match.start(),
                'level': 3,
                'parent': '.'.join(match.group(1).split('.')[:2])
            }
            structure['sub_sub_clauses'].append(sub_clause)
            structure['hierarchy_map'][sub_clause['number']] = sub_clause
        
        # Extract special sections
        for pattern_name in ['annexure', 'appendix']:
            for match in self.patterns[pattern_name].finditer(text):
                special = {
                    'number': match.group(1),
                    'title': match.group(2).strip(),
                    'position': match.start(),
                    'type': pattern_name,
                    'level': 0  # Special level
                }
                structure['special_sections'].append(special)
                structure['hierarchy_map'][special['number']] = special
        
        # Extract cross-references
        for match in self.patterns['section_ref'].finditer(text):
            ref = {
                'referenced_section': match.group(1),
                'position': match.start(),
                'context': text[max(0, match.start()-50):match.end()+50]
            }
            structure['cross_references'].append(ref)
        
        return structure
    
    def _extract_enhanced_title(self, pages_data: List[Dict[str, Any]], full_text: str) -> Optional[str]:
        """
        Enhanced title extraction using multiple heuristics.
        
        Args:
            pages_data: List of page data
            full_text: Complete document text
            
        Returns:
            Extracted title or None
        """
        # Strategy 1: Look for emphasized titles in first few pages
        for page in pages_data[:3]:
            page_text = page.get('text', '')
            
            # Look for potential titles
            potential_titles = self.patterns['potential_title'].findall(page_text)
            for title in potential_titles:
                title = title.strip()
                if (len(title) > 15 and len(title) < 100 and 
                    "INDIAN ROADS CONGRESS" not in title.upper() and
                    "IRC" not in title and
                    not title.isdigit()):
                    return title
        
        # Strategy 2: Look for first significant line after IRC code
        lines = full_text.split('\n')
        irc_found = False
        
        for line in lines:
            line = line.strip()
            
            # Skip until we find IRC code
            if not irc_found and self.patterns['irc_code'].search(line):
                irc_found = True
                continue
            
            # After IRC code, look for title
            if irc_found and len(line) > 15 and len(line) < 100:
                # Skip common headers
                skip_patterns = [
                    "indian roads congress", "guidelines", "specification",
                    "page", "revision", "contents", "index"
                ]
                if not any(skip in line.lower() for skip in skip_patterns):
                    return line
        
        # Strategy 3: Fallback to first chapter title
        structure = self._extract_hierarchical_structure(full_text)
        if structure['main_chapters']:
            # Look for a descriptive chapter title
            for chapter in structure['main_chapters']:
                if chapter['number'] in ['1', '2'] and len(chapter['title']) > 10:
                    return chapter['title']
        
        return None
    
    def _map_elements_to_pages(self, structure: Dict[str, Any], 
                              pages_data: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Create a comprehensive map of structural elements to pages.
        
        Args:
            structure: Hierarchical structure data
            pages_data: List of page data
            
        Returns:
            Dictionary mapping element numbers to page lists
        """
        element_page_map = {}
        
        # Combine all structural elements
        all_elements = (structure['main_chapters'] + 
                       structure['sub_clauses'] + 
                       structure['sub_sub_clauses'] + 
                       structure['special_sections'])
        
        for element in all_elements:
            element_number = element['number']
            element_page_map[element_number] = []
            
            for page in pages_data:
                page_num = page.get('page_number')
                page_text = page.get('text', '')
                
                if not page_num:
                    continue
                
                # Check if element appears in this page
                if element_number in page_text:
                    # Verify it's not just a cross-reference
                    # Look for the element in context (with title)
                    element_pattern = re.compile(
                        rf'\b{re.escape(element_number)}\b.*?{re.escape(element["title"][:20])}',
                        re.IGNORECASE | re.DOTALL
                    )
                    
                    if element_pattern.search(page_text):
                        element_page_map[element_number].append(page_num)
        
        return element_page_map
    
    def _extract_document_statistics(self, text: str, pages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract comprehensive document statistics.
        
        Args:
            text: Full document text
            pages_data: List of page data
            
        Returns:
            Dictionary with document statistics
        """
        return {
            'total_pages': len(pages_data),
            'total_characters': len(text),
            'total_words': len(text.split()),
            'total_lines': len(text.split('\n')),
            'table_count': len(self.patterns['table_ref'].findall(text)),
            'figure_count': len(self.patterns['figure_ref'].findall(text)),
            'cross_reference_count': len(self.patterns['section_ref'].findall(text))
        }
    
    def extract_metadata(self, data_input: Union[Path, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Extract comprehensive metadata from parsed document data or JSON file.
        
        Args:
            data_input: Either a Path to JSON file or parsed document dictionary
            
        Returns:
            Comprehensive metadata dictionary, or None if an error occurs
        """
        try:
            # Handle both file paths and direct data input
            if isinstance(data_input, Path):
                if not data_input.is_file():
                    logger.error(f"JSON file not found: {data_input}")
                    return None
                
                with open(data_input, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                source_file = data_input.name
                
            elif isinstance(data_input, dict):
                data = data_input
                source_file = data.get('metadata', {}).get('source_file', 'unknown')
                
            else:
                logger.error(f"Invalid input type: {type(data_input)}. Expected Path or dict.")
                return None
            
            # Extract basic data
            full_text = data.get("full_text", "")
            pages_data = data.get("page_mapping", [])
            
            if not full_text:
                logger.warning(f"No text content found in {source_file}")
                return None
            
            # Extract basic patterns
            irc_codes = self._extract_references(full_text, self.patterns['irc_code'])
            tables = self._extract_references(full_text, self.patterns['table_ref'])
            figures = self._extract_references(full_text, self.patterns['figure_ref'])
            revisions = self._extract_references(full_text, self.patterns['revision'])
            years = self._extract_references(full_text, self.patterns['year'])
            
            # Extract hierarchical structure
            structure = self._extract_hierarchical_structure(full_text)
            
            # Extract enhanced title
            title = self._extract_enhanced_title(pages_data, full_text)
            
            # Map elements to pages
            element_page_map = self._map_elements_to_pages(structure, pages_data)
            
            # Extract document statistics
            statistics = self._extract_document_statistics(full_text, pages_data)
            
            # Determine primary IRC code and revision year
            primary_irc_code = irc_codes[0] if irc_codes else None
            revision_year = None
            
            # Try to extract year from IRC code first
            if primary_irc_code:
                year_match = re.search(r'(\d{4})$', primary_irc_code)
                if year_match:
                    revision_year = year_match.group(1)
            
            # If no year in IRC code, use latest year found in document
            if not revision_year and years:
                revision_year = max(years)
            
            # Compile comprehensive metadata
            enhanced_metadata = {
                # Basic identification
                "source_file": source_file,
                "title": title,
                "irc_code": primary_irc_code,
                "all_irc_codes": irc_codes,
                "revision_year": revision_year,
                "revision_type": revisions[0] if revisions else None,
                
                # Hierarchical structure
                "structure": structure,
                "main_chapters": [ch['number'] + ': ' + ch['title'] for ch in structure['main_chapters']],
                "sub_clauses": [sc['number'] + ': ' + sc['title'] for sc in structure['sub_clauses']],
                "special_sections": [ss['number'] + ': ' + ss['title'] for ss in structure['special_sections']],
                
                # References and cross-links
                "tables": tables,
                "figures": figures,
                "element_page_map": element_page_map,
                "cross_references": structure['cross_references'],
                
                # Document statistics
                "statistics": statistics,
                
                # Legacy compatibility
                "clauses": [ch['number'] for ch in structure['main_chapters']] + 
                          [sc['number'] for sc in structure['sub_clauses']] +
                          [ssc['number'] for ssc in structure['sub_sub_clauses']],
                "clause_page_map": element_page_map,
                
                # Processing metadata
                "extraction_version": "2.0_hierarchical",
                "extracted_at": None  # Will be set by calling code if needed
            }
            
            logger.info(f"Successfully extracted enhanced metadata from {source_file}")
            logger.info(f"Found {len(structure['main_chapters'])} chapters, "
                       f"{len(structure['sub_clauses'])} sub-clauses, "
                       f"{len(structure['special_sections'])} special sections")
            
            return enhanced_metadata
            
        except Exception as e:
            logger.error(f"Error extracting enhanced metadata: {e}", exc_info=True)
            return None

# For backward compatibility
IRCMetadataExtractor = EnhancedIRCMetadataExtractor