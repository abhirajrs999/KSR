#!/usr/bin/env python3
"""
Test script to validate hierarchical chunking implementation.
Run this script to test the new chunking system before processing all documents.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config.settings import Settings
from processing.chunker import HierarchicalIRCChunker
from processing.metadata_extractor import EnhancedIRCMetadataExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalChunkingTester:
    """Test the hierarchical chunking system with sample documents."""
    
    def __init__(self):
        self.settings = Settings()
        self.chunker = HierarchicalIRCChunker()
        self.metadata_extractor = EnhancedIRCMetadataExtractor()
        
        # Setup paths
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
    def test_sample_document(self, sample_text: str, doc_name: str) -> Dict[str, Any]:
        """
        Test hierarchical chunking on a sample IRC document text.
        
        Args:
            sample_text: Sample IRC document text
            doc_name: Name for the test document
            
        Returns:
            Test results dictionary
        """
        logger.info(f"Testing hierarchical chunking on {doc_name}")
        
        # Create mock parsed document
        parsed_doc = {
            "full_text": sample_text,
            "metadata": {
                "source_file": f"{doc_name}.pdf",
                "irc_code": "IRC:TEST-2024"
            },
            "page_mapping": [
                {"page_number": 1, "text": sample_text[:len(sample_text)//2]},
                {"page_number": 2, "text": sample_text[len(sample_text)//2:]}
            ]
        }
        
        # Extract metadata
        metadata = self.metadata_extractor.extract_metadata(parsed_doc)
        if not metadata:
            return {"status": "error", "message": "Failed to extract metadata"}
        
        # Create chunks
        chunks = self.chunker.create_chunks(parsed_doc, metadata)
        
        # Analyze results
        results = {
            "status": "success",
            "document_name": doc_name,
            "total_chunks": len(chunks),
            "chunk_types": {},
            "hierarchy_levels": {},
            "chunks_summary": []
        }
        
        # Analyze chunk distribution
        for chunk in chunks:
            chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
            hierarchy_path = chunk['metadata'].get('hierarchy_path', 'unknown')
            
            # Count chunk types
            results["chunk_types"][chunk_type] = results["chunk_types"].get(chunk_type, 0) + 1
            
            # Count hierarchy levels
            level_depth = len(hierarchy_path.split(' > ')) if hierarchy_path != 'unknown' else 0
            results["hierarchy_levels"][level_depth] = results["hierarchy_levels"].get(level_depth, 0) + 1
            
            # Create chunk summary
            chunk_summary = {
                "chunk_id": chunk['metadata'].get('chunk_id'),
                "chunk_type": chunk_type,
                "hierarchy_path": hierarchy_path,
                "element_number": chunk['metadata'].get('element_number'),
                "element_title": chunk['metadata'].get('element_title', '')[:50] + "...",
                "content_length": len(chunk['content']),
                "pages": chunk['metadata'].get('pages', [])
            }
            results["chunks_summary"].append(chunk_summary)
        
        # Save detailed results
        output_file = self.test_dir / f"{doc_name}_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": metadata,
                "chunks": chunks,
                "test_results": results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test results saved to {output_file}")
        return results
    
    def create_sample_irc_document(self) -> str:
        """Create a sample IRC document text for testing."""
        return """
IRC:57-2018

GUIDELINES FOR DESIGN OF JOINTS IN CONCRETE PAVEMENTS

INDIAN ROADS CONGRESS

1. Introduction

This code provides guidelines for the design and construction of joints in concrete pavements for roads and airfields. The proper design of joints is essential for the performance and durability of concrete pavements.

2. Scope

This code covers the design principles, construction methods, and maintenance requirements for various types of joints in concrete pavements. It applies to both new construction and rehabilitation projects.

3. General

Joints in concrete pavements serve multiple purposes including controlling cracking, accommodating thermal movements, and facilitating construction. The design must consider traffic loads, environmental conditions, and pavement geometry.

4. Shape of the Joint Sealing Groove

The shape of the joint sealing groove shall be designed to accommodate the expected movement and provide adequate seal performance. Standard groove shapes include rectangular, modified rectangular, and hourglass configurations.

5. Type of Joints

Concrete pavement joints can be classified into several categories based on their function and construction method.

5.1: Contraction Joints

Contraction joints are provided to control random cracking of concrete due to drying shrinkage and thermal contraction. These joints should be spaced at regular intervals and extend through the full depth of the slab.

5.2: Longitudinal Joints

Longitudinal joints are constructed parallel to the centerline of the pavement. They may be constructed joints formed during paving operations or contraction joints sawed after concrete placement.

5.3: Expansion Joints

Expansion joints are designed to accommodate thermal expansion of concrete slabs. They are typically provided at structures, intersections, and at intervals along the pavement length as determined by design requirements.

5.4: Construction Joints

Construction joints are necessary when concrete placement is interrupted. These joints must be designed to transfer loads effectively while maintaining pavement integrity.

6. Sealing Details

Proper sealing of joints is critical for pavement performance and longevity.

6.1: Steps involved

The sealing process involves several key steps including joint preparation, primer application, sealant installation, and quality control inspection.

6.2: Sawing of Groove

Joint grooves shall be sawed to the specified dimensions using appropriate equipment. The timing of sawing is critical to prevent random cracking while ensuring clean joint formation.

6.3: Cleaning of Groove

All joints must be thoroughly cleaned of debris, loose concrete, and contaminants before sealant application.

6.4: Application of Primer

Primer shall be applied to joint faces when specified to improve sealant adhesion.

6.5: Installation of Backer Rod

Backer rod shall be installed at the proper depth to control sealant shape and prevent three-sided adhesion.

7. Preformed Seals

Preformed seals may be used as an alternative to liquid sealants in certain applications. These seals must be designed for the expected joint movement and traffic conditions.

8. Resealing Old Joints

Existing joints may require resealing as part of pavement maintenance. The process involves removal of old sealant, joint preparation, and installation of new sealing materials.

Annexure-I: Sample calculation

This annexure provides sample calculations for joint spacing and seal design based on typical pavement conditions and materials.

Table 1: Recommended Joint Spacing

Figure 1: Joint Details

"""
    
    def run_comprehensive_test(self):
        """Run comprehensive tests on the hierarchical chunking system."""
        print("\n" + "="*60)
        print("HIERARCHICAL CHUNKING SYSTEM TEST")
        print("="*60)
        
        # Test 1: Sample IRC document
        sample_text = self.create_sample_irc_document()
        results1 = self.test_sample_document(sample_text, "IRC57_sample")
        
        # Print results
        self.print_test_results(results1)
        
        # Test 2: Check if any real documents exist
        if self.settings.parsed_docs_dir.exists():
            parsed_files = list(self.settings.parsed_docs_dir.glob("*.json"))
            parsed_files = [f for f in parsed_files if 'processing_summary' not in f.name]
            
            if parsed_files:
                print(f"\nTesting on real document: {parsed_files[0].name}")
                try:
                    with open(parsed_files[0], 'r', encoding='utf-8') as f:
                        real_doc = json.load(f)
                    
                    metadata = self.metadata_extractor.extract_metadata(real_doc)
                    chunks = self.chunker.create_chunks(real_doc, metadata or {})
                    
                    print(f"✅ Real document test successful!")
                    print(f"   - Created {len(chunks)} chunks")
                    print(f"   - Document: {real_doc.get('metadata', {}).get('source_file', 'unknown')}")
                    
                    # Show chunk type distribution
                    chunk_types = {}
                    for chunk in chunks:
                        chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
                        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    
                    print(f"   - Chunk types: {chunk_types}")
                    
                except Exception as e:
                    print(f"❌ Real document test failed: {e}")
            else:
                print("\nNo real documents found for testing")
        
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print(f"Test outputs saved in: {self.test_dir.absolute()}")
        
    def print_test_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        if results["status"] != "success":
            print(f"❌ Test failed: {results.get('message', 'Unknown error')}")
            return
        
        print(f"\n✅ Test successful for {results['document_name']}")
        print(f"   Total chunks created: {results['total_chunks']}")
        print(f"   Chunk type distribution:")
        for chunk_type, count in results['chunk_types'].items():
            print(f"     - {chunk_type}: {count}")
        
        print(f"   Hierarchy level distribution:")
        for level, count in results['hierarchy_levels'].items():
            print(f"     - Level {level}: {count}")
        
        print(f"\n   Sample chunks:")
        for i, chunk in enumerate(results['chunks_summary'][:5]):  # Show first 5
            print(f"     {i+1}. {chunk['chunk_type']} - {chunk['element_number']} - {chunk['element_title']}")
            print(f"        Path: {chunk['hierarchy_path']}")
            print(f"        Length: {chunk['content_length']} chars, Pages: {chunk['pages']}")

def main():
    """Main function to run the hierarchical chunking tests."""
    try:
        tester = HierarchicalChunkingTester()
        tester.run_comprehensive_test()
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()