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
from database.vector_store import ChromaVectorStore
from api.gemini_chat import GeminiChatEngine
from database.query_engine import IRCQueryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemTester:
    """Comprehensive system testing for the IRC RAG system."""

    # Test queries covering different aspects of IRC documents
    TEST_QUERIES = [
        "What is congestion factor for 40m spans?",
        "Explain the design criteria for IRC bridges"
        # "What are the requirements for bridge deck waterproofing?",
        # "How to calculate live load for highway bridges?",
        # "What are the specifications for concrete mix design?",
        # "Explain the quality control measures for bridge construction",
        # "What are the safety requirements for bridge maintenance?",
        # "How to determine bridge load capacity?",
        # "What are the environmental considerations in bridge design?",
        # "Explain the inspection procedures for bridges"
    ]

    def __init__(self):
        """Initialize the system tester with all components."""
        self.settings = Settings()
        self.vector_store = ChromaVectorStore()
        self.chat_engine = GeminiChatEngine()
        self.query_engine = IRCQueryEngine(
            vector_store=self.vector_store,
            chat_engine=self.chat_engine
        )

        # Setup paths
        self.vector_db_dir = self.settings.vector_db_dir
        self.test_results_dir = Path("test_results")
        self.test_results_dir.mkdir(exist_ok=True)

    async def test_vector_database_connection(self) -> Dict[str, Any]:
        """Test vector database connection and basic functionality."""
        result = {
            "test_name": "Vector Database Connection",
            "status": "failed",
            "details": [],
            "errors": []
        }

        try:
            logger.info("Testing vector database connection...")

            # Initialize vector store
            await self.vector_store.initialize()
            result["details"].append("Vector store initialized successfully")

            # Check if database has data
            collection = self.vector_store.get_collection()
            if collection:
                count = collection.count()
                result["details"].append(f"Database contains {count} documents")

                if count > 0:
                    result["status"] = "passed"
                else:
                    result["errors"].append("Database is empty - no documents found")
            else:
                result["errors"].append("Failed to get collection from vector store")

        except Exception as e:
            result["errors"].append(f"Vector database connection failed: {e}")
            logger.error(f"Vector database test failed: {e}")

        return result

    async def test_vector_search_functionality(self) -> Dict[str, Any]:
        """Test vector search functionality with sample queries."""
        result = {
            "test_name": "Vector Search Functionality",
            "status": "failed",
            "details": [],
            "errors": [],
            "search_results": []
        }

        try:
            logger.info("Testing vector search functionality...")

            # Test basic search - now sync method
            test_query = "bridge design"
            search_results = self.vector_store.search_by_text(
                query=test_query,
                n_results=5
            )

            if search_results:
                result["details"].append(f"Search returned {len(search_results)} results")
                result["search_results"] = search_results[:2]  # Store first 2 results for inspection

                # Check result structure
                first_result = search_results[0]
                required_fields = ['text', 'metadata', 'distance']
                missing_fields = [field for field in required_fields if field not in first_result]

                if not missing_fields:
                    result["status"] = "passed"
                    result["details"].append("Search results have correct structure")
                else:
                    result["errors"].append(f"Missing fields in search results: {missing_fields}")
            else:
                result["errors"].append("Search returned no results")

        except Exception as e:
            result["errors"].append(f"Vector search test failed: {e}")
            logger.error(f"Vector search test failed: {e}")

        return result

    async def test_gemini_api_connection(self) -> Dict[str, Any]:
        """Test Gemini API connection and basic functionality."""
        result = {
            "test_name": "Gemini API Connection",
            "status": "failed",
            "details": [],
            "errors": []
        }

        try:
            logger.info("Testing Gemini API connection...")

            # Test basic query
            test_query = "Hello, this is a test message."
            response = await self.chat_engine.query_irc_technical(
                query=test_query,
                irc_code="TEST",
                context="This is a test context for API validation."
            )

            if response:
                result["status"] = "passed"
                result["details"].append("Gemini API responded successfully")
                result["details"].append(f"Response length: {len(response)} characters")
            else:
                result["errors"].append("Gemini API returned no response")

        except Exception as e:
            result["errors"].append(f"Gemini API test failed: {e}")
            logger.error(f"Gemini API test failed: {e}")

        return result

    async def test_end_to_end_query(self, query: str) -> Dict[str, Any]:
        """Test end-to-end query processing."""
        result = {
            "test_name": f"End-to-End Query: {query[:50]}...",
            "query": query,
            "status": "failed",
            "details": [],
            "errors": [],
            "search_results_count": 0,
            "response_length": 0,
            "processing_time": 0
        }

        start_time = datetime.now()

        try:
            logger.info(f"Testing end-to-end query: {query}")

            # Process query through the complete pipeline
            query_result = await self.query_engine.query(query=query, limit=5)

            if query_result and "error" not in query_result:
                result["status"] = "passed"
                result["search_results_count"] = len(query_result.get("relevant_chunks", []))
                result["response_length"] = len(query_result.get("response", ""))
                result["details"].append(f"Query processed successfully")
                result["details"].append(f"Found {result['search_results_count']} relevant chunks")
                result["details"].append(f"Generated {result['response_length']} character response")
                result["details"].append(f"Citations: {len(query_result.get('citations', []))}")
            else:
                result["errors"].append(f"Query processing failed: {query_result.get('error', 'Unknown error')}")

        except Exception as e:
            result["errors"].append(f"End-to-end query test failed: {e}")
            logger.error(f"End-to-end query test failed for '{query}': {e}")

        result["processing_time"] = (datetime.now() - start_time).total_seconds()
        return result

    async def test_irc_specific_queries(self) -> Dict[str, Any]:
        """Test IRC-specific queries to validate domain knowledge."""
        result = {
            "test_name": "IRC-Specific Query Testing",
            "status": "failed",
            "details": [],
            "errors": [],
            "query_results": []
        }

        try:
            logger.info("Testing IRC-specific queries...")

            passed_queries = 0
            total_queries = len(self.TEST_QUERIES)

            for query in self.TEST_QUERIES:
                query_result = await self.test_end_to_end_query(query)
                result["query_results"].append(query_result)

                if query_result["status"] == "passed":
                    passed_queries += 1

            success_rate = (passed_queries / total_queries) * 100
            result["details"].append(f"Success rate: {success_rate:.1f}% ({passed_queries}/{total_queries})")

            if success_rate >= 70:  # At least 70% success rate
                result["status"] = "passed"
            else:
                result["errors"].append(f"Success rate too low: {success_rate:.1f}%")

        except Exception as e:
            result["errors"].append(f"IRC-specific query testing failed: {e}")
            logger.error(f"IRC-specific query testing failed: {e}")

        return result

    async def validate_system_setup(self) -> Dict[str, Any]:
        """Validate overall system setup and configuration."""
        result = {
            "test_name": "System Setup Validation",
            "status": "failed",
            "details": [],
            "errors": []
        }

        try:
            logger.info("Validating system setup...")

            # Check required directories
            required_dirs = [
                self.settings.vector_db_dir,
                self.settings.raw_pdfs_dir,
                self.settings.parsed_docs_dir,
                self.settings.chunks_dir,
                self.settings.metadata_dir
            ]

            for dir_path in required_dirs:
                if Path(dir_path).exists():
                    result["details"].append(f"Directory exists: {dir_path}")
                else:
                    result["errors"].append(f"Missing directory: {dir_path}")

            # Check API keys
            if self.settings.LLAMA_PARSE_API_KEY and self.settings.LLAMA_PARSE_API_KEY != "your_llama_parse_api_key":
                result["details"].append("LlamaParse API key configured")
            else:
                result["errors"].append("LlamaParse API key not configured")

            if self.settings.GOOGLE_API_KEY and self.settings.GOOGLE_API_KEY != "your_google_api_key":
                result["details"].append("Google API key configured")
            else:
                result["errors"].append("Google API key not configured")

            # Check if vector database has data
            try:
                collection = self.vector_store.get_collection()
                if collection and collection.count() > 0:
                    result["details"].append("Vector database contains data")
                else:
                    result["errors"].append("Vector database is empty")
            except:
                result["errors"].append("Cannot access vector database")

            if not result["errors"]:
                result["status"] = "passed"

        except Exception as e:
            result["errors"].append(f"System setup validation failed: {e}")
            logger.error(f"System setup validation failed: {e}")

        return result

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests and generate comprehensive report."""
        logger.info("Starting comprehensive system testing...")

        all_tests = [
            await self.validate_system_setup(),
            await self.test_vector_database_connection(),
            await self.test_vector_search_functionality(),
            await self.test_gemini_api_connection(),
            await self.test_irc_specific_queries()
        ]

        # Calculate overall results
        passed_tests = sum(1 for test in all_tests if test["status"] == "passed")
        total_tests = len(all_tests)
        overall_success_rate = (passed_tests / total_tests) * 100

        summary = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "passed" if overall_success_rate >= 80 else "failed",
            "success_rate": overall_success_rate,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "test_results": all_tests
        }

        # Save detailed results
        results_file = self.test_results_dir / f"system_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Test results saved to: {results_file}")
        return summary

    def print_test_results(self, summary: Dict[str, Any]):
        """Print formatted test results."""
        print("\n" + "="*80)
        print("IRC RAG SYSTEM TEST RESULTS")
        print("="*80)
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"Timestamp: {summary['timestamp']}")

        print("\n" + "-"*80)
        print("DETAILED TEST RESULTS")
        print("-"*80)

        for test in summary['test_results']:
            status_icon = "✅" if test['status'] == 'passed' else "❌"
            print(f"\n{status_icon} {test['test_name']}")
            print(f"   Status: {test['status']}")

            if test['details']:
                print("   Details:")
                for detail in test['details']:
                    print(f"     • {detail}")

            if test['errors']:
                print("   Errors:")
                for error in test['errors']:
                    print(f"     • {error}")

        print("\n" + "="*80)

        if summary['overall_status'] == 'passed':
            print("🎉 SYSTEM TEST PASSED - RAG system is ready for use!")
        else:
            print("⚠️  SYSTEM TEST FAILED - Please check the errors above and fix issues.")

        print("="*80 + "\n")

async def main():
    """Main function to run the comprehensive system test."""
    try:
        tester = SystemTester()

        # Run all tests
        summary = await tester.run_all_tests()

        # Print results
        tester.print_test_results(summary)

        # Exit with appropriate code
        if summary['overall_status'] == 'passed':
            print("System test completed successfully!")
            return 0
        else:
            print("System test failed. Please review the errors above.")
            return 1

    except Exception as e:
        logger.error(f"Fatal error in system testing: {e}")
        print(f"\nFatal error: {e}")
        print("Check the log file for more details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)