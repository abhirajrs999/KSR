#!/usr/bin/env python3
"""
Test script for IRC RAG API deployed on Railway.
This script validates the API endpoints and functionality after deployment.
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, Any, List
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class APITester:
    """Test suite for IRC RAG API."""
    
    def __init__(self, api_url: str, timeout: int = 30):
        """
        Initialize API tester.
        
        Args:
            api_url: Base URL of the deployed API
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        
        # Test results
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
    
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
    
    def record_test_result(self, test_name: str, success: bool, details: Dict[str, Any] = None):
        """Record a test result."""
        result = {
            'test_name': test_name,
            'success': success,
            'details': details or {},
            'timestamp': time.time()
        }
        self.test_results.append(result)
        
        if success:
            self.passed_tests += 1
            logger.info(f"✅ {test_name}")
        else:
            self.failed_tests += 1
            logger.error(f"❌ {test_name}")
            if details and 'error' in details:
                logger.error(f"   Error: {details['error']}")
    
    async def test_root_endpoint(self) -> bool:
        """Test the root endpoint."""
        try:
            async with self.session.get(f"{self.api_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    expected_fields = ['message', 'version', 'description']
                    has_fields = all(field in data for field in expected_fields)
                    
                    self.record_test_result(
                        "Root Endpoint",
                        has_fields,
                        {'response': data, 'has_required_fields': has_fields}
                    )
                    return has_fields
                else:
                    self.record_test_result(
                        "Root Endpoint",
                        False,
                        {'error': f"HTTP {response.status}", 'response_text': await response.text()}
                    )
                    return False
        except Exception as e:
            self.record_test_result("Root Endpoint", False, {'error': str(e)})
            return False
    
    async def test_health_endpoint(self) -> bool:
        """Test the health check endpoint."""
        try:
            async with self.session.get(f"{self.api_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ['status', 'timestamp', 'vector_db_documents', 'api_version', 'deployment_info']
                    has_fields = all(field in data for field in required_fields)
                    is_healthy = data.get('status') == 'healthy'
                    
                    success = has_fields and is_healthy
                    self.record_test_result(
                        "Health Check",
                        success,
                        {
                            'response': data,
                            'has_required_fields': has_fields,
                            'is_healthy': is_healthy,
                            'document_count': data.get('vector_db_documents', 0)
                        }
                    )
                    return success
                else:
                    self.record_test_result(
                        "Health Check",
                        False,
                        {'error': f"HTTP {response.status}", 'response_text': await response.text()}
                    )
                    return False
        except Exception as e:
            self.record_test_result("Health Check", False, {'error': str(e)})
            return False
    
    async def test_documents_endpoint(self) -> bool:
        """Test the documents listing endpoint."""
        try:
            async with self.session.get(f"{self.api_url}/documents") as response:
                if response.status == 200:
                    data = await response.json()
                    is_list = isinstance(data, list)
                    
                    # Check document structure if any documents exist
                    valid_structure = True
                    if data and is_list:
                        required_doc_fields = ['source_file', 'title', 'irc_code', 'chunk_count']
                        for doc in data[:3]:  # Check first 3 documents
                            if not all(field in doc for field in required_doc_fields):
                                valid_structure = False
                                break
                    
                    success = is_list and valid_structure
                    self.record_test_result(
                        "Documents Listing",
                        success,
                        {
                            'document_count': len(data) if is_list else 0,
                            'is_list': is_list,
                            'valid_structure': valid_structure,
                            'sample_documents': data[:2] if is_list and data else []
                        }
                    )
                    return success
                else:
                    self.record_test_result(
                        "Documents Listing",
                        False,
                        {'error': f"HTTP {response.status}", 'response_text': await response.text()}
                    )
                    return False
        except Exception as e:
            self.record_test_result("Documents Listing", False, {'error': str(e)})
            return False
    
    async def test_search_endpoint(self) -> bool:
        """Test the search endpoint."""
        test_query = "pavement design"
        
        try:
            params = {'query': test_query, 'limit': 3}
            async with self.session.get(f"{self.api_url}/search", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ['query', 'results', 'total_results']
                    has_fields = all(field in data for field in required_fields)
                    correct_query = data.get('query') == test_query
                    has_results_list = isinstance(data.get('results'), list)
                    
                    success = has_fields and correct_query and has_results_list
                    self.record_test_result(
                        "Search Endpoint",
                        success,
                        {
                            'query': test_query,
                            'has_required_fields': has_fields,
                            'correct_query': correct_query,
                            'has_results_list': has_results_list,
                            'result_count': len(data.get('results', [])),
                            'total_results': data.get('total_results', 0)
                        }
                    )
                    return success
                else:
                    self.record_test_result(
                        "Search Endpoint",
                        False,
                        {'error': f"HTTP {response.status}", 'response_text': await response.text()}
                    )
                    return False
        except Exception as e:
            self.record_test_result("Search Endpoint", False, {'error': str(e)})
            return False
    
    async def test_query_endpoint(self) -> bool:
        """Test the RAG query endpoint."""
        test_query = {
            "query": "What are the key requirements for pavement design?",
            "limit": 3
        }
        
        try:
            async with self.session.post(
                f"{self.api_url}/query",
                json=test_query,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ['response', 'citations', 'relevant_chunks', 'processing_time', 'query']
                    has_fields = all(field in data for field in required_fields)
                    has_response = bool(data.get('response', '').strip())
                    has_citations = isinstance(data.get('citations'), list)
                    has_chunks = isinstance(data.get('relevant_chunks'), list)
                    correct_query = data.get('query') == test_query['query']
                    
                    success = has_fields and has_response and has_citations and has_chunks and correct_query
                    self.record_test_result(
                        "RAG Query",
                        success,
                        {
                            'query': test_query['query'],
                            'has_required_fields': has_fields,
                            'has_response': has_response,
                            'has_citations': has_citations,
                            'has_chunks': has_chunks,
                            'correct_query': correct_query,
                            'processing_time': data.get('processing_time', 0),
                            'response_length': len(data.get('response', '')),
                            'citation_count': len(data.get('citations', [])),
                            'chunk_count': len(data.get('relevant_chunks', []))
                        }
                    )
                    return success
                else:
                    error_text = await response.text()
                    self.record_test_result(
                        "RAG Query",
                        False,
                        {'error': f"HTTP {response.status}", 'response_text': error_text}
                    )
                    return False
        except Exception as e:
            self.record_test_result("RAG Query", False, {'error': str(e)})
            return False
    
    async def test_api_documentation(self) -> bool:
        """Test API documentation endpoints."""
        docs_endpoints = ['/docs', '/redoc']
        success_count = 0
        
        for endpoint in docs_endpoints:
            try:
                async with self.session.get(f"{self.api_url}{endpoint}") as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        is_html = 'text/html' in content_type
                        if is_html:
                            success_count += 1
                            logger.info(f"✅ Documentation endpoint {endpoint} accessible")
                        else:
                            logger.warning(f"⚠️ Documentation endpoint {endpoint} returned non-HTML content")
                    else:
                        logger.error(f"❌ Documentation endpoint {endpoint} failed: HTTP {response.status}")
            except Exception as e:
                logger.error(f"❌ Documentation endpoint {endpoint} error: {e}")
        
        success = success_count == len(docs_endpoints)
        self.record_test_result(
            "API Documentation",
            success,
            {'accessible_endpoints': success_count, 'total_endpoints': len(docs_endpoints)}
        )
        return success
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests."""
        logger.info(f"Starting API tests for: {self.api_url}")
        logger.info("="*60)
        
        # Run tests in order
        tests = [
            self.test_root_endpoint,
            self.test_health_endpoint,
            self.test_api_documentation,
            self.test_documents_endpoint,
            self.test_search_endpoint,
            self.test_query_endpoint,
        ]
        
        for test in tests:
            await test()
            await asyncio.sleep(1)  # Small delay between tests
        
        # Generate summary
        total_tests = len(self.test_results)
        success_rate = (self.passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        summary = {
            'api_url': self.api_url,
            'total_tests': total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'success_rate': success_rate,
            'test_results': self.test_results
        }
        
        self.print_summary(summary)
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary."""
        logger.info("\n" + "="*60)
        logger.info("API TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"API URL: {summary['api_url']}")
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        
        if summary['failed_tests'] > 0:
            logger.info("\nFailed tests:")
            for result in summary['test_results']:
                if not result['success']:
                    logger.info(f"  - {result['test_name']}: {result['details'].get('error', 'Unknown error')}")
        
        logger.info("="*60)

async def main():
    """Main function for API testing."""
    parser = argparse.ArgumentParser(description="Test IRC RAG API deployed on Railway")
    parser.add_argument("api_url", help="Base URL of the deployed API (e.g., https://your-app.up.railway.app)")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds (default: 30)")
    parser.add_argument("--output", help="Output file for test results (JSON)")
    
    args = parser.parse_args()
    
    # Run tests
    async with APITester(args.api_url, args.timeout) as tester:
        summary = await tester.run_all_tests()
        
        # Save results if output file specified
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                logger.info(f"Test results saved to: {args.output}")
            except Exception as e:
                logger.error(f"Failed to save test results: {e}")
        
        # Return appropriate exit code
        return 0 if summary['failed_tests'] == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
