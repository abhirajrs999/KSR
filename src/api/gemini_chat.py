import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import content_types

from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for Gemini API calls."""
    def __init__(self, max_requests: int = 15, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make an API call."""
        async with self.lock:
            now = time.time()
            # Remove old requests
            self.requests = [t for t in self.requests if now - t < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                wait_time = self.requests[0] + self.time_window - now
                if wait_time > 0:
                    logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    # Recursive call after waiting
                    await self.acquire()
                    return
            
            self.requests.append(now)

class GeminiChatEngine:
    """
    Chat engine using Google's Gemini models.
    Implements rate limiting, fallback options, and specialized IRC prompts.
    """

    # IRC-specific prompt templates
    PROMPT_TEMPLATES = {
        "technical_query": """You are an expert in Indian Roads Congress (IRC) standards and specifications.
        Please analyze the following query about IRC code {irc_code}:
        
        Query: {query}
        
        Context from IRC document:
        {context}
        
        IMPORTANT: When providing your response, always clearly mention the source document name (e.g., "according to IRC document mort.250.2013" or "as per IRC-37-2019").
        
        Provide a detailed technical response, citing specific clauses and sections where applicable.
        If there are any relevant tables, format them clearly with proper headers and aligned columns.
        Always include the source document reference in your citations.
        """,
        
        "clause_explanation": """Explain the following clause from IRC {irc_code}:
        
        {clause_text}
        
        Provide:
        1. A clear explanation of the requirements
        2. The technical rationale behind them
        3. Any related clauses or cross-references
        4. Practical implications for implementation
        """,
        
        "comparison": """Compare the following specifications from different IRC codes:
        
        Code 1: {code1} - {spec1}
        Code 2: {code2} - {spec2}
        
        Analyze:
        1. Key differences in requirements
        2. Technical implications
        3. Which version is more current/stringent
        4. Practical considerations for implementation
        """
    }

    def __init__(
        self,
        primary_model: str = "gemini-2.0-flash-exp",
        fallback_model: str = "gemini-1.5-flash",
        temperature: float = 0.3,
        top_p: float = 0.8,
        top_k: int = 40,
        max_output_tokens: int = 2048,
    ):
        """
        Initialize the Gemini chat engine.

        Args:
            primary_model: Primary Gemini model to use
            fallback_model: Fallback model if primary fails
            temperature: Temperature for response generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_output_tokens: Maximum output length
        """
        if not hasattr(settings, 'GOOGLE_API_KEY'):
            raise ValueError("Google API key not found in settings")

        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }
        
        self.rate_limiter = RateLimiter()
        self.usage_stats = {
            "total_requests": 0,
            "fallback_requests": 0,
            "errors": 0,
            "last_reset": datetime.now()
        }

    async def _get_model(self, use_fallback: bool = False) -> Any:
        """Get the appropriate model instance."""
        model_name = self.fallback_model if use_fallback else self.primary_model
        try:
            return genai.GenerativeModel(model_name)
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {e}")
            if not use_fallback:
                logger.info("Attempting fallback model")
                return await self._get_model(use_fallback=True)
            raise

    async def _make_request(
        self,
        prompt: str,
        use_fallback: bool = False,
        retry_count: int = 0
    ) -> Optional[str]:
        """Make a rate-limited request to the Gemini API."""
        if retry_count >= 3:
            logger.error("Max retries reached")
            return None

        await self.rate_limiter.acquire()
        
        try:
            model = await self._get_model(use_fallback)
            response = await model.generate_content_async(
                prompt,
                generation_config=self.generation_config
            )
            
            self.usage_stats["total_requests"] += 1
            if use_fallback:
                self.usage_stats["fallback_requests"] += 1
            
            return response.text

        except Exception as e:
            self.usage_stats["errors"] += 1
            logger.error(f"Error in Gemini API call: {e}")
            
            if not use_fallback:
                logger.info("Attempting fallback model")
                return await self._make_request(prompt, use_fallback=True, retry_count=retry_count)
            
            if retry_count < 2:
                wait_time = (retry_count + 1) * 2
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                return await self._make_request(prompt, use_fallback, retry_count + 1)
            
            return None

    async def query_irc_technical(
        self,
        query: str,
        irc_code: str,
        context: str
    ) -> Optional[str]:
        """
        Process a technical query about IRC specifications.

        Args:
            query: The technical question
            irc_code: The IRC code being queried
            context: Relevant context from the IRC document

        Returns:
            Generated response or None if failed
        """
        prompt = self.PROMPT_TEMPLATES["technical_query"].format(
            query=query,
            irc_code=irc_code,
            context=context
        )
        return await self._make_request(prompt)

    async def explain_clause(
        self,
        irc_code: str,
        clause_text: str
    ) -> Optional[str]:
        """
        Generate a detailed explanation of an IRC clause.

        Args:
            irc_code: The IRC code containing the clause
            clause_text: The text of the clause to explain

        Returns:
            Generated explanation or None if failed
        """
        prompt = self.PROMPT_TEMPLATES["clause_explanation"].format(
            irc_code=irc_code,
            clause_text=clause_text
        )
        return await self._make_request(prompt)

    async def compare_specifications(
        self,
        code1: str,
        spec1: str,
        code2: str,
        spec2: str
    ) -> Optional[str]:
        """
        Compare specifications from different IRC codes.

        Args:
            code1: First IRC code
            spec1: First specification text
            code2: Second IRC code
            spec2: Second specification text

        Returns:
            Generated comparison or None if failed
        """
        prompt = self.PROMPT_TEMPLATES["comparison"].format(
            code1=code1,
            spec1=spec1,
            code2=code2,
            spec2=spec2
        )
        return await self._make_request(prompt)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        stats = self.usage_stats.copy()
        stats["uptime"] = str(datetime.now() - self.usage_stats["last_reset"])
        stats["success_rate"] = (
            (stats["total_requests"] - stats["errors"]) / stats["total_requests"] * 100
            if stats["total_requests"] > 0 else 0
        )
        stats["fallback_rate"] = (
            stats["fallback_requests"] / stats["total_requests"] * 100
            if stats["total_requests"] > 0 else 0
        )
        return stats

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_stats = {
            "total_requests": 0,
            "fallback_requests": 0,
            "errors": 0,
            "last_reset": datetime.now()
        } 