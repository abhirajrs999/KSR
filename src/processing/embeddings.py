import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingModel(str, Enum):
    """Supported embedding models"""
    E5_LARGE = "intfloat/e5-large-v2"
    MPNET = "all-mpnet-base-v2"

class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using sentence-transformers models.
    Supports local embedding generation with fallback options and batch processing.
    """

    def __init__(
        self,
        model_name: str = EmbeddingModel.E5_LARGE.value,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the embedding model to use
            batch_size: Batch size for processing
            device: Device to use for inference ("cuda" or "cpu")
        """
        self.batch_size = batch_size
        self.device = device
        self.model_name = model_name
        
        try:
            self.model = self._load_model(model_name)
            logger.info(f"Successfully loaded model {model_name} on {device}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            fallback_model = EmbeddingModel.MPNET.value
            logger.info(f"Falling back to {fallback_model}")
            self.model = self._load_model(fallback_model)
            self.model_name = fallback_model

    def _load_model(self, model_name: str) -> SentenceTransformer:
        """Load and configure the sentence-transformer model"""
        model = SentenceTransformer(model_name)
        model.to(self.device)
        return model

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings in batches using sentence-transformers"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            try:
                with torch.no_grad():
                    embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                all_embeddings.append(embeddings)
            except Exception as e:
                logger.error(f"Error encoding batch {i}: {e}")
                raise

        return np.vstack(all_embeddings)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embeddings (converted from numpy array for ChromaDB compatibility)
        """
        if not texts:
            raise ValueError("Empty text list provided")

        try:
            # Generate embeddings as numpy array
            embeddings_array = self._batch_encode(texts)
            
            # Convert to list of lists for ChromaDB compatibility
            return embeddings_array.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    @staticmethod
    def compare_models(
        texts: List[str],
        models: List[str] = [EmbeddingModel.E5_LARGE.value, EmbeddingModel.MPNET.value],
        sample_size: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different embedding models on a sample of texts.
        
        Args:
            texts: List of texts to compare embeddings for
            models: List of model names to compare
            sample_size: Number of texts to sample for comparison
            
        Returns:
            Dict with model comparison metrics
        """
        if len(texts) > sample_size:
            import random
            texts = random.sample(texts, sample_size)

        results = {}
        for model_name in models:
            try:
                generator = EmbeddingGenerator(model_name=model_name)
                start_time = time.time()
                embeddings = generator.generate_embeddings(texts)
                end_time = time.time()

                # Convert back to numpy for metrics calculation
                embeddings_array = np.array(embeddings)

                results[model_name] = {
                    "embedding_dim": embeddings_array.shape[1],
                    "processing_time": end_time - start_time,
                    "avg_time_per_text": (end_time - start_time) / len(texts),
                    "memory_usage_mb": embeddings_array.nbytes / (1024 * 1024)
                }
            except Exception as e:
                logger.error(f"Error comparing model {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def process_chunk_file(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Process a chunk file and generate embeddings for all chunks.

        Args:
            input_path: Path to the input chunks JSON file
            output_path: Optional path for the output file with embeddings

        Returns:
            Path to the output file
        """
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_embeddings.json"

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.generate_embeddings(texts)

            # Add embeddings to chunks (embeddings are already lists)
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2)

            logger.info(f"Successfully processed {len(chunks)} chunks and saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error processing chunk file {input_path}: {e}")
            raise