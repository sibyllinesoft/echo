"""Semantic embeddings using GraphCodeBERT-mini."""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from .normalize import NormalizedBlock
from .config import EchoConfig

logger = logging.getLogger(__name__)


class SemanticEmbedder:
    """GraphCodeBERT-mini based semantic embeddings."""
    
    def __init__(self, config: EchoConfig, model_name: str = 'microsoft/graphcodebert-base'):
        self.config = config
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model will be loaded lazily
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self._model_loaded = False
        
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        if self._model_loaded:
            return
            
        try:
            logger.info(f"Loading semantic model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            logger.info(f"Model loaded on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            raise
            
    def embed_block(self, block: NormalizedBlock) -> np.ndarray:
        """Compute semantic embedding for a code block."""
        self._load_model()
        
        # TODO: Implement semantic embedding
        # 1. Prepare input text from normalized tokens
        # 2. Tokenize with GraphCodeBERT tokenizer
        # 3. Get model embeddings
        # 4. Pool/reduce to fixed size (256-D)
        # 5. Return as numpy array
        raise NotImplementedError("Semantic embedding pending")
        
    def embed_blocks(self, blocks: List[NormalizedBlock], batch_size: int = 16) -> np.ndarray:
        """Compute embeddings for multiple blocks in batches."""
        self._load_model()
        
        # TODO: Implement batch embedding
        # Process blocks in batches for efficiency
        raise NotImplementedError("Batch embedding pending")
        
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(emb1, emb2) / (norm1 * norm2)
        
    def find_semantic_neighbors(self, target_embedding: np.ndarray,
                               candidate_embeddings: List[np.ndarray],
                               threshold: float = 0.83) -> List[Tuple[int, float]]:
        """Find semantically similar embeddings above threshold."""
        neighbors = []
        
        for i, candidate_emb in enumerate(candidate_embeddings):
            similarity = self.cosine_similarity(target_embedding, candidate_emb)
            if similarity >= threshold:
                neighbors.append((i, similarity))
                
        # Sort by similarity (descending)
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors
        

def compute_semantic_similarity(block1: NormalizedBlock, block2: NormalizedBlock,
                               embedder: SemanticEmbedder) -> float:
    """Compute semantic similarity between two blocks."""
    emb1 = embedder.embed_block(block1)
    emb2 = embedder.embed_block(block2)
    return embedder.cosine_similarity(emb1, emb2)
    
    
def batch_semantic_similarity(target_block: NormalizedBlock,
                             candidate_blocks: List[NormalizedBlock],
                             embedder: SemanticEmbedder,
                             threshold: float = 0.83) -> List[Tuple[NormalizedBlock, float]]:
    """Compute semantic similarities for multiple candidates."""
    # TODO: Implement efficient batch similarity computation
    raise NotImplementedError("Batch semantic similarity pending")