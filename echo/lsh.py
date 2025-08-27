"""MinHash and LSH implementation for candidate generation.

This module implements MinHash signatures and Locality-Sensitive Hashing (LSH) 
for efficient duplicate code detection. It's the first stage of the echo pipeline,
responsible for quickly finding candidate duplicates without comparing every pair.

Key Features:
- MinHash signatures for Jaccard similarity estimation
- LSH banding for sub-linear candidate retrieval
- Configurable parameters (hash count, bands, shingle size)
- Thread-safe operations with concurrent access support
- Database persistence integration
- Incremental indexing (add, update, remove blocks)

Performance Characteristics:
- Time complexity: O(k) for insertion/query (k = num_hashes)
- Space complexity: O(n*k + b*m) where n=blocks, b=bands, m=buckets
- Target: Handle large codebases (>100K blocks) efficiently

Usage:
    from echo.lsh import MinHashLSH, create_lsh_from_config
    from echo.config import EchoConfig
    
    # Initialize with configuration
    config = EchoConfig()
    lsh = create_lsh_from_config(config, db_session)
    
    # Add normalized blocks
    signature = lsh.add_block(normalized_block)
    
    # Query for candidates
    candidates = lsh.query_candidates(query_block, max_candidates=50)
    
    # Each candidate is a tuple: (block_id, jaccard_similarity)
    for block_id, similarity in candidates:
        if similarity > 0.7:  # High similarity threshold
            print(f"Potential duplicate: {block_id} (sim={similarity:.3f})")

Integration with Echo Pipeline:
    1. parser.py extracts CodeBlocks from source files
    2. normalize.py creates NormalizedBlocks with token normalization
    3. lsh.py (this module) generates MinHash signatures and finds candidates
    4. verify.py performs detailed similarity verification on candidates
    5. embed.py provides semantic similarity for high-precision ranking
"""

from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
import random
import threading
import base64
from collections import defaultdict

import numpy as np

from .normalize import NormalizedBlock


@dataclass
class MinHashSignature:
    """MinHash signature for a code block."""
    block_id: str
    signature: np.ndarray
    shingles: Set[str]
    

class MinHashLSH:
    """MinHash LSH for efficient duplicate candidate generation."""
    
    def __init__(self, num_hashes: int = 128, num_bands: int = 16, shingle_size: int = 5, db_session=None):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.shingle_size = shingle_size
        self.db_session = db_session
        
        # Ensure parameters are valid
        if num_hashes % num_bands != 0:
            raise ValueError(f"num_hashes ({num_hashes}) must be divisible by num_bands ({num_bands})")
        
        # Random hash functions for MinHash - use large primes for better distribution
        self.hash_functions = []
        random.seed(42)  # For reproducible results
        
        # Use larger primes for better hash distribution
        LARGE_PRIME = 2**61 - 1
        for _ in range(num_hashes):
            a = random.randint(1, LARGE_PRIME)
            b = random.randint(0, LARGE_PRIME)
            self.hash_functions.append((a, b))
            
        # LSH buckets: band_id -> bucket_hash -> set of block_ids
        self.buckets: Dict[int, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # Cache for MinHash signatures: block_id -> signature
        self.signature_cache: Dict[str, np.ndarray] = {}
        
        # Threading lock for thread safety
        self._lock = threading.RLock()
        
    def create_shingles(self, tokens: List[str]) -> Set[str]:
        """Create k-shingles from token sequence."""
        if len(tokens) < self.shingle_size:
            # For very short sequences, use the entire sequence as a single shingle
            return {' '.join(tokens)}
            
        shingles = set()
        for i in range(len(tokens) - self.shingle_size + 1):
            shingle = ' '.join(tokens[i:i + self.shingle_size])
            shingles.add(shingle)
        return shingles
        
    def compute_minhash(self, shingles: Set[str]) -> np.ndarray:
        """Compute MinHash signature for a set of shingles."""
        if not shingles:
            # Empty set gets random signature
            return np.random.randint(0, 2**32, size=self.num_hashes, dtype=np.int64)
            
        signature = np.full(self.num_hashes, 2**63 - 1, dtype=np.int64)
        
        for shingle in shingles:
            shingle_bytes = shingle.encode('utf-8')
            # Use built-in hash for consistency across Python versions
            base_hash = hash(shingle_bytes)
            
            for i, (a, b) in enumerate(self.hash_functions):
                # Use modular arithmetic to avoid overflow
                hash_val = (base_hash * a + b) % (2**63 - 1)
                signature[i] = min(signature[i], hash_val)
                
        return signature
        
    def add_block(self, block: NormalizedBlock) -> MinHashSignature:
        """Add a block to the LSH index."""
        with self._lock:
            block_id = block.original.file_path.as_posix() + f":{block.original.start_line}"
            
            # Create shingles from normalized tokens
            shingles = self.create_shingles(block.normalized_tokens)
            
            # Compute MinHash signature
            signature = self.compute_minhash(shingles)
            
            # Store in cache
            self.signature_cache[block_id] = signature
            
            # Add to LSH buckets
            self._add_to_buckets(block_id, signature)
            
            # Persist signature if database session available
            if self.db_session:
                self._persist_signature(block_id, signature, block)
            
            return MinHashSignature(
                block_id=block_id,
                signature=signature,
                shingles=shingles
            )
    
    def _add_to_buckets(self, block_id: str, signature: np.ndarray):
        """Add signature to LSH buckets."""
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            band = signature[start_idx:end_idx]
            
            bucket_hash = self._band_hash(band)
            self.buckets[band_idx][bucket_hash].add(block_id)
    
    def _persist_signature(self, block_id: str, signature: np.ndarray, block: NormalizedBlock):
        """Persist MinHash signature to database."""
        try:
            from .storage import BlockRecord
            
            # Check if block record exists, update or create
            existing = self.db_session.query(BlockRecord).filter_by(id=block_id).first()
            if existing:
                # Update existing record with MinHash data
                existing.norm_hash = self._signature_to_string(signature)
            else:
                # Create new block record
                block_record = BlockRecord(
                    id=block_id,
                    file_path=block.original.file_path.as_posix(),
                    start=block.original.start_line,
                    end=block.original.end_line,
                    lang=block.original.lang,
                    tokens=len(block.normalized_tokens),
                    norm_hash=self._signature_to_string(signature),
                    churn=0
                )
                self.db_session.add(block_record)
            
            self.db_session.commit()
        except Exception as e:
            print(f"Warning: Failed to persist signature for {block_id}: {e}")
            if self.db_session:
                self.db_session.rollback()
    
    def _signature_to_string(self, signature: np.ndarray) -> str:
        """Convert signature to string for storage."""
        import base64
        return base64.b64encode(signature.tobytes()).decode('ascii')
    
    def _string_to_signature(self, signature_str: str) -> np.ndarray:
        """Convert string back to signature."""
        import base64
        bytes_data = base64.b64decode(signature_str.encode('ascii'))
        return np.frombuffer(bytes_data, dtype=np.int64)
        
    def query_candidates(self, block: NormalizedBlock, max_candidates: int = 50) -> List[Tuple[str, float]]:
        """Query LSH for duplicate candidates with similarity scores."""
        with self._lock:
            # Compute MinHash for query block
            shingles = self.create_shingles(block.normalized_tokens)
            query_signature = self.compute_minhash(shingles)
            
            # Collect candidates from LSH buckets
            candidates = set()
            
            for band_idx in range(self.num_bands):
                start_idx = band_idx * self.rows_per_band
                end_idx = start_idx + self.rows_per_band
                band = query_signature[start_idx:end_idx]
                
                bucket_hash = self._band_hash(band)
                if bucket_hash in self.buckets[band_idx]:
                    candidates.update(self.buckets[band_idx][bucket_hash])
            
            # Remove self if present (for updates)
            query_block_id = block.original.file_path.as_posix() + f":{block.original.start_line}"
            candidates.discard(query_block_id)
            
            # Compute Jaccard similarities and sort by similarity
            candidate_scores = []
            for candidate_id in candidates:
                if candidate_id in self.signature_cache:
                    candidate_sig = self.signature_cache[candidate_id]
                    similarity = self.jaccard_similarity(query_signature, candidate_sig)
                    candidate_scores.append((candidate_id, similarity))
            
            # Sort by similarity (descending) and limit to max_candidates
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            return candidate_scores[:max_candidates]
    
    def remove_block(self, block_id: str):
        """Remove a block from the LSH index."""
        with self._lock:
            if block_id not in self.signature_cache:
                return
                
            signature = self.signature_cache[block_id]
            
            # Remove from LSH buckets
            for band_idx in range(self.num_bands):
                start_idx = band_idx * self.rows_per_band
                end_idx = start_idx + self.rows_per_band
                band = signature[start_idx:end_idx]
                
                bucket_hash = self._band_hash(band)
                if bucket_hash in self.buckets[band_idx]:
                    self.buckets[band_idx][bucket_hash].discard(block_id)
                    
                    # Clean up empty buckets
                    if not self.buckets[band_idx][bucket_hash]:
                        del self.buckets[band_idx][bucket_hash]
            
            # Remove from cache
            del self.signature_cache[block_id]
    
    def update_block(self, block: NormalizedBlock) -> MinHashSignature:
        """Update an existing block in the LSH index."""
        block_id = block.original.file_path.as_posix() + f":{block.original.start_line}"
        
        # Remove old version if it exists
        self.remove_block(block_id)
        
        # Add updated version
        return self.add_block(block)
    
    def bulk_add_blocks(self, blocks: List[NormalizedBlock]) -> List[MinHashSignature]:
        """Efficiently add multiple blocks."""
        signatures = []
        with self._lock:
            for block in blocks:
                try:
                    sig = self.add_block(block)
                    signatures.append(sig)
                except Exception as e:
                    print(f"Warning: Failed to add block {block.original.file_path}:{block.original.start_line}: {e}")
                    continue
        return signatures
    
    def load_from_database(self):
        """Load existing MinHash signatures from database."""
        if not self.db_session:
            return
            
        try:
            from .storage import BlockRecord
            
            with self._lock:
                # Clear current state
                self.buckets.clear()
                self.signature_cache.clear()
                
                # Load all block records with MinHash signatures
                blocks = self.db_session.query(BlockRecord).filter(
                    BlockRecord.norm_hash.isnot(None)
                ).all()
                
                for block_record in blocks:
                    try:
                        signature = self._string_to_signature(block_record.norm_hash)
                        self.signature_cache[block_record.id] = signature
                        self._add_to_buckets(block_record.id, signature)
                    except Exception as e:
                        print(f"Warning: Failed to load signature for {block_record.id}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Warning: Failed to load signatures from database: {e}")
    
    def jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Compute Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have same length")
        return float(np.sum(sig1 == sig2)) / len(sig1)
        
    def _band_hash(self, signature_band: np.ndarray) -> str:
        """Hash a band of the signature for LSH bucketing."""
        return hashlib.md5(signature_band.tobytes()).hexdigest()
    
    def estimate_threshold(self) -> float:
        """Estimate the LSH threshold for current parameters."""
        # Theoretical threshold where probability crosses 0.5
        r = self.rows_per_band
        b = self.num_bands
        return (1.0 / b) ** (1.0 / r)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get LSH statistics."""
        with self._lock:
            total_blocks = len(self.signature_cache)
            total_buckets = sum(len(band_buckets) for band_buckets in self.buckets.values())
            non_empty_buckets = sum(
                len([bucket for bucket in band_buckets.values() if bucket])
                for band_buckets in self.buckets.values()
            )
            
            # Calculate bucket distribution statistics
            bucket_sizes = []
            for band_buckets in self.buckets.values():
                for bucket in band_buckets.values():
                    if bucket:
                        bucket_sizes.append(len(bucket))
            
            avg_bucket_size = sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0
            max_bucket_size = max(bucket_sizes) if bucket_sizes else 0
            
            return {
                'total_blocks': total_blocks,
                'total_bands': len(self.buckets),
                'total_buckets': total_buckets,
                'non_empty_buckets': non_empty_buckets,
                'avg_bucket_size': round(avg_bucket_size, 2),
                'max_bucket_size': max_bucket_size,
                'estimated_threshold': round(self.estimate_threshold(), 3),
                'signature_size': self.num_hashes,
                'shingle_size': self.shingle_size
            }
    
    def get_block_signature(self, block_id: str) -> Optional[np.ndarray]:
        """Get cached signature for a block."""
        with self._lock:
            return self.signature_cache.get(block_id)
    
    def clear(self):
        """Clear all LSH data."""
        with self._lock:
            self.buckets.clear()
            self.signature_cache.clear()


# Factory function for creating LSH with configuration
def create_lsh_from_config(config, db_session=None) -> MinHashLSH:
    """Create MinHashLSH instance from configuration."""
    return MinHashLSH(
        num_hashes=config.num_hashes,
        num_bands=config.num_bands,
        shingle_size=config.shingle_size,
        db_session=db_session
    )