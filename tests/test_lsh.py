"""Tests for MinHash LSH implementation."""

import pytest
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

# Mock the dependencies for testing
@dataclass
class MockCodeBlock:
    """Mock CodeBlock for testing."""
    lang: str
    file_path: Path
    start_line: int
    end_line: int
    start_byte: int = 0
    end_byte: int = 100
    tokens: List[str] = None
    raw_content: str = ""
    node_type: str = "function"

@dataclass
class MockNormalizedBlock:
    """Mock NormalizedBlock for testing."""
    original: MockCodeBlock
    normalized_tokens: List[str]
    token_mapping: Dict[str, str] = None
    reverse_mapping: Dict[str, List[str]] = None
    hash_signature: str = ""

# Import after mocking
import sys
sys.path.insert(0, '/home/nathan/Projects/echo')

from echo.lsh import MinHashLSH, MinHashSignature


class TestMinHashLSH:
    """Test MinHashLSH functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.lsh = MinHashLSH(num_hashes=64, num_bands=8, shingle_size=3)
        
    def test_initialization(self):
        """Test LSH initialization."""
        assert self.lsh.num_hashes == 64
        assert self.lsh.num_bands == 8
        assert self.lsh.rows_per_band == 8
        assert self.lsh.shingle_size == 3
        assert len(self.lsh.hash_functions) == 64
        
    def test_invalid_parameters(self):
        """Test validation of LSH parameters."""
        with pytest.raises(ValueError):
            MinHashLSH(num_hashes=65, num_bands=8)  # Not divisible
            
    def test_create_shingles(self):
        """Test shingle creation."""
        tokens = ["def", "foo", "bar", "baz", "end"]
        shingles = self.lsh.create_shingles(tokens)
        
        expected = {
            "def foo bar",
            "foo bar baz", 
            "bar baz end"
        }
        assert shingles == expected
        
    def test_create_shingles_short_sequence(self):
        """Test shingle creation with short token sequence."""
        tokens = ["a", "b"]
        shingles = self.lsh.create_shingles(tokens)
        assert shingles == {"a b"}
        
    def test_compute_minhash(self):
        """Test MinHash computation."""
        shingles = {"hello world", "world test", "test code"}
        signature = self.lsh.compute_minhash(shingles)
        
        assert isinstance(signature, np.ndarray)
        assert len(signature) == 64
        assert signature.dtype == np.int64
        
    def test_compute_minhash_empty(self):
        """Test MinHash computation with empty shingles."""
        signature = self.lsh.compute_minhash(set())
        assert isinstance(signature, np.ndarray)
        assert len(signature) == 64
        
    def test_jaccard_similarity(self):
        """Test Jaccard similarity computation."""
        sig1 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        sig2 = np.array([1, 2, 8, 9, 5], dtype=np.int64)  # 3 matches out of 5
        
        similarity = self.lsh.jaccard_similarity(sig1, sig2)
        assert similarity == 0.6  # 3/5
        
    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity with identical signatures."""
        sig = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        similarity = self.lsh.jaccard_similarity(sig, sig)
        assert similarity == 1.0
        
    def test_add_block(self):
        """Test adding a block to the LSH index."""
        code_block = MockCodeBlock(
            lang="python",
            file_path=Path("/test/file.py"),
            start_line=10,
            end_line=20
        )
        
        norm_block = MockNormalizedBlock(
            original=code_block,
            normalized_tokens=["def", "func", "VAR_1", "return", "VAR_1"]
        )
        
        signature = self.lsh.add_block(norm_block)
        
        assert isinstance(signature, MinHashSignature)
        assert signature.block_id == "/test/file.py:10"
        assert len(signature.signature) == 64
        assert len(signature.shingles) > 0
        
        # Check that block was added to cache
        assert "/test/file.py:10" in self.lsh.signature_cache
        
    def test_query_candidates(self):
        """Test querying for candidates."""
        # Add some blocks first
        blocks = []
        for i in range(5):
            code_block = MockCodeBlock(
                lang="python",
                file_path=Path(f"/test/file{i}.py"),
                start_line=10,
                end_line=20
            )
            
            # Create similar but not identical token sequences
            base_tokens = ["def", "func", "VAR_1", "return", "VAR_1"]
            if i < 3:
                # First 3 blocks are similar
                tokens = base_tokens + [f"extra_{i}"]
            else:
                # Last 2 blocks are different
                tokens = ["class", "MyClass", "pass", "end"] + [f"extra_{i}"]
                
            norm_block = MockNormalizedBlock(
                original=code_block,
                normalized_tokens=tokens
            )
            blocks.append(norm_block)
            self.lsh.add_block(norm_block)
        
        # Query with a similar block
        query_block = MockNormalizedBlock(
            original=MockCodeBlock(
                lang="python",
                file_path=Path("/test/query.py"),
                start_line=1,
                end_line=10
            ),
            normalized_tokens=["def", "func", "VAR_1", "return", "VAR_1", "extra_new"]
        )
        
        candidates = self.lsh.query_candidates(query_block, max_candidates=10)
        
        assert isinstance(candidates, list)
        assert len(candidates) <= 10
        
        # Should return tuples of (block_id, similarity_score)
        for candidate_id, score in candidates:
            assert isinstance(candidate_id, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            
    def test_remove_block(self):
        """Test removing a block from the index."""
        code_block = MockCodeBlock(
            lang="python",
            file_path=Path("/test/file.py"),
            start_line=10,
            end_line=20
        )
        
        norm_block = MockNormalizedBlock(
            original=code_block,
            normalized_tokens=["def", "func", "return"]
        )
        
        # Add block
        self.lsh.add_block(norm_block)
        block_id = "/test/file.py:10"
        
        assert block_id in self.lsh.signature_cache
        
        # Remove block
        self.lsh.remove_block(block_id)
        
        assert block_id not in self.lsh.signature_cache
        
    def test_update_block(self):
        """Test updating a block in the index."""
        code_block = MockCodeBlock(
            lang="python",
            file_path=Path("/test/file.py"),
            start_line=10,
            end_line=20
        )
        
        # Original block
        norm_block1 = MockNormalizedBlock(
            original=code_block,
            normalized_tokens=["def", "func", "return"]
        )
        
        sig1 = self.lsh.add_block(norm_block1)
        
        # Updated block with different content
        norm_block2 = MockNormalizedBlock(
            original=code_block,
            normalized_tokens=["def", "func", "VAR_1", "return", "VAR_1"]
        )
        
        sig2 = self.lsh.update_block(norm_block2)
        
        # Signatures should be different
        assert not np.array_equal(sig1.signature, sig2.signature)
        
        # Only one entry should exist in cache
        assert len(self.lsh.signature_cache) == 1
        
    def test_bulk_add_blocks(self):
        """Test bulk adding blocks."""
        blocks = []
        for i in range(10):
            code_block = MockCodeBlock(
                lang="python",
                file_path=Path(f"/test/file{i}.py"),
                start_line=10,
                end_line=20
            )
            
            norm_block = MockNormalizedBlock(
                original=code_block,
                normalized_tokens=["def", "func", f"var_{i}", "return"]
            )
            blocks.append(norm_block)
            
        signatures = self.lsh.bulk_add_blocks(blocks)
        
        assert len(signatures) == 10
        assert len(self.lsh.signature_cache) == 10
        
    def test_signature_serialization(self):
        """Test signature serialization/deserialization."""
        signature = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        
        # Convert to string and back
        sig_str = self.lsh._signature_to_string(signature)
        recovered_sig = self.lsh._string_to_signature(sig_str)
        
        assert np.array_equal(signature, recovered_sig)
        
    def test_estimate_threshold(self):
        """Test LSH threshold estimation."""
        threshold = self.lsh.estimate_threshold()
        
        assert isinstance(threshold, float)
        assert 0.0 < threshold < 1.0
        
    def test_get_statistics(self):
        """Test statistics collection."""
        # Add some blocks
        for i in range(5):
            code_block = MockCodeBlock(
                lang="python",
                file_path=Path(f"/test/file{i}.py"),
                start_line=10,
                end_line=20
            )
            
            norm_block = MockNormalizedBlock(
                original=code_block,
                normalized_tokens=["def", "func", f"var_{i}", "return"]
            )
            self.lsh.add_block(norm_block)
            
        stats = self.lsh.get_statistics()
        
        expected_keys = {
            'total_blocks', 'total_bands', 'total_buckets', 
            'non_empty_buckets', 'avg_bucket_size', 'max_bucket_size',
            'estimated_threshold', 'signature_size', 'shingle_size'
        }
        
        assert set(stats.keys()) == expected_keys
        assert stats['total_blocks'] == 5
        assert stats['total_bands'] == 8
        assert stats['signature_size'] == 64
        assert stats['shingle_size'] == 3
        
    def test_clear(self):
        """Test clearing the LSH index."""
        # Add a block
        code_block = MockCodeBlock(
            lang="python",
            file_path=Path("/test/file.py"),
            start_line=10,
            end_line=20
        )
        
        norm_block = MockNormalizedBlock(
            original=code_block,
            normalized_tokens=["def", "func", "return"]
        )
        
        self.lsh.add_block(norm_block)
        assert len(self.lsh.signature_cache) == 1
        
        # Clear and verify
        self.lsh.clear()
        assert len(self.lsh.signature_cache) == 0
        assert len(self.lsh.buckets) == 0
        
    def test_thread_safety(self):
        """Test thread safety with concurrent operations."""
        import threading
        import time
        
        def add_blocks(start_idx, count):
            for i in range(start_idx, start_idx + count):
                code_block = MockCodeBlock(
                    lang="python",
                    file_path=Path(f"/test/file{i}.py"),
                    start_line=10,
                    end_line=20
                )
                
                norm_block = MockNormalizedBlock(
                    original=code_block,
                    normalized_tokens=["def", "func", f"var_{i}", "return"]
                )
                self.lsh.add_block(norm_block)
                time.sleep(0.001)  # Small delay to increase chance of race conditions
                
        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=add_blocks, args=(i*10, 10))
            threads.append(t)
            t.start()
            
        # Wait for all threads to complete
        for t in threads:
            t.join()
            
        # Verify all blocks were added
        assert len(self.lsh.signature_cache) == 30


if __name__ == "__main__":
    pytest.main([__file__])