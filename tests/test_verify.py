"""Tests for token-level verification module."""

import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

from echo.verify import (
    longest_common_subsequence,
    compute_edit_distance, 
    compute_jaccard_similarity,
    verify_near_miss,
    batch_verify,
    verify_candidate_pairs,
    get_verification_statistics,
    VerificationResult
)
from echo.config import EchoConfig


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


class TestLongestCommonSubsequence:
    """Test LCS algorithm implementation."""
    
    def test_identical_sequences(self):
        """Test LCS of identical sequences."""
        seq = ["a", "b", "c", "d"]
        assert longest_common_subsequence(seq, seq) == 4
        
    def test_completely_different(self):
        """Test LCS of completely different sequences."""
        seq1 = ["a", "b", "c"]
        seq2 = ["x", "y", "z"]
        assert longest_common_subsequence(seq1, seq2) == 0
        
    def test_partial_overlap(self):
        """Test LCS of sequences with partial overlap."""
        seq1 = ["a", "b", "c", "d", "e"]
        seq2 = ["x", "b", "y", "d", "z"]
        # Common subsequence: "b", "d"
        assert longest_common_subsequence(seq1, seq2) == 2
        
    def test_empty_sequences(self):
        """Test LCS with empty sequences."""
        seq1 = ["a", "b", "c"]
        seq2 = []
        assert longest_common_subsequence(seq1, seq2) == 0
        assert longest_common_subsequence(seq2, seq1) == 0
        assert longest_common_subsequence([], []) == 0
        
    def test_single_element(self):
        """Test LCS with single element sequences."""
        assert longest_common_subsequence(["a"], ["a"]) == 1
        assert longest_common_subsequence(["a"], ["b"]) == 0
        
    def test_code_tokens(self):
        """Test LCS with realistic code token sequences."""
        tokens1 = ["def", "VAR_1", "(", "PARAM_1", ")", ":", "return", "VAR_1", "+", "NUM_LIT"]
        tokens2 = ["def", "VAR_2", "(", "PARAM_1", ")", ":", "return", "VAR_2", "*", "NUM_LIT"]
        # Common: def, (, PARAM_1, ), :, return, NUM_LIT = 7 tokens
        assert longest_common_subsequence(tokens1, tokens2) == 7


class TestEditDistance:
    """Test edit distance algorithm implementation."""
    
    def test_identical_sequences(self):
        """Test edit distance of identical sequences."""
        seq = ["a", "b", "c", "d"]
        assert compute_edit_distance(seq, seq) == 0
        
    def test_single_substitution(self):
        """Test single substitution."""
        seq1 = ["a", "b", "c"]
        seq2 = ["a", "x", "c"]
        assert compute_edit_distance(seq1, seq2) == 1
        
    def test_single_insertion(self):
        """Test single insertion."""
        seq1 = ["a", "b"]
        seq2 = ["a", "x", "b"]
        assert compute_edit_distance(seq1, seq2) == 1
        
    def test_single_deletion(self):
        """Test single deletion."""
        seq1 = ["a", "x", "b"]
        seq2 = ["a", "b"]
        assert compute_edit_distance(seq1, seq2) == 1
        
    def test_empty_sequences(self):
        """Test edit distance with empty sequences."""
        seq1 = ["a", "b", "c"]
        seq2 = []
        assert compute_edit_distance(seq1, seq2) == 3
        assert compute_edit_distance(seq2, seq1) == 3
        assert compute_edit_distance([], []) == 0
        
    def test_completely_different(self):
        """Test edit distance of completely different sequences."""
        seq1 = ["a", "b", "c"]
        seq2 = ["x", "y", "z"]
        assert compute_edit_distance(seq1, seq2) == 3
        
    def test_code_tokens(self):
        """Test edit distance with code tokens."""
        tokens1 = ["def", "func", "(", "x", ")", ":", "return", "x", "+", "1"]
        tokens2 = ["def", "func", "(", "y", ")", ":", "return", "y", "*", "2"]
        # Substitutions: x->y (2 times), +->*, 1->2 = 4 edits
        assert compute_edit_distance(tokens1, tokens2) == 4


class TestJaccardSimilarity:
    """Test Jaccard similarity computation."""
    
    def test_identical_sequences(self):
        """Test Jaccard of identical sequences."""
        seq = ["a", "b", "c"]
        assert compute_jaccard_similarity(seq, seq) == 1.0
        
    def test_no_overlap(self):
        """Test Jaccard with no overlap."""
        seq1 = ["a", "b", "c"]
        seq2 = ["x", "y", "z"]
        assert compute_jaccard_similarity(seq1, seq2) == 0.0
        
    def test_partial_overlap(self):
        """Test Jaccard with partial overlap."""
        seq1 = ["a", "b", "c", "d"]
        seq2 = ["c", "d", "e", "f"]
        # Intersection: {c, d} = 2, Union: {a, b, c, d, e, f} = 6
        # Jaccard = 2/6 = 1/3
        assert abs(compute_jaccard_similarity(seq1, seq2) - 1/3) < 0.001
        
    def test_empty_sequences(self):
        """Test Jaccard with empty sequences."""
        seq1 = ["a", "b"]
        assert compute_jaccard_similarity(seq1, []) == 0.0
        assert compute_jaccard_similarity([], seq1) == 0.0
        assert compute_jaccard_similarity([], []) == 1.0
        
    def test_duplicates_in_sequence(self):
        """Test Jaccard handles duplicates correctly (set-based)."""
        seq1 = ["a", "a", "b", "b", "c"]
        seq2 = ["a", "b", "b", "c", "c"]
        # Both have same unique elements: {a, b, c}
        assert compute_jaccard_similarity(seq1, seq2) == 1.0


class TestVerifyNearMiss:
    """Test near-miss duplicate verification."""
    
    def create_mock_block(self, tokens: List[str], file_path: str = "test.py", line: int = 1):
        """Helper to create mock normalized block."""
        code_block = MockCodeBlock(
            lang="python",
            file_path=Path(file_path),
            start_line=line,
            end_line=line + 10
        )
        return MockNormalizedBlock(
            original=code_block,
            normalized_tokens=tokens
        )
        
    def test_identical_blocks(self):
        """Test verification of identical blocks."""
        tokens = ["def", "VAR_1", "(", "PARAM_1", ")", ":", "return", "VAR_1"]
        block1 = self.create_mock_block(tokens)
        block2 = self.create_mock_block(tokens, "other.py", 5)
        
        result = verify_near_miss(block1, block2)
        
        assert result.is_duplicate is True
        assert result.overlap_score == 1.0
        assert result.jaccard_score == 1.0
        assert result.edit_distance == 0
        assert result.edit_density == 0.0
        assert result.lcs_length == len(tokens)
        assert result.similarity_confidence == 1.0
        
    def test_high_overlap_duplicate(self):
        """Test blocks with high overlap (>= 0.75)."""
        tokens1 = ["def", "VAR_1", "(", "PARAM_1", ")", ":", "return", "VAR_1", "+", "NUM_LIT"]  # 10 tokens
        tokens2 = ["def", "VAR_2", "(", "PARAM_1", ")", ":", "return", "VAR_2", "+", "NUM_LIT"]  # 10 tokens
        # LCS: def, (, PARAM_1, ), :, return, +, NUM_LIT = 8 tokens
        # Overlap: 8/10 = 0.8 >= 0.75
        
        block1 = self.create_mock_block(tokens1)
        block2 = self.create_mock_block(tokens2)
        
        result = verify_near_miss(block1, block2)
        
        assert result.is_duplicate is True
        assert result.overlap_score >= 0.75
        assert result.lcs_length == 8
        
    def test_medium_overlap_low_edits_duplicate(self):
        """Test blocks with medium overlap (>= 0.6) and low edit density (<= 0.25)."""
        tokens1 = ["def", "VAR_1", "(", "PARAM_1", ")", ":", "return", "VAR_1", "+", "NUM_LIT"]  # 10 tokens  
        tokens2 = ["def", "VAR_2", "(", "PARAM_2", ")", ":", "return", "VAR_2", "*", "NUM_LIT"]  # 10 tokens
        # LCS should be around 6-7, edit distance around 2-3
        
        block1 = self.create_mock_block(tokens1)
        block2 = self.create_mock_block(tokens2)
        
        result = verify_near_miss(block1, block2, overlap_threshold=0.75, edit_density_threshold=0.25)
        
        # Should pass secondary criteria: overlap >= 0.6 AND edit_density <= 0.25
        if result.overlap_score >= 0.6 and result.edit_density <= 0.25:
            assert result.is_duplicate is True
        
    def test_low_similarity_not_duplicate(self):
        """Test blocks with low similarity."""
        tokens1 = ["def", "func1", "(", "x", ")", ":", "return", "x", "+", "1"]
        tokens2 = ["class", "MyClass", ":", "pass"]
        
        block1 = self.create_mock_block(tokens1)
        block2 = self.create_mock_block(tokens2)
        
        result = verify_near_miss(block1, block2)
        
        assert result.is_duplicate is False
        assert result.overlap_score < 0.6
        
    def test_empty_blocks(self):
        """Test verification with empty blocks."""
        block1 = self.create_mock_block([])
        block2 = self.create_mock_block([])
        
        result = verify_near_miss(block1, block2)
        assert result.is_duplicate is True
        
        # One empty, one non-empty
        block3 = self.create_mock_block(["def", "func", "(", ")"])
        result2 = verify_near_miss(block1, block3)
        assert result2.is_duplicate is False
        
    def test_custom_thresholds(self):
        """Test verification with custom thresholds."""
        tokens1 = ["def", "VAR_1", "(", ")", "return", "NUM_LIT"]
        tokens2 = ["def", "VAR_2", "(", ")", "return", "NUM_LIT"] 
        
        block1 = self.create_mock_block(tokens1)
        block2 = self.create_mock_block(tokens2)
        
        # Strict thresholds
        result_strict = verify_near_miss(block1, block2, overlap_threshold=0.9, edit_density_threshold=0.1)
        
        # Lenient thresholds  
        result_lenient = verify_near_miss(block1, block2, overlap_threshold=0.5, edit_density_threshold=0.5)
        
        # Lenient should be more likely to accept
        assert result_lenient.is_duplicate or not result_strict.is_duplicate


class TestBatchVerify:
    """Test batch verification functionality."""
    
    def create_mock_block(self, tokens: List[str], file_path: str = "test.py", line: int = 1):
        """Helper to create mock normalized block."""
        code_block = MockCodeBlock(
            lang="python", 
            file_path=Path(file_path),
            start_line=line,
            end_line=line + 10
        )
        return MockNormalizedBlock(
            original=code_block,
            normalized_tokens=tokens
        )
        
    def test_batch_verify_basic(self):
        """Test basic batch verification."""
        target_tokens = ["def", "VAR_1", "(", "PARAM_1", ")", ":", "return", "VAR_1"]
        target_block = self.create_mock_block(target_tokens)
        
        # Create candidates: one duplicate, one not
        duplicate_tokens = ["def", "VAR_2", "(", "PARAM_1", ")", ":", "return", "VAR_2"]  # Very similar
        non_duplicate_tokens = ["class", "TYPE_1", ":", "pass"]  # Different
        
        candidates = [
            self.create_mock_block(duplicate_tokens, "dup.py", 10),
            self.create_mock_block(non_duplicate_tokens, "other.py", 20)
        ]
        
        results = batch_verify(target_block, candidates)
        
        # Should find 1 duplicate
        assert len(results) == 1
        candidate_block, result = results[0]
        assert result.is_duplicate is True
        assert candidate_block.normalized_tokens == duplicate_tokens
        
    def test_batch_verify_with_config(self):
        """Test batch verification with custom config."""
        config = EchoConfig(
            overlap_threshold=0.5,
            edit_density_threshold=0.3,
            max_matches_per_block=2
        )
        
        target_tokens = ["def", "func", "(", "x", ")", ":", "return", "x"]
        target_block = self.create_mock_block(target_tokens)
        
        # Create multiple similar candidates
        candidates = [
            self.create_mock_block(["def", "func", "(", "y", ")", ":", "return", "y"], "c1.py", 1),
            self.create_mock_block(["def", "func", "(", "z", ")", ":", "return", "z"], "c2.py", 1), 
            self.create_mock_block(["def", "other", "(", "a", ")", ":", "return", "a"], "c3.py", 1),
        ]
        
        results = batch_verify(target_block, candidates, config)
        
        # Results should be limited by max_matches_per_block and sorted by confidence
        assert len(results) <= config.max_matches_per_block
        if len(results) > 1:
            # Should be sorted by confidence (descending)
            assert results[0][1].similarity_confidence >= results[1][1].similarity_confidence
            
    def test_batch_verify_empty_candidates(self):
        """Test batch verification with empty candidate list."""
        target_block = self.create_mock_block(["def", "func", "(", ")"])
        results = batch_verify(target_block, [])
        assert results == []
        
    def test_batch_verify_self_exclusion(self):
        """Test that batch verification excludes self-comparison."""
        tokens = ["def", "func", "(", ")", "return", "NUM_LIT"]
        target_block = self.create_mock_block(tokens, "test.py", 10)
        
        # Include target block in candidates (should be excluded)
        candidates = [
            target_block,  # Self - should be excluded
            self.create_mock_block(tokens, "other.py", 20),  # Different file - should match
        ]
        
        results = batch_verify(target_block, candidates)
        
        assert len(results) == 1
        assert results[0][0].original.file_path == Path("other.py")


class TestVerificationStatistics:
    """Test verification statistics computation."""
    
    def test_statistics_basic(self):
        """Test basic statistics computation."""
        results = [
            VerificationResult(
                is_duplicate=True, overlap_score=0.8, jaccard_score=0.7,
                edit_distance=2, edit_density=0.2, lcs_length=8,
                token_diff_count=3, similarity_confidence=0.8
            ),
            VerificationResult(
                is_duplicate=False, overlap_score=0.4, jaccard_score=0.3,
                edit_distance=6, edit_density=0.6, lcs_length=4,
                token_diff_count=8, similarity_confidence=0.3
            ),
            VerificationResult(
                is_duplicate=True, overlap_score=0.9, jaccard_score=0.85,
                edit_distance=1, edit_density=0.1, lcs_length=9,
                token_diff_count=2, similarity_confidence=0.9
            )
        ]
        
        stats = get_verification_statistics(results)
        
        assert stats['total_verifications'] == 3
        assert stats['duplicate_count'] == 2
        assert stats['duplicate_rate'] == 2/3
        assert abs(stats['avg_overlap_score'] - (0.8 + 0.4 + 0.9)/3) < 0.001
        assert abs(stats['avg_overlap_score_duplicates'] - (0.8 + 0.9)/2) < 0.001
        
    def test_statistics_empty(self):
        """Test statistics with empty results."""
        stats = get_verification_statistics([])
        assert stats == {}
        
    def test_statistics_no_duplicates(self):
        """Test statistics when no duplicates found."""
        results = [
            VerificationResult(
                is_duplicate=False, overlap_score=0.3, jaccard_score=0.2,
                edit_distance=7, edit_density=0.7, lcs_length=3,
                token_diff_count=10, similarity_confidence=0.2
            )
        ]
        
        stats = get_verification_statistics(results)
        
        assert stats['duplicate_count'] == 0
        assert stats['duplicate_rate'] == 0.0
        assert 'avg_overlap_score_duplicates' not in stats


if __name__ == "__main__":
    pytest.main([__file__])