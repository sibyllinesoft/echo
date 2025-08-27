"""Token-level verification for near-miss duplicate detection.

This module performs detailed token-level comparison to verify if LSH candidates
are truly near-miss duplicates. It implements:
- Efficient dynamic programming algorithms for LCS and edit distance
- Multiple similarity metrics (overlap, Jaccard, edit density)
- Configurable thresholds for accepting duplicates
- Batch verification for performance optimization

The verification stage filters false positives from the LSH candidate generation
and provides detailed similarity metrics for ranking and explanation.

Key Algorithms:
- Token-LCS: Longest Common Subsequence for token sequences using DP
- Edit Distance: Levenshtein distance optimized for token arrays
- Jaccard Similarity: Set-based similarity for additional validation

Acceptance Criteria:
- Primary: overlap_ratio ≥ 0.75 (high similarity threshold)
- Secondary: overlap_ratio ≥ 0.6 AND edit_density ≤ 0.25 (medium similarity with low edits)

Performance Optimizations:
- Early termination for obvious non-matches
- Memory-efficient DP implementations
- Batch processing to amortize overhead
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

from .normalize import NormalizedBlock
from .config import EchoConfig

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of near-miss verification with detailed metrics."""
    is_duplicate: bool
    overlap_score: float          # LCS length / max(len1, len2)
    jaccard_score: float          # |intersection| / |union|
    edit_distance: int            # Levenshtein distance
    edit_density: float           # edit_distance / max(len1, len2)
    lcs_length: int              # Longest common subsequence length
    token_diff_count: int        # Number of different tokens
    similarity_confidence: float  # Combined confidence score [0,1]


def longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """Compute longest common subsequence length using dynamic programming.
    
    Optimized implementation with space complexity O(min(m,n)) instead of O(m*n).
    
    Args:
        seq1: First token sequence
        seq2: Second token sequence
        
    Returns:
        Length of the longest common subsequence
    """
    if not seq1 or not seq2:
        return 0
    
    # Optimize by using shorter sequence for inner loop
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1
    
    m, n = len(seq1), len(seq2)
    
    # Use rolling arrays to optimize space - only need previous row
    prev_row = [0] * (m + 1)
    curr_row = [0] * (m + 1)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq2[i-1] == seq1[j-1]:
                curr_row[j] = prev_row[j-1] + 1
            else:
                curr_row[j] = max(prev_row[j], curr_row[j-1])
        
        # Swap rows for next iteration
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[m]


def compute_edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute Levenshtein edit distance between token sequences.
    
    Optimized implementation with early termination and space optimization.
    Uses Wagner-Fischer algorithm with O(min(m,n)) space complexity.
    
    Args:
        seq1: First token sequence
        seq2: Second token sequence
        
    Returns:
        Minimum edit distance (insertions + deletions + substitutions)
    """
    if not seq1:
        return len(seq2)
    if not seq2:
        return len(seq1)
    
    # Optimize by using shorter sequence for columns
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1
    
    m, n = len(seq1), len(seq2)
    
    # Use two arrays instead of full matrix
    prev_row = list(range(m + 1))
    curr_row = [0] * (m + 1)
    
    for i in range(1, n + 1):
        curr_row[0] = i
        
        for j in range(1, m + 1):
            if seq2[i-1] == seq1[j-1]:
                curr_row[j] = prev_row[j-1]  # No operation needed
            else:
                curr_row[j] = min(
                    prev_row[j] + 1,      # Deletion
                    curr_row[j-1] + 1,    # Insertion  
                    prev_row[j-1] + 1     # Substitution
                )
        
        # Swap rows
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[m]


def compute_jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute Jaccard similarity coefficient between token sequences.
    
    Jaccard = |intersection| / |union|
    
    Args:
        tokens1: First token sequence
        tokens2: Second token sequence
        
    Returns:
        Jaccard similarity coefficient [0.0, 1.0]
    """
    if not tokens1 and not tokens2:
        return 1.0
    
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    intersection_size = len(set1 & set2)
    union_size = len(set1 | set2)
    
    return intersection_size / union_size if union_size > 0 else 0.0


def _compute_similarity_confidence(overlap_score: float, jaccard_score: float, 
                                  edit_density: float) -> float:
    """Compute combined similarity confidence score.
    
    Combines multiple metrics into a single confidence score using
    weighted harmonic mean to penalize outliers.
    
    Args:
        overlap_score: LCS-based overlap ratio
        jaccard_score: Set-based Jaccard similarity  
        edit_density: Edit distance density (lower is better)
        
    Returns:
        Combined confidence score [0.0, 1.0]
    """
    # Convert edit density to similarity (lower density = higher similarity)
    edit_similarity = max(0.0, 1.0 - edit_density)
    
    # Weighted harmonic mean (overlap weighted higher as most important)
    weights = [0.5, 0.3, 0.2]  # overlap, jaccard, edit_similarity
    scores = [overlap_score, jaccard_score, edit_similarity]
    
    # Harmonic mean calculation with weights
    if any(score <= 0 for score in scores):
        return 0.0
    
    weighted_sum = sum(w / s for w, s in zip(weights, scores))
    return sum(weights) / weighted_sum if weighted_sum > 0 else 0.0


def verify_near_miss(block1: NormalizedBlock, block2: NormalizedBlock,
                     overlap_threshold: float = 0.75,
                     edit_density_threshold: float = 0.25,
                     min_confidence: float = 0.7) -> VerificationResult:
    """Verify if two blocks are near-miss duplicates.
    
    Performs detailed token-level comparison using multiple similarity metrics.
    
    Acceptance criteria:
    - Primary: overlap_ratio >= overlap_threshold (default 0.75)
    - Secondary: overlap_ratio >= 0.6 AND edit_density <= edit_density_threshold (default 0.25)
    - Additional: similarity_confidence >= min_confidence for edge cases
    
    Args:
        block1: First normalized block
        block2: Second normalized block  
        overlap_threshold: Minimum overlap ratio for accepting as duplicate (0.75)
        edit_density_threshold: Maximum edit density for secondary criteria (0.25)
        min_confidence: Minimum confidence score for edge cases (0.7)
        
    Returns:
        VerificationResult with is_duplicate decision and detailed metrics
    """
    tokens1 = block1.normalized_tokens
    tokens2 = block2.normalized_tokens
    
    # Early exit for trivial cases
    if not tokens1 and not tokens2:
        logger.debug("Both blocks are empty - considering as duplicates")
        return VerificationResult(
            is_duplicate=True, overlap_score=1.0, jaccard_score=1.0,
            edit_distance=0, edit_density=0.0, lcs_length=0,
            token_diff_count=0, similarity_confidence=1.0
        )
    
    if not tokens1 or not tokens2:
        logger.debug("One block is empty - not duplicates")
        return VerificationResult(
            is_duplicate=False, overlap_score=0.0, jaccard_score=0.0,
            edit_distance=max(len(tokens1), len(tokens2)), edit_density=1.0,
            lcs_length=0, token_diff_count=max(len(tokens1), len(tokens2)),
            similarity_confidence=0.0
        )
    
    # Quick size-based filtering (if one block is much smaller/larger)
    len1, len2 = len(tokens1), len(tokens2)
    size_ratio = min(len1, len2) / max(len1, len2)
    if size_ratio < 0.5:  # One block is less than half the size of the other
        logger.debug(f"Size ratio {size_ratio:.3f} too low - likely not duplicates")
        # Still compute metrics for completeness, but lower confidence
        
    # Compute core similarity metrics
    lcs_length = longest_common_subsequence(tokens1, tokens2)
    edit_distance = compute_edit_distance(tokens1, tokens2)
    jaccard_score = compute_jaccard_similarity(tokens1, tokens2)
    
    # Calculate derived metrics
    max_length = max(len1, len2)
    overlap_score = lcs_length / max_length if max_length > 0 else 0.0
    edit_density = edit_distance / max_length if max_length > 0 else 1.0
    
    # Count different tokens for additional insight
    set1, set2 = set(tokens1), set(tokens2)
    token_diff_count = len(set1.symmetric_difference(set2))
    
    # Compute combined confidence score
    similarity_confidence = _compute_similarity_confidence(
        overlap_score, jaccard_score, edit_density
    )
    
    # Apply acceptance criteria (in order of priority)
    is_duplicate = False
    decision_reason = ""
    
    if overlap_score >= overlap_threshold:
        is_duplicate = True
        decision_reason = f"High overlap: {overlap_score:.3f} >= {overlap_threshold}"
    elif overlap_score >= 0.6 and edit_density <= edit_density_threshold:
        is_duplicate = True  
        decision_reason = f"Medium overlap + low edits: {overlap_score:.3f} >= 0.6 and {edit_density:.3f} <= {edit_density_threshold}"
    elif similarity_confidence >= min_confidence:
        is_duplicate = True
        decision_reason = f"High confidence: {similarity_confidence:.3f} >= {min_confidence}"
    else:
        decision_reason = f"No criteria met: overlap={overlap_score:.3f}, edit_density={edit_density:.3f}, confidence={similarity_confidence:.3f}"
    
    logger.debug(f"Verification: {block1.original.file_path}:{block1.original.start_line} vs "
                f"{block2.original.file_path}:{block2.original.start_line} - {decision_reason}")
    
    return VerificationResult(
        is_duplicate=is_duplicate,
        overlap_score=overlap_score,
        jaccard_score=jaccard_score,
        edit_distance=edit_distance,
        edit_density=edit_density,
        lcs_length=lcs_length,
        token_diff_count=token_diff_count,
        similarity_confidence=similarity_confidence
    )


def batch_verify(target_block: NormalizedBlock, 
                candidate_blocks: List[NormalizedBlock],
                config: Optional[EchoConfig] = None) -> List[Tuple[NormalizedBlock, VerificationResult]]:
    """Verify multiple candidate blocks against a target block.
    
    Performs batch verification with optimizations and configurable thresholds.
    
    Args:
        target_block: The target block to compare against
        candidate_blocks: List of candidate blocks from LSH stage
        config: Configuration object with thresholds (optional)
        
    Returns:
        List of (block, result) tuples for verified duplicates only
    """
    if config is None:
        config = EchoConfig()  # Use default configuration
    
    if not candidate_blocks:
        logger.debug("No candidate blocks provided for verification")
        return []
    
    logger.debug(f"Batch verifying {len(candidate_blocks)} candidates against target "
                f"{target_block.original.file_path}:{target_block.original.start_line}")
    
    results = []
    verification_count = 0
    duplicate_count = 0
    
    for candidate in candidate_blocks:
        # Skip self-comparison
        if (candidate.original.file_path == target_block.original.file_path and 
            candidate.original.start_line == target_block.original.start_line):
            continue
        
        verification_count += 1
        result = verify_near_miss(
            target_block, candidate,
            overlap_threshold=config.overlap_threshold,
            edit_density_threshold=config.edit_density_threshold
        )
        
        if result.is_duplicate:
            duplicate_count += 1
            results.append((candidate, result))
    
    # Sort by similarity confidence for best matches first
    results.sort(key=lambda x: x[1].similarity_confidence, reverse=True)
    
    # Apply max matches limit if configured
    if hasattr(config, 'max_matches_per_block') and config.max_matches_per_block > 0:
        results = results[:config.max_matches_per_block]
    
    logger.info(f"Verified {duplicate_count}/{verification_count} candidates as duplicates "
               f"for target {target_block.original.file_path}:{target_block.original.start_line}")
    
    return results


def verify_candidate_pairs(candidate_pairs: List[Tuple[NormalizedBlock, NormalizedBlock]],
                          config: Optional[EchoConfig] = None) -> List[Tuple[NormalizedBlock, NormalizedBlock, VerificationResult]]:
    """Verify a list of candidate pairs for duplicates.
    
    Useful for processing LSH results in batch where candidates are already paired.
    
    Args:
        candidate_pairs: List of (block1, block2) tuples to verify
        config: Configuration object with thresholds (optional)
        
    Returns:
        List of (block1, block2, result) tuples for verified duplicates only
    """
    if config is None:
        config = EchoConfig()
    
    if not candidate_pairs:
        logger.debug("No candidate pairs provided for verification")
        return []
    
    logger.debug(f"Verifying {len(candidate_pairs)} candidate pairs")
    
    results = []
    duplicate_count = 0
    
    for block1, block2 in candidate_pairs:
        result = verify_near_miss(
            block1, block2,
            overlap_threshold=config.overlap_threshold,
            edit_density_threshold=config.edit_density_threshold
        )
        
        if result.is_duplicate:
            duplicate_count += 1
            results.append((block1, block2, result))
    
    logger.info(f"Verified {duplicate_count}/{len(candidate_pairs)} pairs as duplicates")
    
    return results


def get_verification_statistics(results: List[VerificationResult]) -> Dict[str, float]:
    """Compute statistics for a set of verification results.
    
    Args:
        results: List of verification results
        
    Returns:
        Dictionary with statistical metrics
    """
    if not results:
        return {}
    
    duplicate_results = [r for r in results if r.is_duplicate]
    
    stats = {
        'total_verifications': len(results),
        'duplicate_count': len(duplicate_results),
        'duplicate_rate': len(duplicate_results) / len(results),
        'avg_overlap_score': sum(r.overlap_score for r in results) / len(results),
        'avg_jaccard_score': sum(r.jaccard_score for r in results) / len(results),
        'avg_edit_density': sum(r.edit_density for r in results) / len(results),
        'avg_confidence': sum(r.similarity_confidence for r in results) / len(results),
    }
    
    if duplicate_results:
        stats.update({
            'avg_overlap_score_duplicates': sum(r.overlap_score for r in duplicate_results) / len(duplicate_results),
            'avg_jaccard_score_duplicates': sum(r.jaccard_score for r in duplicate_results) / len(duplicate_results),
            'avg_edit_density_duplicates': sum(r.edit_density for r in duplicate_results) / len(duplicate_results),
            'avg_confidence_duplicates': sum(r.similarity_confidence for r in duplicate_results) / len(duplicate_results),
        })
    
    return stats