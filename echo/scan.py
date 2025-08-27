"""Main orchestrator for duplicate code detection pipeline."""

import hashlib
import logging
import math
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from .config import EchoConfig
from .embed import SemanticEmbedder, compute_semantic_similarity
from .lsh import MinHashLSH, create_lsh_from_config
from .normalize import NormalizedBlock, normalize_blocks
from .parser import CodeBlock, extract_blocks
from .storage import BlockRecord, DuplicateFinding, EchoDatabase, create_database
from .verify import VerificationResult, verify_near_miss

logger = logging.getLogger(__name__)


@dataclass
class ScanStatistics:
    """Statistics from a duplicate detection scan."""

    files_processed: int = 0
    blocks_extracted: int = 0
    blocks_normalized: int = 0
    blocks_indexed: int = 0
    lsh_queries: int = 0
    candidates_generated: int = 0
    near_miss_verifications: int = 0
    semantic_comparisons: int = 0
    findings_generated: int = 0
    findings_filtered: int = 0
    errors: int = 0
    skipped_files: int = 0
    scan_time_ms: int = 0
    stage_times_ms: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "files_processed": self.files_processed,
            "blocks_extracted": self.blocks_extracted,
            "blocks_normalized": self.blocks_normalized,
            "blocks_indexed": self.blocks_indexed,
            "lsh_queries": self.lsh_queries,
            "candidates_generated": self.candidates_generated,
            "near_miss_verifications": self.near_miss_verifications,
            "semantic_comparisons": self.semantic_comparisons,
            "findings_generated": self.findings_generated,
            "findings_filtered": self.findings_filtered,
            "errors": self.errors,
            "skipped_files": self.skipped_files,
            "scan_time_ms": self.scan_time_ms,
            "stage_times_ms": self.stage_times_ms,
        }


@dataclass
class ScanResult:
    """Result of a duplicate detection scan."""

    findings: List[DuplicateFinding]
    statistics: ScanStatistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "findings": [finding.__dict__ for finding in self.findings],
            "statistics": self.statistics.to_dict(),
        }


class DuplicateScanner:
    """Main orchestrator for duplicate detection pipeline.

    Coordinates the complete duplicate detection process through multiple stages:
    1. S1 LSH candidate generation: shingle 5-grams, MinHash, LSH bands
    2. S2 Near-miss verification: token-LCS with overlap ≥0.75 OR (≥0.6 & edit density ≤0.25)
    3. S3 Semantic reranking: embed with GraphCodeBERT-mini; accept if cosine ≥ τ_sem
    """

    def __init__(
        self,
        config: EchoConfig,
        database: EchoDatabase,
        lsh: Optional[MinHashLSH] = None,
        semantic_embedder: Optional[SemanticEmbedder] = None,
    ):
        self.config = config
        self.database = database
        self.lsh = lsh or create_lsh_from_config(config, database.session)
        self.semantic_embedder = semantic_embedder
        self._block_cache: Dict[str, NormalizedBlock] = {}
        self._progress_callback: Optional[callable] = None

        logger.info(
            f"Initialized DuplicateScanner with config: "
            f"min_tokens={config.min_tokens}, "
            f"tau_semantic={config.tau_semantic}, "
            f"max_candidates={config.max_candidates}"
        )

    def set_progress_callback(self, callback: callable) -> None:
        """Set callback for progress reporting."""
        self._progress_callback = callback

    def scan_repository(
        self, repo_path: Path, budget_ms: Optional[int] = None
    ) -> ScanResult:
        """Scan entire repository for duplicates.

        Performs comprehensive repository-wide scan with optional time budget.

        Args:
            repo_path: Root path of repository to scan
            budget_ms: Optional time budget in milliseconds

        Returns:
            ScanResult with findings and statistics
        """
        start_time = time.time()
        stats = ScanStatistics()

        try:
            logger.info(f"Starting repository scan of {repo_path}")
            if budget_ms:
                logger.info(f"Time budget: {budget_ms}ms")

            # Extract all code blocks from repository
            stage_start = time.time()
            all_blocks = self._extract_repository_blocks(repo_path, stats)
            stats.stage_times_ms["extraction"] = int((time.time() - stage_start) * 1000)

            if not all_blocks:
                logger.warning("No code blocks extracted from repository")
                stats.scan_time_ms = int((time.time() - start_time) * 1000)
                return ScanResult(findings=[], statistics=stats)

            # Check time budget after extraction
            if budget_ms and (time.time() - start_time) * 1000 > budget_ms * 0.8:
                logger.warning("Approaching time budget after extraction phase")

            # Run the complete detection pipeline
            findings = self._run_detection_pipeline(
                all_blocks, stats, budget_ms, start_time
            )

            stats.scan_time_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Repository scan completed in {stats.scan_time_ms}ms. "
                f"Found {len(findings)} duplicate findings."
            )

            return ScanResult(findings=findings, statistics=stats)

        except Exception as e:
            logger.error(f"Repository scan failed: {e}")
            stats.errors += 1
            stats.scan_time_ms = int((time.time() - start_time) * 1000)
            return ScanResult(findings=[], statistics=stats)

    def scan_changed_files(self, file_paths: List[Path]) -> ScanResult:
        """Scan only changed files for duplicates (incremental scan).

        Optimized for changed files - queries existing index for matches
        and updates index with new/modified blocks.

        Args:
            file_paths: List of file paths that have changed

        Returns:
            ScanResult with findings and statistics
        """
        start_time = time.time()
        stats = ScanStatistics()

        try:
            logger.info(f"Starting incremental scan of {len(file_paths)} files")

            # Extract blocks only from changed files
            stage_start = time.time()
            changed_blocks = self._extract_blocks_from_files(file_paths, stats)
            stats.stage_times_ms["extraction"] = int((time.time() - stage_start) * 1000)

            if not changed_blocks:
                logger.info("No code blocks extracted from changed files")
                stats.scan_time_ms = int((time.time() - start_time) * 1000)
                return ScanResult(findings=[], statistics=stats)

            # Load LSH index from database if not already loaded
            stage_start = time.time()
            self.lsh.load_from_database()
            stats.stage_times_ms["lsh_load"] = int((time.time() - stage_start) * 1000)

            # Run detection pipeline on changed blocks only
            findings = self._run_detection_pipeline(changed_blocks, stats)

            # Update the index with new/modified blocks
            stage_start = time.time()
            self._update_index_with_blocks(changed_blocks, stats)
            stats.stage_times_ms["index_update"] = int(
                (time.time() - stage_start) * 1000
            )

            stats.scan_time_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Incremental scan completed in {stats.scan_time_ms}ms. "
                f"Found {len(findings)} duplicate findings."
            )

            return ScanResult(findings=findings, statistics=stats)

        except Exception as e:
            logger.error(f"Incremental scan failed: {e}")
            stats.errors += 1
            stats.scan_time_ms = int((time.time() - start_time) * 1000)
            return ScanResult(findings=[], statistics=stats)

    def _extract_repository_blocks(
        self, repo_path: Path, stats: ScanStatistics
    ) -> List[NormalizedBlock]:
        """Extract and normalize all code blocks from repository."""
        all_blocks = []

        # Find all supported files
        file_paths = []
        for pattern in [
            "**/*.py",
            "**/*.js",
            "**/*.ts",
            "**/*.java",
            "**/*.cpp",
            "**/*.c",
        ]:
            file_paths.extend(repo_path.glob(pattern))

        # Filter out ignored patterns
        filtered_paths = []
        for path in file_paths:
            if not any(pattern in str(path) for pattern in self.config.ignore_patterns):
                filtered_paths.append(path)

        logger.info(f"Found {len(filtered_paths)} files to process")

        return self._extract_blocks_from_files(filtered_paths, stats)

    def _extract_blocks_from_files(
        self, file_paths: List[Path], stats: ScanStatistics
    ) -> List[NormalizedBlock]:
        """Extract and normalize code blocks from list of files."""
        all_blocks = []

        for file_path in file_paths:
            try:
                if self._progress_callback:
                    self._progress_callback(f"Processing {file_path}")

                # Extract raw blocks
                raw_blocks = extract_blocks(file_path)
                stats.files_processed += 1
                stats.blocks_extracted += len(raw_blocks)

                # Filter blocks by minimum token count and suppress single-line
                filtered_blocks = []
                for block in raw_blocks:
                    if (
                        block.token_count >= self.config.min_tokens and block.lines > 1
                    ):  # Suppress single-line matches
                        filtered_blocks.append(block)

                if filtered_blocks:
                    # Normalize blocks
                    normalized_blocks = normalize_blocks(filtered_blocks)
                    stats.blocks_normalized += len(normalized_blocks)
                    all_blocks.extend(normalized_blocks)

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                stats.errors += 1
                stats.skipped_files += 1

        logger.info(
            f"Extracted {len(all_blocks)} normalized blocks from {stats.files_processed} files"
        )
        return all_blocks

    def _update_index_with_blocks(
        self, blocks: List[NormalizedBlock], stats: ScanStatistics
    ) -> None:
        """Update LSH index with new blocks."""
        try:
            for block in blocks:
                self.lsh.add_block(block)
                stats.blocks_indexed += 1
        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            stats.errors += 1

    def _run_detection_pipeline(
        self,
        blocks: List[NormalizedBlock],
        stats: ScanStatistics,
        budget_ms: Optional[int] = None,
        start_time: Optional[float] = None,
    ) -> List[DuplicateFinding]:
        """Run the multi-stage detection pipeline.

        Pipeline stages:
        1. S1 LSH candidate generation: shingle 5-grams, MinHash, LSH bands
        2. S2 Near-miss verification: token-LCS; accept if overlap ≥0.75 OR (≥0.6 & edit density ≤0.25)
        3. S3 Semantic reranking: embed with GraphCodeBERT-mini; accept if cosine ≥ τ_sem
        """
        findings = []
        stage_start = time.time()

        logger.info(f"Running detection pipeline on {len(blocks)} blocks")

        try:
            # Process blocks and collect all findings
            for i, block in enumerate(blocks):
                if self._progress_callback and i % 50 == 0:
                    self._progress_callback(f"Processing block {i + 1}/{len(blocks)}")

                # Check time budget
                if budget_ms and start_time:
                    elapsed_ms = (time.time() - start_time) * 1000
                    if elapsed_ms > budget_ms * 0.95:
                        logger.warning(
                            f"Approaching time budget at block {i}/{len(blocks)}"
                        )
                        break

                block_findings = self._process_block(block, stats)
                findings.extend(block_findings)

            stats.stage_times_ms["pipeline"] = int((time.time() - stage_start) * 1000)

            # Final filtering and ranking
            stage_start = time.time()
            filtered_findings = self._final_filter_and_rank(findings, stats)
            stats.stage_times_ms["final_filter"] = int(
                (time.time() - stage_start) * 1000
            )

            logger.info(
                f"Pipeline generated {len(findings)} findings, filtered to {len(filtered_findings)}"
            )
            return filtered_findings

        except Exception as e:
            logger.error(f"Detection pipeline failed: {e}")
            logger.error(traceback.format_exc())
            stats.errors += 1
            return findings

    def _process_block(
        self, block: NormalizedBlock, stats: ScanStatistics
    ) -> List[DuplicateFinding]:
        """Process a single block through all pipeline stages."""
        block_findings = []

        try:
            # Stage 1: LSH candidate generation
            candidates = self._generate_lsh_candidates(block, stats)

            # Stage 2: Near-miss verification
            verified_candidates = self._verify_near_miss_candidates(
                block, candidates, stats
            )

            # Stage 3: Semantic reranking (if enabled)
            if self.config.tau_semantic > 0 and self.semantic_embedder:
                verified_candidates = self._semantic_rerank_candidates(
                    block, verified_candidates, stats
                )

            # Convert to findings and apply per-block filtering
            block_findings = self._create_findings_from_candidates(
                block, verified_candidates, stats
            )

        except Exception as e:
            logger.warning(
                f"Failed to process block {block.original.file_path}:{block.original.start_line}: {e}"
            )
            stats.errors += 1

        return block_findings

    def _generate_lsh_candidates(
        self, block: NormalizedBlock, stats: ScanStatistics
    ) -> List[str]:
        """Stage 1: Generate LSH candidates using shingle 5-grams, MinHash, LSH bands."""
        try:
            # Add block to LSH if not already present
            if not self.lsh.get_block_signature(block):
                self.lsh.add_block(block)
                stats.blocks_indexed += 1

            # Query for candidates
            candidates = self.lsh.query_candidates(
                block, topk=self.config.max_candidates
            )
            stats.lsh_queries += 1
            stats.candidates_generated += len(candidates)

            # Filter out self-matches
            block_hash = self._get_block_hash(block)
            candidates = [c for c in candidates if c != block_hash]

            return candidates

        except Exception as e:
            logger.warning(f"LSH candidate generation failed: {e}")
            stats.errors += 1
            return []

    def _verify_near_miss_candidates(
        self, block: NormalizedBlock, candidates: List[str], stats: ScanStatistics
    ) -> List[Tuple[str, VerificationResult]]:
        """Stage 2: Near-miss verification using token-LCS."""
        verified = []

        for candidate_id in candidates:
            try:
                candidate_block = self._load_candidate_block(candidate_id)
                if candidate_block is None:
                    continue

                # Verify using near-miss criteria
                verification_result = verify_near_miss(
                    block,
                    candidate_block,
                    overlap_threshold=self.config.overlap_threshold,
                    edit_density_threshold=self.config.edit_density_threshold,
                )

                stats.near_miss_verifications += 1

                if verification_result.is_duplicate:
                    verified.append((candidate_id, verification_result))

            except Exception as e:
                logger.warning(
                    f"Near-miss verification failed for candidate {candidate_id}: {e}"
                )
                stats.errors += 1

        return verified

    def _semantic_rerank_candidates(
        self,
        block: NormalizedBlock,
        verified_candidates: List[Tuple[str, VerificationResult]],
        stats: ScanStatistics,
    ) -> List[Tuple[str, VerificationResult]]:
        """Stage 3: Semantic reranking with GraphCodeBERT-mini embeddings."""
        if not self.semantic_embedder:
            logger.warning("Semantic reranking requested but no embedder available")
            return verified_candidates

        semantic_verified = []

        for candidate_id, verification_result in verified_candidates:
            try:
                candidate_block = self._load_candidate_block(candidate_id)
                if candidate_block is None:
                    continue

                # Compute semantic similarity
                semantic_similarity = compute_semantic_similarity(
                    block, candidate_block, self.semantic_embedder
                )

                stats.semantic_comparisons += 1

                # Accept if semantic similarity >= threshold
                if semantic_similarity >= self.config.tau_semantic:
                    # Update verification result with semantic score
                    verification_result.semantic_similarity = semantic_similarity
                    semantic_verified.append((candidate_id, verification_result))

            except Exception as e:
                logger.warning(
                    f"Semantic reranking failed for candidate {candidate_id}: {e}"
                )
                stats.errors += 1

        return semantic_verified

    def _create_findings_from_candidates(
        self,
        block: NormalizedBlock,
        verified_candidates: List[Tuple[str, VerificationResult]],
        stats: ScanStatistics,
    ) -> List[DuplicateFinding]:
        """Convert verified candidates to duplicate findings with scoring."""
        findings = []

        # Compute refactor-worthiness scores and create findings
        for candidate_id, verification_result in verified_candidates:
            try:
                # Create finding object
                finding_id = self._generate_finding_id(block, candidate_id)
                block_id = self._get_block_hash(block)

                # Compute refactor score
                refactor_score = self._compute_refactor_score(
                    block, len(verified_candidates)
                )

                # Collect all scores
                scores = {
                    "overlap_score": verification_result.overlap_score,
                    "jaccard_score": verification_result.jaccard_score,
                    "edit_density": verification_result.edit_density,
                    "similarity_confidence": verification_result.similarity_confidence,
                }

                if hasattr(verification_result, "semantic_similarity"):
                    scores["semantic_similarity"] = (
                        verification_result.semantic_similarity
                    )

                finding = DuplicateFinding(
                    id=finding_id,
                    block_id=block_id,
                    match_block_id=candidate_id,
                    scores=scores,
                    type=(
                        "near_miss"
                        if not hasattr(verification_result, "semantic_similarity")
                        else "semantic"
                    ),
                    confidence=verification_result.similarity_confidence,
                    refactor_score=refactor_score,
                )

                findings.append(finding)
                stats.findings_generated += 1

            except Exception as e:
                logger.warning(f"Failed to create finding: {e}")
                stats.errors += 1

        return findings

    def _final_filter_and_rank(
        self, findings: List[DuplicateFinding], stats: ScanStatistics
    ) -> List[DuplicateFinding]:
        """Apply final filtering and ranking heuristics."""
        if not findings:
            return findings

        # Apply refactor-worthiness filtering (R ≥ 200)
        filtered_findings = [
            f for f in findings if f.refactor_score >= self.config.min_refactor_score
        ]

        # Group by block and limit to k=5 matches per block
        findings_by_block = defaultdict(list)
        for finding in filtered_findings:
            findings_by_block[finding.block_id].append(finding)

        final_findings = []
        for block_id, block_findings in findings_by_block.items():
            # Sort by refactor score (descending) and take top k
            sorted_findings = sorted(
                block_findings, key=lambda f: f.refactor_score, reverse=True
            )
            final_findings.extend(sorted_findings[: self.config.max_matches_per_block])

        stats.findings_filtered = len(findings) - len(final_findings)

        return final_findings

    def _compute_refactor_score(
        self, block: NormalizedBlock, duplicate_count: int
    ) -> float:
        """Compute refactor-worthiness score R.

        Formula: R = tokens * (1 + log1p(dup_count)) * dispersion_factor
        """
        # Basic formula without dispersion for now
        # TODO: Implement dispersion factor based on file/directory spread
        dispersion_factor = 1.0

        if duplicate_count > 1:
            # More duplicates = higher refactor value
            dispersion_factor = 1.0 + (duplicate_count - 1) * 0.1

        score = (
            block.token_count * (1 + math.log1p(duplicate_count)) * dispersion_factor
        )
        return score

    def _load_candidate_block(self, candidate_id: str) -> Optional[NormalizedBlock]:
        """Load a candidate block from cache or database."""
        # Check cache first
        if candidate_id in self._block_cache:
            return self._block_cache[candidate_id]

        # Load from database
        try:
            # Query database for block record
            block_record = (
                self.database.session.query(BlockRecord)
                .filter_by(hash_signature=candidate_id)
                .first()
            )

            if block_record:
                # Reconstruct normalized block
                # This is a simplified reconstruction - in practice might need more data
                raw_block = CodeBlock(
                    lang=block_record.language,
                    file_path=Path(block_record.file_path),
                    start_line=block_record.start_line,
                    end_line=block_record.end_line,
                    start_byte=0,  # Not stored in DB
                    end_byte=0,  # Not stored in DB
                    tokens=block_record.tokens.split() if block_record.tokens else [],
                    raw_content="",  # Not stored in DB
                    node_type=block_record.node_type or "unknown",
                )

                normalized_block = NormalizedBlock(
                    original=raw_block,
                    normalized_tokens=(
                        block_record.normalized_tokens.split()
                        if block_record.normalized_tokens
                        else []
                    ),
                    token_mapping={},  # Not stored in DB
                    reverse_mapping={},  # Not stored in DB
                    hash_signature=candidate_id,
                )

                # Cache for future use
                self._block_cache[candidate_id] = normalized_block
                return normalized_block

        except Exception as e:
            logger.warning(f"Failed to load candidate block {candidate_id}: {e}")

        return None

    def _get_block_hash(self, block: NormalizedBlock) -> str:
        """Generate a unique hash for a normalized block."""
        if hasattr(block, "hash_signature") and block.hash_signature:
            return block.hash_signature

        # Generate hash from normalized tokens
        token_string = " ".join(block.normalized_tokens)
        return hashlib.sha256(token_string.encode()).hexdigest()[:16]

    def _generate_finding_id(self, block: NormalizedBlock, candidate_id: str) -> str:
        """Generate unique finding ID."""
        block_id = self._get_block_hash(block)
        combined = f"{block_id}:{candidate_id}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


# Convenience functions for the public API
def scan_repository(
    repo_path: Path,
    config: Optional[EchoConfig] = None,
    budget_ms: Optional[int] = None,
    progress_callback: Optional[callable] = None,
) -> ScanResult:
    """Scan a repository for duplicate code.

    High-level API function that sets up the complete pipeline and scans
    a repository for duplicate code blocks.

    Args:
        repo_path: Path to repository root
        config: Configuration object (uses defaults if None)
        budget_ms: Optional time budget in milliseconds
        progress_callback: Optional progress callback function

    Returns:
        ScanResult with findings and statistics
    """
    if config is None:
        config = EchoConfig()

    # Initialize database
    cache_dir = config.get_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    db_path = cache_dir / "echo.db"

    database = create_database(db_path)

    try:
        # Initialize scanner with LSH
        lsh = create_lsh_from_config(config, database.session)
        scanner = DuplicateScanner(config, database, lsh)

        if progress_callback:
            scanner.set_progress_callback(progress_callback)

        # Run repository scan
        result = scanner.scan_repository(repo_path, budget_ms)

        logger.info(
            f"Repository scan completed: {len(result.findings)} findings in {result.statistics.scan_time_ms}ms"
        )
        return result

    except Exception as e:
        logger.error(f"Repository scan failed: {e}")
        # Return empty result with error statistics
        stats = ScanStatistics()
        stats.errors = 1
        stats.scan_time_ms = 0
        return ScanResult(findings=[], statistics=stats)
    finally:
        database.session.close()


def scan_changed_files(
    file_paths: List[Path],
    config: Optional[EchoConfig] = None,
    progress_callback: Optional[callable] = None,
) -> ScanResult:
    """Scan changed files for duplicate code (incremental scan).

    High-level API function for incremental scanning of changed files.
    Optimized for CI/CD workflows where only specific files have changed.

    Args:
        file_paths: List of file paths that have changed
        config: Configuration object (uses defaults if None)
        progress_callback: Optional progress callback function

    Returns:
        ScanResult with findings and statistics
    """
    if config is None:
        config = EchoConfig()

    # Initialize database
    cache_dir = config.get_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    db_path = cache_dir / "echo.db"

    database = create_database(db_path)

    try:
        # Initialize scanner with LSH (will load existing index)
        lsh = create_lsh_from_config(config, database.session)
        scanner = DuplicateScanner(config, database, lsh)

        if progress_callback:
            scanner.set_progress_callback(progress_callback)

        # Run incremental scan
        result = scanner.scan_changed_files(file_paths)

        logger.info(
            f"Incremental scan completed: {len(result.findings)} findings in {result.statistics.scan_time_ms}ms"
        )
        return result

    except Exception as e:
        logger.error(f"Incremental scan failed: {e}")
        # Return empty result with error statistics
        stats = ScanStatistics()
        stats.errors = 1
        stats.scan_time_ms = 0
        return ScanResult(findings=[], statistics=stats)
    finally:
        database.session.close()


def scan_files_batch(
    file_paths: List[Path], config: Optional[EchoConfig] = None, max_workers: int = 4
) -> ScanResult:
    """Scan a batch of files with parallel processing.

    Uses ThreadPoolExecutor for parallel file processing to improve
    performance on multi-core systems.

    Args:
        file_paths: List of file paths to scan
        config: Configuration object (uses defaults if None)
        max_workers: Maximum number of worker threads

    Returns:
        ScanResult with findings and statistics
    """
    if config is None:
        config = EchoConfig()

    start_time = time.time()
    stats = ScanStatistics()
    all_findings = []

    # Initialize database
    cache_dir = config.get_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    db_path = cache_dir / "echo.db"

    database = create_database(db_path)

    try:
        # Process files in batches with ThreadPoolExecutor
        chunk_size = max(1, len(file_paths) // max_workers)
        file_chunks = [
            file_paths[i: i + chunk_size]
            for i in range(0, len(file_paths), chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {}

            for chunk in file_chunks:
                # Create separate scanner for each thread
                lsh = create_lsh_from_config(config, database.session)
                scanner = DuplicateScanner(config, database, lsh)

                future = executor.submit(scanner.scan_changed_files, chunk)
                future_to_chunk[future] = chunk

            # Collect results
            for future in as_completed(future_to_chunk):
                try:
                    chunk_result = future.result()
                    all_findings.extend(chunk_result.findings)

                    # Aggregate statistics
                    chunk_stats = chunk_result.statistics
                    stats.files_processed += chunk_stats.files_processed
                    stats.blocks_extracted += chunk_stats.blocks_extracted
                    stats.blocks_normalized += chunk_stats.blocks_normalized
                    stats.blocks_indexed += chunk_stats.blocks_indexed
                    stats.lsh_queries += chunk_stats.lsh_queries
                    stats.candidates_generated += chunk_stats.candidates_generated
                    stats.near_miss_verifications += chunk_stats.near_miss_verifications
                    stats.semantic_comparisons += chunk_stats.semantic_comparisons
                    stats.findings_generated += chunk_stats.findings_generated
                    stats.findings_filtered += chunk_stats.findings_filtered
                    stats.errors += chunk_stats.errors
                    stats.skipped_files += chunk_stats.skipped_files

                except Exception as e:
                    logger.error(f"Batch processing failed for chunk: {e}")
                    stats.errors += 1

        stats.scan_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Batch scan completed: {len(all_findings)} findings in {stats.scan_time_ms}ms"
        )
        return ScanResult(findings=all_findings, statistics=stats)

    except Exception as e:
        logger.error(f"Batch scan failed: {e}")
        stats.errors += 1
        stats.scan_time_ms = int((time.time() - start_time) * 1000)
        return ScanResult(findings=[], statistics=stats)
    finally:
        database.session.close()


# Additional utility functions
def create_scanner(
    config: Optional[EchoConfig] = None, db_path: Optional[Path] = None
) -> DuplicateScanner:
    """Create a DuplicateScanner instance with proper initialization.

    Utility function for creating scanner instances with consistent setup.

    Args:
        config: Configuration object (uses defaults if None)
        db_path: Database path (uses config cache dir if None)

    Returns:
        Initialized DuplicateScanner instance
    """
    if config is None:
        config = EchoConfig()

    if db_path is None:
        cache_dir = config.get_cache_dir()
        cache_dir.mkdir(exist_ok=True)
        db_path = cache_dir / "echo.db"

    database = create_database(db_path)
    lsh = create_lsh_from_config(config, database.session)

    return DuplicateScanner(config, database, lsh)


def scan_with_semantic_reranking(
    repo_path: Path,
    config: Optional[EchoConfig] = None,
    embedder: Optional[SemanticEmbedder] = None,
) -> ScanResult:
    """Scan repository with semantic reranking enabled.

    Convenience function that ensures semantic reranking is enabled.

    Args:
        repo_path: Path to repository root
        config: Configuration object (enables semantic if None)
        embedder: Semantic embedder instance (creates default if None)

    Returns:
        ScanResult with semantic-enhanced findings
    """
    if config is None:
        config = EchoConfig()
        config.tau_semantic = 0.83  # Enable semantic reranking

    if config.tau_semantic <= 0:
        config.tau_semantic = 0.83  # Ensure semantic reranking is enabled

    # Initialize components
    cache_dir = config.get_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    db_path = cache_dir / "echo.db"

    database = create_database(db_path)
    lsh = create_lsh_from_config(config, database.session)

    if embedder is None:
        embedder = SemanticEmbedder()  # Create default embedder

    scanner = DuplicateScanner(config, database, lsh, embedder)

    try:
        result = scanner.scan_repository(repo_path)
        logger.info(
            f"Semantic-enhanced scan completed: {len(result.findings)} findings"
        )
        return result
    finally:
        database.session.close()
