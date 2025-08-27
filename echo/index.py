"""File indexing and block extraction coordination."""

import hashlib
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pathspec

from .config import EchoConfig
from .normalize import NormalizedBlock, normalize_blocks
from .parser import CodeBlock, detect_language, extract_blocks
from .storage import EchoDatabase, create_database

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result of repository indexing."""

    files_processed: int
    files_skipped: int
    blocks_extracted: int
    errors: List[str]
    processing_time_ms: int


class RepositoryIndexer:
    """Handles repository file scanning and indexing with comprehensive ignore support."""

    def __init__(self, config: EchoConfig, database: EchoDatabase):
        self.config = config
        self.database = database
        self.ignore_spec: Optional[pathspec.PathSpec] = None

    def _build_ignore_spec(self, repo_path: Path) -> pathspec.PathSpec:
        """Build pathspec for ignored files from .gitignore, .dupesignore, and config."""
        patterns = []

        # Default patterns from spec
        default_patterns = [
            "tests/",
            "test/",
            "__tests__/",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "node_modules/",
            ".npm/",
            "npm-debug.log*",
            "vendor/",
            "vendors/",
            "migrations/",
            "migrate/",
            "*.min.js",
            "*.min.css",
            "*.bundle.js",
            "*.bundle.css",
            ".git/",
            ".svn/",
            ".hg/",
            ".bzr/",
            "build/",
            "dist/",
            "target/",
            "out/",
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
            ".DS_Store",
            "Thumbs.db",
            "*.log",
            "*.tmp",
            "*.temp",
            "coverage/",
            ".coverage",
            ".nyc_output/",
            "*.generated.*",
            "*_pb2.py",
            "*.pb.go",
        ]
        patterns.extend(default_patterns)

        # Add patterns from config
        patterns.extend(self.config.ignore_patterns)

        # Read .gitignore
        gitignore_path = repo_path / ".gitignore"
        if gitignore_path.exists():
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    git_patterns = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
                    patterns.extend(git_patterns)
                    logger.debug(f"Loaded {len(git_patterns)} patterns from .gitignore")
            except Exception as e:
                logger.warning(f"Could not read .gitignore: {e}")

        # Read .dupesignore (takes precedence)
        dupesignore_path = repo_path / ".dupesignore"
        if dupesignore_path.exists():
            try:
                with open(dupesignore_path, "r", encoding="utf-8") as f:
                    dupes_patterns = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
                    patterns.extend(dupes_patterns)
                    logger.debug(
                        f"Loaded {len(dupes_patterns)} patterns from .dupesignore"
                    )
            except Exception as e:
                logger.warning(f"Could not read .dupesignore: {e}")

        # Store effective patterns in database metadata
        self.database.set_metadata("ignore_patterns", "\n".join(patterns))

        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def index_repository(
        self, repo_path: Path, reindex: bool = False
    ) -> IndexingResult:
        """Index all files in repository with incremental support."""
        start_time = datetime.now()
        logger.info(f"Starting repository index: {repo_path} (reindex={reindex})")

        # Build ignore specification
        self.ignore_spec = self._build_ignore_spec(repo_path)

        files_processed = 0
        files_skipped = 0
        blocks_extracted = 0
        errors = []

        try:
            # Scan directory tree for supported files
            all_files = self._scan_directory(repo_path)
            logger.info(f"Found {len(all_files)} candidate files")

            # Filter files to process
            files_to_process = []
            for file_path in all_files:
                if self._should_skip_file(file_path):
                    files_skipped += 1
                    continue

                # Check if file needs reindexing
                if not reindex and not self._file_needs_reindex(file_path):
                    files_skipped += 1
                    continue

                files_to_process.append(file_path)

            logger.info(f"Processing {len(files_to_process)} files")

            # Process files in batches for better performance
            batch_size = 50
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i: i + batch_size]
                batch_result = self._process_file_batch(batch)

                files_processed += batch_result[0]
                blocks_extracted += batch_result[1]
                errors.extend(batch_result[2])

                # Progress reporting
                if i % (batch_size * 5) == 0:
                    progress = min(100, (i + batch_size) * 100 // len(files_to_process))
                    logger.info(
                        f"Indexing progress: {progress}% ({i + batch_size}/{len(files_to_process)} files)"
                    )

        except Exception as e:
            logger.error(f"Repository indexing failed: {e}")
            errors.append(f"Repository indexing failed: {e}")

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Update last indexed timestamp
        self.database.set_metadata("last_indexed", datetime.utcnow().isoformat())

        result = IndexingResult(
            files_processed=files_processed,
            files_skipped=files_skipped,
            blocks_extracted=blocks_extracted,
            errors=errors,
            processing_time_ms=processing_time,
        )

        logger.info(
            f"Repository indexing complete: {files_processed} files, "
            f"{blocks_extracted} blocks, {len(errors)} errors in {processing_time}ms"
        )

        return result

    def _scan_directory(self, repo_path: Path) -> List[Path]:
        """Recursively scan directory for code files."""
        files = []

        try:
            for root, dirs, filenames in os.walk(repo_path):
                root_path = Path(root)

                # Skip directories that match ignore patterns
                dirs[:] = [
                    d for d in dirs if not self._should_skip_directory(root_path / d)
                ]

                for filename in filenames:
                    file_path = root_path / filename

                    # Basic file filtering
                    if self._has_supported_extension(file_path):
                        files.append(file_path)

        except Exception as e:
            logger.error(f"Error scanning directory {repo_path}: {e}")

        return files

    def _should_skip_directory(self, dir_path: Path) -> bool:
        """Check if directory should be skipped entirely."""
        if self.ignore_spec and self.ignore_spec.match_file(str(dir_path.name)):
            return True

        # Quick checks for common skip patterns
        skip_dirs = {
            ".git",
            ".svn",
            ".hg",
            "__pycache__",
            "node_modules",
            ".idea",
            ".vscode",
            "coverage",
            ".nyc_output",
            "build",
            "dist",
        }
        return dir_path.name in skip_dirs

    def _has_supported_extension(self, file_path: Path) -> bool:
        """Quick check if file has supported extension."""
        extension_map = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".tf": "hcl",
            ".hcl": "hcl",
        }
        return file_path.suffix.lower() in extension_map

    def _process_file_batch(self, file_paths: List[Path]) -> Tuple[int, int, List[str]]:
        """Process a batch of files efficiently."""
        files_processed = 0
        blocks_extracted = 0
        errors = []

        # Group files by language for batch processing
        files_by_lang: Dict[str, List[Path]] = {}

        for file_path in file_paths:
            try:
                language = detect_language(file_path)
                if language and language in self.config.supported_languages:
                    if language not in files_by_lang:
                        files_by_lang[language] = []
                    files_by_lang[language].append(file_path)

            except Exception as e:
                errors.append(f"Language detection failed for {file_path}: {e}")

        # Process each language group
        for language, lang_files in files_by_lang.items():
            try:
                # Extract blocks from all files of this language
                blocks = extract_blocks(lang_files, [language])

                if blocks:
                    # Normalize blocks
                    normalized_blocks = normalize_blocks(blocks)

                    # Store file records and blocks
                    for file_path in lang_files:
                        try:
                            file_hash = self._compute_file_hash(file_path)
                            file_size = file_path.stat().st_size

                            # Count lines in file
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    lines = sum(1 for _ in f)
                            except (OSError, UnicodeDecodeError):
                                lines = 0

                            # Store file record
                            self.database.store_file(
                                file_path, language, file_hash, file_size, lines
                            )

                            # Clear old blocks for this file
                            self.database.delete_blocks_by_file(file_path)

                            files_processed += 1

                        except Exception as e:
                            errors.append(
                                f"File processing failed for {file_path}: {e}"
                            )

                    # Store blocks with normalized hashes
                    block_data = []
                    for original, normalized in zip(blocks, normalized_blocks):
                        try:
                            norm_hash = self._compute_normalized_hash(normalized)
                            complexity = (
                                len(normalized.tokens) / 10.0
                            )  # Simple complexity metric
                            block_data.append((original, norm_hash, complexity))
                        except Exception as e:
                            errors.append(f"Block normalization failed: {e}")

                    # Batch store blocks
                    if block_data:
                        stored_count = self.database.store_blocks_batch(block_data)
                        blocks_extracted += stored_count

            except Exception as e:
                errors.append(f"Language group processing failed for {language}: {e}")

        return files_processed, blocks_extracted, errors

    def index_files(self, file_paths: List[Path]) -> IndexingResult:
        """Index specific files with validation and error handling."""
        start_time = datetime.now()

        files_processed = 0
        files_skipped = 0
        blocks_extracted = 0
        errors = []

        # Filter valid files
        valid_files = []
        for file_path in file_paths:
            if not file_path.exists():
                errors.append(f"File does not exist: {file_path}")
                continue

            if not file_path.is_file():
                errors.append(f"Not a file: {file_path}")
                continue

            if self._should_skip_file(file_path):
                files_skipped += 1
                continue

            valid_files.append(file_path)

        if valid_files:
            try:
                # Process in batch
                batch_result = self._process_file_batch(valid_files)
                files_processed = batch_result[0]
                blocks_extracted = batch_result[1]
                errors.extend(batch_result[2])

            except Exception as e:
                logger.error(f"Batch file processing failed: {e}")
                errors.append(f"Batch processing failed: {e}")

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return IndexingResult(
            files_processed=files_processed,
            files_skipped=files_skipped,
            blocks_extracted=blocks_extracted,
            errors=errors,
            processing_time_ms=processing_time,
        )

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped based on patterns and language."""
        # Check ignore patterns
        if self.ignore_spec and self.ignore_spec.match_file(str(file_path)):
            return True

        # Check supported languages
        language = detect_language(file_path)
        if language not in self.config.supported_languages:
            return True

        # Skip very large files (> 1MB by default)
        try:
            if file_path.stat().st_size > 1024 * 1024:
                return True
        except OSError:
            return True  # Can't stat, skip

        return False

    def _file_needs_reindex(self, file_path: Path) -> bool:
        """Check if file needs reindexing based on modification time and hash."""
        try:
            current_hash = self._compute_file_hash(file_path)

            # Check database for existing file record
            with self.database.get_session() as session:
                from .storage import FileRecord

                record = (
                    session.query(FileRecord)
                    .filter_by(path=file_path.as_posix())
                    .first()
                )

                # File not in database, needs indexing
                if not record:
                    return True

                # Hash changed, needs reindexing
                if record.hash != current_hash:
                    return True

                return False

        except Exception as e:
            logger.warning(f"Could not check reindex status for {file_path}: {e}")
            return True  # When in doubt, reindex

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return ""

    def _compute_normalized_hash(self, normalized_block: NormalizedBlock) -> str:
        """Compute hash of normalized token sequence."""
        token_string = " ".join(normalized_block.tokens)
        return hashlib.sha256(token_string.encode("utf-8")).hexdigest()[
            :16
        ]  # 16 chars sufficient

    def get_changed_files(self, repo_path: Path) -> List[Path]:
        """Get list of files that changed since last index using git."""
        changed_files = []

        try:
            # Try to get changed files from git
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        # Parse git status format
                        status = line[:2]
                        filepath = line[3:]

                        # Include modified, added, and renamed files
                        if any(s in status for s in ["M", "A", "R"]):
                            file_path = repo_path / filepath
                            if file_path.exists() and file_path.is_file():
                                changed_files.append(file_path)

                logger.info(f"Found {len(changed_files)} changed files via git")

            else:
                logger.warning("Git status failed, falling back to file comparison")

        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(
                f"Git command failed: {e}, using file modification comparison"
            )

        # Fallback: compare modification times if git failed
        if not changed_files:
            changed_files = self._get_changed_files_by_mtime(repo_path)

        return changed_files

    def _get_changed_files_by_mtime(self, repo_path: Path) -> List[Path]:
        """Get changed files by comparing modification times with database."""
        changed_files = []

        try:
            last_indexed_str = self.database.get_metadata("last_indexed")
            if not last_indexed_str:
                # No previous index, return all files
                return self._scan_directory(repo_path)

            last_indexed = datetime.fromisoformat(
                last_indexed_str.replace("Z", "+00:00")
            )

            for file_path in self._scan_directory(repo_path):
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime > last_indexed:
                        changed_files.append(file_path)
                except Exception as e:
                    logger.debug(f"Could not check mtime for {file_path}: {e}")

        except Exception as e:
            logger.error(f"Failed to get changed files by mtime: {e}")

        return changed_files


def index_repository(
    repo_path: Path, config: Optional[EchoConfig] = None, reindex: bool = False
) -> IndexingResult:
    """Index a repository for duplicate detection."""
    if config is None:
        config = EchoConfig()

    # Initialize database
    db_path = config.get_cache_dir() / "echo.db"
    database = create_database(db_path, config)

    # Create indexer and run
    indexer = RepositoryIndexer(config, database)
    return indexer.index_repository(repo_path, reindex)
