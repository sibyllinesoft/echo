"""SQLite and FAISS storage backend for Echo.

This module provides the central data persistence layer for Echo's duplicate code detection.
It manages SQLite for metadata and FAISS for semantic embeddings with full CRUD operations,
transaction support, and thread-safe concurrent access.

Key Features:
- SQLAlchemy ORM for type-safe database operations
- FAISS index management for high-performance semantic search
- Transaction support with automatic rollback on failures
- Connection pooling for concurrent access
- Migration system for schema evolution
- Comprehensive error handling and logging
- Thread-safe operations with proper locking
"""

import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import faiss
import numpy as np
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    and_,
    create_engine,
    event,
    func,
    or_,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool

from .config import EchoConfig
from .normalize import NormalizedBlock
from .parser import CodeBlock
from .utils import (
    JsonHandler,
    PathHandler,
    ProgressReporter,
    StorageError,
    database_session_handler,
    handle_errors,
)

logger = logging.getLogger(__name__)

Base = declarative_base()

# Database schema version for migrations
SCHEMA_VERSION = "1.0.0"


class FileRecord(Base):
    """SQLite table for tracked files with comprehensive metadata."""

    __tablename__ = "files"

    path = Column(String, primary_key=True, index=True)
    lang = Column(String, nullable=False, index=True)
    hash = Column(String, nullable=False)  # File content hash
    last_indexed = Column(DateTime, nullable=False, default=datetime.utcnow)
    ignored = Column(Boolean, default=False, nullable=False, index=True)
    size_bytes = Column(Integer, default=0)
    lines = Column(Integer, default=0)

    # Relationship to blocks
    blocks = relationship(
        "BlockRecord", back_populates="file", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<FileRecord(path='{self.path}', lang='{self.lang}', blocks={len(self.blocks)})>"


class BlockRecord(Base):
    """SQLite table for code blocks with enhanced metadata."""

    __tablename__ = "blocks"

    id = Column(String, primary_key=True, index=True)
    file_path = Column(String, ForeignKey("files.path"), nullable=False, index=True)
    start = Column(Integer, nullable=False)
    end = Column(Integer, nullable=False)
    lang = Column(String, nullable=False, index=True)
    tokens = Column(Integer, nullable=False)
    norm_hash = Column(String, index=True)  # MinHash signature as base64
    churn = Column(Integer, default=0, nullable=False)
    node_type = Column(String)  # Tree-sitter node type
    complexity = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    file = relationship("FileRecord", back_populates="blocks")
    findings_as_source = relationship(
        "FindingRecord",
        foreign_keys="FindingRecord.block_id",
        back_populates="source_block",
    )
    findings_as_match = relationship(
        "FindingRecord",
        foreign_keys="FindingRecord.match_block_id",
        back_populates="match_block",
    )

    # Composite indexes for performance
    __table_args__ = (
        Index("idx_blocks_file_range", "file_path", "start", "end"),
        Index("idx_blocks_lang_tokens", "lang", "tokens"),
        Index("idx_blocks_norm_hash_lang", "norm_hash", "lang"),
    )

    def __repr__(self) -> str:
        return f"<BlockRecord(id='{self.id}', file='{self.file_path}', range={self.start}-{self.end})>"


class FindingRecord(Base):
    """SQLite table for duplicate code findings with detailed scoring."""

    __tablename__ = "findings"

    id = Column(String, primary_key=True, index=True)
    block_id = Column(String, ForeignKey("blocks.id"), nullable=False, index=True)
    match_block_id = Column(String, ForeignKey("blocks.id"), nullable=False, index=True)
    scores_json = Column(Text, nullable=False)  # JSON serialized scores
    type = Column(
        String, nullable=False, index=True
    )  # 'exact', 'near_miss', 'semantic'
    confidence = Column(Float, default=0.0)  # Overall confidence score
    refactor_score = Column(Float, default=0.0)  # R score for prioritization
    status = Column(
        String, default="active", index=True
    )  # 'active', 'suppressed', 'false_positive'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    source_block = relationship(
        "BlockRecord", foreign_keys=[block_id], back_populates="findings_as_source"
    )
    match_block = relationship(
        "BlockRecord", foreign_keys=[match_block_id], back_populates="findings_as_match"
    )

    # Composite indexes
    __table_args__ = (
        Index("idx_findings_blocks", "block_id", "match_block_id"),
        Index("idx_findings_type_status", "type", "status"),
        Index("idx_findings_score", "refactor_score"),
    )

    @property
    def scores(self) -> Dict[str, float]:
        """Parse scores from JSON."""
        return JsonHandler.deserialize(self.scores_json or "{}")

    @scores.setter
    def scores(self, scores: Dict[str, float]) -> None:
        """Serialize scores to JSON."""
        self.scores_json = JsonHandler.serialize(scores)

    def __repr__(self) -> str:
        return f"<FindingRecord(id='{self.id}', type='{self.type}', score={self.refactor_score:.1f})>"


class MetadataRecord(Base):
    """SQLite table for system metadata and configuration."""

    __tablename__ = "metadata"

    key = Column(String, primary_key=True)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<MetadataRecord(key='{self.key}')>"


class MinHashRecord(Base):
    """SQLite table for LSH MinHash signatures and bands."""

    __tablename__ = "minhash_signatures"

    block_id = Column(String, ForeignKey("blocks.id"), primary_key=True, index=True)
    signature = Column(Text, nullable=False)  # Base64 encoded numpy array
    bands = Column(Text, nullable=False)  # JSON array of band hashes
    shingle_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<MinHashRecord(block_id='{self.block_id}', shingles={self.shingle_count})>"


@dataclass
class DuplicateFinding:
    """Represents a duplicate code finding with comprehensive metadata."""

    id: str
    block_id: str
    match_block_id: str
    scores: Dict[str, float]
    type: str
    confidence: float
    refactor_score: float
    status: str = "active"
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class EchoDatabase:
    """Main database interface for Echo storage with comprehensive functionality.

    Features:
    - Thread-safe operations with connection pooling
    - Transaction management with automatic rollback
    - Batch operations for performance
    - Migration support for schema evolution
    - Comprehensive error handling and logging
    """

    def __init__(self, db_path: Path, config: Optional[EchoConfig] = None):
        """Initialize database with proper configuration.

        Args:
            db_path: Path to SQLite database file
            config: Optional configuration for connection tuning
        """
        self.db_path = db_path
        self.config = config or EchoConfig()

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure SQLite engine with performance optimizations
        connect_args = {
            "check_same_thread": False,
            "timeout": 30,
            "isolation_level": None,  # Enable autocommit mode for better concurrency
        }

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args=connect_args,
            poolclass=StaticPool,
            pool_pre_ping=True,
            echo=False,  # Set to True for SQL debugging
        )

        # Thread-safe scoped session
        self.SessionLocal = scoped_session(
            sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        )

        # Thread synchronization
        self._lock = threading.RLock()

        # Initialize schema and metadata
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema and metadata."""
        try:
            # Create all tables
            Base.metadata.create_all(self.engine)

            # Configure SQLite pragmas for performance
            self._configure_sqlite()

            # Initialize metadata
            self._init_metadata()

            logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _configure_sqlite(self) -> None:
        """Configure SQLite for optimal performance."""
        pragmas = [
            "PRAGMA journal_mode=WAL",  # Write-Ahead Logging for concurrency
            "PRAGMA synchronous=NORMAL",  # Balance safety and performance
            "PRAGMA cache_size=-64000",  # 64MB cache
            "PRAGMA temp_store=MEMORY",  # Use memory for temp tables
            "PRAGMA mmap_size=268435456",  # 256MB memory mapping
            "PRAGMA optimize",  # Optimize query planner
        ]

        with self.engine.connect() as conn:
            for pragma in pragmas:
                conn.execute(pragma)
            conn.commit()

    def _init_metadata(self) -> None:
        """Initialize system metadata."""
        with self.get_session() as session:
            # Set schema version
            self.set_metadata("schema_version", SCHEMA_VERSION, session)
            # Initialize statistics
            self.set_metadata("initialized_at", datetime.utcnow().isoformat(), session)

    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_metadata(
        self, key: str, session: Optional[Session] = None
    ) -> Optional[str]:
        """Get metadata value by key."""
        use_session = session or self.SessionLocal()
        try:
            record = use_session.query(MetadataRecord).filter_by(key=key).first()
            return record.value if record else None
        except SQLAlchemyError as e:
            logger.error(f"Failed to get metadata {key}: {e}")
            return None
        finally:
            if session is None:
                use_session.close()

    def set_metadata(
        self, key: str, value: str, session: Optional[Session] = None
    ) -> bool:
        """Set metadata key-value pair."""
        use_session = session or self.SessionLocal()
        try:
            record = use_session.query(MetadataRecord).filter_by(key=key).first()
            if record:
                record.value = value
                record.updated_at = datetime.utcnow()
            else:
                record = MetadataRecord(key=key, value=value)
                use_session.add(record)

            if session is None:
                use_session.commit()
            return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to set metadata {key}: {e}")
            if session is None:
                use_session.rollback()
            return False
        finally:
            if session is None:
                use_session.close()

    @handle_errors("store file record", default_return=False)
    def store_file(
        self,
        file_path: Path,
        lang: str,
        file_hash: str,
        size_bytes: int = 0,
        lines: int = 0,
    ) -> bool:
        """Store or update file record with comprehensive metadata."""
        with self.get_session() as session:
            path_str = PathHandler.normalize_path(file_path)

            # Check for existing record
            existing = session.query(FileRecord).filter_by(path=path_str).first()

            if existing:
                # Update existing record
                existing.lang = lang
                existing.hash = file_hash
                existing.last_indexed = datetime.utcnow()
                existing.size_bytes = size_bytes
                existing.lines = lines
                existing.ignored = False
                logger.debug(f"Updated file record: {path_str}")
            else:
                # Create new record
                record = FileRecord(
                    path=path_str,
                    lang=lang,
                    hash=file_hash,
                    size_bytes=size_bytes,
                    lines=lines,
                    ignored=False,
                )
                session.add(record)
                logger.debug(f"Created file record: {path_str}")

            return True

    def store_block(
        self,
        block: CodeBlock,
        normalized: Optional[NormalizedBlock] = None,
        norm_hash: Optional[str] = None,
        complexity: float = 0.0,
    ) -> bool:
        """Store a code block with optional normalization data."""
        with self.get_session() as session:
            try:
                # Generate block ID
                block_id = (
                    f"{block.file_path.as_posix()}:{block.start_line}:{block.end_line}"
                )

                # Check for existing block
                existing = session.query(BlockRecord).filter_by(id=block_id).first()

                if existing:
                    # Update existing block
                    existing.tokens = block.token_count
                    existing.norm_hash = norm_hash
                    existing.complexity = complexity
                    existing.updated_at = datetime.utcnow()
                    existing.churn += 1  # Track modification frequency
                    logger.debug(f"Updated block record: {block_id}")
                else:
                    # Create new block
                    record = BlockRecord(
                        id=block_id,
                        file_path=block.file_path.as_posix(),
                        start=block.start_line,
                        end=block.end_line,
                        lang=block.lang,
                        tokens=block.token_count,
                        norm_hash=norm_hash,
                        node_type=block.node_type,
                        complexity=complexity,
                        churn=0,
                    )
                    session.add(record)
                    logger.debug(f"Created block record: {block_id}")

                return True

            except SQLAlchemyError as e:
                logger.error(
                    f"Failed to store block from {block.file_path}:{block.start_line}: {e}"
                )
                return False

    def store_blocks_batch(
        self, blocks: List[Tuple[CodeBlock, Optional[str], float]]
    ) -> int:
        """Efficiently store multiple blocks in a single transaction."""
        stored_count = 0

        with self.get_session() as session:
            try:
                records_to_add = []

                for block, norm_hash, complexity in blocks:
                    block_id = f"{block.file_path.as_posix()}:{block.start_line}:{block.end_line}"

                    # Check if exists
                    existing = session.query(BlockRecord).filter_by(id=block_id).first()

                    if existing:
                        existing.tokens = block.token_count
                        existing.norm_hash = norm_hash
                        existing.complexity = complexity
                        existing.updated_at = datetime.utcnow()
                        existing.churn += 1
                        stored_count += 1
                    else:
                        record = BlockRecord(
                            id=block_id,
                            file_path=block.file_path.as_posix(),
                            start=block.start_line,
                            end=block.end_line,
                            lang=block.lang,
                            tokens=block.token_count,
                            norm_hash=norm_hash,
                            node_type=block.node_type,
                            complexity=complexity,
                            churn=0,
                        )
                        records_to_add.append(record)

                # Bulk insert new records
                if records_to_add:
                    session.add_all(records_to_add)
                    stored_count += len(records_to_add)

                logger.info(f"Stored {stored_count} blocks in batch")
                return stored_count

            except SQLAlchemyError as e:
                logger.error(f"Failed to store blocks batch: {e}")
                return 0

    def store_finding(self, finding: DuplicateFinding) -> bool:
        """Store a duplicate code finding with comprehensive metadata."""
        with self.get_session() as session:
            try:
                # Check for existing finding
                existing = session.query(FindingRecord).filter_by(id=finding.id).first()

                if existing:
                    # Update existing finding
                    existing.scores = finding.scores
                    existing.confidence = finding.confidence
                    existing.refactor_score = finding.refactor_score
                    existing.status = finding.status
                    existing.updated_at = datetime.utcnow()
                    logger.debug(f"Updated finding: {finding.id}")
                else:
                    # Create new finding
                    record = FindingRecord(
                        id=finding.id,
                        block_id=finding.block_id,
                        match_block_id=finding.match_block_id,
                        scores_json=JsonHandler.serialize(finding.scores),
                        type=finding.type,
                        confidence=finding.confidence,
                        refactor_score=finding.refactor_score,
                        status=finding.status,
                    )
                    session.add(record)
                    logger.debug(f"Created finding: {finding.id}")

                return True

            except SQLAlchemyError as e:
                logger.error(f"Failed to store finding {finding.id}: {e}")
                return False

    def store_findings_batch(self, findings: List[DuplicateFinding]) -> int:
        """Efficiently store multiple findings in a single transaction."""
        stored_count = 0

        with self.get_session() as session:
            try:
                for finding in findings:
                    existing = (
                        session.query(FindingRecord).filter_by(id=finding.id).first()
                    )

                    if existing:
                        existing.scores = finding.scores
                        existing.confidence = finding.confidence
                        existing.refactor_score = finding.refactor_score
                        existing.status = finding.status
                        existing.updated_at = datetime.utcnow()
                    else:
                        record = FindingRecord(
                            id=finding.id,
                            block_id=finding.block_id,
                            match_block_id=finding.match_block_id,
                            scores_json=JsonHandler.serialize(finding.scores),
                            type=finding.type,
                            confidence=finding.confidence,
                            refactor_score=finding.refactor_score,
                            status=finding.status,
                        )
                        session.add(record)

                    stored_count += 1

                logger.info(f"Stored {stored_count} findings in batch")
                return stored_count

            except SQLAlchemyError as e:
                logger.error(f"Failed to store findings batch: {e}")
                return 0

    def get_blocks_by_hash(
        self, norm_hash: str, lang: Optional[str] = None
    ) -> List[BlockRecord]:
        """Get blocks by normalized hash with optional language filtering."""
        with self.get_session() as session:
            try:
                query = session.query(BlockRecord).filter(
                    BlockRecord.norm_hash == norm_hash
                )

                if lang:
                    query = query.filter(BlockRecord.lang == lang)

                return query.all()

            except SQLAlchemyError as e:
                logger.error(f"Failed to get blocks by hash {norm_hash}: {e}")
                return []

    def get_blocks_by_file(self, file_path: Union[Path, str]) -> List[BlockRecord]:
        """Get all blocks for a specific file."""
        path_str = file_path.as_posix() if isinstance(file_path, Path) else file_path

        with self.get_session() as session:
            try:
                return (
                    session.query(BlockRecord)
                    .filter(BlockRecord.file_path == path_str)
                    .order_by(BlockRecord.start)
                    .all()
                )

            except SQLAlchemyError as e:
                logger.error(f"Failed to get blocks for file {path_str}: {e}")
                return []

    def get_findings_by_block(
        self, block_id: str, status: str = "active"
    ) -> List[FindingRecord]:
        """Get all findings for a specific block."""
        with self.get_session() as session:
            try:
                return (
                    session.query(FindingRecord)
                    .filter(
                        or_(
                            FindingRecord.block_id == block_id,
                            FindingRecord.match_block_id == block_id,
                        ),
                        FindingRecord.status == status,
                    )
                    .order_by(FindingRecord.refactor_score.desc())
                    .all()
                )

            except SQLAlchemyError as e:
                logger.error(f"Failed to get findings for block {block_id}: {e}")
                return []

    def get_top_findings(
        self,
        limit: int = 50,
        min_refactor_score: float = 200.0,
        finding_type: Optional[str] = None,
    ) -> List[FindingRecord]:
        """Get top findings by refactor score with filtering."""
        with self.get_session() as session:
            try:
                query = session.query(FindingRecord).filter(
                    FindingRecord.status == "active",
                    FindingRecord.refactor_score >= min_refactor_score,
                )

                if finding_type:
                    query = query.filter(FindingRecord.type == finding_type)

                return (
                    query.order_by(FindingRecord.refactor_score.desc())
                    .limit(limit)
                    .all()
                )

            except SQLAlchemyError as e:
                logger.error(f"Failed to get top findings: {e}")
                return []

    def delete_file(self, file_path: Union[Path, str]) -> bool:
        """Delete a file and all associated blocks/findings."""
        path_str = file_path.as_posix() if isinstance(file_path, Path) else file_path

        with self.get_session() as session:
            try:
                # Delete file record (cascades to blocks and findings)
                deleted = session.query(FileRecord).filter_by(path=path_str).delete()

                if deleted:
                    logger.info(f"Deleted file and associated data: {path_str}")
                    return True
                else:
                    logger.warning(f"File not found for deletion: {path_str}")
                    return False

            except SQLAlchemyError as e:
                logger.error(f"Failed to delete file {path_str}: {e}")
                return False

    def delete_blocks_by_file(self, file_path: Union[Path, str]) -> int:
        """Delete all blocks for a specific file."""
        path_str = file_path.as_posix() if isinstance(file_path, Path) else file_path

        with self.get_session() as session:
            try:
                deleted = (
                    session.query(BlockRecord).filter_by(file_path=path_str).delete()
                )
                logger.info(f"Deleted {deleted} blocks for file: {path_str}")
                return deleted

            except SQLAlchemyError as e:
                logger.error(f"Failed to delete blocks for file {path_str}: {e}")
                return 0

    def suppress_finding(
        self, finding_id: str, reason: str = "user_suppressed"
    ) -> bool:
        """Mark a finding as suppressed."""
        with self.get_session() as session:
            try:
                finding = session.query(FindingRecord).filter_by(id=finding_id).first()
                if finding:
                    finding.status = "suppressed"
                    finding.updated_at = datetime.utcnow()
                    # Store suppression reason in metadata
                    self.set_metadata(f"suppression:{finding_id}", reason, session)
                    logger.info(f"Suppressed finding: {finding_id}")
                    return True
                else:
                    logger.warning(f"Finding not found for suppression: {finding_id}")
                    return False

            except SQLAlchemyError as e:
                logger.error(f"Failed to suppress finding {finding_id}: {e}")
                return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        with self.get_session() as session:
            try:
                stats = {}

                # File statistics
                files_query = session.query(FileRecord)
                stats["files"] = {
                    "total": files_query.count(),
                    "by_language": dict(
                        session.query(FileRecord.lang, func.count(FileRecord.path))
                        .group_by(FileRecord.lang)
                        .all()
                    ),
                    "ignored": files_query.filter_by(ignored=True).count(),
                }

                # Block statistics
                blocks_query = session.query(BlockRecord)
                stats["blocks"] = {
                    "total": blocks_query.count(),
                    "by_language": dict(
                        session.query(BlockRecord.lang, func.count(BlockRecord.id))
                        .group_by(BlockRecord.lang)
                        .all()
                    ),
                    "avg_tokens": session.query(func.avg(BlockRecord.tokens)).scalar()
                    or 0.0,
                    "total_tokens": session.query(func.sum(BlockRecord.tokens)).scalar()
                    or 0,
                }

                # Finding statistics
                findings_query = session.query(FindingRecord).filter_by(status="active")
                stats["findings"] = {
                    "total": findings_query.count(),
                    "by_type": dict(
                        session.query(FindingRecord.type, func.count(FindingRecord.id))
                        .filter_by(status="active")
                        .group_by(FindingRecord.type)
                        .all()
                    ),
                    "suppressed": session.query(FindingRecord)
                    .filter_by(status="suppressed")
                    .count(),
                    "avg_refactor_score": session.query(
                        func.avg(FindingRecord.refactor_score)
                    )
                    .filter_by(status="active")
                    .scalar()
                    or 0.0,
                }

                # System metadata
                stats["system"] = {
                    "schema_version": self.get_metadata("schema_version", session),
                    "initialized_at": self.get_metadata("initialized_at", session),
                    "database_size_mb": self.db_path.stat().st_size / 1024 / 1024,
                    "last_updated": datetime.utcnow().isoformat(),
                }

                return stats

            except SQLAlchemyError as e:
                logger.error(f"Failed to get database statistics: {e}")
                return {}

    def vacuum_database(self) -> bool:
        """Perform database maintenance and optimization."""
        try:
            with self.engine.connect() as conn:
                # Vacuum to reclaim space and optimize
                conn.execute("VACUUM")
                # Analyze to update query planner statistics
                conn.execute("ANALYZE")
                # Optimize SQLite
                conn.execute("PRAGMA optimize")
                conn.commit()

            logger.info("Database vacuum and optimization completed")
            return True

        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            return False

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        try:
            self.SessionLocal.remove()
            self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")


class FAISSIndex:
    """FAISS-based semantic embedding index with persistence and thread safety.

    Features:
    - High-performance similarity search with FAISS
    - Persistent storage with automatic loading
    - Thread-safe operations with proper locking
    - Batch operations for efficiency
    - Flexible distance metrics and search parameters
    """

    def __init__(self, index_path: Path, dimension: int = 256, metric: str = "cosine"):
        """Initialize FAISS index with configuration.

        Args:
            index_path: Directory to store FAISS index files
            dimension: Embedding vector dimension
            metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.index_path = index_path
        self.dimension = dimension
        self.metric = metric.lower()

        # Create directory if needed
        self.index_path.mkdir(parents=True, exist_ok=True)

        # File paths
        self.index_file = self.index_path / "faiss_index.bin"
        self.mapping_file = self.index_path / "block_mapping.json"

        # Thread safety
        self._lock = threading.RLock()

        # Initialize FAISS index
        self.index = self._create_index()
        self.block_id_mapping: Dict[int, str] = {}  # FAISS index -> block_id
        self.reverse_mapping: Dict[str, int] = {}  # block_id -> FAISS index

        # Load existing data
        self._load_index()

        logger.info(
            f"FAISS index initialized: {self.dimension}D {self.metric} at {self.index_path}"
        )

    def _create_index(self) -> faiss.Index:
        """Create appropriate FAISS index based on metric."""
        if self.metric == "cosine":
            # Use inner product with normalized vectors for cosine similarity
            index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            # Euclidean distance
            index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "ip":
            # Inner product
            index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        return index

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            return vectors / norms
        return vectors

    def add_embeddings(self, block_ids: List[str], embeddings: np.ndarray) -> bool:
        """Add embeddings to the index with block ID mapping.

        Args:
            block_ids: List of block IDs corresponding to embeddings
            embeddings: Numpy array of shape (n, dimension) with dtype float32

        Returns:
            True if successful, False otherwise
        """
        if len(block_ids) != embeddings.shape[0]:
            raise ValueError("Number of block IDs must match number of embeddings")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != expected {self.dimension}"
            )

        with self._lock:
            try:
                # Convert to float32 if needed
                if embeddings.dtype != np.float32:
                    embeddings = embeddings.astype(np.float32)

                # Normalize for cosine similarity
                embeddings = self._normalize_vectors(embeddings)

                # Add to FAISS index
                start_idx = self.index.ntotal
                self.index.add(embeddings)

                # Update mappings
                for i, block_id in enumerate(block_ids):
                    faiss_idx = start_idx + i
                    self.block_id_mapping[faiss_idx] = block_id
                    self.reverse_mapping[block_id] = faiss_idx

                logger.debug(f"Added {len(block_ids)} embeddings to FAISS index")
                return True

            except Exception as e:
                logger.error(f"Failed to add embeddings to FAISS index: {e}")
                return False

    def search_similar(
        self, embedding: np.ndarray, k: int = 20, exclude_ids: Optional[Set[str]] = None
    ) -> Tuple[List[float], List[str]]:
        """Search for similar embeddings.

        Args:
            embedding: Query embedding of shape (dimension,)
            k: Number of nearest neighbors to return
            exclude_ids: Block IDs to exclude from results

        Returns:
            Tuple of (distances, block_ids) sorted by similarity
        """
        with self._lock:
            try:
                if self.index.ntotal == 0:
                    return [], []

                # Ensure correct shape and type
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)

                if embedding.dtype != np.float32:
                    embedding = embedding.astype(np.float32)

                # Normalize for cosine similarity
                embedding = self._normalize_vectors(embedding)

                # Search with buffer for exclusions
                search_k = min(k * 2, self.index.ntotal) if exclude_ids else k
                distances, indices = self.index.search(embedding, search_k)

                # Convert to lists and map indices to block IDs
                distances = distances[0].tolist()
                indices = indices[0].tolist()

                results_distances = []
                results_block_ids = []

                exclude_set = exclude_ids or set()

                for dist, idx in zip(distances, indices):
                    if idx == -1:  # Invalid index from FAISS
                        continue

                    block_id = self.block_id_mapping.get(idx)
                    if block_id and block_id not in exclude_set:
                        # Convert distance to similarity score for cosine metric
                        if self.metric == "cosine":
                            similarity = float(dist)  # Already inner product (cosine)
                        elif self.metric == "l2":
                            similarity = 1.0 / (
                                1.0 + float(dist)
                            )  # Convert L2 to similarity
                        else:
                            similarity = float(dist)

                        results_distances.append(similarity)
                        results_block_ids.append(block_id)

                        if len(results_block_ids) >= k:
                            break

                logger.debug(f"Found {len(results_block_ids)} similar embeddings")
                return results_distances, results_block_ids

            except Exception as e:
                logger.error(f"Failed to search FAISS index: {e}")
                return [], []

    def update_embedding(self, block_id: str, embedding: np.ndarray) -> bool:
        """Update embedding for existing block ID.

        Note: FAISS doesn't support in-place updates, so this removes and re-adds.
        For frequent updates, consider rebuilding the index periodically.
        """
        with self._lock:
            try:
                if block_id not in self.reverse_mapping:
                    logger.warning(f"Block ID {block_id} not found for update")
                    return False

                # Remove old embedding
                self.remove_embedding(block_id)

                # Add new embedding
                return self.add_embeddings([block_id], embedding.reshape(1, -1))

            except Exception as e:
                logger.error(f"Failed to update embedding for {block_id}: {e}")
                return False

    def remove_embedding(self, block_id: str) -> bool:
        """Remove embedding for a block ID.

        Note: FAISS doesn't support efficient removal, so this marks as invalid.
        Consider periodic index rebuilding for better performance.
        """
        with self._lock:
            try:
                if block_id not in self.reverse_mapping:
                    return True  # Already removed

                faiss_idx = self.reverse_mapping[block_id]

                # Remove from mappings
                del self.reverse_mapping[block_id]
                del self.block_id_mapping[faiss_idx]

                logger.debug(f"Removed embedding for block {block_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to remove embedding for {block_id}: {e}")
                return False

    def save_index(self) -> bool:
        """Save FAISS index and mappings to disk."""
        with self._lock:
            try:
                # Save FAISS index
                faiss.write_index(self.index, str(self.index_file))

                # Save mappings
                mapping_data = {
                    "block_id_mapping": {
                        str(k): v for k, v in self.block_id_mapping.items()
                    },
                    "reverse_mapping": self.reverse_mapping,
                    "dimension": self.dimension,
                    "metric": self.metric,
                    "total_embeddings": self.index.ntotal,
                }

                JsonHandler.write_file(mapping_data, self.mapping_file)

                logger.info(f"Saved FAISS index with {self.index.ntotal} embeddings")
                return True

            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
                return False

    def _load_index(self) -> bool:
        """Load FAISS index and mappings from disk."""
        try:
            if not self.index_file.exists() or not self.mapping_file.exists():
                logger.info("No existing FAISS index found, starting fresh")
                return True

            with self._lock:
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_file))

                # Load mappings
                mapping_data = JsonHandler.read_file(self.mapping_file)

                # Restore mappings with proper types
                self.block_id_mapping = {
                    int(k): v for k, v in mapping_data["block_id_mapping"].items()
                }
                self.reverse_mapping = mapping_data["reverse_mapping"]

                # Validate consistency
                if self.index.ntotal != len(self.block_id_mapping):
                    logger.warning(
                        f"Index size mismatch: FAISS={self.index.ntotal}, "
                        f"mapping={len(self.block_id_mapping)}"
                    )

                logger.info(f"Loaded FAISS index with {self.index.ntotal} embeddings")
                return True

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Reset to empty index on load failure
            self.index = self._create_index()
            self.block_id_mapping.clear()
            self.reverse_mapping.clear()
            return False

    def rebuild_index(self) -> bool:
        """Rebuild index to remove gaps from deletions."""
        with self._lock:
            try:
                if self.index.ntotal == 0:
                    return True

                # Extract all valid embeddings
                valid_embeddings = []
                valid_block_ids = []

                for faiss_idx in sorted(self.block_id_mapping.keys()):
                    block_id = self.block_id_mapping[faiss_idx]
                    if block_id:  # Valid entry
                        # Retrieve embedding from index
                        embedding = self.index.reconstruct(faiss_idx)
                        valid_embeddings.append(embedding)
                        valid_block_ids.append(block_id)

                if not valid_embeddings:
                    # Empty index
                    self.index = self._create_index()
                    self.block_id_mapping.clear()
                    self.reverse_mapping.clear()
                    logger.info("Rebuilt empty FAISS index")
                    return True

                # Create new index
                old_total = self.index.ntotal
                self.index = self._create_index()
                self.block_id_mapping.clear()
                self.reverse_mapping.clear()

                # Add all valid embeddings
                embeddings_array = np.array(valid_embeddings, dtype=np.float32)
                success = self.add_embeddings(valid_block_ids, embeddings_array)

                if success:
                    logger.info(
                        f"Rebuilt FAISS index: {old_total} -> {self.index.ntotal} embeddings"
                    )
                    return True
                else:
                    logger.error("Failed to rebuild FAISS index")
                    return False

            except Exception as e:
                logger.error(f"Failed to rebuild FAISS index: {e}")
                return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get FAISS index statistics."""
        with self._lock:
            return {
                "total_embeddings": self.index.ntotal,
                "dimension": self.dimension,
                "metric": self.metric,
                "index_file_exists": self.index_file.exists(),
                "mapping_file_exists": self.mapping_file.exists(),
                "mapping_consistency": len(self.block_id_mapping) == self.index.ntotal,
                "index_size_mb": (
                    self.index_file.stat().st_size / 1024 / 1024
                    if self.index_file.exists()
                    else 0
                ),
            }

    def clear(self) -> None:
        """Clear all embeddings and mappings."""
        with self._lock:
            self.index = self._create_index()
            self.block_id_mapping.clear()
            self.reverse_mapping.clear()
            logger.info("Cleared FAISS index")


# Factory functions for easy integration
def create_database(db_path: Path, config: Optional[EchoConfig] = None) -> EchoDatabase:
    """Create EchoDatabase instance with configuration."""
    return EchoDatabase(db_path, config)


def create_faiss_index(
    index_path: Path, dimension: int = 256, metric: str = "cosine"
) -> FAISSIndex:
    """Create FAISSIndex instance with configuration."""
    return FAISSIndex(index_path, dimension, metric)
