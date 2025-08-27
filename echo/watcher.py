"""File system watcher for real-time duplicate detection."""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Set

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from .config import EchoConfig
from .index import RepositoryIndexer
from .parser import detect_language
from .storage import EchoDatabase, create_database

logger = logging.getLogger(__name__)


@dataclass
class FileChangeEvent:
    """Represents a file system change event."""

    path: Path
    event_type: str  # 'created', 'modified', 'deleted', 'moved'
    timestamp: float
    old_path: Optional[Path] = None  # For move events


@dataclass
class WatcherStatistics:
    """Statistics about file watcher activity."""

    events_processed: int
    files_indexed: int
    blocks_extracted: int
    errors: int
    started_at: datetime
    last_activity: Optional[datetime] = None


class EventBatcher:
    """Batches file system events to avoid excessive processing."""

    def __init__(
        self,
        callback: Callable[[List[FileChangeEvent]], None],
        batch_delay: float = 2.0,
        max_batch_size: int = 100,
    ):
        self.callback = callback
        self.batch_delay = batch_delay
        self.max_batch_size = max_batch_size
        self.pending_events: Dict[Path, FileChangeEvent] = {}
        self.batch_timer: Optional[threading.Timer] = None
        self.lock = threading.Lock()

    def add_event(self, event: FileChangeEvent) -> None:
        """Add an event to the batch, replacing previous events for same file."""
        with self.lock:
            # Latest event for a file supersedes previous ones
            self.pending_events[event.path] = event

            # Cancel existing timer
            if self.batch_timer:
                self.batch_timer.cancel()

            # Check if we should flush immediately
            if len(self.pending_events) >= self.max_batch_size:
                self._flush_events()
            else:
                # Start new timer
                self.batch_timer = threading.Timer(self.batch_delay, self._flush_events)
                self.batch_timer.start()

    def _flush_events(self) -> None:
        """Flush accumulated events to callback."""
        with self.lock:
            if self.pending_events:
                events = list(self.pending_events.values())
                self.pending_events.clear()

                if self.batch_timer:
                    self.batch_timer.cancel()
                    self.batch_timer = None

                try:
                    self.callback(events)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")

    def flush(self) -> None:
        """Immediately flush all pending events."""
        self._flush_events()


class EchoFileHandler(FileSystemEventHandler):
    """Enhanced file system event handler for Echo with filtering and batching."""

    def __init__(
        self,
        config: EchoConfig,
        indexer: RepositoryIndexer,
        change_callback: Callable[[List[FileChangeEvent]], None],
    ):
        self.config = config
        self.indexer = indexer
        self.batcher = EventBatcher(change_callback, batch_delay=2.0)
        self.stats = WatcherStatistics(
            events_processed=0,
            files_indexed=0,
            blocks_extracted=0,
            errors=0,
            started_at=datetime.now(),
        )

        # Rate limiting to prevent event storms
        self.event_counts: Deque[float] = deque(maxlen=1000)  # Track last 1000 events
        self.rate_limit_window = 60.0  # 1 minute window
        self.max_events_per_minute = 500

    def on_modified(self, event) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if self._should_process_event(file_path):
            change_event = FileChangeEvent(file_path, "modified", time.time())
            self._add_event(change_event)

    def on_created(self, event) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if self._should_process_event(file_path):
            change_event = FileChangeEvent(file_path, "created", time.time())
            self._add_event(change_event)

    def on_deleted(self, event) -> None:
        """Handle file deletion events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        # Process delete events for any previously indexed file
        change_event = FileChangeEvent(file_path, "deleted", time.time())
        self._add_event(change_event)

    def on_moved(self, event) -> None:
        """Handle file move/rename events."""
        if event.is_directory:
            return

        old_path = Path(event.src_path)
        new_path = Path(event.dest_path)

        if self._should_process_event(new_path):
            change_event = FileChangeEvent(
                new_path, "moved", time.time(), old_path=old_path
            )
            self._add_event(change_event)

    def _should_process_event(self, file_path: Path) -> bool:
        """Check if file event should be processed."""
        # Rate limiting check
        if not self._check_rate_limit():
            return False

        # Check if file exists and is accessible
        try:
            if not file_path.exists() or not file_path.is_file():
                return False
        except (OSError, PermissionError):
            return False

        # Check supported languages
        language = detect_language(file_path)
        if not language or language not in self.config.supported_languages:
            return False

        # Check ignore patterns using indexer's logic
        if self.indexer._should_skip_file(file_path):
            return False

        # Skip temporary files and backups
        if file_path.name.startswith(".") or file_path.suffix in {
            ".tmp",
            ".bak",
            ".swp",
        }:
            return False

        return True

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()

        # Add current event timestamp
        self.event_counts.append(current_time)

        # Count events in the last minute
        cutoff = current_time - self.rate_limit_window
        recent_events = sum(1 for t in self.event_counts if t > cutoff)

        if recent_events > self.max_events_per_minute:
            logger.warning(
                f"Rate limit exceeded: {recent_events} events in last minute"
            )
            return False

        return True

    def _add_event(self, event: FileChangeEvent) -> None:
        """Add event to batch processor."""
        self.stats.events_processed += 1
        self.stats.last_activity = datetime.now()
        self.batcher.add_event(event)

    def flush_events(self) -> None:
        """Flush any pending events immediately."""
        self.batcher.flush()

    def get_statistics(self) -> WatcherStatistics:
        """Get current watcher statistics."""
        return self.stats


class FileWatcher:
    """Enhanced file system watcher for Echo duplicate detection."""

    def __init__(self, config: EchoConfig, database: EchoDatabase):
        self.config = config
        self.database = database
        self.observer: Optional[Observer] = None
        self.indexer = RepositoryIndexer(config, database)
        self.handler: Optional[EchoFileHandler] = None
        self.watch_paths: List[Path] = []
        self.is_running = False

        # Performance monitoring
        self.total_processed = 0
        self.total_errors = 0
        self.start_time: Optional[datetime] = None

    def start_watching(
        self,
        watch_paths: List[Path],
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> bool:
        """Start watching file system for changes with advanced configuration."""
        if self.observer and self.observer.is_alive():
            logger.warning("File watcher already running")
            return False

        try:
            self.observer = Observer()
            self.handler = EchoFileHandler(
                self.config, self.indexer, self._on_files_changed
            )

            # Validate and schedule watch paths
            valid_paths = []
            for path in watch_paths:
                if path.exists() and path.is_dir():
                    self.observer.schedule(self.handler, str(path), recursive=True)
                    valid_paths.append(path)
                    logger.info(f"Watching directory: {path}")
                else:
                    logger.warning(f"Cannot watch non-existent directory: {path}")

            if not valid_paths:
                logger.error("No valid directories to watch")
                return False

            self.watch_paths = valid_paths
            self.observer.start()
            self.is_running = True
            self.start_time = datetime.now()

            logger.info(
                f"File watcher started monitoring {len(valid_paths)} directories"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            return False

    def stop_watching(self) -> None:
        """Stop file system watching gracefully."""
        if not self.is_running:
            return

        self.is_running = False

        try:
            # Flush any pending events
            if self.handler:
                self.handler.flush_events()

            # Stop observer
            if self.observer and self.observer.is_alive():
                self.observer.stop()
                self.observer.join(timeout=5)

            logger.info("File watcher stopped")

        except Exception as e:
            logger.error(f"Error stopping file watcher: {e}")

    def _on_files_changed(self, changes: List[FileChangeEvent]) -> None:
        """Handle batch of file changes with comprehensive processing."""
        if not changes:
            return

        start_time = time.time()
        logger.info(f"Processing batch of {len(changes)} file changes")

        # Group changes by type
        changes_by_type = defaultdict(list)
        for change in changes:
            changes_by_type[change.event_type].append(change)

        # Process deletions first
        if "deleted" in changes_by_type:
            self._process_deleted_files(changes_by_type["deleted"])

        # Process moves (handle as delete old + create new)
        if "moved" in changes_by_type:
            self._process_moved_files(changes_by_type["moved"])

        # Process created and modified files
        files_to_index = []
        for event_type in ["created", "modified"]:
            if event_type in changes_by_type:
                for change in changes_by_type[event_type]:
                    files_to_index.append(change.path)

        # Index files if any
        if files_to_index:
            try:
                result = self.indexer.index_files(files_to_index)

                # Update statistics
                self.total_processed += result.files_processed
                self.total_errors += len(result.errors)

                if self.handler:
                    self.handler.stats.files_indexed += result.files_processed
                    self.handler.stats.blocks_extracted += result.blocks_extracted
                    self.handler.stats.errors += len(result.errors)

                processing_time = int((time.time() - start_time) * 1000)

                logger.info(
                    f"Indexed {result.files_processed} files, "
                    f"extracted {result.blocks_extracted} blocks in {processing_time}ms"
                )

                if result.errors:
                    logger.warning(
                        f"Indexing errors: {result.errors[:3]}..."
                    )  # Show first 3 errors

            except Exception as e:
                logger.error(f"Error processing file changes: {e}")
                self.total_errors += len(files_to_index)

    def _process_deleted_files(self, deleted_changes: List[FileChangeEvent]) -> None:
        """Process deleted files by removing from database."""
        for change in deleted_changes:
            try:
                if self.database.delete_file(change.path):
                    logger.debug(f"Removed deleted file from database: {change.path}")
            except Exception as e:
                logger.error(f"Error removing deleted file {change.path}: {e}")

    def _process_moved_files(self, moved_changes: List[FileChangeEvent]) -> None:
        """Process moved files by updating paths in database."""
        for change in moved_changes:
            try:
                # Remove old path and let the indexing process handle the new path
                if change.old_path:
                    self.database.delete_file(change.old_path)
                    logger.debug(
                        f"Processed file move: {change.old_path} -> {change.path}"
                    )
            except Exception as e:
                logger.error(
                    f"Error processing file move {change.old_path} -> {change.path}: {e}"
                )

    def is_watching(self) -> bool:
        """Check if file watcher is active."""
        return (
            self.is_running and self.observer is not None and self.observer.is_alive()
        )

    def get_status(self) -> Dict:
        """Get comprehensive watcher status and statistics."""
        status = {
            "running": self.is_watching(),
            "watch_paths": [str(p) for p in self.watch_paths],
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
        }

        if self.start_time:
            status["uptime_seconds"] = (
                datetime.now() - self.start_time
            ).total_seconds()

        if self.handler:
            handler_stats = self.handler.get_statistics()
            status.update(
                {
                    "events_processed": handler_stats.events_processed,
                    "files_indexed": handler_stats.files_indexed,
                    "blocks_extracted": handler_stats.blocks_extracted,
                    "handler_errors": handler_stats.errors,
                    "last_activity": (
                        handler_stats.last_activity.isoformat()
                        if handler_stats.last_activity
                        else None
                    ),
                }
            )

        return status

    def pause_watching(self) -> None:
        """Temporarily pause file watching without fully stopping."""
        if self.observer:
            self.observer.unschedule_all()
            logger.info("File watching paused")

    def resume_watching(self) -> None:
        """Resume file watching after pause."""
        if self.observer and self.handler:
            for path in self.watch_paths:
                self.observer.schedule(self.handler, str(path), recursive=True)
            logger.info("File watching resumed")


async def watch_repository(
    repo_path: Path,
    config: Optional[EchoConfig] = None,
    callback: Optional[Callable] = None,
) -> None:
    """Watch a repository for file changes and automatically re-index."""
    if config is None:
        config = EchoConfig()

    # Initialize database
    db_path = config.get_cache_dir() / "echo.db"
    database = create_database(db_path, config)

    # Create and start watcher
    watcher = FileWatcher(config, database)

    try:
        # Start watching
        if not watcher.start_watching([repo_path]):
            logger.error("Failed to start repository watching")
            return

        logger.info(f"Started watching repository: {repo_path}")

        # Run indefinitely or until interrupted
        try:
            while watcher.is_watching():
                await asyncio.sleep(1)

                # Optional periodic callback for status updates
                if callback:
                    try:
                        await callback(watcher.get_status())
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

        except KeyboardInterrupt:
            logger.info("Repository watching interrupted")

    finally:
        # Clean shutdown
        watcher.stop_watching()
        database.close()
        logger.info("Repository watching stopped")


class WatcherManager:
    """Manages multiple file watchers for different repositories."""

    def __init__(self, config: Optional[EchoConfig] = None):
        self.config = config or EchoConfig()
        self.watchers: Dict[str, FileWatcher] = {}

    def start_watching_repo(self, repo_path: Path, name: Optional[str] = None) -> bool:
        """Start watching a repository."""
        watch_name = name or str(repo_path)

        if watch_name in self.watchers:
            logger.warning(f"Already watching repository: {watch_name}")
            return False

        # Initialize database for this repo
        db_path = self.config.get_cache_dir() / f"echo_{hash(watch_name) % 10000}.db"
        database = create_database(db_path, self.config)

        # Create watcher
        watcher = FileWatcher(self.config, database)

        if watcher.start_watching([repo_path]):
            self.watchers[watch_name] = watcher
            logger.info(f"Started watching repository '{watch_name}': {repo_path}")
            return True
        else:
            database.close()
            return False

    def stop_watching_repo(self, name: str) -> bool:
        """Stop watching a repository."""
        if name not in self.watchers:
            logger.warning(f"No watcher found for: {name}")
            return False

        watcher = self.watchers.pop(name)
        watcher.stop_watching()
        logger.info(f"Stopped watching repository: {name}")
        return True

    def get_status(self) -> Dict:
        """Get status of all managed watchers."""
        return {name: watcher.get_status() for name, watcher in self.watchers.items()}

    def stop_all(self) -> None:
        """Stop all watchers."""
        for name in list(self.watchers.keys()):
            self.stop_watching_repo(name)
