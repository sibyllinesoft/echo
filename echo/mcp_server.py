"""MCP server implementation for Echo duplicate detection."""

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncIterator, Union
from dataclasses import asdict
import logging
from datetime import datetime

from .config import EchoConfig
from .scan import DuplicateScanner, ScanResult, create_scanner
from .index import index_repository, RepositoryIndexer
from .storage import EchoDatabase, DuplicateFinding, create_database
from .normalize import create_diff_explanation
from .utils import JsonHandler, handle_errors

logger = logging.getLogger(__name__)


class EchoMCPServer:
    """MCP server for Echo duplicate code detection."""
    
    def __init__(self, config: Optional[EchoConfig] = None):
        self.config = config or EchoConfig()
        self.database: Optional[EchoDatabase] = None
        self.scanner: Optional[DuplicateScanner] = None
        self.indexer: Optional[RepositoryIndexer] = None
        self._indexing_in_progress = False
        self._indexing_stats = {}
        self._current_repo_root: Optional[Path] = None
        
    async def initialize(self, repo_root: Optional[Path] = None) -> None:
        """Initialize the MCP server."""
        if repo_root:
            self._current_repo_root = repo_root
            
        cache_dir = self.config.get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = cache_dir / 'echo.db'
        self.database = await asyncio.to_thread(create_database, db_path)
        
        # Initialize scanner and indexer
        self.scanner = create_scanner(self.config, db_path)
        self.indexer = RepositoryIndexer(self.config, self.database)
        
        logger.info(f"Echo MCP server initialized with cache: {cache_dir}")
        
    async def index_repo(self, root: Union[str, Path], reindex: bool = False) -> Dict[str, Any]:
        """Index a repository for duplicate detection."""
        if not self.database:
            await self.initialize()
            
        root_path = Path(root) if isinstance(root, str) else root
        self._current_repo_root = root_path
        
        if not root_path.exists():
            raise ValueError(f"Repository path does not exist: {root_path}")
            
        if self._indexing_in_progress:
            return {
                'started': False,
                'error': 'Indexing already in progress',
                'stats': self._indexing_stats
            }
            
        try:
            self._indexing_in_progress = True
            
            # Run indexing in thread pool to avoid blocking
            result = await asyncio.to_thread(
                self._run_indexing, root_path, reindex
            )
            
            self._indexing_stats = result['stats']
            
            logger.info(f"Completed indexing repository: {root_path}")
            return result
            
        except Exception as e:
            logger.error(f"Repository indexing failed: {e}")
            return {
                'started': False,
                'error': str(e),
                'stats': {}
            }
        finally:
            self._indexing_in_progress = False
        
    async def index_status(self) -> Dict[str, Any]:
        """Get indexing status and statistics."""
        if not self.database:
            await self.initialize()
            
        try:
            # Get database statistics
            stats = await asyncio.to_thread(self.database.get_statistics)
            
            # Check if index is dirty (files changed since last index)
            dirty = False
            if self._current_repo_root:
                dirty = await asyncio.to_thread(self._check_index_dirty)
            
            # Calculate completion percentages for multi-stage processing
            total_blocks = stats.get('total_blocks', 0)
            lsh_indexed = stats.get('lsh_indexed_blocks', 0)
            verified_pairs = stats.get('verified_pairs', 0)
            semantic_pairs = stats.get('semantic_pairs', 0)
            
            lsh_percent = f"{(lsh_indexed * 100 // total_blocks) if total_blocks > 0 else 0}%"
            verify_percent = f"{(verified_pairs * 100 // max(lsh_indexed, 1))}%"
            semantic_percent = f"{(semantic_pairs * 100 // max(verified_pairs, 1))}%"
            
            return {
                'passes': {
                    '0': lsh_percent,     # LSH pass
                    '1': verify_percent,  # Verification pass 
                    '2': semantic_percent # Semantic pass
                },
                'dirty': dirty,
                'errors': stats.get('errors', []),
                'stats': stats,
                'indexing_in_progress': self._indexing_in_progress
            }
            
        except Exception as e:
            logger.error(f"Failed to get index status: {e}")
            return {
                'passes': {'0': '0%', '1': '0%', '2': '0%'},
                'dirty': True,
                'errors': [str(e)],
                'stats': {},
                'indexing_in_progress': self._indexing_in_progress
            }
        
    async def scan_changed(self, paths: Optional[List[Union[str, Path]]] = None,
                          options: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
        """Stream findings for changed files."""
        if not self.database or not self.scanner:
            await self.initialize()
            
        try:
            # Convert string paths to Path objects
            file_paths = []
            if paths:
                file_paths = [Path(p) if isinstance(p, str) else p for p in paths]
            else:
                # Auto-detect changed files if none provided
                if self._current_repo_root:
                    file_paths = await asyncio.to_thread(self._get_changed_files)
                else:
                    yield {
                        'type': 'error',
                        'message': 'No repository root set and no paths provided'
                    }
                    return
            
            if not file_paths:
                yield {
                    'type': 'info',
                    'message': 'No changed files to scan'
                }
                return
                
            yield {
                'type': 'progress',
                'message': f'Starting scan of {len(file_paths)} changed files'
            }
            
            # Run scan in thread pool
            result = await asyncio.to_thread(
                self.scanner.scan_changed_files, file_paths
            )
            
            # Stream findings as JSON resources
            for finding in result.findings:
                resource = await self._finding_to_resource(finding)
                yield {
                    'type': 'finding',
                    'resource': resource
                }
            
            # Final statistics
            yield {
                'type': 'complete',
                'statistics': asdict(result.statistics),
                'total_findings': len(result.findings)
            }
            
        except Exception as e:
            logger.error(f"Changed files scan failed: {e}")
            yield {
                'type': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
        
    async def scan_repo(self, scope: str = 'all',
                       budget_ms: Optional[int] = None,
                       options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Scan repository for duplicates."""
        if not self.database or not self.scanner:
            await self.initialize()
            
        if not self._current_repo_root:
            raise ValueError("No repository root set. Use index_repo first.")
            
        try:
            logger.info(f"Repository scan requested (scope={scope}, budget={budget_ms}ms)")
            
            # Run scan in thread pool to avoid blocking
            result = await asyncio.to_thread(
                self.scanner.scan_repository, 
                self._current_repo_root,
                budget_ms
            )
            
            # Convert findings to JSON resources
            resources = []
            for finding in result.findings:
                resource = await self._finding_to_resource(finding)
                resources.append(resource)
            
            # Apply scope filtering if needed
            if scope != 'all':
                resources = self._filter_by_scope(resources, scope)
            
            logger.info(f"Repository scan completed: {len(resources)} findings")
            return resources
            
        except Exception as e:
            logger.error(f"Repository scan failed: {e}")
            raise
        
    async def explain(self, finding_id: str) -> Dict[str, Any]:
        """Explain a specific duplicate finding."""
        if not self.database:
            await self.initialize()
            
        try:
            # Load finding from database (using available method)
            findings = await asyncio.to_thread(
                self._get_finding_by_id, finding_id
            )
            
            if not findings:
                raise ValueError(f"Finding not found: {finding_id}")
            
            finding = findings[0]
            
            # Load the blocks involved in the finding
            source_block = await asyncio.to_thread(
                self._get_block_by_id, finding.block_id
            )
            match_block = await asyncio.to_thread(
                self._get_block_by_id, finding.match_block_id
            )
            
            if not source_block or not match_block:
                raise ValueError("Could not load blocks for finding")
            
            # Generate code frames with context
            source_frame = await self._generate_codeframe(
                source_block.file_path,
                source_block.start,
                source_block.end
            )
            
            match_frame = await self._generate_codeframe(
                match_block.file_path,
                match_block.start,
                match_block.end
            )
            
            # Generate normalized diff using available function
            # Note: create_diff_explanation expects NormalizedBlock objects
            # For now, provide a simple explanation
            normalized_diff = f"Block comparison between {source_block.file_path}:{source_block.start}-{source_block.end} and {match_block.file_path}:{match_block.start}-{match_block.end}"
            
            return {
                'finding_id': finding_id,
                'normalized_diff': normalized_diff,
                'codeframes': [source_frame, match_frame],
                'scores': finding.scores,
                'type': finding.type,
                'confidence': finding.confidence
            }
            
        except Exception as e:
            logger.error(f"Explain finding failed for {finding_id}: {e}")
            raise
        
    async def configure(self, policy_json: Dict[str, Any]) -> Dict[str, bool]:
        """Update configuration from JSON policy."""
        try:
            # Validate and apply configuration changes
            await asyncio.to_thread(self._update_config, policy_json)
            
            # Reinitialize scanner with new config if needed
            if self.database:
                cache_dir = self.config.get_cache_dir()
                db_path = cache_dir / 'echo.db'
                self.scanner = create_scanner(self.config, db_path)
                
            logger.info("Configuration updated from policy")
            return {'ok': True}
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return {'ok': False, 'error': str(e)}
            
    async def clear_index(self) -> Dict[str, bool]:
        """Clear the duplicate detection index."""
        if not self.database:
            await self.initialize()
            
        try:
            # Clear database tables (implement basic clear functionality)
            await asyncio.to_thread(self._clear_database)
            
            # Clear FAISS index if exists
            faiss_path = self.config.get_cache_dir() / 'faiss'
            if faiss_path.exists():
                await asyncio.to_thread(self._clear_faiss_index, faiss_path)
            
            # Reset indexing stats
            self._indexing_stats = {}
            
            logger.info("Index cleared successfully")
            return {'ok': True}
            
        except Exception as e:
            logger.error(f"Index clear failed: {e}")
            return {'ok': False, 'error': str(e)}
    
    # Helper methods
    
    def _run_indexing(self, root_path: Path, reindex: bool) -> Dict[str, Any]:
        """Run repository indexing synchronously."""
        try:
            # Use RepositoryIndexer directly since the standalone function isn't implemented yet
            if self.indexer is None:
                self.indexer = RepositoryIndexer(self.config, self.database)
            
            result = self.indexer.index_repository(root_path, reindex)
            
            return {
                'started': True,
                'stats': {
                    'files_processed': result.files_processed,
                    'blocks_extracted': result.blocks_extracted,
                    'blocks_normalized': getattr(result, 'blocks_normalized', 0),
                    'processing_time_ms': result.processing_time_ms,
                    'errors': result.errors
                }
            }
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return {
                'started': False,
                'error': str(e),
                'stats': {}
            }
    
    def _check_index_dirty(self) -> bool:
        """Check if repository has changes since last index."""
        try:
            # Simple implementation - check if any tracked files are newer than index
            # In production, this would use git status or file modification times
            return False  # Placeholder
        except Exception:
            return True
    
    def _get_changed_files(self) -> List[Path]:
        """Get list of changed files in repository."""
        try:
            # Placeholder - would use git diff or file watching
            return []
        except Exception:
            return []
    
    async def _finding_to_resource(self, finding: DuplicateFinding) -> Dict[str, Any]:
        """Convert DuplicateFinding to MCP resource format."""
        try:
            # Load source and match blocks
            source_block = await asyncio.to_thread(
                self._get_block_by_id, finding.block_id
            )
            match_block = await asyncio.to_thread(
                self._get_block_by_id, finding.match_block_id
            )
            
            if not source_block or not match_block:
                raise ValueError("Could not load blocks for finding")
            
            # Build resource in required JSON format
            resource = {
                'id': finding.id,
                'unit': {
                    'lang': source_block.lang,
                    'path': str(source_block.file_path),
                    'start': source_block.start,
                    'end': source_block.end
                },
                'matches': [{
                    'path': str(match_block.file_path),
                    'start': match_block.start,
                    'end': match_block.end,
                    'scores': {
                        'jaccard': finding.scores.get('jaccard_score', 0.0),
                        'overlap': finding.scores.get('overlap_score', 0.0),
                        'cosine': finding.scores.get('semantic_similarity', 0.0),
                        'R': finding.refactor_score
                    },
                    'type': finding.type
                }]
            }
            
            return resource
            
        except Exception as e:
            logger.error(f"Failed to convert finding to resource: {e}")
            raise
    
    def _filter_by_scope(self, resources: List[Dict[str, Any]], scope: str) -> List[Dict[str, Any]]:
        """Filter resources by scope (e.g., 'high_confidence', 'semantic_only')."""
        if scope == 'all':
            return resources
        elif scope == 'high_confidence':
            return [r for r in resources if r['matches'][0]['scores']['R'] >= 1000]
        elif scope == 'semantic_only':
            return [r for r in resources if r['matches'][0]['type'] == 'semantic']
        else:
            return resources
    
    async def _generate_codeframe(self, file_path: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """Generate a code frame with context lines."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            content = await asyncio.to_thread(path.read_text, encoding='utf-8')
            lines = content.splitlines()
            
            # Add context lines (3 before and after)
            context_start = max(0, start_line - 4)  # -1 for 0-based, -3 for context
            context_end = min(len(lines), end_line + 3)
            
            context_lines = lines[context_start:context_end]
            
            return {
                'file': file_path,
                'start_line': start_line,
                'end_line': end_line,
                'context_start': context_start + 1,  # 1-based for display
                'context_end': context_end,
                'content': '\n'.join(context_lines)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate codeframe for {file_path}: {e}")
            return {
                'file': file_path,
                'start_line': start_line,
                'end_line': end_line,
                'content': f'# Error loading content: {e}',
                'error': str(e)
            }
    
    def _get_finding_by_id(self, finding_id: str) -> List[DuplicateFinding]:
        """Get finding by ID from database."""
        try:
            # Query database for finding record by ID
            with self.database.get_session() as session:
                from .storage import FindingRecord
                finding_record = session.query(FindingRecord).filter_by(id=finding_id).first()
                
                if finding_record:
                    # Convert to DuplicateFinding
                    scores = JsonHandler.deserialize(finding_record.scores_json or "{}")
                    finding = DuplicateFinding(
                        id=finding_record.id,
                        block_id=finding_record.block_id,
                        match_block_id=finding_record.match_block_id,
                        scores=scores,
                        type=finding_record.type,
                        confidence=scores.get('confidence', 0.0),
                        refactor_score=scores.get('R', 0.0),
                        created_at=finding_record.created_at
                    )
                    return [finding]
                return []
        except Exception as e:
            logger.error(f"Failed to get finding {finding_id}: {e}")
            return []
    
    def _get_block_by_id(self, block_id: str):
        """Get block by ID from database."""
        try:
            # For now, we'll search by hash since that's what we have
            blocks = self.database.get_blocks_by_hash(block_id)
            return blocks[0] if blocks else None
        except Exception as e:
            logger.error(f"Failed to get block {block_id}: {e}")
            return None
    
    def _clear_database(self):
        """Clear all data from database."""
        try:
            with self.database.get_session() as session:
                from .storage import FileRecord, BlockRecord, FindingRecord, MinHashRecord
                
                # Delete all records
                session.query(FindingRecord).delete()
                session.query(MinHashRecord).delete()
                session.query(BlockRecord).delete()
                session.query(FileRecord).delete()
                
                session.commit()
                logger.info("Cleared all database records")
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            raise
    
    def _update_config(self, policy_json: Dict[str, Any]) -> None:
        """Update configuration from policy JSON."""
        # Validate and apply configuration changes
        valid_keys = {
            'min_tokens', 'tau_semantic', 'overlap_threshold', 
            'edit_density_threshold', 'max_candidates', 'min_refactor_score',
            'ignore_patterns'
        }
        
        for key, value in policy_json.items():
            if key in valid_keys:
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"Updated config {key} = {value}")
    
    def _clear_faiss_index(self, faiss_path: Path) -> None:
        """Clear FAISS index files."""
        import shutil
        if faiss_path.exists():
            shutil.rmtree(faiss_path)
            logger.info(f"Cleared FAISS index at {faiss_path}")


# MCP Protocol Implementation
class MCPProtocolHandler:
    """Handle MCP JSON-RPC protocol communication."""
    
    def __init__(self, server: EchoMCPServer):
        self.server = server
        self.tools = {
            'index_repo': self._handle_index_repo,
            'index_status': self._handle_index_status,
            'scan_changed': self._handle_scan_changed,
            'scan_repo': self._handle_scan_repo,
            'explain': self._handle_explain,
            'configure': self._handle_configure,
            'clear_index': self._handle_clear_index
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request."""
        try:
            method = request.get('method')
            params = request.get('params', {})
            request_id = request.get('id')
            
            if method in self.tools:
                result = await self.tools[method](params)
                return {
                    'jsonrpc': '2.0',
                    'id': request_id,
                    'result': result
                }
            else:
                return {
                    'jsonrpc': '2.0',
                    'id': request_id,
                    'error': {
                        'code': -32601,
                        'message': f'Method not found: {method}'
                    }
                }
                
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return {
                'jsonrpc': '2.0',
                'id': request.get('id'),
                'error': {
                    'code': -32603,
                    'message': 'Internal error',
                    'data': str(e)
                }
            }
    
    async def _handle_index_repo(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle index_repo tool call."""
        root = params.get('root')
        reindex = params.get('reindex', False)
        
        if not root:
            raise ValueError("'root' parameter is required")
        
        return await self.server.index_repo(root, reindex)
    
    async def _handle_index_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle index_status tool call."""
        return await self.server.index_status()
    
    async def _handle_scan_changed(self, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Handle scan_changed tool call (streaming)."""
        paths = params.get('paths')
        options = params.get('options', {})
        
        async for result in self.server.scan_changed(paths, options):
            yield result
    
    async def _handle_scan_repo(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle scan_repo tool call."""
        scope = params.get('scope', 'all')
        budget_ms = params.get('budget_ms')
        options = params.get('options', {})
        
        return await self.server.scan_repo(scope, budget_ms, options)
    
    async def _handle_explain(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle explain tool call."""
        finding_id = params.get('finding_id')
        
        if not finding_id:
            raise ValueError("'finding_id' parameter is required")
        
        return await self.server.explain(finding_id)
    
    async def _handle_configure(self, params: Dict[str, Any]) -> Dict[str, bool]:
        """Handle configure tool call."""
        policy_json = params.get('policy_json', {})
        return await self.server.configure(policy_json)
    
    async def _handle_clear_index(self, params: Dict[str, Any]) -> Dict[str, bool]:
        """Handle clear_index tool call."""
        return await self.server.clear_index()


async def serve_mcp(server: EchoMCPServer) -> None:
    """Serve the MCP protocol via stdin/stdout."""
    handler = MCPProtocolHandler(server)
    
    await server.initialize()
    logger.info("Echo MCP server ready")
    
    try:
        # Read JSON-RPC requests from stdin and write responses to stdout
        while True:
            try:
                # Read line from stdin
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                # Parse JSON-RPC request
                request = json.loads(line)
                
                # Handle special case for streaming methods
                if request.get('method') == 'scan_changed':
                    # Handle streaming response
                    async for result in handler._handle_scan_changed(request.get('params', {})):
                        response = {
                            'jsonrpc': '2.0',
                            'id': request.get('id'),
                            'result': result
                        }
                        print(json.dumps(response), flush=True)
                    
                    # Send final completion marker
                    response = {
                        'jsonrpc': '2.0',
                        'id': request.get('id'),
                        'result': {'type': 'stream_complete'}
                    }
                    print(json.dumps(response), flush=True)
                else:
                    # Handle regular request
                    response = await handler.handle_request(request)
                    print(json.dumps(response), flush=True)
                    
            except json.JSONDecodeError as e:
                error_response = {
                    'jsonrpc': '2.0',
                    'id': None,
                    'error': {
                        'code': -32700,
                        'message': 'Parse error',
                        'data': str(e)
                    }
                }
                print(json.dumps(error_response), flush=True)
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                error_response = {
                    'jsonrpc': '2.0',
                    'id': None,
                    'error': {
                        'code': -32603,
                        'message': 'Internal error',
                        'data': str(e)
                    }
                }
                print(json.dumps(error_response), flush=True)
                
    except KeyboardInterrupt:
        logger.info("MCP server shutting down")
    except Exception as e:
        logger.error(f"MCP server error: {e}")
        
        
if __name__ == '__main__':
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    server = EchoMCPServer()
    
    try:
        asyncio.run(serve_mcp(server))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)