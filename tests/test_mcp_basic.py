#!/usr/bin/env python3
"""Basic test for MCP server functionality without heavy dependencies using proper pytest isolation."""

import json
import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# Simple mock classes to test the MCP server logic without dependencies
class MockConfig:
    def __init__(self):
        self.min_tokens = 40
        self.tau_semantic = 0.83
        self.overlap_threshold = 0.75
        self.edit_density_threshold = 0.25
        self.max_candidates = 50
        self.min_refactor_score = 200
        self.ignore_patterns = []
    
    def get_cache_dir(self):
        return Path("/tmp/echo_test")


class MockDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def get_statistics(self):
        return {
            'total_blocks': 100,
            'lsh_indexed_blocks': 90,
            'verified_pairs': 20,
            'semantic_pairs': 15,
            'errors': []
        }
    
    def get_session(self):
        return MockSession()


class MockSession:
    def query(self, model):
        return MockQuery()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class MockQuery:
    def filter_by(self, **kwargs):
        return self
    
    def first(self):
        return None
    
    def delete(self):
        return 0


class MockScanner:
    def scan_repository(self, repo_path, budget_ms=None):
        return MockScanResult()
    
    def scan_changed_files(self, file_paths):
        return MockScanResult()


class MockScanResult:
    def __init__(self):
        self.findings = []
        self.statistics = MockStatistics()


class MockStatistics:
    def __init__(self):
        pass


class MockIndexer:
    def __init__(self, config=None, database=None):
        self.config = config
        self.database = database
    
    def index_repository(self, repo_path, reindex=False):
        return MockIndexingResult()


class MockIndexingResult:
    def __init__(self):
        self.files_processed = 10
        self.blocks_extracted = 50
        self.processing_time_ms = 1000
        self.errors = []


def mock_create_diff_explanation(a, b):
    return "Mock diff explanation"


# Test fixtures with proper mocking
@pytest.fixture
def mocked_dependencies():
    """Fixture that mocks all the heavy dependencies for MCP tests."""
    with patch('echo.config.EchoConfig', MockConfig), \
         patch('echo.scan.DuplicateScanner', MockScanner), \
         patch('echo.scan.create_scanner', lambda config, db_path: MockScanner()), \
         patch('echo.index.RepositoryIndexer', MockIndexer), \
         patch('echo.storage.create_database', lambda db_path: MockDatabase(db_path)), \
         patch('echo.normalize.create_diff_explanation', mock_create_diff_explanation):
        yield


@pytest.mark.asyncio
async def test_basic_functionality(mocked_dependencies):
    """Test basic MCP server functionality."""
    from echo.mcp_server import EchoMCPServer
    
    # Create server instance
    server = EchoMCPServer()
    
    # Test initialization
    await server.initialize(Path("/tmp/test_repo"))
    
    # Test index status
    status = await server.index_status()
    assert isinstance(status, dict)
    
    # Test configuration update
    config_result = await server.configure({
        'min_tokens': 50,
        'tau_semantic': 0.9
    })
    assert isinstance(config_result, dict)
    
    # Test clear index
    clear_result = await server.clear_index()
    assert isinstance(clear_result, dict)


@pytest.mark.asyncio
async def test_protocol_handler(mocked_dependencies):
    """Test MCP protocol handler."""
    from echo.mcp_server import EchoMCPServer, MCPProtocolHandler
    
    server = EchoMCPServer()
    await server.initialize()
    
    handler = MCPProtocolHandler(server)
    
    # Test index_status request
    request = {
        'method': 'index_status',
        'params': {},
        'id': 1
    }
    
    response = await handler.handle_request(request)
    assert isinstance(response, dict)
    assert 'result' in response
    
    # Test unknown method
    request = {
        'method': 'unknown_method',
        'params': {},
        'id': 2
    }
    
    response = await handler.handle_request(request)
    assert isinstance(response, dict)
    assert 'error' in response


def test_json_serialization():
    """Test JSON serialization of responses."""
    # Test a typical response
    response = {
        'jsonrpc': '2.0',
        'id': 1,
        'result': {
            'passes': {'0': '90%', '1': '80%', '2': '75%'},
            'dirty': False,
            'errors': [],
            'stats': {'total_blocks': 100}
        }
    }
    
    # Test serialization and deserialization
    json_str = json.dumps(response)
    parsed = json.loads(json_str)
    assert parsed == response
    assert isinstance(json_str, str)
    assert len(json_str) > 0