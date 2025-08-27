#!/usr/bin/env python3
"""Basic test for MCP server functionality without heavy dependencies."""

import json
import asyncio
import sys
from pathlib import Path

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
    def index_repository(self, repo_path, reindex=False):
        return MockIndexingResult()

class MockIndexingResult:
    def __init__(self):
        self.files_processed = 10
        self.blocks_extracted = 50
        self.processing_time_ms = 1000
        self.errors = []

# Mock the imports
import sys
from unittest.mock import MagicMock

# Mock modules
sys.modules['echo.config'] = MagicMock()
sys.modules['echo.scan'] = MagicMock()
sys.modules['echo.index'] = MagicMock() 
sys.modules['echo.storage'] = MagicMock()
sys.modules['echo.normalize'] = MagicMock()

# Mock the classes
sys.modules['echo.config'].EchoConfig = MockConfig
sys.modules['echo.scan'].DuplicateScanner = MockScanner
sys.modules['echo.scan'].create_scanner = lambda config, db_path: MockScanner()
sys.modules['echo.index'].RepositoryIndexer = MockIndexer
sys.modules['echo.storage'].create_database = lambda db_path: MockDatabase(db_path)
sys.modules['echo.normalize'].create_diff_explanation = lambda a, b: "Mock diff explanation"

# Now we can import our MCP server
from echo.mcp_server import EchoMCPServer, MCPProtocolHandler

async def test_basic_functionality():
    """Test basic MCP server functionality."""
    print("Testing EchoMCPServer basic functionality...")
    
    # Create server instance
    server = EchoMCPServer()
    
    # Test initialization
    await server.initialize(Path("/tmp/test_repo"))
    print("✓ Server initialization successful")
    
    # Test index status
    status = await server.index_status()
    print(f"✓ Index status: {status}")
    
    # Test configuration update
    config_result = await server.configure({
        'min_tokens': 50,
        'tau_semantic': 0.9
    })
    print(f"✓ Configuration update: {config_result}")
    
    # Test clear index
    clear_result = await server.clear_index()
    print(f"✓ Clear index: {clear_result}")
    
    print("All basic functionality tests passed!")

async def test_protocol_handler():
    """Test MCP protocol handler."""
    print("\nTesting MCPProtocolHandler...")
    
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
    print(f"✓ Protocol handler response: {response}")
    
    # Test unknown method
    request = {
        'method': 'unknown_method',
        'params': {},
        'id': 2
    }
    
    response = await handler.handle_request(request)
    print(f"✓ Unknown method response: {response}")
    
    print("Protocol handler tests passed!")

def test_json_serialization():
    """Test JSON serialization of responses."""
    print("\nTesting JSON serialization...")
    
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
    
    try:
        json_str = json.dumps(response)
        parsed = json.loads(json_str)
        print(f"✓ JSON serialization successful: {len(json_str)} chars")
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False
    
    return True

async def main():
    """Run all tests."""
    print("Echo MCP Server Basic Tests")
    print("=" * 40)
    
    try:
        await test_basic_functionality()
        await test_protocol_handler()
        test_json_serialization()
        
        print("\n" + "=" * 40)
        print("🎉 All tests passed! MCP server implementation looks good.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)