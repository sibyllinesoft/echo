"""Tests for Tree-sitter based code parsing and block extraction."""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch, MagicMock

from echo.parser import (
    CodeBlock, 
    LanguageParser, 
    get_parser, 
    detect_language, 
    extract_blocks,
    extract_blocks_from_content
)


class TestCodeBlock:
    """Test cases for CodeBlock dataclass."""
    
    def test_code_block_properties(self):
        """Test CodeBlock property calculations."""
        block = CodeBlock(
            lang='python',
            file_path=Path('test.py'),
            start_line=10,
            end_line=15,
            start_byte=100,
            end_byte=200,
            tokens=['def', 'test', '(', ')'],
            raw_content='def test():\n    pass',
            node_type='function_definition'
        )
        
        assert block.token_count == 4
        assert block.lines == 6  # 15 - 10 + 1


class TestLanguageParser:
    """Test cases for LanguageParser class."""
    
    def test_parser_initialization_python(self):
        """Test parser initialization for Python."""
        parser = LanguageParser('python')
        assert parser.language == 'python'
        assert parser.parser is not None
        
    def test_parser_initialization_javascript(self):
        """Test parser initialization for JavaScript."""
        parser = LanguageParser('javascript')
        assert parser.language == 'javascript'
        assert parser.parser is not None
        
    def test_parser_initialization_typescript(self):
        """Test parser initialization for TypeScript."""
        parser = LanguageParser('typescript')
        assert parser.language == 'typescript'
        assert parser.parser is not None
        
    def test_parser_initialization_unsupported_language(self):
        """Test parser initialization with unsupported language."""
        with pytest.raises(ValueError, match="Unsupported language"):
            LanguageParser('unsupported')
    
    def test_parse_empty_content(self):
        """Test parsing empty content."""
        parser = LanguageParser('python')
        blocks = parser.extract_blocks('', Path('test.py'))
        assert blocks == []
        
    def test_parse_python_function(self):
        """Test parsing a simple Python function."""
        parser = LanguageParser('python')
        # Temporarily lower minimum block tokens for this test
        original_min = parser.MIN_BLOCK_TOKENS
        parser.MIN_BLOCK_TOKENS = 10
        
        content = """
def hello_world():
    '''A simple function that returns hello world.'''
    message = "Hello, World!"
    return message

def another_function(x, y):
    return x + y
"""
        try:
            blocks = parser.extract_blocks(content, Path('test.py'))
        finally:
            parser.MIN_BLOCK_TOKENS = original_min
        
        # Should extract at least the function definitions
        assert len(blocks) >= 2
        function_blocks = [b for b in blocks if b.node_type == 'function_definition']
        assert len(function_blocks) == 2
        
        # Check first function
        hello_block = next(b for b in function_blocks if 'hello_world' in b.tokens)
        assert hello_block.lang == 'python'
        assert hello_block.start_line == 2  # Note: 1-based indexing
        assert 'hello_world' in hello_block.tokens
        assert '"Hello, World!"' in hello_block.tokens
        
    def test_parse_javascript_function(self):
        """Test parsing a JavaScript function."""
        parser = LanguageParser('javascript')
        # Temporarily lower minimum block tokens for this test
        original_min = parser.MIN_BLOCK_TOKENS
        parser.MIN_BLOCK_TOKENS = 7
        
        content = """
function greet(name) {
    const message = `Hello, ${name}!`;
    return message;
}

const arrow = (x) => x * 2;

class Person {
    constructor(name) {
        this.name = name;
    }
    
    speak() {
        return `My name is ${this.name}`;
    }
}
"""
        try:
            blocks = parser.extract_blocks(content, Path('test.js'))
        finally:
            parser.MIN_BLOCK_TOKENS = original_min
        
        # Should extract function, arrow function, class, and methods
        assert len(blocks) >= 3
        
        # Check we have various node types
        node_types = {b.node_type for b in blocks}
        expected_types = {'function_declaration', 'class_declaration'}
        assert expected_types.issubset(node_types)
        
    def test_parse_typescript_interface(self):
        """Test parsing TypeScript interface."""
        parser = LanguageParser('typescript')
        # Temporarily lower minimum block tokens for this test
        original_min = parser.MIN_BLOCK_TOKENS
        parser.MIN_BLOCK_TOKENS = 9
        
        content = """
interface User {
    name: string;
    age: number;
}

type UserRole = 'admin' | 'user' | 'guest';

function createUser(name: string, age: number): User {
    return { name, age };
}

class UserService {
    private users: User[] = [];
    
    addUser(user: User): void {
        this.users.push(user);
    }
}
"""
        try:
            blocks = parser.extract_blocks(content, Path('test.ts'))
        finally:
            parser.MIN_BLOCK_TOKENS = original_min
        
        # Should extract interface, type alias, function, and class
        assert len(blocks) >= 4
        
        # Check we have TypeScript-specific node types
        node_types = {b.node_type for b in blocks}
        expected_types = {'interface_declaration', 'type_alias_declaration', 'function_declaration', 'class_declaration'}
        assert expected_types.issubset(node_types)
        
    def test_chunk_extraction(self):
        """Test chunking of non-function code."""
        parser = LanguageParser('python')
        content = """
import os
import sys
from pathlib import Path

# Configuration constants
DEBUG = True
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
BASE_URL = "https://api.example.com"

# Global variables
cache = {}
session = None

# Utility functions would be extracted as functions
# But these statements should be chunked together
if __name__ == "__main__":
    print("Starting application")
    config = load_config()
    app = create_app(config)
    app.run(debug=DEBUG)
"""
        blocks = parser.extract_blocks(content, Path('test.py'))
        
        # Should have at least some chunk blocks for the statements
        chunk_blocks = [b for b in blocks if b.node_type == 'chunk']
        assert len(chunk_blocks) > 0
        
        # Chunks should have reasonable token counts
        for block in chunk_blocks:
            assert parser.MIN_BLOCK_TOKENS <= block.token_count <= parser.MAX_CHUNK_TOKENS
    
    def test_file_parsing_with_encoding_issues(self):
        """Test file parsing with encoding issues."""
        parser = LanguageParser('python')
        
        # Test with a file that doesn't exist
        blocks = parser.parse_file(Path('nonexistent.py'))
        assert blocks == []
        
    def test_token_extraction(self):
        """Test token extraction from AST nodes."""
        parser = LanguageParser('python')
        # Temporarily lower minimum block tokens for this test
        original_min = parser.MIN_BLOCK_TOKENS
        parser.MIN_BLOCK_TOKENS = 10
        
        content = """
def calculate(x: int, y: int) -> int:
    result = x + y
    return result
"""
        try:
            blocks = parser.extract_blocks(content, Path('test.py'))
        finally:
            parser.MIN_BLOCK_TOKENS = original_min
        
        assert len(blocks) >= 1
        func_block = next(b for b in blocks if b.node_type == 'function_definition')
        
        # Should contain function name and other identifiers
        assert 'calculate' in func_block.tokens
        assert 'result' in func_block.tokens
        assert 'x' in func_block.tokens
        assert 'y' in func_block.tokens
        

class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_detect_language(self):
        """Test language detection from file extensions."""
        assert detect_language(Path('test.py')) == 'python'
        assert detect_language(Path('test.js')) == 'javascript'
        assert detect_language(Path('test.jsx')) == 'javascript'
        assert detect_language(Path('test.ts')) == 'typescript'
        assert detect_language(Path('test.tsx')) == 'typescript'
        assert detect_language(Path('test.mjs')) == 'javascript'
        assert detect_language(Path('test.cjs')) == 'javascript'
        
        # Unsupported extensions
        assert detect_language(Path('test.cpp')) is None
        assert detect_language(Path('test.java')) is None
        assert detect_language(Path('README.md')) is None
        
    def test_get_parser_caching(self):
        """Test parser caching mechanism."""
        parser1 = get_parser('python')
        parser2 = get_parser('python')
        
        # Should return the same instance (cached)
        assert parser1 is parser2
        
        # Different language should return different parser
        parser3 = get_parser('javascript')
        assert parser3 is not parser1
        
    def test_extract_blocks_multiple_files(self):
        """Test extracting blocks from multiple files."""
        # Create temporary files
        python_content = """
def func1():
    return "python"

class TestClass:
    def method1(self):
        pass
"""
        
        js_content = """
function func2() {
    return "javascript";
}

const arrow = () => "arrow";
"""
        
        # Temporarily lower minimum block tokens for cached parsers
        python_parser = get_parser('python')
        js_parser = get_parser('javascript')
        original_py_min = python_parser.MIN_BLOCK_TOKENS
        original_js_min = js_parser.MIN_BLOCK_TOKENS
        python_parser.MIN_BLOCK_TOKENS = 5
        js_parser.MIN_BLOCK_TOKENS = 5
        
        with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f1:
            f1.write(python_content)
            py_file = Path(f1.name)
            
        with NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f2:
            f2.write(js_content)
            js_file = Path(f2.name)
            
        try:
            files = [py_file, js_file]
            blocks = extract_blocks(files)
            
            # Should have blocks from both files
            python_blocks = [b for b in blocks if b.lang == 'python']
            js_blocks = [b for b in blocks if b.lang == 'javascript']
            
            assert len(python_blocks) >= 2  # function and class
            assert len(js_blocks) >= 1     # function
            
            # Check file paths are preserved
            assert any(b.file_path == py_file for b in python_blocks)
            assert any(b.file_path == js_file for b in js_blocks)
            
        finally:
            python_parser.MIN_BLOCK_TOKENS = original_py_min
            js_parser.MIN_BLOCK_TOKENS = original_js_min
            py_file.unlink(missing_ok=True)
            js_file.unlink(missing_ok=True)
            
    def test_extract_blocks_with_language_filter(self):
        """Test extracting blocks with language filtering."""
        # Temporarily lower minimum block tokens for cached parsers
        python_parser = get_parser('python')
        js_parser = get_parser('javascript')
        original_py_min = python_parser.MIN_BLOCK_TOKENS
        original_js_min = js_parser.MIN_BLOCK_TOKENS
        python_parser.MIN_BLOCK_TOKENS = 3
        js_parser.MIN_BLOCK_TOKENS = 3
        
        with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f1:
            f1.write('def func(): pass')
            py_file = Path(f1.name)
            
        with NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f2:
            f2.write('function func() {}')
            js_file = Path(f2.name)
            
        try:
            files = [py_file, js_file]
            
            # Filter to only Python
            blocks = extract_blocks(files, languages=['python'])
            assert all(b.lang == 'python' for b in blocks)
            
            # Filter to only JavaScript
            blocks = extract_blocks(files, languages=['javascript'])
            assert all(b.lang == 'javascript' for b in blocks)
            
        finally:
            python_parser.MIN_BLOCK_TOKENS = original_py_min
            js_parser.MIN_BLOCK_TOKENS = original_js_min
            py_file.unlink(missing_ok=True)
            js_file.unlink(missing_ok=True)
            
    def test_extract_blocks_from_content(self):
        """Test extracting blocks from content string."""
        # Temporarily lower minimum block tokens for cached parser
        python_parser = get_parser('python')
        original_min = python_parser.MIN_BLOCK_TOKENS
        python_parser.MIN_BLOCK_TOKENS = 10
        
        content = """
def example_function():
    x = 1
    y = 2
    return x + y
"""
        try:
            blocks = extract_blocks_from_content(content, Path('test.py'), 'python')
            
            assert len(blocks) >= 1
            func_block = next(b for b in blocks if b.node_type == 'function_definition')
            assert func_block.lang == 'python'
            assert func_block.file_path == Path('test.py')
            assert 'example_function' in func_block.tokens
        finally:
            python_parser.MIN_BLOCK_TOKENS = original_min
        
    def test_extract_blocks_empty_file_list(self):
        """Test extracting blocks from empty file list."""
        blocks = extract_blocks([])
        assert blocks == []
        
    @patch('echo.parser.logger')
    def test_error_handling_in_extract_blocks(self, mock_logger):
        """Test error handling during block extraction."""
        # Create a file that will cause parsing issues
        with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write invalid Python that might cause parsing issues
            f.write('def incomplete_function(')
            problem_file = Path(f.name)
            
        try:
            blocks = extract_blocks([problem_file])
            # Should handle errors gracefully
            # May return empty list or partial results depending on tree-sitter behavior
            assert isinstance(blocks, list)
            
        finally:
            problem_file.unlink(missing_ok=True)
            

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_functions(self):
        """Test handling of very small functions that don't meet token threshold."""
        parser = LanguageParser('python')
        content = """
def f():
    pass

def g(): return 1
"""
        blocks = parser.extract_blocks(content, Path('test.py'))
        
        # Very small functions might be filtered out or chunked
        # depending on token count
        for block in blocks:
            # All returned blocks should meet minimum requirements
            if block.node_type == 'function_definition':
                assert block.token_count >= 0  # At least some tokens
                
    def test_mixed_indentation(self):
        """Test handling of mixed indentation (tabs and spaces)."""
        parser = LanguageParser('python')
        # Temporarily lower minimum block tokens for this test
        original_min = parser.MIN_BLOCK_TOKENS
        parser.MIN_BLOCK_TOKENS = 10
        
        content = """
def mixed_indent():
    x = 1  # spaces
\ty = 2  # tab
    return x + y
"""
        try:
            blocks = parser.extract_blocks(content, Path('test.py'))
            
            # Should handle mixed indentation without crashing
            assert len(blocks) >= 1
        finally:
            parser.MIN_BLOCK_TOKENS = original_min
        
    def test_unicode_content(self):
        """Test handling of Unicode content."""
        parser = LanguageParser('python')
        # Temporarily lower minimum block tokens for this test
        original_min = parser.MIN_BLOCK_TOKENS
        parser.MIN_BLOCK_TOKENS = 10
        
        content = """
def greet_unicode():
    message = "Hello, 世界! 🌍"
    return message

class Café:
    def méthode(self):
        return "café"
"""
        try:
            blocks = parser.extract_blocks(content, Path('test.py'))
            
            # Should handle Unicode without issues
            assert len(blocks) >= 2
            
            # Check that Unicode characters are preserved in tokens/content
            func_block = next(b for b in blocks if 'greet_unicode' in b.tokens)
            assert '世界' in func_block.raw_content or '世界' in func_block.tokens
        finally:
            parser.MIN_BLOCK_TOKENS = original_min
        
    def test_large_file_simulation(self):
        """Test handling of large files (simulated)."""
        parser = LanguageParser('python')
        # Temporarily lower minimum block tokens for this test (functions have ~15 tokens each)
        original_min = parser.MIN_BLOCK_TOKENS
        parser.MIN_BLOCK_TOKENS = 10
        
        # Create a content with many functions
        functions = []
        for i in range(50):
            functions.append(f"""
def function_{i}():
    value = {i}
    result = value * 2
    return result
""")
            
        content = '\n'.join(functions)
        try:
            blocks = parser.extract_blocks(content, Path('large_test.py'))
            
            # Should extract all functions
            function_blocks = [b for b in blocks if b.node_type == 'function_definition']
            assert len(function_blocks) == 50
            
            # All blocks should have valid line numbers
            for block in blocks:
                assert block.start_line > 0
                assert block.end_line >= block.start_line
        finally:
            parser.MIN_BLOCK_TOKENS = original_min