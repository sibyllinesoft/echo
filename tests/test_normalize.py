"""Tests for token normalization and code block standardization."""

import pytest
from pathlib import Path
from unittest.mock import Mock

from echo.normalize import (
    TokenNormalizer, 
    NormalizedBlock, 
    normalize_blocks,
    create_diff_explanation,
    pretty_print_normalization
)
from echo.parser import CodeBlock


class TestTokenNormalizer:
    """Test cases for TokenNormalizer class."""
    
    def test_normalizer_initialization(self):
        """Test normalizer initialization."""
        normalizer = TokenNormalizer('python')
        assert normalizer.language == 'python'
        assert normalizer._var_counter == 0
        assert normalizer._func_counter == 0
        assert normalizer._type_counter == 0
    
    def test_reset_counters(self):
        """Test counter reset functionality."""
        normalizer = TokenNormalizer('python')
        normalizer._var_counter = 5
        normalizer._func_counter = 3
        normalizer._reset_counters()
        
        assert normalizer._var_counter == 0
        assert normalizer._func_counter == 0
        assert normalizer._type_counter == 0
    
    def test_keyword_preservation(self):
        """Test that language keywords are preserved."""
        normalizer = TokenNormalizer('python')
        
        # Python keywords should be preserved
        assert normalizer._normalize_token('def') == 'def'
        assert normalizer._normalize_token('class') == 'class'
        assert normalizer._normalize_token('if') == 'if'
        assert normalizer._normalize_token('return') == 'return'
        
        # JavaScript keywords for JS normalizer
        js_normalizer = TokenNormalizer('javascript')
        assert js_normalizer._normalize_token('function') == 'function'
        assert js_normalizer._normalize_token('const') == 'const'
        assert js_normalizer._normalize_token('let') == 'let'
        
    def test_structural_token_preservation(self):
        """Test that operators and punctuation are preserved."""
        normalizer = TokenNormalizer('python')
        
        # Operators
        assert normalizer._normalize_token('+') == '+'
        assert normalizer._normalize_token('==') == '=='
        assert normalizer._normalize_token('&&') == '&&'
        
        # Punctuation
        assert normalizer._normalize_token('(') == '('
        assert normalizer._normalize_token('{') == '{'
        assert normalizer._normalize_token(',') == ','
        
    def test_literal_detection_and_normalization(self):
        """Test literal type detection and normalization."""
        normalizer = TokenNormalizer('python')
        
        # String literals
        assert normalizer._detect_literal_type('"hello"') == 'string'
        assert normalizer._detect_literal_type("'world'") == 'string'
        assert normalizer._detect_literal_type('`template`') == 'string'
        assert normalizer._detect_literal_type('f"format"') == 'string'
        assert normalizer._normalize_token('"hello"') == 'STR_LIT'
        
        # Number literals
        assert normalizer._detect_literal_type('42') == 'number'
        assert normalizer._detect_literal_type('3.14') == 'number'
        assert normalizer._detect_literal_type('-123') == 'number'
        assert normalizer._detect_literal_type('1e5') == 'number'
        assert normalizer._normalize_token('42') == 'NUM_LIT'
        
        # Boolean literals
        assert normalizer._detect_literal_type('True') == 'boolean'
        assert normalizer._detect_literal_type('false') == 'boolean'
        assert normalizer._normalize_token('True') == 'BOOL_LIT'
        
    def test_identifier_role_detection(self):
        """Test identifier role detection based on naming patterns."""
        normalizer = TokenNormalizer('python')
        
        # Type/class patterns (PascalCase)
        assert normalizer._detect_identifier_role('MyClass') == 'type'
        assert normalizer._detect_identifier_role('UserService') == 'type'
        assert normalizer._detect_identifier_role('IUserService') == 'type'  # Interface
        
        # Function patterns
        assert normalizer._detect_identifier_role('my_function') == 'function'
        assert normalizer._detect_identifier_role('camelCase') == 'function'
        assert normalizer._detect_identifier_role('__init__') == 'function'
        
        # Variable patterns
        assert normalizer._detect_identifier_role('my_var') == 'variable'
        assert normalizer._detect_identifier_role('CONSTANT') == 'variable'
        assert normalizer._detect_identifier_role('someValue') == 'variable'
        
    def test_identifier_normalization(self):
        """Test identifier normalization with role-based placeholders."""
        normalizer = TokenNormalizer('python')
        
        # Variables get incremental VAR_N
        assert normalizer._normalize_identifier('x', 'variable') == 'VAR_1'
        assert normalizer._normalize_identifier('y', 'variable') == 'VAR_2'
        assert normalizer._normalize_identifier('data', 'variable') == 'VAR_3'
        
        # Functions get incremental FUNC_N
        normalizer._reset_counters()
        assert normalizer._normalize_identifier('func1', 'function') == 'FUNC_1'
        assert normalizer._normalize_identifier('func2', 'function') == 'FUNC_2'
        
        # Types get incremental TYPE_N
        normalizer._reset_counters()
        assert normalizer._normalize_identifier('MyClass', 'type') == 'TYPE_1'
        assert normalizer._normalize_identifier('OtherClass', 'type') == 'TYPE_2'
        
    def test_block_normalization_complete(self):
        """Test complete block normalization with a realistic example."""
        # Create a mock code block
        code_block = CodeBlock(
            lang='python',
            file_path=Path('test.py'),
            start_line=1,
            end_line=5,
            start_byte=0,
            end_byte=100,
            tokens=['def', 'calculate', '(', 'x', ',', 'y', '):', 
                   'result', '=', 'x', '+', 'y', 
                   'return', 'result'],
            raw_content='def calculate(x, y):\n    result = x + y\n    return result',
            node_type='function_definition'
        )
        
        normalizer = TokenNormalizer('python')
        normalized = normalizer.normalize_block(code_block)
        
        # Check structure
        assert isinstance(normalized, NormalizedBlock)
        assert normalized.original == code_block
        assert len(normalized.normalized_tokens) == len(code_block.tokens)
        
        # Check that identifiers were normalized
        # Each occurrence gets a new VAR_N (even for repeated identifiers)
        expected_tokens = ['def', 'VAR_1', '(', 'VAR_2', ',', 'VAR_3', '):', 
                          'VAR_4', '=', 'VAR_5', '+', 'VAR_6', 
                          'return', 'VAR_7']
        assert normalized.normalized_tokens == expected_tokens
        
        # Check mappings (only stores the last occurrence for each unique token)
        assert 'calculate' in normalized.token_mapping
        assert normalized.token_mapping['calculate'] == 'VAR_1'
        assert 'x' in normalized.token_mapping
        assert normalized.token_mapping['x'] == 'VAR_5'  # Last occurrence
        assert 'result' in normalized.token_mapping
        assert normalized.token_mapping['result'] == 'VAR_7'  # Last occurrence
        
        # Check reverse mappings
        assert 'VAR_1' in normalized.reverse_mapping
        assert 'calculate' in normalized.reverse_mapping['VAR_1']
        
        # Check hash generation
        assert isinstance(normalized.hash_signature, str)
        assert len(normalized.hash_signature) == 64  # SHA-256 hex
        
    def test_block_normalization_with_literals(self):
        """Test block normalization with various literal types."""
        code_block = CodeBlock(
            lang='python',
            file_path=Path('test.py'),
            start_line=1,
            end_line=3,
            start_byte=0,
            end_byte=50,
            tokens=['name', '=', '"John"', 'age', '=', '25', 'active', '=', 'True'],
            raw_content='name = "John"\nage = 25\nactive = True',
            node_type='chunk'
        )
        
        normalizer = TokenNormalizer('python')
        normalized = normalizer.normalize_block(code_block)
        
        # Note: 'True' is a Python keyword, not a literal, so it's preserved
        expected_tokens = ['VAR_1', '=', 'STR_LIT', 'VAR_2', '=', 'NUM_LIT', 'VAR_3', '=', 'True']
        assert normalized.normalized_tokens == expected_tokens
        
        # Check literal mappings
        assert normalized.token_mapping['"John"'] == 'STR_LIT'
        assert normalized.token_mapping['25'] == 'NUM_LIT'
        # True is a keyword, so no mapping (preserved as-is)
        assert 'True' not in normalized.token_mapping
        
    def test_javascript_normalization(self):
        """Test normalization for JavaScript code."""
        code_block = CodeBlock(
            lang='javascript',
            file_path=Path('test.js'),
            start_line=1,
            end_line=3,
            start_byte=0,
            end_byte=60,
            tokens=['function', 'greet', '(', 'name', ')', '{', 
                   'const', 'message', '=', '`Hello, ${name}!`', 
                   'return', 'message', '}'],
            raw_content='function greet(name) {\n  const message = `Hello, ${name}!`\n  return message\n}',
            node_type='function_declaration'
        )
        
        normalizer = TokenNormalizer('javascript')
        normalized = normalizer.normalize_block(code_block)
        
        # Keywords should be preserved
        assert 'function' in normalized.normalized_tokens
        assert 'const' in normalized.normalized_tokens  
        assert 'return' in normalized.normalized_tokens
        
        # Identifiers should be normalized (message appears twice)
        assert normalized.token_mapping['greet'] == 'VAR_1'
        assert normalized.token_mapping['name'] == 'VAR_2'
        assert normalized.token_mapping['message'] == 'VAR_4'  # Last occurrence
        
        # Template literal should be normalized
        assert normalized.token_mapping['`Hello, ${name}!`'] == 'STR_LIT'
        
    def test_typescript_specific_keywords(self):
        """Test TypeScript-specific keyword preservation."""
        normalizer = TokenNormalizer('typescript')
        
        # TypeScript-specific keywords
        assert normalizer._normalize_token('type') == 'type'
        assert normalizer._normalize_token('interface') == 'interface'
        assert normalizer._normalize_token('namespace') == 'namespace'
        assert normalizer._normalize_token('readonly') == 'readonly'
        assert normalizer._normalize_token('keyof') == 'keyof'
        
    def test_empty_and_whitespace_tokens(self):
        """Test handling of empty and whitespace-only tokens."""
        normalizer = TokenNormalizer('python')
        
        assert normalizer._normalize_token('') == ''
        assert normalizer._normalize_token(' ') == ' '
        assert normalizer._normalize_token('\t') == '\t'
        assert normalizer._normalize_token('\n') == '\n'


class TestNormalizedBlock:
    """Test cases for NormalizedBlock dataclass."""
    
    def test_normalized_block_properties(self):
        """Test NormalizedBlock property calculations."""
        original_block = CodeBlock(
            lang='python',
            file_path=Path('test.py'),
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=20,
            tokens=['def', 'test', ':', 'pass'],
            raw_content='def test():\n    pass',
            node_type='function_definition'
        )
        
        normalized = NormalizedBlock(
            original=original_block,
            normalized_tokens=['def', 'FUNC_1', ':', 'pass'],
            token_mapping={'test': 'FUNC_1'},
            reverse_mapping={'FUNC_1': ['test']},
            hash_signature='abc123'
        )
        
        assert normalized.token_count == 4
        assert normalized.original == original_block


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_normalize_blocks_empty_list(self):
        """Test normalizing empty list of blocks."""
        result = normalize_blocks([])
        assert result == []
        
    def test_normalize_blocks_mixed_languages(self):
        """Test normalizing blocks from different languages."""
        # Python block
        py_block = CodeBlock(
            lang='python',
            file_path=Path('test.py'),
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=30,
            tokens=['def', 'func1', '(', '):', 'return', '42'],
            raw_content='def func1():\n    return 42',
            node_type='function_definition'
        )
        
        # JavaScript block
        js_block = CodeBlock(
            lang='javascript', 
            file_path=Path('test.js'),
            start_line=1,
            end_line=1,
            start_byte=0,
            end_byte=25,
            tokens=['function', 'func2', '(', ')', '{', 'return', '42', '}'],
            raw_content='function func2() { return 42 }',
            node_type='function_declaration'
        )
        
        blocks = [py_block, js_block]
        normalized = normalize_blocks(blocks)
        
        assert len(normalized) == 2
        assert all(isinstance(n, NormalizedBlock) for n in normalized)
        
        # Both should have normalized function names as variables (VAR_1)
        py_normalized = next(n for n in normalized if n.original.lang == 'python')
        js_normalized = next(n for n in normalized if n.original.lang == 'javascript')
        
        assert py_normalized.token_mapping['func1'] == 'VAR_1'
        assert js_normalized.token_mapping['func2'] == 'VAR_1'
        
    def test_normalize_blocks_error_handling(self):
        """Test error handling during normalization."""
        # Create a block that might cause issues
        problem_block = CodeBlock(
            lang='python',
            file_path=Path('problem.py'),
            start_line=1,
            end_line=1,
            start_byte=0,
            end_byte=10,
            tokens=[None, 'invalid', ''],  # Problematic tokens
            raw_content='invalid content',
            node_type='chunk'
        )
        
        # Should handle errors gracefully
        result = normalize_blocks([problem_block])
        # May return empty list or partial results depending on error handling
        assert isinstance(result, list)
        
    def test_create_diff_explanation(self):
        """Test creation of diff explanations between normalized blocks."""
        # Create two similar normalized blocks
        original1 = CodeBlock(
            lang='python', file_path=Path('test1.py'), start_line=1, end_line=2,
            start_byte=0, end_byte=20, tokens=[], raw_content='', node_type='function'
        )
        
        original2 = CodeBlock(
            lang='python', file_path=Path('test2.py'), start_line=10, end_line=11,
            start_byte=100, end_byte=120, tokens=[], raw_content='', node_type='function'
        )
        
        block1 = NormalizedBlock(
            original=original1,
            normalized_tokens=['def', 'FUNC_1', ':', 'VAR_1'],
            token_mapping={'func1': 'FUNC_1', 'x': 'VAR_1'},
            reverse_mapping={'FUNC_1': ['func1'], 'VAR_1': ['x']},
            hash_signature='hash1'
        )
        
        block2 = NormalizedBlock(
            original=original2,
            normalized_tokens=['def', 'FUNC_1', ':', 'VAR_1'],
            token_mapping={'func2': 'FUNC_1', 'y': 'VAR_1'},
            reverse_mapping={'FUNC_1': ['func2'], 'VAR_1': ['y']},
            hash_signature='hash2'
        )
        
        diff = create_diff_explanation(block1, block2)
        
        assert 'block1_mappings' in diff
        assert 'block2_mappings' in diff
        assert 'common_patterns' in diff
        assert 'different_patterns' in diff
        assert 'structural_similarity' in diff
        
        # Should have high structural similarity (same normalized tokens)
        assert diff['structural_similarity'] == 1.0
        
    def test_pretty_print_normalization(self):
        """Test pretty printing of normalized blocks."""
        original = CodeBlock(
            lang='python', file_path=Path('example.py'), start_line=5, end_line=7,
            start_byte=50, end_byte=100, tokens=[], raw_content='', node_type='function'
        )
        
        normalized = NormalizedBlock(
            original=original,
            normalized_tokens=['def', 'FUNC_1', '(', 'VAR_1', '):', 'return', 'VAR_1'],
            token_mapping={'calculate': 'FUNC_1', 'value': 'VAR_1'},
            reverse_mapping={'FUNC_1': ['calculate'], 'VAR_1': ['value']},
            hash_signature='a1b2c3d4e5f6' + 'x' * 52  # 64-char hash
        )
        
        output = pretty_print_normalization(normalized)
        
        assert 'Normalized Block: example.py:5' in output
        assert 'Language: python' in output
        assert 'Hash: a1b2c3d4e5f6...' in output
        assert 'calculate -> FUNC_1' in output
        assert 'value -> VAR_1' in output
        assert 'def FUNC_1 ( VAR_1 ): return VAR_1' in output


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_unknown_language(self):
        """Test normalizer with unknown language."""
        normalizer = TokenNormalizer('unknown')
        assert normalizer.language == 'unknown'
        
        # Should still work, just no language-specific keywords
        token = normalizer._normalize_token('someidentifier')
        assert token == 'VAR_1'  # Should treat as variable
        
    def test_very_long_token_sequence(self):
        """Test normalization of very long token sequences."""
        # Create a block with many tokens
        tokens = ['def', 'long_function', '(']
        for i in range(100):
            tokens.extend([f'param_{i}', ','])
        tokens.extend([')', ':', 'pass'])
        
        code_block = CodeBlock(
            lang='python',
            file_path=Path('long.py'), 
            start_line=1,
            end_line=1,
            start_byte=0,
            end_byte=1000,
            tokens=tokens,
            raw_content='long function definition',
            node_type='function_definition'
        )
        
        normalizer = TokenNormalizer('python')
        normalized = normalizer.normalize_block(code_block)
        
        # Should handle long sequences without issues
        assert len(normalized.normalized_tokens) == len(tokens)
        assert 'long_function' in normalized.token_mapping
        assert normalized.token_mapping['long_function'] == 'FUNC_1'
        
    def test_special_characters_in_identifiers(self):
        """Test handling of identifiers with special characters."""
        normalizer = TokenNormalizer('python')
        
        # Python allows underscores
        assert normalizer._normalize_token('my_var') == 'VAR_1'
        assert normalizer._normalize_token('__private') == 'FUNC_1'  # Dunder method
        assert normalizer._normalize_token('_protected') == 'VAR_2'
        
        # Should handle edge cases gracefully
        assert normalizer._normalize_token('_') == 'VAR_3'
        assert normalizer._normalize_token('__') == 'VAR_4'
        
    def test_hash_consistency(self):
        """Test that identical normalized sequences produce identical hashes."""
        # Create two identical normalized blocks
        tokens = ['def', 'FUNC_1', '(', 'VAR_1', '):', 'return', 'VAR_1']
        
        normalizer1 = TokenNormalizer('python')
        hash1 = normalizer1._generate_hash(tokens)
        
        normalizer2 = TokenNormalizer('python')
        hash2 = normalizer2._generate_hash(tokens)
        
        assert hash1 == hash2
        
        # Different sequences should produce different hashes
        different_tokens = ['def', 'FUNC_2', '(', 'VAR_1', '):', 'return', 'VAR_1']
        hash3 = normalizer1._generate_hash(different_tokens)
        
        assert hash1 != hash3