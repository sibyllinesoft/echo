"""Token normalization and code block standardization.

This module provides comprehensive token normalization for duplicate code detection.
It transforms code into a canonical representation by:
- Replacing identifiers with role-based placeholders (VAR_1, FUNC_1, etc.)
- Normalizing literals to type-based tokens (STR_LIT, NUM_LIT, BOOL_LIT)
- Stripping comments and normalizing whitespace
- Preserving structural tokens (keywords, operators, punctuation)
- Maintaining bidirectional mappings for explanations
"""

import re
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from .parser import CodeBlock

logger = logging.getLogger(__name__)


@dataclass
class NormalizedBlock:
    """A normalized code block with token mappings."""
    original: CodeBlock
    normalized_tokens: List[str]
    token_mapping: Dict[str, str]  # original -> normalized
    reverse_mapping: Dict[str, List[str]]  # normalized -> original(s)
    hash_signature: str
    
    @property
    def token_count(self) -> int:
        return len(self.normalized_tokens)


class TokenNormalizer:
    """Normalizes code tokens for duplicate detection comparison.
    
    Uses role-based identifier replacement and type-based literal normalization
    to create canonical representations that preserve code structure while
    abstracting away naming differences.
    """
    
    # Language-specific keywords to preserve
    KEYWORDS = {
        'python': {
            'False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class',
            'continue', 'def', 'del', 'elif', 'else', 'except', 'finally',
            'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
            'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
            'while', 'with', 'yield', 'async', 'await'
        },
        'javascript': {
            'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger',
            'default', 'delete', 'do', 'else', 'enum', 'export', 'extends',
            'false', 'finally', 'for', 'function', 'if', 'implements',
            'import', 'in', 'instanceof', 'interface', 'let', 'new', 'null',
            'package', 'private', 'protected', 'public', 'return', 'static',
            'super', 'switch', 'this', 'throw', 'true', 'try', 'typeof',
            'var', 'void', 'while', 'with', 'yield', 'async', 'await'
        },
        'typescript': {
            'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger',
            'default', 'delete', 'do', 'else', 'enum', 'export', 'extends',
            'false', 'finally', 'for', 'function', 'if', 'implements',
            'import', 'in', 'instanceof', 'interface', 'let', 'new', 'null',
            'package', 'private', 'protected', 'public', 'return', 'static',
            'super', 'switch', 'this', 'throw', 'true', 'try', 'typeof',
            'var', 'void', 'while', 'with', 'yield', 'async', 'await',
            'type', 'namespace', 'declare', 'module', 'abstract', 'readonly',
            'keyof', 'infer', 'is', 'asserts'
        }
    }
    
    # Operators and punctuation to preserve as-is
    STRUCTURAL_TOKENS = {
        '+', '-', '*', '/', '%', '**', '//', '&', '|', '^', '~', '<<', '>>', 
        '=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=',
        '==', '!=', '<', '>', '<=', '>=', '===', '!==', '&&', '||', '!',
        '(', ')', '[', ']', '{', '}', ',', ';', ':', '.', '->', '=>',
        '?', '??', '?.', '@', '#'
    }
    
    # Patterns for literal recognition
    STRING_PATTERNS = [
        re.compile(r'^".*"$'),       # Double quotes
        re.compile(r"^'.*'$"),       # Single quotes  
        re.compile(r'^`.*`$'),       # Template literals
        re.compile(r'^f".*"$'),      # Python f-strings
        re.compile(r"^f'.*'$"),      # Python f-strings
        re.compile(r'^r".*"$'),      # Raw strings
        re.compile(r"^r'.*'$"),      # Raw strings
    ]
    
    NUMBER_PATTERN = re.compile(r'^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$')
    BOOLEAN_PATTERN = re.compile(r'^(?:true|false|True|False)$')
    
    # Common identifier patterns by role
    VARIABLE_PATTERNS = [
        re.compile(r'^[a-z_][a-zA-Z0-9_]*$'),  # snake_case, camelCase
        re.compile(r'^[A-Z][A-Z0-9_]*$'),      # CONSTANTS
    ]
    
    FUNCTION_PATTERNS = [
        re.compile(r'^[a-z_][a-zA-Z0-9_]*$'),  # snake_case, camelCase functions
        re.compile(r'^__[a-zA-Z0-9_]*__$'),    # Python dunder methods
    ]
    
    TYPE_PATTERNS = [
        re.compile(r'^[A-Z][a-zA-Z0-9]*$'),    # PascalCase types/classes
        re.compile(r'^I[A-Z][a-zA-Z0-9]*$'),   # Interface naming (TypeScript)
    ]
    
    def __init__(self, language: str):
        """Initialize normalizer for specific language."""
        self.language = language.lower()
        self._reset_counters()
        
    def _reset_counters(self):
        """Reset all identifier counters."""
        self._var_counter = 0
        self._func_counter = 0
        self._type_counter = 0
        self._method_counter = 0
        self._param_counter = 0
        self._prop_counter = 0
        
    def normalize_block(self, block: CodeBlock) -> NormalizedBlock:
        """Normalize a code block for duplicate detection.
        
        Args:
            block: The code block to normalize
            
        Returns:
            NormalizedBlock with normalized tokens and mappings
        """
        self._reset_counters()
        
        # Track mappings for explanations
        token_mapping = {}
        reverse_mapping = defaultdict(list)
        normalized_tokens = []
        
        # Process each token
        for token in block.tokens:
            normalized = self._normalize_token(token)
            
            # Track mapping if changed
            if normalized != token:
                token_mapping[token] = normalized
                reverse_mapping[normalized].append(token)
            
            normalized_tokens.append(normalized)
        
        # Generate hash signature from normalized tokens
        hash_signature = self._generate_hash(normalized_tokens)
        
        logger.debug(f"Normalized block from {block.file_path}:{block.start_line} "
                    f"with {len(token_mapping)} replacements")
        
        return NormalizedBlock(
            original=block,
            normalized_tokens=normalized_tokens,
            token_mapping=token_mapping,
            reverse_mapping=dict(reverse_mapping),
            hash_signature=hash_signature
        )
    
    def _normalize_token(self, token: str) -> str:
        """Normalize a single token based on its type and role.
        
        Args:
            token: The token to normalize
            
        Returns:
            The normalized token
        """
        # Skip empty or whitespace-only tokens
        if not token or token.isspace():
            return token
            
        # Preserve keywords
        if self._is_keyword(token):
            return token
            
        # Preserve structural tokens (operators, punctuation)
        if token in self.STRUCTURAL_TOKENS:
            return token
            
        # Normalize literals
        literal_type = self._detect_literal_type(token)
        if literal_type:
            return self._normalize_literal(token, literal_type)
            
        # Normalize identifiers based on role detection
        role = self._detect_identifier_role(token)
        if role:
            return self._normalize_identifier(token, role)
            
        # If we can't classify it, keep as-is
        return token
    
    def _is_keyword(self, token: str) -> bool:
        """Check if token is a language keyword."""
        keywords = self.KEYWORDS.get(self.language, set())
        return token in keywords
    
    def _detect_literal_type(self, token: str) -> Optional[str]:
        """Detect the type of a literal token.
        
        Args:
            token: The token to analyze
            
        Returns:
            'string', 'number', 'boolean', or None
        """
        # Check for string literals
        for pattern in self.STRING_PATTERNS:
            if pattern.match(token):
                return 'string'
        
        # Check for numeric literals
        if self.NUMBER_PATTERN.match(token):
            return 'number'
            
        # Check for boolean literals
        if self.BOOLEAN_PATTERN.match(token):
            return 'boolean'
            
        return None
    
    def _detect_identifier_role(self, token: str) -> Optional[str]:
        """Detect the role of an identifier token.
        
        This is a heuristic-based approach since we don't have full
        semantic analysis. Uses naming conventions and patterns.
        
        Args:
            token: The identifier token to analyze
            
        Returns:
            'variable', 'function', 'type', 'method', 'parameter', or 'property'
        """
        # Check for type/class naming patterns (PascalCase) - most specific first
        for pattern in self.TYPE_PATTERNS:
            if pattern.match(token):
                return 'type'
        
        # Check for Python dunder methods (specific function pattern)
        if re.match(r'^__[a-zA-Z0-9_]*__$', token):
            return 'function'
        
        # Check for ALL_CAPS constants
        if re.match(r'^[A-Z][A-Z0-9_]*$', token):
            return 'variable'
            
        # Default all lowercase/snake_case/camelCase identifiers to variable
        # In practice, without context we can't distinguish functions from variables
        # So we'll treat them all as variables for consistency
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            return 'variable'
            
        return None
    
    def _normalize_identifier(self, token: str, role: str) -> str:
        """Normalize an identifier based on its role.
        
        Args:
            token: The identifier token
            role: The detected role ('variable', 'function', etc.)
            
        Returns:
            The normalized placeholder token
        """
        if role == 'variable':
            self._var_counter += 1
            return f'VAR_{self._var_counter}'
        elif role == 'function':
            self._func_counter += 1
            return f'FUNC_{self._func_counter}'
        elif role == 'type':
            self._type_counter += 1
            return f'TYPE_{self._type_counter}'
        elif role == 'method':
            self._method_counter += 1
            return f'METHOD_{self._method_counter}'
        elif role == 'parameter':
            self._param_counter += 1
            return f'PARAM_{self._param_counter}'
        elif role == 'property':
            self._prop_counter += 1
            return f'PROP_{self._prop_counter}'
        else:
            # Fallback to variable
            self._var_counter += 1
            return f'VAR_{self._var_counter}'
    
    def _normalize_literal(self, token: str, literal_type: str) -> str:
        """Normalize literals by type.
        
        Args:
            token: The literal token
            literal_type: The type ('string', 'number', 'boolean')
            
        Returns:
            The normalized literal token
        """
        if literal_type == 'string':
            return 'STR_LIT'
        elif literal_type == 'number':
            return 'NUM_LIT'
        elif literal_type == 'boolean':
            return 'BOOL_LIT'
        else:
            return token
    
    def _generate_hash(self, tokens: List[str]) -> str:
        """Generate a hash signature from normalized tokens.
        
        Args:
            tokens: The normalized token sequence
            
        Returns:
            Hex-encoded SHA-256 hash of the token sequence
        """
        # Create a stable string representation
        token_str = ' '.join(tokens)
        return hashlib.sha256(token_str.encode('utf-8')).hexdigest()


def normalize_blocks(blocks: List[CodeBlock]) -> List[NormalizedBlock]:
    """Normalize multiple code blocks.
    
    Args:
        blocks: List of code blocks to normalize
        
    Returns:
        List of normalized blocks
    """
    if not blocks:
        return []
        
    normalized = []
    
    # Group blocks by language for consistent normalization
    blocks_by_lang = defaultdict(list)
    for block in blocks:
        blocks_by_lang[block.lang].append(block)
    
    # Normalize each language group
    for language, lang_blocks in blocks_by_lang.items():
        logger.debug(f"Normalizing {len(lang_blocks)} {language} blocks")
        
        normalizer = TokenNormalizer(language)
        
        for block in lang_blocks:
            try:
                normalized_block = normalizer.normalize_block(block)
                normalized.append(normalized_block)
            except Exception as e:
                logger.warning(f"Failed to normalize block from "
                             f"{block.file_path}:{block.start_line}: {e}")
                
    logger.info(f"Normalized {len(normalized)}/{len(blocks)} blocks")
    return normalized


def create_diff_explanation(block1: NormalizedBlock, block2: NormalizedBlock) -> Dict[str, any]:
    """Create an explanation of differences between two normalized blocks.
    
    Args:
        block1: First normalized block
        block2: Second normalized block
        
    Returns:
        Dictionary containing difference analysis and mappings
    """
    # Find common normalized patterns
    tokens1 = set(block1.normalized_tokens)
    tokens2 = set(block2.normalized_tokens)
    
    common_patterns = tokens1 & tokens2
    different_patterns = (tokens1 | tokens2) - common_patterns
    
    # Build mapping explanations
    mappings = {
        'block1_mappings': block1.token_mapping,
        'block2_mappings': block2.token_mapping,
        'common_patterns': list(common_patterns),
        'different_patterns': list(different_patterns),
        'structural_similarity': len(common_patterns) / max(len(tokens1), len(tokens2))
    }
    
    return mappings


def pretty_print_normalization(normalized_block: NormalizedBlock) -> str:
    """Pretty print a normalized block showing original->normalized mappings.
    
    Args:
        normalized_block: The normalized block to display
        
    Returns:
        Formatted string showing the normalization
    """
    lines = [
        f"=== Normalized Block: {normalized_block.original.file_path}:{normalized_block.original.start_line} ===",
        f"Language: {normalized_block.original.lang}",
        f"Original tokens: {normalized_block.original.token_count}",
        f"Normalized tokens: {normalized_block.token_count}",
        f"Hash: {normalized_block.hash_signature[:16]}...",
        "",
        "Token mappings:",
    ]
    
    if normalized_block.token_mapping:
        for original, normalized in sorted(normalized_block.token_mapping.items()):
            lines.append(f"  {original} -> {normalized}")
    else:
        lines.append("  (no mappings - all tokens preserved)")
    
    lines.extend([
        "",
        "Normalized sequence:",
        " ".join(normalized_block.normalized_tokens[:50]) + 
        ("..." if len(normalized_block.normalized_tokens) > 50 else "")
    ])
    
    return "\n".join(lines)