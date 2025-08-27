"""Tree-sitter based code parsing and block extraction."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import tree_sitter as ts
from tree_sitter import Language, Node, Parser

# Language-specific imports
try:
    import tree_sitter_javascript as ts_javascript
    import tree_sitter_python as ts_python
    import tree_sitter_typescript as ts_typescript
except ImportError as e:
    logging.error(f"Failed to import tree-sitter language bindings: {e}")
    raise ImportError(
        "Tree-sitter language bindings are required. Install with: pip install tree-sitter-python tree-sitter-javascript tree-sitter-typescript"
    )

logger = logging.getLogger(__name__)


@dataclass
class CodeBlock:
    """Represents a parsed code block."""

    lang: str
    file_path: Path
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    tokens: List[str]
    raw_content: str
    node_type: str

    @property
    def token_count(self) -> int:
        """Get the number of tokens in this block."""
        return len(self.tokens)

    @property
    def lines(self) -> int:
        """Get the number of lines in this block."""
        return self.end_line - self.start_line + 1


class LanguageParser:
    """Parser for a specific programming language using Tree-sitter."""

    # Node types to extract as code blocks for each language
    EXTRACTABLE_NODES = {
        "python": {
            "function_definition",
            "async_function_definition",
            "class_definition",
            "module",
        },
        "javascript": {
            "function_declaration",
            "function_expression",
            "arrow_function",
            "method_definition",
            "class_declaration",
            "export_statement",
            "program",
        },
        "typescript": {
            "function_declaration",
            "function_expression",
            "arrow_function",
            "method_definition",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "export_statement",
            "program",
        },
    }

    # Minimum tokens per block - blocks smaller than this will be candidates for chunking
    MIN_BLOCK_TOKENS = 40
    MAX_CHUNK_TOKENS = 120

    def __init__(self, language: str, grammar_path: Optional[Path] = None):
        """Initialize parser for a specific language."""
        self.language = language.lower()
        self.parser = Parser()
        self._setup_language()

    def _setup_language(self) -> None:
        """Set up the Tree-sitter language grammar."""
        try:
            if self.language == "python":
                lang = Language(ts_python.language())
            elif self.language == "javascript":
                lang = Language(ts_javascript.language())
            elif self.language == "typescript":
                lang = Language(ts_typescript.language())
            else:
                raise ValueError(f"Unsupported language: {self.language}")

            self.parser.language = lang
            logger.debug(f"Initialized Tree-sitter parser for {self.language}")

        except Exception as e:
            logger.error(f"Failed to initialize {self.language} parser: {e}")
            raise

    def parse_file(self, file_path: Path) -> List[CodeBlock]:
        """Parse a file and extract code blocks."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return self.extract_blocks(content, file_path)

        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            return []

    def extract_blocks(self, content: str, file_path: Path) -> List[CodeBlock]:
        """Extract blocks from parsed content."""
        if not content.strip():
            return []

        try:
            # Parse the content
            tree = self.parser.parse(bytes(content, "utf-8"))
            root_node = tree.root_node

            blocks = []
            content_lines = content.split("\n")

            # Extract structured blocks first
            extracted_ranges = set()
            for block in self._extract_structured_blocks(
                root_node, content, file_path, content_lines
            ):
                blocks.append(block)
                # Track byte ranges to avoid overlapping chunks
                extracted_ranges.add((block.start_byte, block.end_byte))

            # Extract chunks from remaining code
            chunk_blocks = self._extract_chunks(
                root_node, content, file_path, content_lines, extracted_ranges
            )
            blocks.extend(chunk_blocks)

            logger.debug(f"Extracted {len(blocks)} blocks from {file_path}")
            return blocks

        except Exception as e:
            logger.error(f"Failed to extract blocks from {file_path}: {e}")
            return []

    def _extract_structured_blocks(
        self, root_node: Node, content: str, file_path: Path, content_lines: List[str]
    ) -> Iterator[CodeBlock]:
        """Extract structured code blocks (functions, classes, etc.)."""
        extractable_types = self.EXTRACTABLE_NODES.get(self.language, set())

        def traverse(node: Node) -> Iterator[CodeBlock]:
            if node.type in extractable_types:
                # Only extract if it's meaningful (not just the root program/module)
                if node.type not in {"program", "module"} or self._is_meaningful_block(
                    node
                ):
                    block = self._create_code_block(
                        node, content, file_path, content_lines
                    )
                    if block and block.token_count >= self.MIN_BLOCK_TOKENS:
                        yield block
                        return  # Don't traverse children of extracted blocks

            # Traverse children
            for child in node.children:
                yield from traverse(child)

        return traverse(root_node)

    def _is_meaningful_block(self, node: Node) -> bool:
        """Check if a program/module node represents a meaningful block."""
        # For top-level nodes, only consider them if they have substantial content
        if not node.children:
            return False

        # Count meaningful child nodes (not just whitespace/comments)
        meaningful_children = 0
        for child in node.children:
            if child.type not in {"comment", "newline"} and child.text.strip():
                meaningful_children += 1

        return meaningful_children >= 3  # At least 3 statements/declarations

    def _extract_chunks(
        self,
        root_node: Node,
        content: str,
        file_path: Path,
        content_lines: List[str],
        extracted_ranges: Set[Tuple[int, int]],
    ) -> List[CodeBlock]:
        """Extract chunks from code that wasn't captured in structured blocks."""
        chunks = []

        # Find statement-level nodes that aren't already covered
        statements = self._find_statement_nodes(root_node)

        current_chunk_statements = []
        current_tokens = 0

        for stmt in statements:
            # Skip if this statement is already covered by an extracted block
            if any(
                start <= stmt.start_byte < end or start < stmt.end_byte <= end
                for start, end in extracted_ranges
            ):
                continue

            stmt_tokens = self._extract_tokens(stmt)

            # If adding this statement would exceed max tokens, finalize current chunk
            if (
                current_tokens + len(stmt_tokens) > self.MAX_CHUNK_TOKENS
                and current_chunk_statements
            ):
                chunk = self._create_chunk_block(
                    current_chunk_statements, content, file_path, content_lines
                )
                if chunk:
                    chunks.append(chunk)
                current_chunk_statements = []
                current_tokens = 0

            current_chunk_statements.append(stmt)
            current_tokens += len(stmt_tokens)

            # If we have enough tokens for a meaningful chunk, consider finalizing
            if current_tokens >= self.MIN_BLOCK_TOKENS:
                # Continue accumulating unless we hit max or natural break
                if current_tokens >= self.MAX_CHUNK_TOKENS:
                    chunk = self._create_chunk_block(
                        current_chunk_statements, content, file_path, content_lines
                    )
                    if chunk:
                        chunks.append(chunk)
                    current_chunk_statements = []
                    current_tokens = 0

        # Handle remaining statements
        if current_chunk_statements and current_tokens >= self.MIN_BLOCK_TOKENS:
            chunk = self._create_chunk_block(
                current_chunk_statements, content, file_path, content_lines
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _find_statement_nodes(self, root_node: Node) -> List[Node]:
        """Find statement-level nodes for chunking."""
        statements = []

        def traverse(node: Node) -> None:
            # Statement types by language
            statement_types = {
                "python": {
                    "expression_statement",
                    "assignment",
                    "augmented_assignment",
                    "if_statement",
                    "for_statement",
                    "while_statement",
                    "try_statement",
                    "with_statement",
                    "import_statement",
                    "import_from_statement",
                    "return_statement",
                    "raise_statement",
                    "assert_statement",
                    "pass_statement",
                    "break_statement",
                    "continue_statement",
                    "global_statement",
                    "nonlocal_statement",
                    "delete_statement",
                },
                "javascript": {
                    "expression_statement",
                    "variable_declaration",
                    "if_statement",
                    "for_statement",
                    "for_in_statement",
                    "while_statement",
                    "do_statement",
                    "try_statement",
                    "throw_statement",
                    "return_statement",
                    "break_statement",
                    "continue_statement",
                    "switch_statement",
                    "labeled_statement",
                },
                "typescript": {
                    "expression_statement",
                    "variable_declaration",
                    "if_statement",
                    "for_statement",
                    "for_in_statement",
                    "while_statement",
                    "do_statement",
                    "try_statement",
                    "throw_statement",
                    "return_statement",
                    "break_statement",
                    "continue_statement",
                    "switch_statement",
                    "labeled_statement",
                },
            }

            lang_statements = statement_types.get(self.language, set())

            if node.type in lang_statements:
                statements.append(node)
            else:
                # Traverse children for compound constructs
                for child in node.children:
                    traverse(child)

        traverse(root_node)
        return statements

    def _create_code_block(
        self, node: Node, content: str, file_path: Path, content_lines: List[str]
    ) -> Optional[CodeBlock]:
        """Create a CodeBlock from a Tree-sitter node."""
        try:
            start_line = node.start_point[0] + 1  # Tree-sitter uses 0-based indexing
            end_line = node.end_point[0] + 1

            # Extract raw content
            raw_content = node.text.decode("utf-8")

            # Extract tokens
            tokens = self._extract_tokens(node)

            if not tokens:
                return None

            return CodeBlock(
                lang=self.language,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                tokens=tokens,
                raw_content=raw_content,
                node_type=node.type,
            )

        except Exception as e:
            logger.error(f"Failed to create code block from node: {e}")
            return None

    def _create_chunk_block(
        self,
        statements: List[Node],
        content: str,
        file_path: Path,
        content_lines: List[str],
    ) -> Optional[CodeBlock]:
        """Create a CodeBlock from a list of statement nodes."""
        if not statements:
            return None

        try:
            first_stmt = statements[0]
            last_stmt = statements[-1]

            start_line = first_stmt.start_point[0] + 1
            end_line = last_stmt.end_point[0] + 1

            # Extract combined raw content
            start_byte = first_stmt.start_byte
            end_byte = last_stmt.end_byte
            raw_content = content[start_byte:end_byte]

            # Extract combined tokens
            tokens = []
            for stmt in statements:
                tokens.extend(self._extract_tokens(stmt))

            if not tokens:
                return None

            return CodeBlock(
                lang=self.language,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                tokens=tokens,
                raw_content=raw_content,
                node_type="chunk",
            )

        except Exception as e:
            logger.error(f"Failed to create chunk block: {e}")
            return None

    def _extract_tokens(self, node: Node) -> List[str]:
        """Extract tokens from a Tree-sitter node."""
        tokens = []

        def traverse(n: Node) -> None:
            if n.type in {
                "identifier",
                "string",
                "number",
                "true",
                "false",
                "null",
                "none",
            }:
                text = n.text.decode("utf-8")
                if text.strip():
                    tokens.append(text)
            elif n.type in {"comment"}:
                # Skip comments for token extraction
                pass
            elif not n.children:
                # Leaf node - include if it's meaningful
                text = n.text.decode("utf-8")
                if text.strip() and not text.isspace():
                    tokens.append(text)
            else:
                # Traverse children
                for child in n.children:
                    traverse(child)

        traverse(node)
        return tokens


# Global parser registry
_PARSERS: Dict[str, LanguageParser] = {}


def get_parser(language: str) -> LanguageParser:
    """Get or create a parser for the specified language."""
    language = language.lower()

    if language not in _PARSERS:
        _PARSERS[language] = LanguageParser(language)

    return _PARSERS[language]


def detect_language(file_path: Path) -> Optional[str]:
    """Detect the programming language of a file based on its extension."""
    suffix = file_path.suffix.lower()

    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".mjs": "javascript",
        ".cjs": "javascript",
    }

    return language_map.get(suffix)


def extract_blocks(
    file_paths: List[Path], languages: Optional[List[str]] = None
) -> List[CodeBlock]:
    """Extract code blocks from multiple files."""
    if not file_paths:
        return []

    supported_languages = (
        set(languages) if languages else {"python", "typescript", "javascript"}
    )
    blocks = []

    for file_path in file_paths:
        try:
            # Detect language
            lang = detect_language(file_path)
            if not lang or lang not in supported_languages:
                logger.debug(f"Skipping {file_path}: unsupported language {lang}")
                continue

            # Get parser and extract blocks
            parser = get_parser(lang)
            file_blocks = parser.parse_file(file_path)
            blocks.extend(file_blocks)

        except Exception as e:
            logger.error(f"Failed to extract blocks from {file_path}: {e}")
            continue

    logger.info(f"Extracted {len(blocks)} blocks from {len(file_paths)} files")
    return blocks


def extract_blocks_from_content(
    content: str, file_path: Path, language: str
) -> List[CodeBlock]:
    """Extract code blocks from content string with known language."""
    try:
        parser = get_parser(language)
        return parser.extract_blocks(content, file_path)
    except Exception as e:
        logger.error(f"Failed to extract blocks from content: {e}")
        return []
