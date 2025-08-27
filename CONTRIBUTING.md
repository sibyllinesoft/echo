# Contributing to Echo

Welcome! We're excited that you're interested in contributing to Echo, the next-generation duplicate code detection tool. This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of duplicate code detection concepts

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/echo.git
   cd echo
   ```

2. **Set Up Development Environment**
   ```bash
   # Run the automated setup script
   bash scripts/dev_setup.sh
   
   # Or manually:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Verify Installation**
   ```bash
   # Run tests
   pytest

   # Run type checking
   mypy echo/

   # Run linting
   black --check echo/
   flake8 echo/
   ```

## 🧪 Development Workflow

### Code Quality Standards

Echo maintains high code quality standards:

- **Test Coverage**: Minimum 80% coverage for new code
- **Type Safety**: All new code must have proper type hints
- **Code Style**: Follow PEP 8, enforced by Black and Flake8
- **Documentation**: All public APIs must have comprehensive docstrings

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=echo --cov-report=html

# Run specific test file
pytest tests/test_parser.py

# Run tests for specific functionality
pytest -k "test_normalize"
```

### Code Formatting

We use Black for code formatting:

```bash
# Check formatting
black --check echo/ tests/

# Auto-format code
black echo/ tests/
```

### Type Checking

We use MyPy for static type checking:

```bash
# Check types
mypy echo/

# Check specific module
mypy echo/parser.py
```

## 📝 Contribution Guidelines

### Issue Reporting

Before creating an issue, please:

1. Check if the issue already exists
2. Use our issue templates
3. Provide minimal reproducible examples
4. Include version information and environment details

### Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow our coding standards
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit Guidelines**
   - Use clear, descriptive commit messages
   - Follow conventional commits format: `type(scope): description`
   - Examples:
     - `feat(parser): add support for Rust language parsing`
     - `fix(lsh): resolve hash collision in MinHash signatures`
     - `docs(readme): update installation instructions`

4. **Submit Pull Request**
   - Use our PR template
   - Link related issues
   - Provide clear description of changes
   - Include testing instructions

### Code Review Process

All contributions go through code review:

- **Automated Checks**: CI runs tests, linting, and type checking
- **Manual Review**: Core maintainers review code quality and design
- **Feedback**: We provide constructive feedback for improvements
- **Approval**: Two approvals required for merge

## 🏗️ Architecture Overview

Understanding Echo's architecture helps with contributions:

### Core Components

1. **Parser** (`echo/parser.py`): Tree-sitter based code block extraction
2. **Normalizer** (`echo/normalize.py`): Token normalization for comparison  
3. **LSH** (`echo/lsh.py`): MinHash signatures and locality-sensitive hashing
4. **Verifier** (`echo/verify.py`): Token-level duplicate verification
5. **Scanner** (`echo/scan.py`): Main orchestration pipeline
6. **Storage** (`echo/storage.py`): SQLite + FAISS persistence

### Design Principles

- **Modularity**: Each component has clear responsibilities
- **Performance**: Optimized for large codebases (250k+ LOC)
- **Extensibility**: Easy to add new languages and detection methods
- **Privacy**: Local-only processing, no cloud dependencies
- **Type Safety**: Comprehensive type hints throughout

## 🎯 Areas for Contribution

### High Priority

- **Language Support**: Add Tree-sitter support for Go, Rust, Java, C++
- **Semantic Analysis**: Improve GraphCodeBERT integration
- **Performance**: Optimize memory usage and processing speed  
- **UI/UX**: Enhance CLI interface and report formatting

### Medium Priority

- **Documentation**: API documentation, tutorials, examples
- **Testing**: Edge case coverage, property-based testing
- **Integration**: IDE plugins, CI/CD integrations
- **Algorithms**: Advanced duplicate detection techniques

### Good First Issues

Look for issues labeled `good-first-issue`:

- Documentation improvements
- Test case additions  
- Code style fixes
- Minor feature enhancements

## 🤖 AI Agent Integration

Echo pioneered MCP (Model Context Protocol) integration for AI agents:

### MCP Tools

When contributing to MCP functionality:

- Maintain JSON-only input/output
- Provide streaming support for long operations
- Follow MCP protocol specifications
- Include comprehensive error handling

### Example Contribution

```python
# Adding a new MCP tool
async def new_analysis_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """New analysis functionality for AI agents."""
    try:
        # Validate parameters
        # Perform analysis
        # Return structured results
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

## 🌟 Recognition

We value all contributions:

- **Contributors**: Listed in GitHub contributors
- **Significant Contributions**: Mentioned in release notes
- **Ongoing Contributors**: Invited to be maintainers

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community discussion  
- **Code Review**: Detailed feedback on pull requests

## 📋 Checklist for Contributors

Before submitting:

- [ ] Tests pass (`pytest`)
- [ ] Type checking passes (`mypy echo/`)
- [ ] Linting passes (`black --check echo/ && flake8 echo/`)
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG.md updated (for notable changes)
- [ ] Tests added for new functionality
- [ ] Type hints added for new code

## 🙏 Thank You

Thank you for considering contributing to Echo! Your contributions help make duplicate code detection better for developers everywhere.

---

**Questions?** Don't hesitate to ask in GitHub issues or discussions. We're here to help make your contribution process smooth and enjoyable.