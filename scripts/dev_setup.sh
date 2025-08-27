#!/bin/bash

# Echo Development Setup Script
# One-step setup for local development environment

set -euo pipefail

echo "🔧 Setting up Echo development environment..."

# Check Python version
if ! python3 --version | grep -E "Python 3\.(9|10|11|12)" > /dev/null; then
    echo "❌ Python 3.9+ is required"
    exit 1
fi

echo "✅ Python version check passed"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🔗 Setting up pre-commit hooks..."
pre-commit install

# Create cache directories
echo "📁 Creating cache directories..."
mkdir -p ~/.echo/models
mkdir -p ~/.echo/index/faiss

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
pytest tests/test_config.py -v

# Run linting to verify tools
echo "✨ Running code quality checks..."
black --check echo/ tests/ || echo "ℹ️  Run 'black echo/ tests/' to format code"
isort --check-only echo/ tests/ || echo "ℹ️  Run 'isort echo/ tests/' to sort imports"
flake8 echo/ tests/ || echo "ℹ️  Fix any linting issues shown above"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run tests: pytest"
echo "  2. Start development: python -m echo.cli --help"
echo "  3. Format code: black echo/ tests/"
echo "  4. Sort imports: isort echo/ tests/"
echo "  5. Type checking: mypy echo/"
echo ""
echo "Happy coding! 🚀"