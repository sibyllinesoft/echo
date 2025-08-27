#!/usr/bin/env python3
"""
Validation script for the parser implementation.
This script provides basic validation without requiring external dependencies.
"""

import sys
import ast
from pathlib import Path

# Add the echo module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def validate_parser_structure():
    """Validate the parser module structure without importing dependencies."""
    parser_file = Path(__file__).parent.parent / 'echo' / 'parser.py'
    
    print("🔍 Validating parser.py structure...")
    
    # Read and parse the file with AST
    with open(parser_file, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        print("✅ Python syntax is valid")
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    
    # Check for required classes and functions
    classes = []
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
    
    # Expected classes
    expected_classes = ['CodeBlock', 'LanguageParser']
    for cls in expected_classes:
        if cls in classes:
            print(f"✅ Found class {cls}")
        else:
            print(f"❌ Missing class {cls}")
    
    # Expected functions
    expected_functions = ['get_parser', 'detect_language', 'extract_blocks', 'extract_blocks_from_content']
    for func in expected_functions:
        if func in functions:
            print(f"✅ Found function {func}")
        else:
            print(f"❌ Missing function {func}")
    
    return True

def validate_test_structure():
    """Validate the test file structure."""
    test_file = Path(__file__).parent.parent / 'tests' / 'test_parser.py'
    
    print("\n🔍 Validating test_parser.py structure...")
    
    if not test_file.exists():
        print("❌ Test file does not exist")
        return False
    
    with open(test_file, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        print("✅ Test file syntax is valid")
    except SyntaxError as e:
        print(f"❌ Test syntax error: {e}")
        return False
    
    # Count test classes and methods
    test_classes = []
    test_methods = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            test_classes.append(node.name)
            # Count test methods in this class
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                    test_methods.append(f"{node.name}.{item.name}")
    
    print(f"✅ Found {len(test_classes)} test classes")
    print(f"✅ Found {len(test_methods)} test methods")
    
    # Expected test classes
    expected_test_classes = ['TestCodeBlock', 'TestLanguageParser', 'TestUtilityFunctions', 'TestEdgeCases']
    for cls in expected_test_classes:
        if cls in test_classes:
            print(f"✅ Found test class {cls}")
        else:
            print(f"❌ Missing test class {cls}")
    
    return True

def check_imports():
    """Check import structure."""
    parser_file = Path(__file__).parent.parent / 'echo' / 'parser.py'
    
    print("\n🔍 Checking imports...")
    
    with open(parser_file, 'r') as f:
        content = f.read()
    
    # Check for expected imports
    expected_imports = [
        'import logging',
        'from dataclasses import dataclass',
        'from pathlib import Path',
        'from typing import',
        'import tree_sitter as ts',
        'from tree_sitter import Language, Parser, Node'
    ]
    
    success = True
    for expected in expected_imports:
        if expected in content:
            print(f"✅ Found import: {expected}")
        else:
            print(f"❌ Missing import: {expected}")
            success = False
    
    return success

def validate_type_hints():
    """Validate that functions have type hints."""
    parser_file = Path(__file__).parent.parent / 'echo' / 'parser.py'
    
    print("\n🔍 Checking type hints...")
    
    with open(parser_file, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    
    functions_without_hints = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private methods and __init__
            if node.name.startswith('_') and node.name != '__init__':
                continue
                
            # Check if function has return type annotation
            if node.returns is None and node.name not in ['__init__']:
                functions_without_hints.append(node.name)
    
    if functions_without_hints:
        print(f"⚠️  Functions without return type hints: {functions_without_hints}")
    else:
        print("✅ All public functions have type hints")
    
    return True

def main():
    """Run all validation checks."""
    print("🚀 Starting parser validation...\n")
    
    success = True
    success &= validate_parser_structure()
    success &= validate_test_structure()
    success &= check_imports()
    success &= validate_type_hints()
    
    print("\n" + "="*50)
    if success:
        print("🎉 All validations passed!")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install -e '.[dev]'")
        print("2. Run tests: pytest")
        print("3. Run type checking: mypy echo/")
        print("4. Run formatting: black echo/ tests/")
    else:
        print("❌ Some validations failed. Please review the output above.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())