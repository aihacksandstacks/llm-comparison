"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Tests for validating imports in app.py to catch missing functions early.
"""

import os
import sys
import ast
import pytest
from pathlib import Path
import importlib.util


def analyze_imports(file_path):
    """
    Parse a Python file and extract all import statements.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        A dictionary mapping module paths to lists of imported names
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    tree = ast.parse(content)
    
    # Dictionary to store imports from each module
    imports = {}
    
    for node in ast.walk(tree):
        # Handle 'from X import Y' statements
        if isinstance(node, ast.ImportFrom):
            module_path = node.module
            imported_names = [name.name for name in node.names]
            
            if module_path in imports:
                imports[module_path].extend(imported_names)
            else:
                imports[module_path] = imported_names
        
        # Handle 'import X' statements
        elif isinstance(node, ast.Import):
            for name in node.names:
                module_path = name.name
                if name.asname:
                    imports[module_path] = [name.asname]
                else:
                    imports[module_path] = [module_path.split('.')[-1]]
    
    return imports


def validate_imports(imports, base_path=None):
    """
    Validate that all imported names are available in their respective modules.
    
    Args:
        imports: Dictionary mapping module paths to lists of imported names
        base_path: Base path for resolving relative imports
        
    Returns:
        A list of error messages for missing imports
    """
    errors = []
    
    for module_path, names in imports.items():
        try:
            # Handle relative imports
            if module_path.startswith('.') and base_path:
                current_dir = os.path.dirname(base_path)
                level = 0
                while module_path.startswith('.'):
                    module_path = module_path[1:]
                    level += 1
                    current_dir = os.path.dirname(current_dir)
                
                if module_path:
                    module_path = os.path.join(current_dir, module_path.replace('.', '/'))
                else:
                    module_path = current_dir
            
            # Import the module
            if module_path:
                try:
                    module = importlib.import_module(module_path)
                except ImportError:
                    errors.append(f"Module not found: {module_path}")
                    continue
                
                # Check if each imported name exists in the module
                for name in names:
                    if name == '*':
                        continue  # Skip wildcard imports
                    
                    if not hasattr(module, name):
                        errors.append(f"Name '{name}' not found in module '{module_path}'")
        
        except Exception as e:
            errors.append(f"Error validating imports from {module_path}: {str(e)}")
    
    return errors


class TestAppImports:
    """Tests for validating imports in app.py."""
    
    def test_app_imports(self):
        """Test that all imports in app.py are valid."""
        # Find the app.py file - adjust path for tests directory
        app_path = Path(__file__).parent.parent.parent.parent / "src" / "features" / "base_ui" / "app.py"
        
        if not app_path.exists():
            pytest.skip(f"App file not found at {app_path}")
        
        # Analyze imports
        imports = analyze_imports(str(app_path))
        
        # Check that we found some imports
        assert imports, "No imports found in app.py"
        
        # Filter for crawler-related imports
        crawler_imports = {k: v for k, v in imports.items() 
                          if k and "crawler" in k}
        
        assert crawler_imports, "No crawler imports found in app.py"
        
        # Validate crawler imports specifically
        for module_path, names in crawler_imports.items():
            try:
                module = importlib.import_module(module_path)
                for name in names:
                    assert hasattr(module, name), f"Name '{name}' not found in module '{module_path}'"
                    assert callable(getattr(module, name)), f"'{name}' in '{module_path}' is not callable"
            except ImportError:
                pytest.fail(f"Module not found: {module_path}")
            except Exception as e:
                pytest.fail(f"Error validating imports from {module_path}: {str(e)}")
    
    def test_app_crawler_import_consistency(self):
        """
        Test that app.py imports from crawler.py are consistent throughout the file.
        This detects cases where the same import statement is repeated with different
        imported names, which can lead to confusion and bugs.
        """
        # Find the app.py file
        app_path = Path(__file__).parent.parent.parent.parent / "src" / "features" / "base_ui" / "app.py"
        
        if not app_path.exists():
            pytest.skip(f"App file not found at {app_path}")
        
        with open(app_path, 'r') as file:
            content = file.read()
        
        # Find all crawler import lines
        import_lines = []
        for line in content.split('\n'):
            if "from src.features.llm_compare.crawler import" in line:
                import_lines.append(line.strip())
        
        # Check that all import lines are the same
        if len(import_lines) > 1:
            first_import = import_lines[0]
            for i, line in enumerate(import_lines[1:], 2):
                assert line == first_import, f"Inconsistent crawler imports: line 1: '{first_import}' vs line {i}: '{line}'"
    
    def test_validate_all_crawler_exports(self):
        """Test that the crawler module exports all functions used by app.py."""
        # Import the crawler module
        from src.features.llm_compare.crawler import WebCrawler, get_web_crawler, crawl_website, simple_http_crawl, simple_http_crawl_sync
        
        # Find the app.py file
        app_path = Path(__file__).parent.parent.parent.parent / "src" / "features" / "base_ui" / "app.py"
        
        if not app_path.exists():
            pytest.skip(f"App file not found at {app_path}")
        
        # Analyze imports
        imports = analyze_imports(str(app_path))
        
        # Find crawler imports
        crawler_imports = {}
        for module_path, names in imports.items():
            if module_path and "crawler" in module_path:
                crawler_imports[module_path] = names
        
        # Check that all imported names are exported by the crawler module
        for module_path, names in crawler_imports.items():
            try:
                module = importlib.import_module(module_path)
                for name in names:
                    assert hasattr(module, name), f"Name '{name}' not found in module '{module_path}'"
            except ImportError:
                pytest.fail(f"Module not found: {module_path}")


if __name__ == "__main__":
    # Run tests directly for debugging
    test = TestAppImports()
    test.test_app_imports()
    test.test_app_crawler_import_consistency()
    test.test_validate_all_crawler_exports()
    print("All tests passed!") 