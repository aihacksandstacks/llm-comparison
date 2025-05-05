"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Tests for API stability in the crawler module.
This test compares the current API to a reference snapshot to catch breaking changes.
"""

import os
import sys
import json
import inspect
import importlib
import pytest
from pathlib import Path

# Add the project root to the path when run directly
if __name__ == "__main__":
    # Get the project root directory
    PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent)
    
    # Add to path if not already there
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        print(f"Added {PROJECT_ROOT} to Python path")


def get_module_api(module_name):
    """
    Extract the public API of a module.
    
    Args:
        module_name: Fully qualified module name
        
    Returns:
        Dictionary with API definition
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise ImportError(f"Could not import module {module_name}")
    
    api = {
        "classes": {},
        "functions": {},
        "constants": {}
    }
    
    # Find all public attributes (not starting with _)
    for name in dir(module):
        if name.startswith('_'):
            continue
        
        attr = getattr(module, name)
        
        if inspect.isclass(attr):
            # For classes, get method signatures
            methods = {}
            for method_name, method in inspect.getmembers(attr, inspect.isfunction):
                if not method_name.startswith('_'):
                    methods[method_name] = {
                        "parameters": [p for p in inspect.signature(method).parameters],
                        "doc": inspect.getdoc(method) or ""
                    }
            
            api["classes"][name] = {
                "methods": methods,
                "doc": inspect.getdoc(attr) or ""
            }
        
        elif inspect.isfunction(attr):
            # For functions, get signatures
            api["functions"][name] = {
                "parameters": [p for p in inspect.signature(attr).parameters],
                "doc": inspect.getdoc(attr) or ""
            }
        
        else:
            # Other attributes (constants, etc.)
            api["constants"][name] = str(type(attr).__name__)
    
    return api


def save_api_snapshot(module_name, output_path):
    """
    Save a snapshot of the module's API to a file.
    
    Args:
        module_name: Fully qualified module name
        output_path: Path to save the snapshot
    """
    api = get_module_api(module_name)
    with open(output_path, 'w') as f:
        json.dump(api, f, indent=2, sort_keys=True)


def compare_apis(current_api, reference_api):
    """
    Compare the current API to a reference API.
    
    Args:
        current_api: Current API definition
        reference_api: Reference API definition
        
    Returns:
        List of incompatible changes
    """
    incompatible_changes = []
    
    # Check for removed functions
    for func_name in reference_api.get("functions", {}):
        if func_name not in current_api.get("functions", {}):
            incompatible_changes.append(f"Function removed: {func_name}")
    
    # Check for changed function signatures
    for func_name, func_def in reference_api.get("functions", {}).items():
        if func_name in current_api.get("functions", {}):
            ref_params = func_def.get("parameters", [])
            current_params = current_api["functions"][func_name].get("parameters", [])
            
            # Check for removed parameters
            for param in ref_params:
                if param not in current_params:
                    incompatible_changes.append(f"Parameter removed from {func_name}: {param}")
    
    # Check for removed classes
    for class_name in reference_api.get("classes", {}):
        if class_name not in current_api.get("classes", {}):
            incompatible_changes.append(f"Class removed: {class_name}")
    
    # Check for removed class methods
    for class_name, class_def in reference_api.get("classes", {}).items():
        if class_name in current_api.get("classes", {}):
            for method_name in class_def.get("methods", {}):
                if method_name not in current_api["classes"][class_name].get("methods", {}):
                    incompatible_changes.append(f"Method removed from {class_name}: {method_name}")
    
    return incompatible_changes


class TestAPIStability:
    """Tests for API stability."""
    
    def test_crawler_api_stability(self, request):
        """Test that the crawler module's API is stable."""
        # Module to test
        module_name = "src.features.llm_compare.crawler"
        
        # Get the current API
        current_api = get_module_api(module_name)
        
        # Reference snapshot path
        reference_path = Path(__file__).parent / "snapshots" / "crawler_api.json"
        
        # Create directory if it doesn't exist
        reference_path.parent.mkdir(exist_ok=True)
        
        # If the reference doesn't exist, create it
        if not reference_path.exists():
            # In CI, we should fail if the reference doesn't exist
            if os.environ.get("CI") == "true":
                pytest.fail(f"Reference API snapshot does not exist: {reference_path}")
            
            save_api_snapshot(module_name, reference_path)
            pytest.skip(f"Created reference API snapshot at {reference_path}")
        
        # Load the reference API
        with open(reference_path, 'r') as f:
            reference_api = json.load(f)
        
        # Compare the APIs
        incompatible_changes = compare_apis(current_api, reference_api)
        
        # If running in CI and --update-api flag is provided, update the reference
        if os.environ.get("CI") != "true" and request.config.getoption("--update-api", default=False):
            save_api_snapshot(module_name, reference_path)
            return
        
        # Fail if there are incompatible changes
        assert not incompatible_changes, "\n".join([
            "Incompatible API changes detected:", 
            *[f"  - {change}" for change in incompatible_changes],
            "\nTo update the reference API snapshot, run pytest with --update-api"
        ])
    
    def test_essential_crawler_exports(self):
        """Test that essential exports are available in the crawler module."""
        # These exports are considered essential and should always be present
        essential_exports = [
            "WebCrawler",
            "get_web_crawler",
            "crawl_website",
            "simple_http_crawl",
            "simple_http_crawl_sync"
        ]
        
        # Import the module
        module = importlib.import_module("src.features.llm_compare.crawler")
        
        # Check that all essential exports are available
        for export in essential_exports:
            assert hasattr(module, export), f"Essential export missing: {export}"
            
            # Check that the export is callable
            attr = getattr(module, export)
            if export != "WebCrawler":  # Skip class
                assert callable(attr), f"Export is not callable: {export}"


# Add a command-line option to update the API reference
def pytest_addoption(parser):
    parser.addoption("--update-api", action="store_true", help="Update API reference snapshots")


# Make snapshots directory if it doesn't exist
if __name__ == "__main__":
    # Run the test directly
    test = TestAPIStability()
    
    # Create the snapshots directory
    snapshots_dir = Path(__file__).parent / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    
    # Create crawler API snapshot
    module_name = "src.features.llm_compare.crawler"
    reference_path = snapshots_dir / "crawler_api.json"
    save_api_snapshot(module_name, reference_path)
    print(f"Created API snapshot at {reference_path}")
    
    # Run essential exports test
    test.test_essential_crawler_exports()
    print("Essential exports test passed") 