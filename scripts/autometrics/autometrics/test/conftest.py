"""
This file contains common fixtures and configuration for pytest.
"""

import pytest
import os
import shutil
import time

# Test-specific cache directory
TEST_CACHE_DIR = "./test_autometrics_cache"

@pytest.fixture(scope="session", autouse=True)
def cleanup_cache():
    """
    Clean up the test cache directory before and after all tests.
    This ensures that tests start with a clean cache
    and don't leave cache files behind.
    """
    # Clean any existing test cache before tests
    if os.path.exists(TEST_CACHE_DIR):
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except (OSError, PermissionError) as e:
            print(f"Warning: Failed to clean test cache directory before tests: {e}")
            # Try to wait a moment and retry once
            time.sleep(0.5)
            try:
                shutil.rmtree(TEST_CACHE_DIR)
            except:
                print(f"Warning: Still failed to clean test cache directory - tests may use existing cache")
    
    yield  # Run the tests
    
    # Clean up after tests
    if os.path.exists(TEST_CACHE_DIR):
        try:
            # Sometimes we need to wait a bit for resources to be released
            time.sleep(0.5)
            shutil.rmtree(TEST_CACHE_DIR)
        except (OSError, PermissionError) as e:
            print(f"Warning: Failed to clean test cache directory after tests: {e}")
            pass

@pytest.fixture(scope="session")
def test_cache_dir():
    """Provide the test cache directory path to tests"""
    return TEST_CACHE_DIR 