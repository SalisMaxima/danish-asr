"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Make scripts/ importable for tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def data_available() -> bool:
    """Check if data is available for testing."""
    data_dir = Path("data/processed")
    return data_dir.exists() and any(data_dir.iterdir()) if data_dir.exists() else False


@pytest.fixture
def skip_if_no_data(data_available):
    """Skip test if data is not available (expected in CI)."""
    if not data_available:
        pytest.skip("Data files not available (expected in CI)")
