"""Pytest configuration and shared fixtures."""

import gc

import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Cleanup GPU memory after each test."""
    yield
    # Cleanup after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def check_gpu_available():
    """Check if GPU is available for tests."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return True
