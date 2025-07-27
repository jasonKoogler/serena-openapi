"""Test configuration for OpenAPI tests."""

import pytest


def pytest_configure(config):
    """Register the openapi marker."""
    config.addinivalue_line("markers", "openapi: marks tests as OpenAPI integration tests")


@pytest.fixture(scope="session")
def embedding_model_cache():
    """Cache the embedding model to avoid re-downloading during tests."""
    # This fixture can be used to manage model caching across test sessions
    # to speed up test execution
