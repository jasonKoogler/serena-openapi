"""Memory usage optimization tests for OpenAPI functionality."""

import gc
import json
import os
import shutil
import tempfile
import tracemalloc
from unittest.mock import Mock, patch

import pytest

from serena.agent import SerenaAgent
from serena.config.serena_config import SerenaConfig
from serena.tools.openapi_tools import OpenApiTool


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_efficient_tool():
    """Create OpenApiTool optimized for memory testing."""
    agent = Mock(spec=SerenaAgent)

    # Use minimal configuration for memory testing
    config = Mock(spec=SerenaConfig)
    openapi_config = Mock()
    openapi_config.embedding_model = "all-MiniLM-L6-v2"  # Smallest model
    openapi_config.index_cache_dir = ".serena/openapi_cache"
    openapi_config.use_redocly_validation = False
    openapi_config.redocly_timeout = 30
    config.openapi = openapi_config
    agent.serena_config = config

    # Mock project methods
    temp_dir = tempfile.mkdtemp()
    agent.get_project_root.return_value = temp_dir

    project = Mock()
    project_config = Mock()
    project_config.openapi_specs = []
    project.project_config = project_config
    project.project_root = temp_dir
    agent.get_active_project.return_value = project

    tool = OpenApiTool(agent)

    yield tool, temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_memory_test_spec(size: str = "medium") -> dict:
    """Create OpenAPI spec for memory testing."""
    sizes = {"small": 20, "medium": 100, "large": 300}
    num_endpoints = sizes.get(size, 100)

    spec = {
        "openapi": "3.0.3",
        "info": {"title": f"Memory Test API ({size})", "version": "1.0.0"},
        "paths": {},
        "components": {"securitySchemes": {}},
        "webhooks": {},
    }

    # Add endpoints with realistic complexity
    for i in range(num_endpoints):
        endpoint_path = f"/api/v1/resource{i}"
        spec["paths"][endpoint_path] = {
            "get": {
                "operationId": f"getResource{i}",
                "summary": f"Retrieve resource {i}",
                "description": f"Get detailed information about resource {i} from the system database",
                "tags": [f"resource-{i % 10}", "get-operations", "data-retrieval"],
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "description": "Unique identifier for the resource",
                        "schema": {"type": "integer", "minimum": 1},
                    },
                    {
                        "name": "include_details",
                        "in": "query",
                        "required": False,
                        "description": "Include detailed information in response",
                        "schema": {"type": "boolean", "default": False},
                    },
                    {
                        "name": "format",
                        "in": "query",
                        "required": False,
                        "description": "Response format",
                        "schema": {"type": "string", "enum": ["json", "xml", "csv"], "default": "json"},
                    },
                ],
                "responses": {
                    "200": {
                        "description": f"Resource {i} retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "name": {"type": "string"},
                                        "description": {"type": "string"},
                                        "metadata": {"type": "object"},
                                    },
                                }
                            }
                        },
                    },
                    "404": {"description": "Resource not found"},
                    "500": {"description": "Internal server error"},
                },
            },
            "post": {
                "operationId": f"createResource{i}",
                "summary": f"Create new resource {i}",
                "description": f"Create a new resource {i} in the system with provided data",
                "tags": [f"resource-{i % 10}", "post-operations", "data-creation"],
                "requestBody": {
                    "required": True,
                    "description": "Resource data to create",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["name", "description"],
                                "properties": {
                                    "name": {"type": "string", "minLength": 1, "maxLength": 255},
                                    "description": {"type": "string", "maxLength": 1000},
                                    "category": {"type": "string", "enum": ["type1", "type2", "type3"]},
                                    "metadata": {"type": "object", "additionalProperties": True},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "201": {"description": f"Resource {i} created successfully"},
                    "400": {"description": "Invalid input data"},
                    "409": {"description": "Resource already exists"},
                    "500": {"description": "Internal server error"},
                },
            },
        }

    # Add webhooks for memory testing
    for i in range(min(10, num_endpoints // 10)):
        spec["webhooks"][f"resourceWebhook{i}"] = {
            "post": {
                "operationId": f"handleResourceWebhook{i}",
                "summary": f"Handle resource webhook {i}",
                "description": f"Process webhook events for resource type {i}",
                "tags": [f"webhook-{i}", "event-handling"],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "event": {"type": "string"},
                                    "resource_id": {"type": "integer"},
                                    "timestamp": {"type": "string", "format": "date-time"},
                                    "data": {"type": "object"},
                                },
                            }
                        }
                    }
                },
                "responses": {"200": {"description": f"Webhook {i} processed successfully"}},
            }
        }

    # Add security schemes
    for i in range(min(5, num_endpoints // 20)):
        spec["components"]["securitySchemes"][f"SecurityScheme{i}"] = {
            "type": "oauth2",
            "description": f"OAuth2 security scheme {i}",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": f"https://auth.example.com/oauth{i}/authorize",
                    "tokenUrl": f"https://auth.example.com/oauth{i}/token",
                    "scopes": {
                        f"read:resource{i}": f"Read access to resource {i}",
                        f"write:resource{i}": f"Write access to resource {i}",
                        f"delete:resource{i}": f"Delete access to resource {i}",
                    },
                }
            },
        }

    return spec


@pytest.mark.openapi
class TestOpenApiMemoryUsage:
    """Memory usage optimization tests."""

    def test_memory_usage_during_indexing(self, memory_efficient_tool):
        """Test memory usage during index creation."""
        tool, temp_dir = memory_efficient_tool

        # Create test spec
        spec = create_memory_test_spec("medium")
        spec_path = os.path.join(temp_dir, "memory_indexing.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Monitor memory during indexing
        tracemalloc.start()

        try:
            result = tool.apply(spec_path=spec_path, query="memory test", rebuild_index=True)
            assert isinstance(result, str)

            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Memory usage should be reasonable (< 100MB peak)
            peak_mb = peak / (1024 * 1024)
            assert peak_mb < 100, f"Peak memory usage too high: {peak_mb:.2f}MB"

        finally:
            if tracemalloc.is_tracing():
                tracemalloc.stop()

    def test_memory_cleanup_after_search(self, memory_efficient_tool):
        """Test that memory is properly cleaned up after searches."""
        tool, temp_dir = memory_efficient_tool

        # Create test spec
        spec = create_memory_test_spec("small")
        spec_path = os.path.join(temp_dir, "memory_cleanup.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index first
        tool.apply(spec_path=spec_path, query="initial", rebuild_index=True)

        # Force garbage collection and measure baseline
        gc.collect()
        tracemalloc.start()
        baseline_current, _ = tracemalloc.get_traced_memory()

        # Perform multiple searches
        for i in range(10):
            result = tool.apply(spec_path=spec_path, query=f"search {i}", top_k=3)
            assert isinstance(result, str)

        # Force garbage collection
        gc.collect()

        # Check memory after searches
        final_current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory growth should be minimal
        memory_growth = (final_current - baseline_current) / (1024 * 1024)
        assert memory_growth < 10, f"Too much memory growth: {memory_growth:.2f}MB"

    def test_large_result_set_memory_efficiency(self, memory_efficient_tool):
        """Test memory efficiency with large result sets."""
        tool, temp_dir = memory_efficient_tool

        # Create larger spec
        spec = create_memory_test_spec("large")
        spec_path = os.path.join(temp_dir, "large_results.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index
        tool.apply(spec_path=spec_path, query="build", rebuild_index=True)

        # Test large result sets
        tracemalloc.start()

        try:
            # Request many results with expensive formatting
            result = tool.apply(
                spec_path=spec_path,
                query="comprehensive search",
                top_k=50,
                output_format="markdown",
                include_examples=True,
            )

            assert isinstance(result, str)

            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)

            # Peak memory for large result processing should be reasonable
            assert peak_mb < 150, f"Peak memory too high for large results: {peak_mb:.2f}MB"

        finally:
            tracemalloc.stop()

    def test_embedding_memory_optimization(self, memory_efficient_tool):
        """Test memory optimization during embedding generation."""
        tool, temp_dir = memory_efficient_tool

        # Create spec with many chunks
        spec = create_memory_test_spec("medium")
        spec_path = os.path.join(temp_dir, "embedding_memory.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Monitor memory during chunk creation and embedding
        tracemalloc.start()

        try:
            # This will trigger embedding generation
            result = tool.apply(spec_path=spec_path, query="embedding test", rebuild_index=True)
            assert isinstance(result, str)

            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)

            # Embedding generation should not use excessive memory
            assert peak_mb < 200, f"Embedding memory usage too high: {peak_mb:.2f}MB"

        finally:
            tracemalloc.stop()

    def test_index_cache_memory_efficiency(self, memory_efficient_tool):
        """Test memory efficiency of index caching."""
        tool, temp_dir = memory_efficient_tool

        # Create test spec
        spec = create_memory_test_spec("medium")
        spec_path = os.path.join(temp_dir, "cache_memory.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # First indexing (creates cache)
        tracemalloc.start()
        result1 = tool.apply(spec_path=spec_path, query="first", rebuild_index=True)
        first_current, first_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert isinstance(result1, str)

        # Force cleanup
        gc.collect()

        # Second search (uses cache)
        tracemalloc.start()
        result2 = tool.apply(spec_path=spec_path, query="second")
        second_current, second_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert isinstance(result2, str)

        # Cached search should use less memory
        first_peak_mb = first_peak / (1024 * 1024)
        second_peak_mb = second_peak / (1024 * 1024)

        assert second_peak_mb < first_peak_mb * 0.7, "Cached search should use significantly less memory"

    def test_concurrent_request_memory_isolation(self, memory_efficient_tool):
        """Test memory isolation between concurrent-like requests."""
        tool, temp_dir = memory_efficient_tool

        # Create test spec
        spec = create_memory_test_spec("small")
        spec_path = os.path.join(temp_dir, "concurrent_memory.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index
        tool.apply(spec_path=spec_path, query="build", rebuild_index=True)
        gc.collect()

        # Simulate multiple concurrent-like requests
        tracemalloc.start()
        baseline_current, _ = tracemalloc.get_traced_memory()

        memory_snapshots = []

        for i in range(5):
            # Simulate different request patterns
            result = tool.apply(spec_path=spec_path, query=f"request {i}", top_k=3, output_format="json")
            assert isinstance(result, str)

            current, _ = tracemalloc.get_traced_memory()
            memory_snapshots.append(current)

        tracemalloc.stop()

        # Memory usage should remain stable across requests
        memory_diffs = [abs(snap - baseline_current) / (1024 * 1024) for snap in memory_snapshots]
        max_diff = max(memory_diffs)

        assert max_diff < 20, f"Memory usage varies too much between requests: {max_diff:.2f}MB"

    def test_filter_operation_memory_efficiency(self, memory_efficient_tool):
        """Test memory efficiency of filtering operations."""
        tool, temp_dir = memory_efficient_tool

        # Create spec with diverse content
        spec = create_memory_test_spec("medium")
        spec_path = os.path.join(temp_dir, "filter_memory.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index
        tool.apply(spec_path=spec_path, query="build", rebuild_index=True)

        # Test different filtering scenarios for memory usage
        filter_tests = [
            {"name": "no_filter", "kwargs": {}},
            {"name": "method", "kwargs": {"method_filter": "GET"}},
            {"name": "path", "kwargs": {"path_filter": "/api/v1/.*"}},
            {"name": "tags", "kwargs": {"tags_filter": ["resource-1", "resource-2"]}},
            {"name": "section", "kwargs": {"section_filter": "operations"}},
            {"name": "combined", "kwargs": {"method_filter": "POST", "tags_filter": ["resource-1"]}},
        ]

        memory_usage = {}

        for test in filter_tests:
            gc.collect()
            tracemalloc.start()

            result = tool.apply(spec_path=spec_path, query="filter test", top_k=5, **test["kwargs"])
            assert isinstance(result, str)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_usage[test["name"]] = peak / (1024 * 1024)

        # All filtering operations should use similar memory
        memory_values = list(memory_usage.values())
        max_memory = max(memory_values)
        min_memory = min(memory_values)

        # Memory usage should not vary dramatically between filter types
        memory_variation = (max_memory - min_memory) / min_memory
        assert memory_variation < 0.5, f"Too much memory variation between filters: {memory_variation:.2%}"

    def test_garbage_collection_effectiveness(self, memory_efficient_tool):
        """Test that objects are properly garbage collected."""
        tool, temp_dir = memory_efficient_tool

        # Create test spec
        spec = create_memory_test_spec("small")
        spec_path = os.path.join(temp_dir, "gc_test.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Get initial object count
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform indexing and searches
        tool.apply(spec_path=spec_path, query="gc test 1", rebuild_index=True)
        tool.apply(spec_path=spec_path, query="gc test 2")
        tool.apply(spec_path=spec_path, query="gc test 3")

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count should not grow excessively
        object_growth = final_objects - initial_objects
        growth_percentage = object_growth / initial_objects

        assert growth_percentage < 0.2, f"Too many objects created: {object_growth} ({growth_percentage:.2%} growth)"

    @patch("psutil.virtual_memory")
    def test_low_memory_handling(self, mock_memory, memory_efficient_tool):
        """Test behavior under low memory conditions."""
        tool, temp_dir = memory_efficient_tool

        # Mock low memory condition
        mock_memory.return_value.available = 100 * 1024 * 1024  # 100MB available

        # Create test spec
        spec = create_memory_test_spec("small")  # Use small spec for low memory
        spec_path = os.path.join(temp_dir, "low_memory.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Should handle low memory gracefully
        try:
            result = tool.apply(spec_path=spec_path, query="low memory test", rebuild_index=True)
            assert isinstance(result, str)
            # If it succeeds, memory management is working
        except MemoryError:
            # Acceptable to fail gracefully under extreme memory pressure
            pass

    def test_chunking_memory_optimization(self, memory_efficient_tool):
        """Test memory optimization in chunk processing."""
        tool, temp_dir = memory_efficient_tool

        # Create spec that will generate many chunks
        spec = create_memory_test_spec("medium")
        spec_path = os.path.join(temp_dir, "chunking_memory.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Monitor memory during chunk creation
        tracemalloc.start()

        try:
            # Create chunks (this happens during indexing)
            chunks = tool._create_semantic_chunks(spec)

            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)

            # Chunking should be memory efficient
            assert peak_mb < 50, f"Chunking uses too much memory: {peak_mb:.2f}MB"
            assert len(chunks) > 0, "Should create chunks"

        finally:
            tracemalloc.stop()

    def test_search_result_memory_streaming(self, memory_efficient_tool):
        """Test memory efficiency in search result processing."""
        tool, temp_dir = memory_efficient_tool

        # Create test spec
        spec = create_memory_test_spec("medium")
        spec_path = os.path.join(temp_dir, "streaming_memory.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index
        tool.apply(spec_path=spec_path, query="build", rebuild_index=True)

        # Test different result sizes for memory usage
        result_sizes = [1, 5, 10, 25, 50]
        memory_usage = []

        for size in result_sizes:
            gc.collect()
            tracemalloc.start()

            result = tool.apply(spec_path=spec_path, query=f"size test {size}", top_k=size)
            assert isinstance(result, str)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_usage.append(peak / (1024 * 1024))

        # Memory usage should scale sub-linearly with result size
        # (good caching/streaming should prevent linear scaling)
        if len(memory_usage) >= 2:
            memory_ratio = memory_usage[-1] / memory_usage[0]  # 50 results vs 1 result
            result_ratio = result_sizes[-1] / result_sizes[0]  # 50x more results

            # Memory should not scale linearly with result count
            assert memory_ratio < result_ratio * 0.7, "Memory scaling too linear with result count"


@pytest.mark.openapi
class TestOpenApiMemoryLeaks:
    """Test for memory leaks in OpenAPI functionality."""

    def test_repeated_indexing_memory_leak(self, memory_efficient_tool):
        """Test for memory leaks during repeated indexing."""
        tool, temp_dir = memory_efficient_tool

        # Create test spec
        spec = create_memory_test_spec("small")
        spec_path = os.path.join(temp_dir, "leak_test.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        memory_snapshots = []

        # Perform multiple index rebuilds
        for i in range(3):
            gc.collect()
            tracemalloc.start()

            result = tool.apply(spec_path=spec_path, query=f"leak test {i}", rebuild_index=True)
            assert isinstance(result, str)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_snapshots.append(peak)

        # Memory usage should not grow significantly across rebuilds
        if len(memory_snapshots) >= 2:
            first_peak = memory_snapshots[0] / (1024 * 1024)
            last_peak = memory_snapshots[-1] / (1024 * 1024)
            growth = (last_peak - first_peak) / first_peak

            assert growth < 0.3, f"Potential memory leak detected: {growth:.2%} growth"

    def test_search_iteration_memory_leak(self, memory_efficient_tool):
        """Test for memory leaks during repeated searches."""
        tool, temp_dir = memory_efficient_tool

        # Create and index spec
        spec = create_memory_test_spec("small")
        spec_path = os.path.join(temp_dir, "search_leak.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index once
        tool.apply(spec_path=spec_path, query="initial", rebuild_index=True)

        memory_snapshots = []

        # Perform many searches
        for i in range(10):
            if i % 3 == 0:  # Sample memory every few iterations
                gc.collect()
                tracemalloc.start()

            result = tool.apply(spec_path=spec_path, query=f"search leak test {i}", top_k=3)
            assert isinstance(result, str)

            if i % 3 == 0:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_snapshots.append(current)

        # Memory should remain stable across searches
        if len(memory_snapshots) >= 2:
            memory_diffs = [abs(memory_snapshots[i] - memory_snapshots[0]) / (1024 * 1024) for i in range(1, len(memory_snapshots))]
            max_diff = max(memory_diffs)

            assert max_diff < 5, f"Memory leak detected in searches: {max_diff:.2f}MB growth"

    def test_model_loading_memory_cleanup(self, memory_efficient_tool):
        """Test that embedding model memory is properly managed."""
        tool, temp_dir = memory_efficient_tool

        # Force model loading
        spec = create_memory_test_spec("small")
        spec_path = os.path.join(temp_dir, "model_cleanup.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # First load (model initialization)
        gc.collect()
        tracemalloc.start()
        result1 = tool.apply(spec_path=spec_path, query="model test 1", rebuild_index=True)
        first_current, first_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert isinstance(result1, str)

        # Subsequent uses (model already loaded)
        gc.collect()
        tracemalloc.start()
        result2 = tool.apply(spec_path=spec_path, query="model test 2")
        second_current, second_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert isinstance(result2, str)

        # Model should be reused, not reloaded
        first_peak_mb = first_peak / (1024 * 1024)
        second_peak_mb = second_peak / (1024 * 1024)

        # Second usage should have much lower peak (no model loading)
        assert second_peak_mb < first_peak_mb * 0.5, "Model appears to be reloaded instead of reused"
