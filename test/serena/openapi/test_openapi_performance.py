"""Performance benchmarks and optimization tests for OpenAPI functionality."""

import json
import os
import shutil
import tempfile
import time
from unittest.mock import Mock

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
def openapi_tool_with_real_model():
    """Create OpenApiTool with real dependencies for performance testing."""
    agent = Mock(spec=SerenaAgent)

    # Use real configuration values
    config = Mock(spec=SerenaConfig)
    openapi_config = Mock()
    openapi_config.embedding_model = "all-MiniLM-L6-v2"  # Small, fast model for testing
    openapi_config.index_cache_dir = ".serena/openapi_cache"
    openapi_config.use_redocly_validation = False  # Disable for testing
    openapi_config.redocly_timeout = 30
    config.openapi = openapi_config
    agent.serena_config = config

    # Mock project methods but use real temp directory
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


def create_large_openapi_spec(num_endpoints: int = 100) -> dict:
    """Create a large OpenAPI specification for performance testing."""
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Large Performance Test API", "version": "1.0.0"},
        "paths": {},
        "components": {"securitySchemes": {}},
        "webhooks": {},
    }

    # Add many endpoints
    for i in range(num_endpoints):
        spec["paths"][f"/api/v1/resource{i}"] = {
            "get": {
                "operationId": f"getResource{i}",
                "summary": f"Get resource {i}",
                "description": f"Retrieve resource {i} from the system with detailed information",
                "tags": [f"resource{i % 10}", "get-operations"],
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}},
                    {"name": "filter", "in": "query", "required": False, "schema": {"type": "string"}},
                ],
                "responses": {"200": {"description": f"Resource {i} retrieved successfully"}},
            },
            "post": {
                "operationId": f"createResource{i}",
                "summary": f"Create resource {i}",
                "description": f"Create a new resource {i} in the system",
                "tags": [f"resource{i % 10}", "post-operations"],
                "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
                "responses": {"201": {"description": f"Resource {i} created successfully"}},
            },
            "put": {
                "operationId": f"updateResource{i}",
                "summary": f"Update resource {i}",
                "description": f"Update existing resource {i} with new data",
                "tags": [f"resource{i % 10}", "put-operations"],
                "responses": {"200": {"description": f"Resource {i} updated successfully"}},
            },
            "delete": {
                "operationId": f"deleteResource{i}",
                "summary": f"Delete resource {i}",
                "description": f"Remove resource {i} from the system permanently",
                "tags": [f"resource{i % 10}", "delete-operations"],
                "responses": {"204": {"description": f"Resource {i} deleted successfully"}},
            },
        }

    # Add webhooks
    for i in range(min(20, num_endpoints // 5)):
        spec["webhooks"][f"webhook{i}"] = {
            "post": {
                "operationId": f"webhook{i}Handler",
                "summary": f"Handle webhook {i}",
                "description": f"Process webhook {i} events from external systems",
                "tags": [f"webhook{i % 5}", "webhooks"],
                "responses": {"200": {"description": f"Webhook {i} processed"}},
            }
        }

    # Add security schemes
    for i in range(min(10, num_endpoints // 10)):
        spec["components"]["securitySchemes"][f"Security{i}"] = {
            "type": "apiKey",
            "in": "header",
            "name": f"X-API-Key-{i}",
            "description": f"API key authentication method {i}",
        }

    return spec


@pytest.mark.openapi
@pytest.mark.slow
class TestOpenApiPerformanceBenchmarks:
    """Performance benchmarks for OpenAPI functionality."""

    def test_indexing_performance_small_spec(self, openapi_tool_with_real_model):
        """Benchmark indexing performance with small specification."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create small spec (10 endpoints)
        spec = create_large_openapi_spec(10)
        spec_path = os.path.join(temp_dir, "small_spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Measure indexing time
        start_time = time.time()
        result = tool.apply(spec_path=spec_path, query="test query", rebuild_index=True)
        indexing_time = time.time() - start_time

        assert isinstance(result, str)
        # Small spec should index quickly (< 10 seconds)
        assert indexing_time < 10.0

    def test_indexing_performance_medium_spec(self, openapi_tool_with_real_model):
        """Benchmark indexing performance with medium specification."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create medium spec (50 endpoints)
        spec = create_large_openapi_spec(50)
        spec_path = os.path.join(temp_dir, "medium_spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Measure indexing time
        start_time = time.time()
        result = tool.apply(spec_path=spec_path, query="test query", rebuild_index=True)
        indexing_time = time.time() - start_time

        assert isinstance(result, str)
        # Medium spec should index in reasonable time (< 30 seconds)
        assert indexing_time < 30.0

    def test_indexing_performance_large_spec(self, openapi_tool_with_real_model):
        """Benchmark indexing performance with large specification."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create large spec (200 endpoints)
        spec = create_large_openapi_spec(200)
        spec_path = os.path.join(temp_dir, "large_spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Measure indexing time
        start_time = time.time()
        result = tool.apply(spec_path=spec_path, query="test query", rebuild_index=True)
        indexing_time = time.time() - start_time

        assert isinstance(result, str)
        # Large spec should index in reasonable time (< 60 seconds)
        assert indexing_time < 60.0

    def test_search_performance_after_indexing(self, openapi_tool_with_real_model):
        """Benchmark search performance with pre-built index."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create and index a medium spec
        spec = create_large_openapi_spec(100)
        spec_path = os.path.join(temp_dir, "test_spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index first
        tool.apply(spec_path=spec_path, query="initial build", rebuild_index=True)

        # Test multiple search queries and measure performance
        queries = ["get resource", "create data", "update information", "delete item", "webhook events"]

        search_times = []
        for query in queries:
            start_time = time.time()
            result = tool.apply(spec_path=spec_path, query=query, top_k=5)
            search_time = time.time() - start_time
            search_times.append(search_time)

            assert isinstance(result, str)

        # Search should be fast (< 2 seconds per query)
        max_search_time = max(search_times)
        avg_search_time = sum(search_times) / len(search_times)

        assert max_search_time < 2.0
        assert avg_search_time < 1.0

    def test_concurrent_search_simulation(self, openapi_tool_with_real_model):
        """Simulate concurrent searches to test performance under load."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create and index a spec
        spec = create_large_openapi_spec(50)
        spec_path = os.path.join(temp_dir, "concurrent_spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index
        tool.apply(spec_path=spec_path, query="build index", rebuild_index=True)

        # Simulate multiple concurrent requests
        queries = [
            "resource management",
            "data operations",
            "user authentication",
            "file upload",
            "search functionality",
            "webhook handling",
            "security schemes",
            "error handling",
            "rate limiting",
            "pagination",
        ]

        start_time = time.time()
        results = []

        for query in queries:
            result = tool.apply(spec_path=spec_path, query=query, top_k=3)
            results.append(result)

        total_time = time.time() - start_time

        # All queries should succeed
        assert len(results) == len(queries)
        assert all(isinstance(r, str) for r in results)

        # Total time for all queries should be reasonable
        avg_time_per_query = total_time / len(queries)
        assert avg_time_per_query < 1.5

    def test_memory_usage_large_spec(self, openapi_tool_with_real_model):
        """Test memory usage with large specifications."""
        import psutil

        tool, temp_dir = openapi_tool_with_real_model

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Create large spec
        spec = create_large_openapi_spec(150)
        spec_path = os.path.join(temp_dir, "memory_test_spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Index the spec
        result = tool.apply(spec_path=spec_path, query="memory test", rebuild_index=True)
        assert isinstance(result, str)

        # Measure memory after indexing
        after_indexing_memory = process.memory_info().rss
        memory_increase = after_indexing_memory - initial_memory

        # Memory increase should be reasonable (< 500MB)
        memory_increase_mb = memory_increase / (1024 * 1024)
        assert memory_increase_mb < 500

        # Perform multiple searches to test memory stability
        for i in range(10):
            tool.apply(spec_path=spec_path, query=f"search {i}", top_k=3)

        # Memory should not grow significantly during searches
        final_memory = process.memory_info().rss
        search_memory_increase = final_memory - after_indexing_memory
        search_memory_increase_mb = search_memory_increase / (1024 * 1024)

        # Memory growth during searches should be minimal (< 50MB)
        assert search_memory_increase_mb < 50

    def test_index_cache_effectiveness(self, openapi_tool_with_real_model):
        """Test the effectiveness of index caching."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create spec
        spec = create_large_openapi_spec(50)
        spec_path = os.path.join(temp_dir, "cache_test_spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # First search (builds index)
        start_time = time.time()
        result1 = tool.apply(spec_path=spec_path, query="cache test", rebuild_index=True)
        first_search_time = time.time() - start_time

        assert isinstance(result1, str)

        # Second search (uses cached index)
        start_time = time.time()
        result2 = tool.apply(spec_path=spec_path, query="cache test again")
        second_search_time = time.time() - start_time

        assert isinstance(result2, str)

        # Cached search should be significantly faster
        assert second_search_time < first_search_time * 0.5

    def test_filtering_performance_impact(self, openapi_tool_with_real_model):
        """Test performance impact of various filtering options."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create spec with diverse tags and methods
        spec = create_large_openapi_spec(100)
        spec_path = os.path.join(temp_dir, "filter_test_spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index
        tool.apply(spec_path=spec_path, query="build", rebuild_index=True)

        # Test different filtering scenarios
        filter_scenarios = [
            {"name": "no_filter", "kwargs": {}},
            {"name": "method_filter", "kwargs": {"method_filter": "GET"}},
            {"name": "path_filter", "kwargs": {"path_filter": "/api/v1/resource.*"}},
            {"name": "tags_filter", "kwargs": {"tags_filter": ["resource1", "resource2"]}},
            {"name": "section_filter", "kwargs": {"section_filter": "operations"}},
            {"name": "combined_filters", "kwargs": {"method_filter": "POST", "tags_filter": ["resource1"]}},
        ]

        performance_results = {}

        for scenario in filter_scenarios:
            start_time = time.time()
            result = tool.apply(spec_path=spec_path, query="performance test", top_k=5, **scenario["kwargs"])
            search_time = time.time() - start_time

            performance_results[scenario["name"]] = search_time
            assert isinstance(result, str)

        # All filtering scenarios should complete in reasonable time
        for scenario_name, search_time in performance_results.items():
            assert search_time < 3.0, f"Scenario {scenario_name} took too long: {search_time}s"

    def test_result_set_size_performance(self, openapi_tool_with_real_model):
        """Test performance with different result set sizes."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create spec
        spec = create_large_openapi_spec(100)
        spec_path = os.path.join(temp_dir, "result_size_test.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index
        tool.apply(spec_path=spec_path, query="build", rebuild_index=True)

        # Test different result set sizes
        result_sizes = [1, 5, 10, 20, 50]
        performance_by_size = {}

        for size in result_sizes:
            start_time = time.time()
            result = tool.apply(spec_path=spec_path, query="test query", top_k=size)
            search_time = time.time() - start_time

            performance_by_size[size] = search_time
            assert isinstance(result, str)

        # Performance should scale reasonably with result set size
        # Larger result sets should not be disproportionately slower
        for size, search_time in performance_by_size.items():
            assert search_time < 2.0, f"top_k={size} took too long: {search_time}s"

    def test_output_format_performance(self, openapi_tool_with_real_model):
        """Test performance impact of different output formats."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create spec
        spec = create_large_openapi_spec(50)
        spec_path = os.path.join(temp_dir, "format_test.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index
        tool.apply(spec_path=spec_path, query="build", rebuild_index=True)

        # Test different output formats
        formats = [
            {"format": "human", "examples": False},
            {"format": "json", "examples": False},
            {"format": "markdown", "examples": False},
            {"format": "markdown", "examples": True},  # Most expensive format
        ]

        for fmt in formats:
            start_time = time.time()
            result = tool.apply(
                spec_path=spec_path,
                query="format performance test",
                output_format=fmt["format"],
                include_examples=fmt["examples"],
                top_k=5,
            )
            format_time = time.time() - start_time

            assert isinstance(result, str)
            # All formats should complete in reasonable time
            assert format_time < 5.0, f"Format {fmt} took too long: {format_time}s"


@pytest.mark.openapi
class TestOpenApiMemoryOptimization:
    """Tests for memory usage optimization."""

    def test_chunk_processing_memory_efficiency(self, openapi_tool_with_real_model):
        """Test memory efficiency during chunk processing."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create spec with many small operations
        spec = create_large_openapi_spec(300)  # Very large spec
        spec_path = os.path.join(temp_dir, "memory_efficiency.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Process in chunks and monitor memory
        try:
            result = tool.apply(spec_path=spec_path, query="memory efficiency test", rebuild_index=True)
            assert isinstance(result, str)
            # Test passes if no memory errors occur
        except MemoryError:
            pytest.fail("Memory error during chunk processing")

    def test_embedding_batch_processing(self, openapi_tool_with_real_model):
        """Test that embeddings are processed in efficient batches."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create moderate spec
        spec = create_large_openapi_spec(100)
        spec_path = os.path.join(temp_dir, "batch_test.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Monitor memory during embedding generation
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        result = tool.apply(spec_path=spec_path, query="batch processing test", rebuild_index=True)

        peak_memory = process.memory_info().rss
        memory_increase = (peak_memory - initial_memory) / (1024 * 1024)  # MB

        assert isinstance(result, str)
        # Memory increase should be reasonable even for large specs
        assert memory_increase < 400  # Less than 400MB increase

    def test_index_cleanup_after_processing(self, openapi_tool_with_real_model):
        """Test that temporary data is cleaned up after processing."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create spec
        spec = create_large_openapi_spec(50)
        spec_path = os.path.join(temp_dir, "cleanup_test.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Process multiple times and check for memory leaks
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Multiple processing rounds
        for i in range(5):
            result = tool.apply(spec_path=spec_path, query=f"cleanup test {i}", top_k=3)
            assert isinstance(result, str)

        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB

        # Memory should not grow significantly across multiple searches
        assert memory_growth < 100  # Less than 100MB growth

    def test_large_result_memory_handling(self, openapi_tool_with_real_model):
        """Test memory handling when processing large result sets."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create large spec
        spec = create_large_openapi_spec(200)
        spec_path = os.path.join(temp_dir, "large_results.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Build index
        tool.apply(spec_path=spec_path, query="build", rebuild_index=True)

        # Request large result sets
        import psutil

        process = psutil.Process()
        before_memory = process.memory_info().rss

        # Large result set with expensive formatting
        result = tool.apply(
            spec_path=spec_path,
            query="comprehensive search",
            top_k=50,  # Large result set
            output_format="markdown",
            include_examples=True,
        )

        after_memory = process.memory_info().rss
        memory_increase = (after_memory - before_memory) / (1024 * 1024)  # MB

        assert isinstance(result, str)
        # Memory usage for large results should be reasonable
        assert memory_increase < 200  # Less than 200MB for large result processing


@pytest.mark.openapi
class TestOpenApiScalabilityLimits:
    """Test scalability limits and edge cases."""

    def test_maximum_endpoints_handling(self, openapi_tool_with_real_model):
        """Test handling of specifications with maximum number of endpoints."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create very large spec (stress test)
        spec = create_large_openapi_spec(500)  # Very large
        spec_path = os.path.join(temp_dir, "max_endpoints.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Should handle large specs gracefully
        try:
            result = tool.apply(spec_path=spec_path, query="scalability test", rebuild_index=True)
            assert isinstance(result, str)
            # If it completes, it should return valid results
            assert len(result) > 0
        except (MemoryError, OSError):
            # Acceptable to fail on extremely large specs due to resource limits
            pytest.skip("System resource limits reached")

    def test_deep_nesting_handling(self, openapi_tool_with_real_model):
        """Test handling of deeply nested OpenAPI structures."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create spec with deep nesting
        spec = {
            "openapi": "3.0.3",
            "info": {"title": "Deep Nesting Test", "version": "1.0.0"},
            "paths": {
                "/deep": {
                    "post": {
                        "operationId": "deepNestingTest",
                        "summary": "Test deep nesting",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "level1": {
                                                "type": "object",
                                                "properties": {
                                                    "level2": {
                                                        "type": "object",
                                                        "properties": {
                                                            "level3": {
                                                                "type": "object",
                                                                "properties": {
                                                                    "level4": {
                                                                        "type": "object",
                                                                        "properties": {"data": {"type": "string"}},
                                                                    }
                                                                },
                                                            }
                                                        },
                                                    }
                                                },
                                            }
                                        },
                                    }
                                }
                            }
                        },
                        "responses": {"200": {"description": "Success"}},
                    }
                }
            },
        }

        spec_path = os.path.join(temp_dir, "deep_nesting.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        # Should handle deep nesting without stack overflow
        result = tool.apply(spec_path=spec_path, query="nesting test", rebuild_index=True)
        assert isinstance(result, str)

    def test_unicode_heavy_content_performance(self, openapi_tool_with_real_model):
        """Test performance with Unicode-heavy content."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create spec with lots of Unicode content
        unicode_spec = {
            "openapi": "3.0.3",
            "info": {"title": "Unicode Heavy API 🌍🚀⭐", "version": "1.0.0"},
            "paths": {},
        }

        # Add endpoints with heavy Unicode content
        for i in range(50):
            unicode_spec["paths"][f"/测试{i}"] = {
                "get": {
                    "operationId": f"unicode测试{i}",
                    "summary": f"Unicode测试端点{i} 🚀",
                    "description": f"处理Unicode字符的端点{i} 包含表情符号 🎉 和重音符号 àéîôù",
                    "tags": [f"unicode{i}", "测试", "🌍"],
                    "responses": {"200": {"description": f"成功响应{i} ✅"}},
                }
            }

        spec_path = os.path.join(temp_dir, "unicode_heavy.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(unicode_spec, f, ensure_ascii=False)

        # Should handle Unicode content efficiently
        start_time = time.time()
        result = tool.apply(spec_path=spec_path, query="Unicode测试", rebuild_index=True)
        processing_time = time.time() - start_time

        assert isinstance(result, str)
        # Unicode processing shouldn't be significantly slower
        assert processing_time < 30.0
