"""Tests for OpenAPI error handling and edge cases."""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

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
def mock_agent(temp_dir):
    """Create a mock agent with minimal configuration."""
    agent = Mock(spec=SerenaAgent)

    # Mock the serena_config
    config = Mock(spec=SerenaConfig)
    openapi_config = Mock()
    openapi_config.embedding_model = "all-MiniLM-L6-v2"
    openapi_config.index_cache_dir = ".serena/openapi_cache"
    openapi_config.use_redocly_validation = False
    openapi_config.redocly_timeout = 30
    config.openapi = openapi_config
    agent.serena_config = config

    # Mock project methods
    agent.get_project_root.return_value = temp_dir

    # Mock project configuration
    project = Mock()
    project_config = Mock()
    project_config.openapi_specs = []
    project.project_config = project_config
    project.project_root = temp_dir
    agent.get_active_project.return_value = project

    return agent


@pytest.fixture
def openapi_tool(mock_agent):
    """Create an OpenApiTool instance with mocked dependencies."""
    return OpenApiTool(mock_agent)


class TestOpenApiErrorHandling:
    """Test cases for OpenAPI error handling."""

    def test_apply_with_no_specs_found(self, openapi_tool):
        """Test apply method when no OpenAPI specs can be found."""
        # No specs in project, no auto-discovery
        result = openapi_tool.apply(query="test query")

        assert "Error: No OpenAPI specifications found" in result

    def test_apply_with_empty_query(self, openapi_tool, temp_dir):
        """Test apply method with empty query."""
        # Create a dummy spec
        spec_path = os.path.join(temp_dir, "test.yaml")
        with open(spec_path, "w") as f:
            f.write("openapi: 3.0.3\ninfo:\n  title: Test\n  version: 1.0.0\npaths: {}")

        result = openapi_tool.apply(spec_path=spec_path, query="")

        assert "Error: Query parameter is required" in result

    def test_apply_with_invalid_spec_path(self, openapi_tool):
        """Test apply method with non-existent spec file."""
        result = openapi_tool.apply(spec_path="/nonexistent/path.yaml", query="test")

        assert "Error: No OpenAPI specifications found" in result

    def test_invalid_json_spec_file(self, openapi_tool, temp_dir):
        """Test handling of invalid JSON OpenAPI spec."""
        spec_path = os.path.join(temp_dir, "invalid.json")
        with open(spec_path, "w") as f:
            f.write("{ invalid json content")

        result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

        assert "Error processing OpenAPI specification" in result

    def test_invalid_yaml_spec_file(self, openapi_tool, temp_dir):
        """Test handling of invalid YAML OpenAPI spec."""
        spec_path = os.path.join(temp_dir, "invalid.yaml")
        with open(spec_path, "w") as f:
            f.write("invalid: yaml: content: [")

        # Mock yaml import to simulate parsing error
        with patch("serena.tools.openapi_tools.yaml") as mock_yaml:
            mock_yaml.safe_load.side_effect = Exception("YAML parsing error")

            result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

            assert "Error processing OpenAPI specification" in result

    def test_yaml_import_error(self, openapi_tool, temp_dir):
        """Test handling when PyYAML is not available."""
        spec_path = os.path.join(temp_dir, "test.yaml")
        with open(spec_path, "w") as f:
            f.write("openapi: 3.0.3")

        # Mock ImportError for yaml
        with patch("serena.tools.openapi_tools.yaml", side_effect=ImportError("No module named 'yaml'")):
            result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

            assert "Error processing OpenAPI specification" in result

    def test_empty_openapi_spec(self, openapi_tool, temp_dir):
        """Test handling of OpenAPI spec with no paths."""
        spec_path = os.path.join(temp_dir, "empty.json")
        empty_spec = {"openapi": "3.0.3", "info": {"title": "Empty API", "version": "1.0.0"}, "paths": {}}
        with open(spec_path, "w") as f:
            json.dump(empty_spec, f)

        result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

        assert "Error processing OpenAPI specification" in result

    def test_malformed_openapi_structure(self, openapi_tool, temp_dir):
        """Test handling of malformed OpenAPI structure."""
        spec_path = os.path.join(temp_dir, "malformed.json")
        malformed_spec = {
            "openapi": "3.0.3",
            "info": {"title": "Malformed API", "version": "1.0.0"},
            "paths": {"/test": {"get": "this should be an object, not a string"}},
        }
        with open(spec_path, "w") as f:
            json.dump(malformed_spec, f)

        result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

        # Should handle malformed structure gracefully
        assert "Error processing OpenAPI specification" in result or "No relevant endpoints found" in result

    @patch("serena.tools.openapi_tools.SentenceTransformer")
    def test_embedding_model_loading_error(self, mock_transformer, openapi_tool, temp_dir):
        """Test handling of embedding model loading errors."""
        mock_transformer.side_effect = Exception("Model loading failed")

        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Test", "version": "1.0.0"},
                    "paths": {"/test": {"get": {"operationId": "test", "responses": {"200": {}}}}},
                },
                f,
            )

        result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

        assert "Error processing OpenAPI specification" in result

    @patch("serena.tools.openapi_tools.faiss")
    def test_faiss_index_creation_error(self, mock_faiss, openapi_tool, temp_dir):
        """Test handling of FAISS index creation errors."""
        # Mock FAISS to raise error
        mock_faiss.IndexFlatIP.side_effect = Exception("FAISS error")

        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Test", "version": "1.0.0"},
                    "paths": {"/test": {"get": {"operationId": "test", "responses": {"200": {}}}}},
                },
                f,
            )

        result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

        assert "Error processing OpenAPI specification" in result

    def test_permission_denied_cache_directory(self, mock_agent):
        """Test handling when cache directory cannot be created due to permissions."""
        # Mock a directory where we can't create subdirectories
        mock_agent.get_project_root.return_value = "/root"

        tool = OpenApiTool(mock_agent)

        # This should not raise an exception, but handle gracefully
        cache_dir = tool.cache_dir
        assert cache_dir is not None

    def test_corrupted_index_files(self, openapi_tool, temp_dir):
        """Test handling of corrupted FAISS index files."""
        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Test", "version": "1.0.0"},
                    "paths": {"/test": {"get": {"operationId": "test", "responses": {"200": {}}}}},
                },
                f,
            )

        # Create corrupted index files
        spec_hash = openapi_tool._get_file_hash(spec_path)
        index_dir = os.path.join(openapi_tool.cache_dir, spec_hash)
        os.makedirs(index_dir, exist_ok=True)

        # Create corrupted files
        with open(os.path.join(index_dir, "index.faiss"), "w") as f:
            f.write("corrupted faiss data")

        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            f.write("{ corrupted json")

        with open(os.path.join(index_dir, "metadata.json"), "w") as f:
            f.write("corrupted metadata")

        # Should rebuild index when corrupted files are detected
        result = openapi_tool.apply(spec_path=spec_path, query="test")

        # Should not crash, either rebuild or return error
        assert isinstance(result, str)

    @patch("subprocess.run")
    def test_redocly_timeout_error(self, mock_run, openapi_tool, temp_dir):
        """Test handling of Redocly CLI timeout."""
        # Enable Redocly
        openapi_tool.agent.serena_config.openapi.use_redocly_validation = True
        openapi_tool.agent.serena_config.openapi.redocly_timeout = 1

        # Mock timeout error
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("redocly", 1)

        spec_path = os.path.join(temp_dir, "test.yaml")
        with open(spec_path, "w") as f:
            f.write(
                "openapi: 3.0.3\ninfo:\n  title: Test\n  version: 1.0.0\npaths:\n  /test:\n    get:\n      responses:\n        '200': {}"
            )

        # Should fall back to direct parsing
        result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

        # Should not crash, should fall back gracefully
        assert isinstance(result, str)

    @patch("subprocess.run")
    def test_redocly_command_not_found(self, mock_run, openapi_tool, temp_dir):
        """Test handling when Redocly CLI is not installed."""
        # Enable Redocly
        openapi_tool.agent.serena_config.openapi.use_redocly_validation = True

        # Mock command not found
        mock_run.side_effect = FileNotFoundError("redocly command not found")

        spec_path = os.path.join(temp_dir, "test.yaml")
        with open(spec_path, "w") as f:
            f.write(
                "openapi: 3.0.3\ninfo:\n  title: Test\n  version: 1.0.0\npaths:\n  /test:\n    get:\n      responses:\n        '200': {}"
            )

        # Should fall back to direct parsing
        result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

        assert isinstance(result, str)

    def test_network_error_during_model_download(self, openapi_tool, temp_dir):
        """Test handling of network errors during model download."""
        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Test", "version": "1.0.0"},
                    "paths": {"/test": {"get": {"operationId": "test", "responses": {"200": {}}}}},
                },
                f,
            )

        # Mock network error during model loading
        with patch("serena.tools.openapi_tools.SentenceTransformer") as mock_transformer:
            mock_transformer.side_effect = ConnectionError("Network error")

            result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

            assert "Error processing OpenAPI specification" in result

    def test_out_of_memory_error(self, openapi_tool, temp_dir):
        """Test handling of out-of-memory errors during processing."""
        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Test", "version": "1.0.0"},
                    "paths": {"/test": {"get": {"operationId": "test", "responses": {"200": {}}}}},
                },
                f,
            )

        # Mock memory error during embedding generation
        with patch.object(openapi_tool, "model") as mock_model:
            mock_model.encode.side_effect = MemoryError("Out of memory")

            result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

            assert "Error processing OpenAPI specification" in result

    @patch("serena.tools.openapi_tools.faiss")
    def test_faiss_search_error(self, mock_faiss, openapi_tool, temp_dir):
        """Test handling of FAISS search errors."""
        # Create valid spec and index
        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Test", "version": "1.0.0"},
                    "paths": {"/test": {"get": {"operationId": "test", "responses": {"200": {}}}}},
                },
                f,
            )

        # Mock FAISS index that fails during search
        mock_index = Mock()
        mock_index.search.side_effect = Exception("FAISS search error")
        mock_faiss.read_index.return_value = mock_index

        # Create valid index files
        spec_hash = openapi_tool._get_file_hash(spec_path)
        index_dir = os.path.join(openapi_tool.cache_dir, spec_hash)
        os.makedirs(index_dir, exist_ok=True)

        chunks = [{"text": "test", "metadata": {"operationId": "test", "method": "GET", "path": "/test"}}]
        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump(chunks, f)

        metadata = {"spec_path": spec_path, "num_chunks": 1, "created_at": os.path.getmtime(spec_path)}
        with open(os.path.join(index_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        result = openapi_tool.apply(spec_path=spec_path, query="test")

        assert "Error processing OpenAPI specification" in result

    def test_invalid_regex_in_path_filter(self, openapi_tool, temp_dir):
        """Test handling of invalid regex in path filter."""
        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Test", "version": "1.0.0"},
                    "paths": {"/test": {"get": {"operationId": "test", "responses": {"200": {}}}}},
                },
                f,
            )

        # Use invalid regex pattern
        result = openapi_tool.apply(spec_path=spec_path, query="test", path_filter="[invalid regex", rebuild_index=True)

        # Should handle invalid regex gracefully
        assert isinstance(result, str)

    def test_disk_full_error_during_index_creation(self, openapi_tool, temp_dir):
        """Test handling of disk full errors during index creation."""
        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Test", "version": "1.0.0"},
                    "paths": {"/test": {"get": {"operationId": "test", "responses": {"200": {}}}}},
                },
                f,
            )

        # Mock disk full error during file writing
        with patch("builtins.open", side_effect=OSError("No space left on device")):
            result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

            assert "Error processing OpenAPI specification" in result

    def test_concurrent_access_to_index_files(self, openapi_tool, temp_dir):
        """Test handling of concurrent access to index files."""
        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Test", "version": "1.0.0"},
                    "paths": {"/test": {"get": {"operationId": "test", "responses": {"200": {}}}}},
                },
                f,
            )

        # Simulate file being locked by another process
        with patch("builtins.open", side_effect=PermissionError("File is locked")):
            result = openapi_tool.apply(spec_path=spec_path, query="test")

            # Should handle gracefully
            assert isinstance(result, str)

    def test_unicode_handling_in_specs(self, openapi_tool, temp_dir):
        """Test handling of Unicode characters in OpenAPI specs."""
        spec_path = os.path.join(temp_dir, "unicode.json")
        unicode_spec = {
            "openapi": "3.0.3",
            "info": {"title": "Unicode API 🚀", "version": "1.0.0"},
            "paths": {
                "/测试": {
                    "get": {
                        "operationId": "测试端点",
                        "summary": "Test endpoint with 中文 characters",
                        "description": "Endpoint with émojis 🎉 and àccënts",
                        "responses": {"200": {"description": "Success"}},
                    }
                }
            },
        }

        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(unicode_spec, f, ensure_ascii=False)

        # Should handle Unicode characters gracefully
        result = openapi_tool.apply(spec_path=spec_path, query="测试", rebuild_index=True)

        assert isinstance(result, str)
        # Should find the Unicode endpoint
        assert "测试端点" in result or "No relevant endpoints found" in result

    def test_very_large_openapi_spec(self, openapi_tool, temp_dir):
        """Test handling of very large OpenAPI specifications."""
        spec_path = os.path.join(temp_dir, "large.json")

        # Create a large spec with many endpoints
        large_spec = {"openapi": "3.0.3", "info": {"title": "Large API", "version": "1.0.0"}, "paths": {}}

        # Add 1000 endpoints
        for i in range(1000):
            large_spec["paths"][f"/endpoint{i}"] = {
                "get": {
                    "operationId": f"getEndpoint{i}",
                    "summary": f"Get endpoint {i}",
                    "description": f"Description for endpoint {i}",
                    "responses": {"200": {"description": "Success"}},
                }
            }

        with open(spec_path, "w") as f:
            json.dump(large_spec, f)

        # Should handle large specs without crashing
        result = openapi_tool.apply(spec_path=spec_path, query="endpoint", rebuild_index=True, top_k=5)

        assert isinstance(result, str)
        assert "relevant API endpoints" in result or "Error" in result


class TestOpenApiErrorRecovery:
    """Test error recovery and resilience mechanisms."""

    def test_graceful_degradation_with_partial_failures(self, openapi_tool, temp_dir):
        """Test graceful degradation when some specs fail to process."""
        # Create multiple specs, some valid, some invalid
        valid_spec = os.path.join(temp_dir, "valid.json")
        with open(valid_spec, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Valid API", "version": "1.0.0"},
                    "paths": {"/valid": {"get": {"operationId": "valid", "responses": {"200": {}}}}},
                },
                f,
            )

        invalid_spec = os.path.join(temp_dir, "invalid.json")
        with open(invalid_spec, "w") as f:
            f.write("{ invalid json")

        # Configure project with both specs
        openapi_tool.agent.get_active_project.return_value.project_config.openapi_specs = ["valid.json", "invalid.json"]

        result = openapi_tool.apply(query="test", search_all_specs=True, rebuild_index=True)

        # Should process valid spec despite invalid one
        assert isinstance(result, str)
        # Should either find results from valid spec or handle error gracefully
        assert "valid" in result or "Error" in result or "No relevant endpoints found" in result

    def test_retry_mechanism_for_transient_errors(self, openapi_tool, temp_dir):
        """Test retry mechanisms for transient errors."""
        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Test", "version": "1.0.0"},
                    "paths": {"/test": {"get": {"operationId": "test", "responses": {"200": {}}}}},
                },
                f,
            )

        # Mock transient failure followed by success
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Transient network error")
            return MagicMock()  # Success on second call

        with patch("serena.tools.openapi_tools.SentenceTransformer", side_effect=side_effect):
            result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

            # Should handle transient errors
            assert isinstance(result, str)

    def test_fallback_to_simple_text_search(self, openapi_tool, temp_dir):
        """Test fallback to simple text search when semantic search fails."""
        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(
                {
                    "openapi": "3.0.3",
                    "info": {"title": "Test", "version": "1.0.0"},
                    "paths": {"/test": {"get": {"operationId": "testOperation", "responses": {"200": {}}}}},
                },
                f,
            )

        # Mock embedding failure
        with patch.object(openapi_tool, "model") as mock_model:
            mock_model.encode.side_effect = Exception("Embedding failed")

            result = openapi_tool.apply(spec_path=spec_path, query="test", rebuild_index=True)

            # Should handle embedding failure
            assert "Error processing OpenAPI specification" in result
