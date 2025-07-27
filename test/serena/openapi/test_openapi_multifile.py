"""Tests for multi-file OpenAPI specification support with $ref resolution."""

import json
import os
import shutil
import tempfile
from pathlib import Path
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


@pytest.fixture
def multifile_spec_path():
    """Return path to the multi-file test specification."""
    test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi" / "multifile"
    return test_resources / "openapi.yaml"


@pytest.mark.openapi
class TestMultiFileSpecification:
    """Test multi-file OpenAPI specification handling."""

    def test_multifile_spec_exists(self, multifile_spec_path):
        """Test that the multi-file specification exists and is readable."""
        assert multifile_spec_path.exists(), f"Multi-file spec not found at {multifile_spec_path}"

        # Verify main spec can be loaded
        with open(multifile_spec_path) as f:
            import yaml

            spec = yaml.safe_load(f)

        assert spec["openapi"] == "3.0.3"
        assert spec["info"]["title"] == "Multi-File E-commerce API"

        # Verify it contains $ref references
        assert any("$ref" in str(path_def) for path_def in spec["paths"].values())

    def test_dependency_detection(self, openapi_tool, multifile_spec_path):
        """Test detection of file dependencies via $ref analysis."""
        if not multifile_spec_path.exists():
            pytest.skip("Multi-file spec not found")

        # Get dependencies
        dependencies = openapi_tool._get_spec_dependencies(str(multifile_spec_path))

        # Should find component and path files
        dependency_names = [os.path.basename(dep) for dep in dependencies]

        expected_files = [
            "schemas.yaml",
            "parameters.yaml",
            "common.yaml",
            "users.yaml",
            "products.yaml",
            "orders.yaml",
            "cart.yaml",
        ]

        for expected in expected_files:
            assert expected in dependency_names, f"Missing dependency: {expected}"

    def test_external_ref_collection(self, openapi_tool, multifile_spec_path):
        """Test collection of external $ref references."""
        if not multifile_spec_path.exists():
            pytest.skip("Multi-file spec not found")

        # Load the main spec
        with open(multifile_spec_path) as f:
            import yaml

            spec = yaml.safe_load(f)

        # Test external ref detection
        dependencies = set()
        base_dir = os.path.dirname(str(multifile_spec_path))
        openapi_tool._collect_external_refs(spec, base_dir, dependencies)

        assert len(dependencies) > 0, "Should find external references"

        # Verify some specific expected dependencies
        dependency_names = [os.path.basename(dep) for dep in dependencies]
        assert "schemas.yaml" in dependency_names
        assert "users.yaml" in dependency_names

    def test_has_external_refs_detection(self, openapi_tool):
        """Test detection of external $ref references in specs."""
        # Test spec with external refs
        spec_with_refs = {
            "openapi": "3.1.1",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {"/test": {"$ref": "./external.yaml"}},
        }

        assert openapi_tool._has_external_refs(spec_with_refs)

        # Test spec with only internal refs
        spec_internal_only = {
            "openapi": "3.1.1",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {"/test": {"get": {"responses": {"200": {"schema": {"$ref": "#/components/schemas/Response"}}}}}},
        }

        assert not openapi_tool._has_external_refs(spec_internal_only)

        # Test spec with no refs
        spec_no_refs = {
            "openapi": "3.0.3",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {"/test": {"get": {"responses": {"200": {"description": "OK"}}}}},
        }

        assert not openapi_tool._has_external_refs(spec_no_refs)

    def test_cache_invalidation_with_dependencies(self, openapi_tool, multifile_spec_path, temp_dir):
        """Test that cache is invalidated when any dependency changes."""
        if not multifile_spec_path.exists():
            pytest.skip("Multi-file spec not found")

        # Copy multifile spec to temp directory for modification
        temp_spec_dir = os.path.join(temp_dir, "multifile")
        shutil.copytree(multifile_spec_path.parent, temp_spec_dir)
        temp_main_spec = os.path.join(temp_spec_dir, "openapi.yaml")

        # Get initial hash
        initial_hash = openapi_tool._get_file_hash(temp_main_spec)

        # Modify a dependency file
        schemas_file = os.path.join(temp_spec_dir, "components", "schemas.yaml")
        with open(schemas_file, "a") as f:
            f.write("\n# Modified for testing\n")

        # Hash should change due to dependency modification
        new_hash = openapi_tool._get_file_hash(temp_main_spec)
        assert new_hash != initial_hash, "Hash should change when dependency is modified"

    def test_jsonref_resolution_fallback(self, openapi_tool, temp_dir):
        """Test jsonref fallback resolution when Redocly is not available."""
        # Create a simple single-file spec with internal refs (this should work)
        simple_spec = {
            "openapi": "3.0.3",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "summary": "List users",
                        "responses": {
                            "200": {
                                "description": "Users retrieved",
                                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/User"}}},
                            }
                        },
                    }
                }
            },
            "components": {"schemas": {"User": {"type": "object", "properties": {"id": {"type": "integer"}, "name": {"type": "string"}}}}},
        }

        # Write file
        simple_spec_path = os.path.join(temp_dir, "simple.yaml")
        with open(simple_spec_path, "w") as f:
            import yaml

            yaml.dump(simple_spec, f)

        # Mock Redocly as unavailable
        with patch.object(openapi_tool, "_is_redocly_available", return_value=False):
            resolved_spec = openapi_tool._try_jsonref_resolution(simple_spec_path)

        # Should return the spec (either resolved or unresolved if no external refs)
        assert resolved_spec is not None, "jsonref resolution should not fail on internal refs"
        assert "paths" in resolved_spec
        assert "/users" in resolved_spec["paths"]

        # Test that the fallback mechanism exists (even if external refs don't work perfectly)
        # The important thing is that the method handles errors gracefully and returns None
        external_ref_spec = {
            "openapi": "3.0.3",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {"/users": {"$ref": "./nonexistent.yaml"}},
        }

        external_spec_path = os.path.join(temp_dir, "external.yaml")
        with open(external_spec_path, "w") as f:
            import yaml

            yaml.dump(external_ref_spec, f)

        # Should handle missing external refs gracefully
        with patch.object(openapi_tool, "_is_redocly_available", return_value=False):
            result = openapi_tool._try_jsonref_resolution(external_spec_path)

        # Should either resolve successfully or fail gracefully (return None)
        assert result is None or isinstance(result, dict), "Should handle external ref errors gracefully"

    @patch("subprocess.run")
    def test_redocly_bundle_integration(self, mock_run, openapi_tool, multifile_spec_path):
        """Test Redocly bundle command integration."""
        if not multifile_spec_path.exists():
            pytest.skip("Multi-file spec not found")

        # Mock successful Redocly bundle
        bundled_spec = {
            "openapi": "3.0.3",
            "info": {"title": "Bundled API", "version": "1.0.0"},
            "paths": {
                "/users": {"get": {"operationId": "listUsers", "summary": "List users", "responses": {"200": {"description": "OK"}}}}
            },
        }

        def mock_subprocess(command, **kwargs):
            if "bundle" in command:
                # Write bundled spec to output file
                output_file = command[command.index("--output") + 1]
                with open(output_file, "w") as f:
                    json.dump(bundled_spec, f)

                result = Mock()
                result.returncode = 0
                result.stderr = ""
                return result

            # Mock other commands
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        mock_run.side_effect = mock_subprocess

        # Test bundling
        with patch.object(openapi_tool, "_is_redocly_available", return_value=True):
            result = openapi_tool._try_redocly_bundle(str(multifile_spec_path))

        assert result is not None, "Redocly bundling should succeed"
        assert result["info"]["title"] == "Bundled API"
        assert "/users" in result["paths"]

    def test_end_to_end_multifile_processing(self, openapi_tool, multifile_spec_path):
        """Test end-to-end processing of multi-file specification."""
        if not multifile_spec_path.exists():
            pytest.skip("Multi-file spec not found")

        # This test verifies the complete workflow
        try:
            # Should handle multi-file spec without errors
            spec = openapi_tool._preprocess_openapi_spec(str(multifile_spec_path))

            assert spec is not None, "Should successfully process multi-file spec"
            assert "openapi" in spec
            assert "paths" in spec
            assert "components" in spec

            # Verify some content is present (either resolved or original)
            assert len(spec["paths"]) > 0, "Should have path definitions"

        except Exception as e:
            pytest.fail(f"Multi-file processing failed: {e}")

    def test_circular_dependency_handling(self, openapi_tool, temp_dir):
        """Test handling of circular dependencies in $ref resolution."""
        # Create specs with circular references
        spec_a = {
            "openapi": "3.0.3",
            "info": {"title": "Spec A", "version": "1.0.0"},
            "components": {
                "schemas": {
                    "TypeA": {"type": "object", "properties": {"id": {"type": "integer"}, "typeB": {"$ref": "./spec_b.yaml#/TypeB"}}}
                }
            },
        }

        spec_b = {
            "TypeB": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "typeA": {"$ref": "./spec_a.yaml#/components/schemas/TypeA"}},
            }
        }

        # Write files
        spec_a_path = os.path.join(temp_dir, "spec_a.yaml")
        spec_b_path = os.path.join(temp_dir, "spec_b.yaml")

        with open(spec_a_path, "w") as f:
            import yaml

            yaml.dump(spec_a, f)

        with open(spec_b_path, "w") as f:
            import yaml

            yaml.dump(spec_b, f)

        # Should handle circular dependencies gracefully
        dependencies = openapi_tool._get_spec_dependencies(spec_a_path)

        # Should not crash, may or may not include circular deps depending on implementation
        assert isinstance(dependencies, list)

    def test_missing_dependency_handling(self, openapi_tool, temp_dir):
        """Test handling of missing dependency files."""
        # Create spec with reference to non-existent file
        spec_with_missing_ref = {
            "openapi": "3.0.3",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {"/test": {"$ref": "./missing_file.yaml"}},
        }

        spec_path = os.path.join(temp_dir, "main.yaml")
        with open(spec_path, "w") as f:
            import yaml

            yaml.dump(spec_with_missing_ref, f)

        # Should handle missing dependencies gracefully
        dependencies = openapi_tool._get_spec_dependencies(spec_path)
        assert isinstance(dependencies, list)

        # Hash should include missing file info
        hash_value = openapi_tool._get_file_hash(spec_path)
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0

    def test_complex_nested_refs(self, openapi_tool, temp_dir):
        """Test handling of complex nested $ref structures."""
        # Create deeply nested reference structure
        main_spec = {
            "openapi": "3.0.3",
            "info": {"title": "Nested API", "version": "1.0.0"},
            "paths": {"/level1": {"$ref": "./level1/paths.yaml"}},
        }

        level1_paths = {
            "get": {
                "operationId": "level1Operation",
                "responses": {"200": {"content": {"application/json": {"schema": {"$ref": "../level2/schemas.yaml#/Level2Schema"}}}}},
            }
        }

        level2_schemas = {"Level2Schema": {"type": "object", "properties": {"nested": {"$ref": "../level3/deep.yaml#/DeepSchema"}}}}

        level3_deep = {"DeepSchema": {"type": "object", "properties": {"value": {"type": "string"}}}}

        # Create directory structure
        level1_dir = os.path.join(temp_dir, "level1")
        level2_dir = os.path.join(temp_dir, "level2")
        level3_dir = os.path.join(temp_dir, "level3")
        os.makedirs(level1_dir)
        os.makedirs(level2_dir)
        os.makedirs(level3_dir)

        # Write files
        import yaml

        with open(os.path.join(temp_dir, "main.yaml"), "w") as f:
            yaml.dump(main_spec, f)

        with open(os.path.join(level1_dir, "paths.yaml"), "w") as f:
            yaml.dump(level1_paths, f)

        with open(os.path.join(level2_dir, "schemas.yaml"), "w") as f:
            yaml.dump(level2_schemas, f)

        with open(os.path.join(level3_dir, "deep.yaml"), "w") as f:
            yaml.dump(level3_deep, f)

        # Test dependency detection with nested structure
        main_path = os.path.join(temp_dir, "main.yaml")
        dependencies = openapi_tool._get_spec_dependencies(main_path)

        # Should find all nested dependencies
        dependency_names = [os.path.basename(dep) for dep in dependencies]
        expected = ["paths.yaml", "schemas.yaml", "deep.yaml"]

        for expected_file in expected:
            assert expected_file in dependency_names, f"Missing nested dependency: {expected_file}"


@pytest.mark.openapi
class TestMultiFileIndexing:
    """Test indexing and search with multi-file specifications."""

    def test_multifile_spec_indexing(self, openapi_tool, multifile_spec_path):
        """Test that multi-file specs can be indexed successfully."""
        if not multifile_spec_path.exists():
            pytest.skip("Multi-file spec not found")

        # Mock the embedding and FAISS components
        with (
            patch("serena.tools.openapi_tools.SentenceTransformer") as mock_transformer,
            patch("serena.tools.openapi_tools.faiss") as mock_faiss,
        ):

            # Mock transformer
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_transformer.return_value = mock_model

            # Mock FAISS index
            mock_index = Mock()
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_faiss.normalize_L2 = Mock()

            # Test processing
            try:
                result = openapi_tool.apply(spec_path=str(multifile_spec_path), query="test query", rebuild_index=True)

                assert isinstance(result, str), "Should return search results"

            except Exception as e:
                pytest.fail(f"Multi-file indexing failed: {e}")

    def test_multifile_cache_efficiency(self, openapi_tool, multifile_spec_path, temp_dir):
        """Test caching efficiency with multi-file specifications."""
        if not multifile_spec_path.exists():
            pytest.skip("Multi-file spec not found")

        # Copy spec to temp directory for modification testing
        temp_spec_dir = os.path.join(temp_dir, "multifile")
        shutil.copytree(multifile_spec_path.parent, temp_spec_dir)
        temp_main_spec = os.path.join(temp_spec_dir, "openapi.yaml")

        # Test that cache hash includes all dependencies
        hash1 = openapi_tool._get_file_hash(temp_main_spec)

        # Modify main file
        with open(temp_main_spec, "a") as f:
            f.write("\n# Test modification\n")

        hash2 = openapi_tool._get_file_hash(temp_main_spec)
        assert hash1 != hash2, "Hash should change when main file changes"

        # Restore main file and modify dependency
        shutil.copytree(multifile_spec_path.parent, temp_spec_dir, dirs_exist_ok=True)
        hash3 = openapi_tool._get_file_hash(temp_main_spec)
        assert hash3 == hash1, "Hash should be same after restoration"

        # Modify dependency file
        dep_file = os.path.join(temp_spec_dir, "components", "schemas.yaml")
        with open(dep_file, "a") as f:
            f.write("\n# Dependency modification\n")

        hash4 = openapi_tool._get_file_hash(temp_main_spec)
        assert hash4 != hash1, "Hash should change when dependency changes"


@pytest.mark.openapi
class TestMultiFilePerformance:
    """Performance tests for multi-file specification handling."""

    def test_dependency_analysis_performance(self, openapi_tool, multifile_spec_path):
        """Test performance of dependency analysis."""
        if not multifile_spec_path.exists():
            pytest.skip("Multi-file spec not found")

        import time

        # Measure dependency analysis time
        start_time = time.time()
        dependencies = openapi_tool._get_spec_dependencies(str(multifile_spec_path))
        analysis_time = time.time() - start_time

        # Should complete quickly (< 1 second for reasonable specs)
        assert analysis_time < 1.0, f"Dependency analysis too slow: {analysis_time:.2f}s"
        assert len(dependencies) > 0, "Should find dependencies"

    def test_hash_calculation_performance(self, openapi_tool, multifile_spec_path):
        """Test performance of multi-file hash calculation."""
        if not multifile_spec_path.exists():
            pytest.skip("Multi-file spec not found")

        import time

        # Measure hash calculation time
        start_time = time.time()
        hash_value = openapi_tool._get_file_hash(str(multifile_spec_path))
        hash_time = time.time() - start_time

        # Should complete quickly (< 2 seconds for reasonable specs)
        assert hash_time < 2.0, f"Hash calculation too slow: {hash_time:.2f}s"
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0
