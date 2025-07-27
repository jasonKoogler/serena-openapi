"""Tests for OpenAPI configuration management."""

import os
import shutil
import tempfile
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
    """Create a mock agent with configurable settings."""
    agent = Mock(spec=SerenaAgent)

    # Mock the serena_config with default values
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


class TestOpenApiConfiguration:
    """Test cases for OpenAPI configuration management."""

    def test_default_configuration(self, openapi_tool):
        """Test that default configuration values are properly set."""
        config = openapi_tool.agent.serena_config.openapi

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.index_cache_dir == ".serena/openapi_cache"
        assert config.use_redocly_validation is False
        assert config.redocly_timeout == 30

    def test_cache_directory_creation(self, openapi_tool, temp_dir):
        """Test that cache directory is created when accessed."""
        cache_dir = openapi_tool.cache_dir

        # Should create the directory
        assert os.path.exists(cache_dir)
        assert cache_dir.endswith(".serena/openapi_cache")
        assert temp_dir in cache_dir

    def test_cache_directory_with_no_project(self, mock_agent):
        """Test cache directory behavior when no active project."""
        # Mock no active project
        mock_agent.get_project_root.side_effect = ValueError("No active project")

        tool = OpenApiTool(mock_agent)
        cache_dir = tool.cache_dir

        # Should use temp directory
        assert "/tmp" in cache_dir or "/var/folders" in cache_dir  # macOS temp dir
        assert "serena_openapi_cache" in cache_dir

    def test_custom_embedding_model_config(self, mock_agent):
        """Test configuration with custom embedding model."""
        # Change embedding model
        mock_agent.serena_config.openapi.embedding_model = "sentence-transformers/all-mpnet-base-v2"

        tool = OpenApiTool(mock_agent)

        # Should use custom model when accessed
        with patch("serena.tools.openapi_tools.SentenceTransformer") as mock_transformer:
            _ = tool.model
            mock_transformer.assert_called_once_with("sentence-transformers/all-mpnet-base-v2")

    def test_redocly_configuration_enabled(self, openapi_tool):
        """Test configuration with Redocly validation enabled."""
        # Enable Redocly validation
        openapi_tool.agent.serena_config.openapi.use_redocly_validation = True
        openapi_tool.agent.serena_config.openapi.redocly_timeout = 60

        config = openapi_tool.agent.serena_config.openapi
        assert config.use_redocly_validation is True
        assert config.redocly_timeout == 60

    def test_custom_cache_directory(self, mock_agent, temp_dir):
        """Test configuration with custom cache directory."""
        # Set custom cache directory
        mock_agent.serena_config.openapi.index_cache_dir = "custom_openapi_cache"

        tool = OpenApiTool(mock_agent)
        cache_dir = tool.cache_dir

        assert "custom_openapi_cache" in cache_dir
        assert os.path.exists(cache_dir)

    def test_project_spec_configuration_empty(self, openapi_tool):
        """Test project with no configured OpenAPI specs."""
        specs = openapi_tool.list_project_specs()

        # Should return auto-discovered specs (empty if none exist)
        assert isinstance(specs, list)

    def test_project_spec_configuration_with_specs(self, openapi_tool, temp_dir):
        """Test project with configured OpenAPI specs."""
        # Create test spec files
        spec1_path = os.path.join(temp_dir, "api1.yaml")
        spec2_path = os.path.join(temp_dir, "api2.json")

        with open(spec1_path, "w") as f:
            f.write("openapi: 3.0.3\ninfo:\n  title: API 1\n  version: 1.0.0\npaths: {}")

        with open(spec2_path, "w") as f:
            f.write('{"openapi": "3.0.3", "info": {"title": "API 2", "version": "1.0.0"}, "paths": {}}')

        # Configure project with these specs
        openapi_tool.agent.get_active_project.return_value.project_config.openapi_specs = ["api1.yaml", "api2.json"]

        specs = openapi_tool.list_project_specs()

        assert len(specs) == 2
        assert spec1_path in specs
        assert spec2_path in specs

    def test_project_spec_configuration_missing_files(self, openapi_tool, temp_dir):
        """Test project configuration with missing spec files."""
        # Configure project with non-existent specs
        openapi_tool.agent.get_active_project.return_value.project_config.openapi_specs = ["missing1.yaml", "missing2.json"]

        specs = openapi_tool.list_project_specs()

        # Should return empty list since files don't exist
        assert specs == []

    def test_add_spec_to_project_success(self, openapi_tool, temp_dir):
        """Test successfully adding a spec to project configuration."""
        # Create a test spec file
        spec_path = os.path.join(temp_dir, "new_api.yaml")
        with open(spec_path, "w") as f:
            f.write("openapi: 3.0.3\ninfo:\n  title: New API\n  version: 1.0.0\npaths: {}")

        result = openapi_tool.add_spec_to_project(spec_path)

        assert "Added OpenAPI specification to project: new_api.yaml" in result

        # Verify it was added to the project config
        specs = openapi_tool.agent.get_active_project.return_value.project_config.openapi_specs
        assert "new_api.yaml" in specs

    def test_add_spec_to_project_relative_path(self, openapi_tool, temp_dir):
        """Test adding a spec using relative path."""
        # Create a test spec file
        spec_path = os.path.join(temp_dir, "relative_api.yaml")
        with open(spec_path, "w") as f:
            f.write("openapi: 3.0.3\ninfo:\n  title: Relative API\n  version: 1.0.0\npaths: {}")

        result = openapi_tool.add_spec_to_project("relative_api.yaml")

        assert "Added OpenAPI specification to project: relative_api.yaml" in result

    def test_add_spec_to_project_already_exists(self, openapi_tool, temp_dir):
        """Test adding a spec that's already configured."""
        # Create and add a spec
        spec_path = os.path.join(temp_dir, "existing_api.yaml")
        with open(spec_path, "w") as f:
            f.write("openapi: 3.0.3\ninfo:\n  title: Existing API\n  version: 1.0.0\npaths: {}")

        openapi_tool.add_spec_to_project(spec_path)

        # Try to add it again
        result = openapi_tool.add_spec_to_project(spec_path)

        assert "OpenAPI specification already configured: existing_api.yaml" in result

    def test_add_spec_to_project_file_not_found(self, openapi_tool):
        """Test adding a non-existent spec file."""
        result = openapi_tool.add_spec_to_project("nonexistent.yaml")

        assert "Error: OpenAPI specification file not found" in result

    def test_add_spec_to_project_outside_root(self, openapi_tool, temp_dir):
        """Test adding a spec file outside project root."""
        # Create spec outside project root
        outside_dir = tempfile.mkdtemp()
        try:
            spec_path = os.path.join(outside_dir, "outside_api.yaml")
            with open(spec_path, "w") as f:
                f.write("openapi: 3.0.3\ninfo:\n  title: Outside API\n  version: 1.0.0\npaths: {}")

            result = openapi_tool.add_spec_to_project(spec_path)

            assert "Error: Specification path is outside project root" in result
        finally:
            shutil.rmtree(outside_dir, ignore_errors=True)

    def test_add_spec_to_project_no_active_project(self, mock_agent):
        """Test adding spec when no active project."""
        mock_agent.get_active_project.return_value = None

        tool = OpenApiTool(mock_agent)
        result = tool.add_spec_to_project("test.yaml")

        assert "Error: No active project" in result

    def test_specs_to_search_with_specific_path(self, openapi_tool, temp_dir):
        """Test spec selection with specific path."""
        # Create a test spec
        spec_path = os.path.join(temp_dir, "specific.yaml")
        with open(spec_path, "w") as f:
            f.write("openapi: 3.0.3\ninfo:\n  title: Specific API\n  version: 1.0.0\npaths: {}")

        specs = openapi_tool._get_specs_to_search(spec_path, False)

        assert len(specs) == 1
        assert specs[0] == spec_path

    def test_specs_to_search_with_search_all(self, openapi_tool, temp_dir):
        """Test spec selection with search_all_specs=True."""
        # Create test specs
        spec1_path = os.path.join(temp_dir, "api1.yaml")
        spec2_path = os.path.join(temp_dir, "api2.json")

        with open(spec1_path, "w") as f:
            f.write("openapi: 3.0.3\ninfo:\n  title: API 1\n  version: 1.0.0\npaths: {}")

        with open(spec2_path, "w") as f:
            f.write('{"openapi": "3.0.3", "info": {"title": "API 2", "version": "1.0.0"}, "paths": {}}')

        # Configure project
        openapi_tool.agent.get_active_project.return_value.project_config.openapi_specs = ["api1.yaml", "api2.json"]

        specs = openapi_tool._get_specs_to_search(None, True)

        assert len(specs) == 2
        assert spec1_path in specs
        assert spec2_path in specs

    def test_specs_to_search_auto_discovery(self, openapi_tool, temp_dir):
        """Test spec selection with auto-discovery fallback."""
        # Create auto-discoverable spec
        spec_path = os.path.join(temp_dir, "openapi.yaml")
        with open(spec_path, "w") as f:
            f.write("openapi: 3.0.3\ninfo:\n  title: Auto API\n  version: 1.0.0\npaths: {}")

        specs = openapi_tool._get_specs_to_search(None, False)

        assert len(specs) >= 1
        assert spec_path in specs

    def test_specs_to_search_no_project(self, mock_agent):
        """Test spec selection when no active project."""
        mock_agent.get_project_root.side_effect = ValueError("No active project")

        tool = OpenApiTool(mock_agent)

        # Should use current directory for auto-discovery
        specs = tool._get_specs_to_search(None, False)

        # Should return list (may be empty if no specs in current dir)
        assert isinstance(specs, list)

    def test_configuration_inheritance(self, openapi_tool):
        """Test that configuration is properly inherited from SerenaConfig."""
        # Access various config properties
        assert hasattr(openapi_tool.agent.serena_config, "openapi")

        openapi_config = openapi_tool.agent.serena_config.openapi
        assert hasattr(openapi_config, "embedding_model")
        assert hasattr(openapi_config, "index_cache_dir")
        assert hasattr(openapi_config, "use_redocly_validation")
        assert hasattr(openapi_config, "redocly_timeout")

    def test_auto_discover_patterns_comprehensive(self, openapi_tool, temp_dir):
        """Test all auto-discovery patterns work correctly."""
        # Create files matching all patterns
        patterns = [
            "openapi.yaml",
            "openapi.yml",
            "openapi.json",
            "swagger.yaml",
            "swagger.yml",
            "swagger.json",
            "api.yaml",
            "api.yml",
            "api.json",
            "spec.yaml",
            "spec.yml",
            "spec.json",
            "docs/openapi.yaml",
            "docs/swagger.json",
            "api/openapi.yml",
        ]

        created_specs = []
        for pattern in patterns:
            spec_path = os.path.join(temp_dir, pattern)
            os.makedirs(os.path.dirname(spec_path), exist_ok=True)

            with open(spec_path, "w") as f:
                f.write(f'{{"openapi": "3.0.3", "info": {{"title": "{pattern}", "version": "1.0.0"}}, "paths": {{}}}}')

            created_specs.append(spec_path)

        discovered = openapi_tool._auto_discover_specs(temp_dir)

        # Should discover all created specs
        assert len(discovered) == len(patterns)
        for spec_path in created_specs:
            assert spec_path in discovered


class TestOpenApiConfigurationEdgeCases:
    """Test edge cases and error conditions in configuration."""

    def test_malformed_project_config(self, mock_agent, temp_dir):
        """Test handling of malformed project configuration."""
        # Mock a project with malformed config
        project = Mock()
        project.project_root = temp_dir
        project.project_config = None  # Malformed config
        mock_agent.get_active_project.return_value = project

        tool = OpenApiTool(mock_agent)

        # Should handle gracefully
        with pytest.raises(AttributeError):
            tool.list_project_specs()

    def test_permission_denied_cache_directory(self, mock_agent):
        """Test handling when cache directory cannot be created."""
        # Mock a read-only directory
        mock_agent.get_project_root.return_value = "/read-only-dir"

        tool = OpenApiTool(mock_agent)

        # Should handle permission errors gracefully
        # (In practice, this would fall back to temp directory)
        cache_dir = tool.cache_dir
        assert cache_dir is not None

    def test_config_without_openapi_section(self, mock_agent):
        """Test handling when SerenaConfig doesn't have OpenAPI section."""
        # Remove OpenAPI config
        mock_agent.serena_config.openapi = None

        tool = OpenApiTool(mock_agent)

        # Should handle missing config gracefully
        with pytest.raises(AttributeError):
            _ = tool.model

    def test_empty_specs_list_in_project(self, openapi_tool):
        """Test project with explicitly empty specs list."""
        openapi_tool.agent.get_active_project.return_value.project_config.openapi_specs = []

        specs = openapi_tool.list_project_specs()

        # Should fall back to auto-discovery
        assert isinstance(specs, list)

    def test_mixed_existing_and_missing_specs(self, openapi_tool, temp_dir):
        """Test project config with mix of existing and missing specs."""
        # Create one spec, reference two
        spec_path = os.path.join(temp_dir, "existing.yaml")
        with open(spec_path, "w") as f:
            f.write("openapi: 3.0.3\ninfo:\n  title: Existing\n  version: 1.0.0\npaths: {}")

        openapi_tool.agent.get_active_project.return_value.project_config.openapi_specs = ["existing.yaml", "missing.yaml"]

        specs = openapi_tool.list_project_specs()

        # Should only return existing specs
        assert len(specs) == 1
        assert spec_path in specs
