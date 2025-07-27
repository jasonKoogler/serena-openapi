"""Tests for OpenAPI index building functionality."""

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
def sample_openapi_spec():
    """Return a simple OpenAPI specification for testing."""
    return {
        "openapi": "3.0.3",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/users": {
                "get": {
                    "operationId": "listUsers",
                    "summary": "List all users",
                    "description": "Retrieve a list of all registered users",
                    "tags": ["users", "admin"],
                    "parameters": [
                        {"name": "limit", "in": "query", "description": "Maximum number of users to return", "schema": {"type": "integer"}}
                    ],
                    "responses": {
                        "200": {
                            "description": "List of users",
                            "content": {"application/json": {"schema": {"type": "array", "items": {"$ref": "#/components/schemas/User"}}}},
                        }
                    },
                },
                "post": {
                    "operationId": "createUser",
                    "summary": "Create new user",
                    "description": "Register a new user account",
                    "tags": ["users", "registration"],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/NewUser"}}},
                    },
                    "responses": {"201": {"description": "User created"}},
                },
            },
            "/users/{userId}": {
                "get": {
                    "operationId": "getUserById",
                    "summary": "Get user by ID",
                    "description": "Retrieve a specific user by their ID",
                    "tags": ["users"],
                    "parameters": [{"name": "userId", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "User details"}, "404": {"description": "User not found"}},
                },
                "delete": {
                    "operationId": "deleteUser",
                    "summary": "Delete user",
                    "description": "Remove a user account",
                    "tags": ["users", "admin"],
                    "parameters": [{"name": "userId", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"204": {"description": "User deleted"}},
                },
            },
            "/posts": {
                "get": {
                    "operationId": "listPosts",
                    "summary": "List posts",
                    "description": "Get all blog posts",
                    "tags": ["posts", "content"],
                    "responses": {"200": {"description": "List of posts"}},
                }
            },
        },
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "name": {"type": "string"}, "email": {"type": "string", "format": "email"}},
                },
                "NewUser": {
                    "type": "object",
                    "required": ["name", "email"],
                    "properties": {"name": {"type": "string"}, "email": {"type": "string", "format": "email"}},
                },
            }
        },
    }


class TestOpenApiIndexing:
    """Test cases for OpenAPI index building."""

    def test_create_semantic_chunks(self, openapi_tool, sample_openapi_spec):
        """Test that semantic chunks are created correctly from OpenAPI spec."""
        chunks = openapi_tool._create_semantic_chunks(sample_openapi_spec)

        # Should create one chunk per operation
        assert len(chunks) == 5  # 2 for /users, 2 for /users/{userId}, 1 for /posts

        # Check first chunk (GET /users)
        first_chunk = chunks[0]
        assert "text" in first_chunk
        assert "metadata" in first_chunk

        metadata = first_chunk["metadata"]
        assert metadata["operationId"] == "listUsers"
        assert metadata["method"] == "GET"
        assert metadata["path"] == "/users"
        assert metadata["summary"] == "List all users"
        assert "users" in metadata["tags"]
        assert "admin" in metadata["tags"]

        # Check that text contains relevant information
        text = first_chunk["text"]
        assert "listUsers" in text
        assert "GET" in text
        assert "/users" in text
        assert "List all users" in text
        assert "users, admin" in text
        assert "limit" in text  # Parameter should be included

    def test_create_semantic_chunks_with_request_body(self, openapi_tool, sample_openapi_spec):
        """Test that chunks include request body information."""
        chunks = openapi_tool._create_semantic_chunks(sample_openapi_spec)

        # Find the POST /users chunk
        post_chunk = next(chunk for chunk in chunks if chunk["metadata"]["method"] == "POST")

        text = post_chunk["text"]
        assert "application/json" in text
        assert "Request Content Types" in text

    def test_create_semantic_chunks_with_responses(self, openapi_tool, sample_openapi_spec):
        """Test that chunks include response information."""
        chunks = openapi_tool._create_semantic_chunks(sample_openapi_spec)

        # Find a chunk with multiple response codes
        get_user_chunk = next(chunk for chunk in chunks if chunk["metadata"]["operationId"] == "getUserById")

        text = get_user_chunk["text"]
        assert "Response Codes" in text
        assert "200, 404" in text

    def test_file_hash_generation(self, openapi_tool, temp_dir):
        """Test that file hash is generated consistently."""
        spec_path = os.path.join(temp_dir, "test.yaml")
        content = "test content"

        with open(spec_path, "w") as f:
            f.write(content)

        hash1 = openapi_tool._get_file_hash(spec_path)
        hash2 = openapi_tool._get_file_hash(spec_path)

        # Same file should produce same hash
        assert hash1 == hash2

        # Modify file
        with open(spec_path, "a") as f:
            f.write("\\nmodified")

        hash3 = openapi_tool._get_file_hash(spec_path)

        # Modified file should produce different hash
        assert hash1 != hash3

    def test_index_validation(self, openapi_tool, temp_dir, sample_openapi_spec):
        """Test index validation logic."""
        spec_path = os.path.join(temp_dir, "test.yaml")
        with open(spec_path, "w") as f:
            json.dump(sample_openapi_spec, f)

        index_dir = os.path.join(temp_dir, "index")

        # Non-existent index should be invalid
        assert not openapi_tool._is_index_valid(index_dir, spec_path)

        # Create partial index (missing files)
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, "index.faiss"), "w") as f:
            f.write("dummy")

        assert not openapi_tool._is_index_valid(index_dir, spec_path)

        # Create complete but outdated index
        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump([], f)

        metadata = {"spec_path": spec_path, "num_chunks": 0, "embedding_model": "test-model", "created_at": 0}  # Very old timestamp
        with open(os.path.join(index_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        assert not openapi_tool._is_index_valid(index_dir, spec_path)

    @patch("serena.tools.openapi_tools.SentenceTransformer")
    @patch("serena.tools.openapi_tools.faiss")
    def test_build_index_success(self, mock_faiss, mock_transformer, openapi_tool, temp_dir, sample_openapi_spec):
        """Test successful index building."""
        # Setup mocks
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Mock embeddings
        mock_transformer.return_value = mock_model

        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        mock_faiss.write_index = Mock()

        # Create spec file
        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(sample_openapi_spec, f)

        index_dir = os.path.join(temp_dir, "index")

        # Override the model property to return our mock
        openapi_tool._model = mock_model

        # Build index
        openapi_tool._build_index(spec_path, index_dir)

        # Verify index directory was created
        assert os.path.exists(index_dir)

        # Verify chunks.json was created
        chunks_path = os.path.join(index_dir, "chunks.json")
        assert os.path.exists(chunks_path)

        with open(chunks_path) as f:
            chunks = json.load(f)
        assert len(chunks) == 5  # Should have 5 operations

        # Verify metadata.json was created
        metadata_path = os.path.join(index_dir, "metadata.json")
        assert os.path.exists(metadata_path)

        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["spec_path"] == spec_path
        assert metadata["num_chunks"] == 5

        # Verify FAISS operations were called
        mock_faiss.IndexFlatIP.assert_called_once()
        mock_index.add.assert_called_once()
        mock_faiss.write_index.assert_called_once()

    def test_build_index_empty_spec(self, openapi_tool, temp_dir):
        """Test building index with empty OpenAPI spec."""
        empty_spec = {"openapi": "3.0.3", "info": {"title": "Empty API", "version": "1.0.0"}, "paths": {}}

        spec_path = os.path.join(temp_dir, "empty.json")
        with open(spec_path, "w") as f:
            json.dump(empty_spec, f)

        index_dir = os.path.join(temp_dir, "index")

        # Should raise ValueError for empty spec
        with pytest.raises(ValueError, match="No API endpoints found"):
            openapi_tool._build_index(spec_path, index_dir)

    def test_build_index_invalid_json(self, openapi_tool, temp_dir):
        """Test building index with invalid JSON file."""
        spec_path = os.path.join(temp_dir, "invalid.json")
        with open(spec_path, "w") as f:
            f.write("invalid json content")

        index_dir = os.path.join(temp_dir, "index")

        # Should raise exception for invalid JSON
        with pytest.raises((json.JSONDecodeError, ValueError)):
            openapi_tool._build_index(spec_path, index_dir)

    @patch("serena.tools.openapi_tools.yaml")
    def test_yaml_file_processing(self, mock_yaml, openapi_tool, temp_dir, sample_openapi_spec):
        """Test processing YAML OpenAPI files."""
        mock_yaml.safe_load.return_value = sample_openapi_spec

        spec_path = os.path.join(temp_dir, "test.yaml")
        with open(spec_path, "w") as f:
            f.write("dummy yaml content")

        result = openapi_tool._preprocess_openapi_spec(spec_path)

        # Should call yaml.safe_load
        mock_yaml.safe_load.assert_called_once()
        assert result == sample_openapi_spec

    def test_yaml_file_without_pyyaml(self, openapi_tool, temp_dir):
        """Test processing YAML file when PyYAML is not available."""
        spec_path = os.path.join(temp_dir, "test.yaml")
        with open(spec_path, "w") as f:
            f.write("dummy yaml content")

        # Mock ImportError for yaml
        with patch("serena.tools.openapi_tools.yaml", side_effect=ImportError):
            with pytest.raises(ValueError, match="PyYAML is required"):
                openapi_tool._preprocess_openapi_spec(spec_path)

    @patch("subprocess.run")
    def test_redocly_preprocessing_success(self, mock_run, openapi_tool, temp_dir, sample_openapi_spec):
        """Test successful Redocly preprocessing."""
        # Enable Redocly in config
        openapi_tool.agent.serena_config.openapi.use_redocly_validation = True

        # Mock Redocly availability check
        mock_run.side_effect = [
            Mock(returncode=0),  # redocly --version
            Mock(returncode=0),  # redocly bundle
            Mock(returncode=0),  # redocly lint
        ]

        spec_path = os.path.join(temp_dir, "test.yaml")
        with open(spec_path, "w") as f:
            f.write("dummy content")

        # Mock the bundled file creation
        def side_effect(*args, **kwargs):
            if args[0][0] == "redocly" and args[0][1] == "bundle":
                output_path = args[0][4]  # --output path
                with open(output_path, "w") as f:
                    json.dump(sample_openapi_spec, f)
            return Mock(returncode=0)

        mock_run.side_effect = side_effect

        result = openapi_tool._preprocess_openapi_spec(spec_path)

        assert result == sample_openapi_spec
        assert mock_run.call_count >= 2  # At least bundle and lint commands

    @patch("subprocess.run")
    def test_redocly_preprocessing_failure_fallback(self, mock_run, openapi_tool, temp_dir, sample_openapi_spec):
        """Test fallback when Redocly preprocessing fails."""
        # Enable Redocly in config
        openapi_tool.agent.serena_config.openapi.use_redocly_validation = True

        # Mock Redocly failure
        mock_run.side_effect = Exception("Redocly error")

        spec_path = os.path.join(temp_dir, "test.json")
        with open(spec_path, "w") as f:
            json.dump(sample_openapi_spec, f)

        result = openapi_tool._preprocess_openapi_spec(spec_path)

        # Should fall back to direct parsing
        assert result == sample_openapi_spec

    def test_auto_discover_specs(self, openapi_tool, temp_dir):
        """Test auto-discovery of OpenAPI specifications."""
        # Create various spec files
        specs = [
            ("openapi.yaml", {"openapi": "3.0.3", "info": {"title": "API 1", "version": "1.0.0"}}),
            ("docs/swagger.json", {"openapi": "3.0.3", "info": {"title": "API 2", "version": "1.0.0"}}),
            ("api/openapi.yml", {"openapi": "3.0.3", "info": {"title": "API 3", "version": "1.0.0"}}),
            ("other.yaml", {"some": "other content"}),  # Not an OpenAPI spec pattern
        ]

        for spec_path, content in specs:
            full_path = os.path.join(temp_dir, spec_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                json.dump(content, f)

        discovered = openapi_tool._auto_discover_specs(temp_dir)

        # Should find 3 specs matching the patterns
        assert len(discovered) == 3

        # Check that all discovered files exist and match expected patterns
        discovered_names = [os.path.basename(path) for path in discovered]
        assert "openapi.yaml" in discovered_names
        assert any("swagger.json" in path for path in discovered)
        assert any("openapi.yml" in path for path in discovered)
        assert "other.yaml" not in discovered_names


@pytest.mark.openapi
class TestOpenApiIndexingIntegration:
    """Integration tests for OpenAPI indexing with real files."""

    def test_index_petstore_spec(self, openapi_tool):
        """Test indexing the petstore sample spec."""
        # Get the path to the sample petstore spec
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        petstore_path = test_resources / "petstore.yaml"

        if not petstore_path.exists():
            pytest.skip("Petstore spec not found")

        # Create temporary cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            spec_hash = openapi_tool._get_file_hash(str(petstore_path))
            index_dir = os.path.join(temp_dir, spec_hash)

            # Build index
            openapi_tool._build_index(str(petstore_path), index_dir)

            # Verify index was created successfully
            assert os.path.exists(index_dir)
            assert os.path.exists(os.path.join(index_dir, "index.faiss"))
            assert os.path.exists(os.path.join(index_dir, "chunks.json"))
            assert os.path.exists(os.path.join(index_dir, "metadata.json"))

            # Verify chunks content
            with open(os.path.join(index_dir, "chunks.json")) as f:
                chunks = json.load(f)

            assert len(chunks) > 0

            # Check that we have expected operations
            operation_ids = [chunk["metadata"]["operationId"] for chunk in chunks]
            assert "listPets" in operation_ids
            assert "createPet" in operation_ids
            assert "getPetById" in operation_ids

    def test_index_blog_spec(self, openapi_tool):
        """Test indexing the blog sample spec."""
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        blog_path = test_resources / "simple-blog.json"

        if not blog_path.exists():
            pytest.skip("Blog spec not found")

        with tempfile.TemporaryDirectory() as temp_dir:
            spec_hash = openapi_tool._get_file_hash(str(blog_path))
            index_dir = os.path.join(temp_dir, spec_hash)

            openapi_tool._build_index(str(blog_path), index_dir)

            # Verify index was created
            assert os.path.exists(os.path.join(index_dir, "chunks.json"))

            with open(os.path.join(index_dir, "chunks.json")) as f:
                chunks = json.load(f)

            # Should have operations for blog posts, authors, and comments
            operation_ids = [chunk["metadata"]["operationId"] for chunk in chunks]
            assert "listPosts" in operation_ids
            assert "createPost" in operation_ids
            assert "listAuthors" in operation_ids
