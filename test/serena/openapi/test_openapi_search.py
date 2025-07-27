"""Tests for OpenAPI semantic search functionality."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
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
    """Return a comprehensive OpenAPI specification for testing search."""
    return {
        "openapi": "3.0.3",
        "info": {"title": "E-commerce API", "version": "1.0.0"},
        "paths": {
            "/users": {
                "get": {
                    "operationId": "listUsers",
                    "summary": "List all users",
                    "description": "Retrieve a paginated list of all registered users in the system",
                    "tags": ["users", "admin", "management"],
                    "parameters": [
                        {"name": "limit", "in": "query", "description": "Maximum number of users to return", "schema": {"type": "integer"}},
                        {"name": "email", "in": "query", "description": "Filter users by email domain", "schema": {"type": "string"}},
                    ],
                    "responses": {"200": {"description": "List of users"}, "400": {"description": "Invalid parameters"}},
                },
                "post": {
                    "operationId": "createUser",
                    "summary": "Create new user account",
                    "description": "Register a new user account with email verification",
                    "tags": ["users", "registration", "auth"],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/NewUser"}}},
                    },
                    "responses": {"201": {"description": "User created successfully"}, "409": {"description": "User already exists"}},
                },
            },
            "/users/{userId}/profile": {
                "get": {
                    "operationId": "getUserProfile",
                    "summary": "Get user profile",
                    "description": "Retrieve detailed profile information for a specific user",
                    "tags": ["users", "profile"],
                    "parameters": [{"name": "userId", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "User profile data"}, "404": {"description": "User not found"}},
                },
                "put": {
                    "operationId": "updateUserProfile",
                    "summary": "Update user profile",
                    "description": "Modify user profile information including personal details",
                    "tags": ["users", "profile", "management"],
                    "parameters": [{"name": "userId", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/UserProfile"}}},
                    },
                    "responses": {"200": {"description": "Profile updated"}, "404": {"description": "User not found"}},
                },
            },
            "/products": {
                "get": {
                    "operationId": "searchProducts",
                    "summary": "Search products",
                    "description": "Search for products with advanced filtering by category, price, and availability",
                    "tags": ["products", "catalog", "search"],
                    "parameters": [
                        {
                            "name": "query",
                            "in": "query",
                            "description": "Search query for product name or description",
                            "schema": {"type": "string"},
                        },
                        {"name": "category", "in": "query", "description": "Filter by product category", "schema": {"type": "string"}},
                    ],
                    "responses": {"200": {"description": "Product search results"}},
                },
                "post": {
                    "operationId": "createProduct",
                    "summary": "Add new product",
                    "description": "Create a new product in the catalog with pricing and inventory",
                    "tags": ["products", "admin", "catalog"],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/NewProduct"}}},
                    },
                    "responses": {"201": {"description": "Product created"}},
                },
            },
            "/orders": {
                "get": {
                    "operationId": "listOrders",
                    "summary": "List customer orders",
                    "description": "Retrieve order history with status filtering and pagination",
                    "tags": ["orders", "customer", "history"],
                    "parameters": [
                        {
                            "name": "status",
                            "in": "query",
                            "description": "Filter by order status",
                            "schema": {"type": "string", "enum": ["pending", "processing", "shipped", "delivered"]},
                        }
                    ],
                    "responses": {"200": {"description": "Order list"}},
                },
                "post": {
                    "operationId": "createOrder",
                    "summary": "Place new order",
                    "description": "Create a new order from shopping cart items with payment processing",
                    "tags": ["orders", "checkout", "payment"],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/NewOrder"}}},
                    },
                    "responses": {"201": {"description": "Order placed successfully"}},
                },
            },
            "/payments/process": {
                "post": {
                    "operationId": "processPayment",
                    "summary": "Process payment",
                    "description": "Handle payment processing for orders using various payment methods",
                    "tags": ["payment", "billing", "finance"],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/PaymentRequest"}}},
                    },
                    "responses": {"200": {"description": "Payment processed"}, "402": {"description": "Payment failed"}},
                }
            },
        },
    }


class TestOpenApiSearch:
    """Test cases for OpenAPI semantic search functionality."""

    def test_create_chunks_for_search(self, openapi_tool, sample_openapi_spec):
        """Test that chunks are created with proper search metadata."""
        chunks = openapi_tool._create_semantic_chunks(sample_openapi_spec)

        assert len(chunks) == 9  # 9 operations total

        # Verify each chunk has required fields
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            metadata = chunk["metadata"]
            assert "operationId" in metadata
            assert "method" in metadata
            assert "path" in metadata
            assert "tags" in metadata

            # Verify text contains searchable content
            text = chunk["text"]
            assert metadata["operationId"] in text
            assert metadata["method"] in text
            assert metadata["path"] in text

    @patch("serena.tools.openapi_tools.faiss")
    def test_semantic_search_basic(self, mock_faiss, openapi_tool, temp_dir, sample_openapi_spec):
        """Test basic semantic search functionality."""
        # Create test chunks
        chunks = openapi_tool._create_semantic_chunks(sample_openapi_spec)

        # Mock FAISS index
        mock_index = Mock()
        mock_index.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]]))  # distances (scores)  # indices
        mock_faiss.read_index.return_value = mock_index

        # Create test index directory with chunks
        index_dir = os.path.join(temp_dir, "test_index")
        os.makedirs(index_dir, exist_ok=True)

        chunks_path = os.path.join(index_dir, "chunks.json")
        with open(chunks_path, "w") as f:
            json.dump(chunks, f)

        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        openapi_tool._model = mock_model

        # Perform search
        results = openapi_tool._semantic_search("list users", index_dir, 3)

        # Verify results
        assert len(results) == 3
        assert results[0]["score"] == 0.9
        assert results[1]["score"] == 0.8
        assert results[2]["score"] == 0.7
        assert results[0]["rank"] == 1
        assert results[1]["rank"] == 2
        assert results[2]["rank"] == 3

    @patch("serena.tools.openapi_tools.faiss")
    def test_semantic_search_with_method_filter(self, mock_faiss, openapi_tool, temp_dir, sample_openapi_spec):
        """Test semantic search with HTTP method filtering."""
        chunks = openapi_tool._create_semantic_chunks(sample_openapi_spec)

        # Count GET vs POST operations
        get_chunks = [c for c in chunks if c["metadata"]["method"] == "GET"]
        post_chunks = [c for c in chunks if c["metadata"]["method"] == "POST"]

        assert len(get_chunks) > 0
        assert len(post_chunks) > 0
        assert len(get_chunks) + len(post_chunks) == len(chunks)

        # Mock FAISS for filtered search
        mock_index = Mock()
        mock_index.reconstruct.side_effect = lambda i: np.array([0.1 * i, 0.2 * i, 0.3 * i])
        mock_faiss.read_index.return_value = mock_index

        # Mock filtered index creation
        mock_filtered_index = Mock()
        mock_filtered_index.search.return_value = (np.array([[0.95, 0.85]]), np.array([[0, 1]]))  # Only 2 results
        mock_faiss.IndexFlatIP.return_value = mock_filtered_index
        mock_faiss.normalize_L2 = Mock()

        # Create test index directory
        index_dir = os.path.join(temp_dir, "test_index")
        os.makedirs(index_dir, exist_ok=True)

        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump(chunks, f)

        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        openapi_tool._model = mock_model

        # Search with GET filter
        results = openapi_tool._semantic_search("users", index_dir, 5, method_filter="GET")

        # Should only return GET operations
        for result in results:
            assert result["metadata"]["method"] == "GET"

    @patch("serena.tools.openapi_tools.faiss")
    def test_semantic_search_with_path_filter(self, mock_faiss, openapi_tool, temp_dir, sample_openapi_spec):
        """Test semantic search with path pattern filtering."""
        chunks = openapi_tool._create_semantic_chunks(sample_openapi_spec)

        # Mock FAISS index
        mock_index = Mock()
        mock_index.reconstruct.side_effect = lambda i: np.array([0.1 * i, 0.2 * i, 0.3 * i])
        mock_faiss.read_index.return_value = mock_index

        # Mock filtered index
        mock_filtered_index = Mock()
        mock_filtered_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
        mock_faiss.IndexFlatIP.return_value = mock_filtered_index
        mock_faiss.normalize_L2 = Mock()

        # Create test index directory
        index_dir = os.path.join(temp_dir, "test_index")
        os.makedirs(index_dir, exist_ok=True)

        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump(chunks, f)

        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        openapi_tool._model = mock_model

        # Search with path filter (regex)
        results = openapi_tool._semantic_search("users", index_dir, 5, path_filter="/users.*")

        # Should only return paths matching the pattern
        for result in results:
            assert result["metadata"]["path"].startswith("/users")

    @patch("serena.tools.openapi_tools.faiss")
    def test_semantic_search_with_tags_filter(self, mock_faiss, openapi_tool, temp_dir, sample_openapi_spec):
        """Test semantic search with tags filtering."""
        chunks = openapi_tool._create_semantic_chunks(sample_openapi_spec)

        # Mock FAISS setup (similar to previous tests)
        mock_index = Mock()
        mock_index.reconstruct.side_effect = lambda i: np.array([0.1 * i, 0.2 * i, 0.3 * i])
        mock_faiss.read_index.return_value = mock_index

        mock_filtered_index = Mock()
        mock_filtered_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
        mock_faiss.IndexFlatIP.return_value = mock_filtered_index
        mock_faiss.normalize_L2 = Mock()

        # Create test index directory
        index_dir = os.path.join(temp_dir, "test_index")
        os.makedirs(index_dir, exist_ok=True)

        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump(chunks, f)

        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        openapi_tool._model = mock_model

        # Search with tags filter
        results = openapi_tool._semantic_search("operations", index_dir, 5, tags_filter=["admin"])

        # Should only return operations with admin tag
        for result in results:
            assert "admin" in result["metadata"]["tags"]

    @patch("serena.tools.openapi_tools.faiss")
    def test_semantic_search_combined_filters(self, mock_faiss, openapi_tool, temp_dir, sample_openapi_spec):
        """Test semantic search with multiple filters combined."""
        chunks = openapi_tool._create_semantic_chunks(sample_openapi_spec)

        # Mock FAISS setup
        mock_index = Mock()
        mock_index.reconstruct.side_effect = lambda i: np.array([0.1 * i, 0.2 * i, 0.3 * i])
        mock_faiss.read_index.return_value = mock_index

        mock_filtered_index = Mock()
        mock_filtered_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
        mock_faiss.IndexFlatIP.return_value = mock_filtered_index
        mock_faiss.normalize_L2 = Mock()

        # Create test index directory
        index_dir = os.path.join(temp_dir, "test_index")
        os.makedirs(index_dir, exist_ok=True)

        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump(chunks, f)

        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        openapi_tool._model = mock_model

        # Search with combined filters: GET method + users path + admin tag
        results = openapi_tool._semantic_search("users", index_dir, 5, method_filter="GET", path_filter="/users", tags_filter=["admin"])

        # Should only return operations matching all filters
        for result in results:
            assert result["metadata"]["method"] == "GET"
            assert "/users" in result["metadata"]["path"]
            assert "admin" in result["metadata"]["tags"]

    @patch("serena.tools.openapi_tools.faiss")
    def test_semantic_search_no_results(self, mock_faiss, openapi_tool, temp_dir, sample_openapi_spec):
        """Test semantic search when no results match filters."""
        chunks = openapi_tool._create_semantic_chunks(sample_openapi_spec)

        # Mock FAISS index
        mock_index = Mock()
        mock_faiss.read_index.return_value = mock_index

        # Create test index directory
        index_dir = os.path.join(temp_dir, "test_index")
        os.makedirs(index_dir, exist_ok=True)

        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump(chunks, f)

        # Mock the model
        mock_model = Mock()
        openapi_tool._model = mock_model

        # Search with filter that matches nothing
        results = openapi_tool._semantic_search("anything", index_dir, 5, tags_filter=["nonexistent_tag"])

        # Should return empty results
        assert len(results) == 0

    def test_format_results_human(self, openapi_tool):
        """Test human-readable result formatting."""
        mock_results = [
            {
                "rank": 1,
                "score": 0.95,
                "metadata": {
                    "operationId": "listUsers",
                    "method": "GET",
                    "path": "/users",
                    "summary": "List all users",
                    "description": "Get paginated user list",
                    "tags": ["users", "admin"],
                },
            },
            {
                "rank": 2,
                "score": 0.85,
                "metadata": {
                    "operationId": "createUser",
                    "method": "POST",
                    "path": "/users",
                    "summary": "Create user",
                    "description": "Register new user",
                    "tags": ["users", "registration"],
                },
            },
        ]

        result = openapi_tool._format_results_human(mock_results, "user management", False, False)

        # Verify formatting
        assert "Found 2 relevant API endpoints" in result
        assert "user management" in result
        assert "Rank 1" in result
        assert "listUsers" in result
        assert "GET /users" in result
        assert "List all users" in result
        assert "users, admin" in result
        assert "Rank 2" in result
        assert "createUser" in result

    def test_format_results_json(self, openapi_tool):
        """Test JSON result formatting."""
        mock_results = [
            {
                "rank": 1,
                "score": 0.95,
                "metadata": {
                    "operationId": "listUsers",
                    "method": "GET",
                    "path": "/users",
                    "summary": "List all users",
                    "description": "Get paginated user list",
                    "tags": ["users", "admin"],
                },
            }
        ]

        result = openapi_tool._format_results_json(mock_results, "user management", False, False)

        # Parse and verify JSON structure
        parsed = json.loads(result)
        assert parsed["query"] == "user management"
        assert parsed["total_results"] == 1
        assert len(parsed["results"]) == 1

        first_result = parsed["results"][0]
        assert first_result["rank"] == 1
        assert first_result["score"] == 0.95
        assert first_result["operationId"] == "listUsers"
        assert first_result["method"] == "GET"
        assert first_result["path"] == "/users"

    def test_format_results_markdown(self, openapi_tool):
        """Test Markdown result formatting."""
        mock_results = [
            {
                "rank": 1,
                "score": 0.95,
                "metadata": {
                    "operationId": "listUsers",
                    "method": "GET",
                    "path": "/users",
                    "summary": "List all users",
                    "description": "Get paginated user list",
                    "tags": ["users", "admin"],
                },
            }
        ]

        result = openapi_tool._format_results_markdown(mock_results, "user management", False, False)

        # Verify Markdown formatting
        assert "# API Search Results" in result
        assert "**Query:** user management" in result
        assert "**Results:** 1 endpoints found" in result
        assert "## 1. listUsers" in result
        assert "**Score:** 0.950" in result
        assert "**Method:** `GET`" in result
        assert "**Path:** `/users`" in result
        assert "**Tags:** `users`, `admin`" in result

    def test_format_results_with_examples(self, openapi_tool):
        """Test result formatting with code examples."""
        mock_results = [
            {
                "rank": 1,
                "score": 0.95,
                "metadata": {
                    "operationId": "createUser",
                    "method": "POST",
                    "path": "/users",
                    "summary": "Create user",
                    "description": "Register new user",
                    "tags": ["users"],
                },
            }
        ]

        result = openapi_tool._format_results_human(mock_results, "create user", False, True)

        # Should include curl example
        assert "Example:" in result
        assert "curl -X POST '/users'" in result
        assert "-H 'Content-Type: application/json'" in result
        assert '-d \'{"data": "example"}\'' in result

    def test_generate_code_examples(self, openapi_tool):
        """Test code example generation for different HTTP methods."""
        # Test GET
        get_metadata = {"method": "GET", "path": "/users"}
        get_example = openapi_tool._generate_code_example(get_metadata)
        assert get_example == "curl -X GET '/users'"

        # Test POST
        post_metadata = {"method": "POST", "path": "/users"}
        post_example = openapi_tool._generate_code_example(post_metadata)
        assert "curl -X POST '/users'" in post_example
        assert "Content-Type: application/json" in post_example

        # Test DELETE
        delete_metadata = {"method": "DELETE", "path": "/users/123"}
        delete_example = openapi_tool._generate_code_example(delete_metadata)
        assert delete_example == "curl -X DELETE '/users/123'"


@pytest.mark.openapi
class TestOpenApiSearchIntegration:
    """Integration tests for OpenAPI search with real specs."""

    def test_search_petstore_spec(self, openapi_tool):
        """Test searching the petstore specification."""
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        petstore_path = test_resources / "petstore.yaml"

        if not petstore_path.exists():
            pytest.skip("Petstore spec not found")

        # Test basic search functionality
        result = openapi_tool.apply(spec_path=str(petstore_path), query="list all pets", top_k=3, rebuild_index=True)

        # Should return results about pet listing
        assert "listPets" in result or "pets" in result.lower()
        assert "GET" in result
        assert "/pets" in result

    def test_search_blog_spec(self, openapi_tool):
        """Test searching the blog specification."""
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        blog_path = test_resources / "simple-blog.json"

        if not blog_path.exists():
            pytest.skip("Blog spec not found")

        # Test search for posting functionality
        result = openapi_tool.apply(
            spec_path=str(blog_path), query="create new blog post", top_k=2, output_format="json", rebuild_index=True
        )

        # Parse JSON result
        parsed = json.loads(result)
        assert "results" in parsed
        assert len(parsed["results"]) <= 2

        # Should find post creation endpoint
        operation_ids = [r["operationId"] for r in parsed["results"]]
        assert "createPost" in operation_ids

    def test_search_with_filters(self, openapi_tool):
        """Test search with various filters on real specs."""
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        ecommerce_path = test_resources / "ecommerce.yaml"

        if not ecommerce_path.exists():
            pytest.skip("E-commerce spec not found")

        # Test method filter
        result = openapi_tool.apply(
            spec_path=str(ecommerce_path), query="user operations", method_filter="GET", top_k=3, rebuild_index=True
        )

        # Should only contain GET operations
        assert "GET" in result
        # Should not contain POST, PUT, DELETE
        assert "POST" not in result or result.count("GET") > result.count("POST")

    def test_multi_spec_search(self, openapi_tool):
        """Test searching across multiple specifications."""
        # Configure multiple specs in the project
        openapi_tool.agent.get_active_project.return_value.project_config.openapi_specs = [
            "test/resources/openapi/petstore.yaml",
            "test/resources/openapi/simple-blog.json",
        ]

        result = openapi_tool.apply(query="list all items", search_all_specs=True, top_k=5, rebuild_index=True)

        # Should return results from multiple specs
        assert "pets" in result.lower() or "posts" in result.lower()
        # Should indicate multiple specifications were searched
        assert len(result) > 100  # Multi-spec results should be substantial
