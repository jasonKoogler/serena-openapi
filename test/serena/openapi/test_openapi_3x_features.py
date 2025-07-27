"""Tests for OpenAPI 3.x advanced features support."""

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
def advanced_openapi_spec():
    """Return OpenAPI 3.x spec with callbacks and webhooks."""
    return {
        "openapi": "3.0.3",
        "info": {"title": "Advanced API", "version": "1.0.0"},
        "paths": {
            "/subscriptions": {
                "post": {
                    "operationId": "createSubscription",
                    "summary": "Create webhook subscription",
                    "description": "Subscribe to receive webhook notifications",
                    "tags": ["subscriptions", "webhooks"],
                    "security": [{"ApiKeyAuth": []}],
                    "callbacks": {
                        "eventNotification": {
                            "{$request.body#/callbackUrl}": {
                                "post": {
                                    "summary": "Event notification callback",
                                    "description": "Webhook callback for events",
                                    "responses": {"200": {"description": "OK"}},
                                }
                            }
                        }
                    },
                    "responses": {"201": {"description": "Created"}},
                }
            },
            "/orders": {
                "post": {
                    "operationId": "createOrder",
                    "summary": "Create new order",
                    "description": "Create order with payment callbacks",
                    "tags": ["orders", "payments"],
                    "security": [{"OAuth2": ["write:orders"]}],
                    "callbacks": {
                        "paymentCallback": {
                            "{$request.body#/paymentUrl}": {
                                "post": {"summary": "Payment status callback", "responses": {"200": {"description": "OK"}}}
                            }
                        }
                    },
                    "responses": {"201": {"description": "Created"}},
                }
            },
        },
        "webhooks": {
            "userCreated": {
                "post": {
                    "operationId": "userCreatedWebhook",
                    "summary": "User created webhook",
                    "description": "Triggered when user is created",
                    "tags": ["webhooks", "users"],
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "orderStatusChanged": {
                "post": {
                    "operationId": "orderStatusWebhook",
                    "summary": "Order status webhook",
                    "description": "Triggered when order status changes",
                    "tags": ["webhooks", "orders"],
                    "responses": {"200": {"description": "OK"}},
                }
            },
        },
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key", "description": "API key authentication"},
                "OAuth2": {
                    "type": "oauth2",
                    "description": "OAuth2 authentication",
                    "flows": {
                        "authorizationCode": {
                            "authorizationUrl": "https://auth.example.com/oauth/authorize",
                            "tokenUrl": "https://auth.example.com/oauth/token",
                            "scopes": {"write:orders": "Create orders", "read:orders": "Read orders"},
                        }
                    },
                },
                "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT", "description": "JWT bearer token"},
            }
        },
    }


class TestOpenApi3xFeatures:
    """Test cases for OpenAPI 3.x advanced features."""

    def test_create_chunks_with_callbacks(self, openapi_tool, advanced_openapi_spec):
        """Test that callbacks are properly extracted into chunks."""
        chunks = openapi_tool._create_semantic_chunks(advanced_openapi_spec)

        # Should create chunks for operations
        operation_chunks = [c for c in chunks if c["metadata"]["type"] == "operation"]
        assert len(operation_chunks) == 2  # createSubscription and createOrder

        # Check callback information is included
        subscription_chunk = next(c for c in operation_chunks if c["metadata"]["operationId"] == "createSubscription")

        assert "Callbacks:" in subscription_chunk["text"]
        assert "eventNotification" in subscription_chunk["text"]
        assert "callbacks" in subscription_chunk["metadata"]
        assert "eventNotification" in subscription_chunk["metadata"]["callbacks"]

    def test_create_chunks_with_webhooks(self, openapi_tool, advanced_openapi_spec):
        """Test that webhooks are properly extracted into chunks."""
        chunks = openapi_tool._create_semantic_chunks(advanced_openapi_spec)

        # Should create chunks for webhooks
        webhook_chunks = [c for c in chunks if c["metadata"]["type"] == "webhook"]
        assert len(webhook_chunks) == 2  # userCreated and orderStatusChanged

        # Check webhook metadata
        user_webhook = next(c for c in webhook_chunks if c["metadata"]["operationId"] == "userCreatedWebhook")

        assert user_webhook["metadata"]["type"] == "webhook"
        assert user_webhook["metadata"]["webhookName"] == "userCreated"
        assert "Webhook:" in user_webhook["text"]
        assert "Type: Webhook/Event Notification" in user_webhook["text"]

    def test_create_chunks_with_security_schemes(self, openapi_tool, advanced_openapi_spec):
        """Test that security schemes are properly extracted."""
        chunks = openapi_tool._create_semantic_chunks(advanced_openapi_spec)

        # Should create chunks for security schemes
        security_chunks = [c for c in chunks if c["metadata"]["type"] == "security_scheme"]
        assert len(security_chunks) == 3  # ApiKeyAuth, OAuth2, BearerAuth

        # Check OAuth2 security scheme
        oauth2_chunk = next(c for c in security_chunks if c["metadata"]["name"] == "OAuth2")

        assert oauth2_chunk["metadata"]["schemeType"] == "oauth2"
        assert "Security Scheme: OAuth2" in oauth2_chunk["text"]
        assert "OAuth2 Flows:" in oauth2_chunk["text"]
        assert "authorizationCode" in oauth2_chunk["text"]

    def test_create_chunks_with_security_metadata(self, openapi_tool, advanced_openapi_spec):
        """Test that security requirements are included in operation metadata."""
        chunks = openapi_tool._create_semantic_chunks(advanced_openapi_spec)

        # Find operation with security
        order_chunk = next(c for c in chunks if c["metadata"].get("operationId") == "createOrder")

        assert "security" in order_chunk["metadata"]
        assert "Security:" in order_chunk["text"]
        assert "OAuth2" in order_chunk["text"]

    def test_section_filter_operations(self, openapi_tool, temp_dir, advanced_openapi_spec):
        """Test filtering by operations section."""
        # Create spec file
        spec_path = os.path.join(temp_dir, "advanced.json")
        with open(spec_path, "w") as f:
            json.dump(advanced_openapi_spec, f)

        # Mock successful search with operations filter
        with patch.object(openapi_tool, "_semantic_search") as mock_search:
            mock_search.return_value = [
                {
                    "rank": 1,
                    "score": 0.9,
                    "metadata": {"operationId": "createSubscription", "type": "operation", "method": "POST", "path": "/subscriptions"},
                }
            ]

            openapi_tool.apply(spec_path=spec_path, query="API operations", section_filter="operations", rebuild_index=True)

            # Should call semantic search with section filter
            mock_search.assert_called_once()
            args, kwargs = mock_search.call_args

            # Check that section_filter was passed as keyword argument
            assert kwargs.get("section_filter") == "operations"

    def test_section_filter_webhooks(self, openapi_tool, temp_dir, advanced_openapi_spec):
        """Test filtering by webhooks section."""
        spec_path = os.path.join(temp_dir, "advanced.json")
        with open(spec_path, "w") as f:
            json.dump(advanced_openapi_spec, f)

        with patch.object(openapi_tool, "_semantic_search") as mock_search:
            mock_search.return_value = [
                {
                    "rank": 1,
                    "score": 0.9,
                    "metadata": {"operationId": "userCreatedWebhook", "type": "webhook", "webhookName": "userCreated"},
                }
            ]

            result = openapi_tool.apply(spec_path=spec_path, query="webhook notifications", section_filter="webhooks", rebuild_index=True)

            assert isinstance(result, str)

    def test_section_filter_security_schemes(self, openapi_tool, temp_dir, advanced_openapi_spec):
        """Test filtering by security schemes section."""
        spec_path = os.path.join(temp_dir, "advanced.json")
        with open(spec_path, "w") as f:
            json.dump(advanced_openapi_spec, f)

        with patch.object(openapi_tool, "_semantic_search") as mock_search:
            mock_search.return_value = [
                {"rank": 1, "score": 0.9, "metadata": {"name": "OAuth2", "type": "security_scheme", "schemeType": "oauth2"}}
            ]

            result = openapi_tool.apply(
                spec_path=spec_path, query="authentication methods", section_filter="security_schemes", rebuild_index=True
            )

            assert isinstance(result, str)

    def test_semantic_search_section_filtering(self, openapi_tool, temp_dir, advanced_openapi_spec):
        """Test that section filtering works in semantic search."""
        chunks = openapi_tool._create_semantic_chunks(advanced_openapi_spec)

        # Create index directory and files
        index_dir = os.path.join(temp_dir, "test_index")
        os.makedirs(index_dir, exist_ok=True)

        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump(chunks, f)

        # Mock FAISS components
        with patch("serena.tools.openapi_tools.faiss") as mock_faiss:
            # Mock main index
            mock_index = Mock()
            mock_index.search.return_value = ([[0.9, 0.8]], [[0, 1]])  # distances  # indices
            mock_index.reconstruct.side_effect = lambda idx: [0.1, 0.2, 0.3]  # Return embedding for any index
            mock_faiss.read_index.return_value = mock_index

            # Mock filtered index creation
            mock_filtered_index = Mock()
            mock_filtered_index.search.return_value = ([[0.9]], [[0]])  # distances  # indices
            mock_faiss.IndexFlatIP.return_value = mock_filtered_index
            mock_faiss.normalize_L2 = Mock()  # Mock normalize function

            # Mock embedding model
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            openapi_tool._model = mock_model

            # Test filtering by webhooks
            results = openapi_tool._semantic_search("webhooks", index_dir, 5, section_filter="webhooks")

            # Should only return webhook chunks (if any exist in filtered results)
            webhook_chunks = [c for c in chunks if c["metadata"]["type"] == "webhook"]
            if webhook_chunks:
                for result in results:
                    assert result["metadata"]["type"] == "webhook"

    def test_combined_filters_with_sections(self, openapi_tool, temp_dir, advanced_openapi_spec):
        """Test combining section filter with other filters."""
        chunks = openapi_tool._create_semantic_chunks(advanced_openapi_spec)

        index_dir = os.path.join(temp_dir, "test_index")
        os.makedirs(index_dir, exist_ok=True)

        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump(chunks, f)

        with patch("serena.tools.openapi_tools.faiss") as mock_faiss:
            # Mock main index
            mock_index = Mock()
            mock_index.search.return_value = ([[0.9]], [[0]])  # distances  # indices
            mock_index.reconstruct.side_effect = lambda idx: [0.1, 0.2, 0.3]  # Return embedding for any index
            mock_faiss.read_index.return_value = mock_index

            # Mock filtered index creation
            mock_filtered_index = Mock()
            mock_filtered_index.search.return_value = ([[0.9]], [[0]])  # distances  # indices
            mock_faiss.IndexFlatIP.return_value = mock_filtered_index
            mock_faiss.normalize_L2 = Mock()  # Mock normalize function

            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            openapi_tool._model = mock_model

            # Test section + method filter
            results = openapi_tool._semantic_search("operations", index_dir, 5, method_filter="POST", section_filter="operations")

            # Should only return operation chunks with POST method (if any exist)
            operation_post_chunks = [c for c in chunks if c["metadata"]["type"] == "operation" and c["metadata"]["method"] == "POST"]
            if operation_post_chunks:
                for result in results:
                    assert result["metadata"]["type"] == "operation"
                    assert result["metadata"]["method"] == "POST"

    def test_external_docs_in_chunks(self, openapi_tool):
        """Test that external documentation is included in chunks."""
        spec_with_external_docs = {
            "openapi": "3.0.3",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOperation",
                        "summary": "Test operation",
                        "externalDocs": {"description": "Additional documentation", "url": "https://docs.example.com/test"},
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }

        chunks = openapi_tool._create_semantic_chunks(spec_with_external_docs)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert "External Docs:" in chunk["text"]
        assert "Additional documentation" in chunk["text"]
        assert "https://docs.example.com/test" in chunk["text"]

    def test_complex_security_schemes(self, openapi_tool):
        """Test complex security scheme extraction."""
        spec_with_complex_security = {
            "openapi": "3.0.3",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
            "components": {
                "securitySchemes": {
                    "ComplexOAuth2": {
                        "type": "oauth2",
                        "description": "Multi-flow OAuth2",
                        "flows": {
                            "authorizationCode": {
                                "authorizationUrl": "https://auth.example.com/authorize",
                                "tokenUrl": "https://auth.example.com/token",
                                "scopes": {"read": "Read access", "write": "Write access"},
                            },
                            "clientCredentials": {"tokenUrl": "https://auth.example.com/token", "scopes": {"admin": "Admin access"}},
                        },
                    },
                    "ApiKeyHeader": {"type": "apiKey", "in": "header", "name": "X-API-Key", "description": "Header API key"},
                    "ApiKeyQuery": {"type": "apiKey", "in": "query", "name": "api_key", "description": "Query parameter API key"},
                }
            },
        }

        chunks = openapi_tool._create_semantic_chunks(spec_with_complex_security)

        security_chunks = [c for c in chunks if c["metadata"]["type"] == "security_scheme"]
        assert len(security_chunks) == 3

        # Check OAuth2 chunk
        oauth2_chunk = next(c for c in security_chunks if c["metadata"]["name"] == "ComplexOAuth2")
        assert "authorizationCode, clientCredentials" in oauth2_chunk["text"]

        # Check API key chunks
        header_key_chunk = next(c for c in security_chunks if c["metadata"]["name"] == "ApiKeyHeader")
        assert "X-API-Key in header" in header_key_chunk["text"]

        query_key_chunk = next(c for c in security_chunks if c["metadata"]["name"] == "ApiKeyQuery")
        assert "api_key in query" in query_key_chunk["text"]


@pytest.mark.openapi
class TestOpenApi3xIntegration:
    """Integration tests for OpenAPI 3.x features."""

    def test_advanced_features_spec_processing(self, openapi_tool):
        """Test processing of the advanced features spec file."""
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        advanced_path = test_resources / "advanced-features.yaml"

        if not advanced_path.exists():
            pytest.skip("Advanced features spec not found")

        # Test that the spec can be processed without errors
        result = openapi_tool.apply(spec_path=str(advanced_path), query="webhook subscriptions", top_k=5, rebuild_index=True)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should find webhook-related content
        assert "webhook" in result.lower() or "subscription" in result.lower()

    def test_callback_search(self, openapi_tool):
        """Test searching for callback-related functionality."""
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        advanced_path = test_resources / "advanced-features.yaml"

        if not advanced_path.exists():
            pytest.skip("Advanced features spec not found")

        result = openapi_tool.apply(spec_path=str(advanced_path), query="payment callbacks and notifications", top_k=3, rebuild_index=True)

        assert isinstance(result, str)
        assert "callback" in result.lower() or "payment" in result.lower()

    def test_webhook_section_filtering(self, openapi_tool):
        """Test filtering to show only webhook sections."""
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        advanced_path = test_resources / "advanced-features.yaml"

        if not advanced_path.exists():
            pytest.skip("Advanced features spec not found")

        result = openapi_tool.apply(
            spec_path=str(advanced_path), query="event notifications", section_filter="webhooks", top_k=5, rebuild_index=True
        )

        assert isinstance(result, str)
        # Should only return webhook-related results
        if "webhook" in result.lower():
            # If webhooks are found, should not contain regular operations
            assert result.lower().count("webhook") >= result.lower().count("endpoint")

    def test_security_scheme_search(self, openapi_tool):
        """Test searching for security schemes."""
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        advanced_path = test_resources / "advanced-features.yaml"

        if not advanced_path.exists():
            pytest.skip("Advanced features spec not found")

        result = openapi_tool.apply(
            spec_path=str(advanced_path),
            query="authentication and authorization methods",
            section_filter="security_schemes",
            top_k=5,
            rebuild_index=True,
        )

        assert isinstance(result, str)
        # Should find security-related content
        if len(result) > 50:  # Non-empty result
            assert any(term in result.lower() for term in ["oauth", "api key", "bearer", "auth"])

    def test_comprehensive_search_across_sections(self, openapi_tool):
        """Test comprehensive search across all sections."""
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        advanced_path = test_resources / "advanced-features.yaml"

        if not advanced_path.exists():
            pytest.skip("Advanced features spec not found")

        # Test different section searches
        sections = [("operations", "order management"), ("webhooks", "event notifications"), ("security_schemes", "authentication")]

        for section, query in sections:
            result = openapi_tool.apply(
                spec_path=str(advanced_path),
                query=query,
                section_filter=section,
                top_k=3,
                rebuild_index=(section == sections[0][0]),  # Only rebuild once
            )

            assert isinstance(result, str)
            # Each section should return some results
            assert len(result) > 0
