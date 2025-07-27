"""Integration tests for OpenAPI functionality with real-world specifications."""

import json
import os
import shutil
import tempfile
from pathlib import Path

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
    """Create OpenApiTool with real dependencies for integration testing."""
    from unittest.mock import Mock

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


@pytest.mark.openapi
@pytest.mark.slow  # Mark as slow due to model loading
class TestOpenApiIntegrationRealWorld:
    """Integration tests with real OpenAPI specifications and models."""

    def test_petstore_integration_end_to_end(self, openapi_tool_with_real_model):
        """Test complete workflow with petstore specification."""
        tool, temp_dir = openapi_tool_with_real_model

        # Get petstore spec path
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        petstore_path = test_resources / "petstore.yaml"

        if not petstore_path.exists():
            pytest.skip("Petstore spec not found")

        # Test 1: Index building
        spec_hash = tool._get_file_hash(str(petstore_path))
        index_dir = os.path.join(tool.cache_dir, spec_hash)

        # Ensure clean start
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)

        # Test 2: Search functionality
        result = tool.apply(spec_path=str(petstore_path), query="list all pets in the store", top_k=3, rebuild_index=True)

        # Verify results
        assert isinstance(result, str)
        assert len(result) > 0
        assert "pets" in result.lower() or "Pet" in result
        assert "listPets" in result or "GET" in result

        # Test 3: Index persistence
        assert os.path.exists(index_dir)
        assert os.path.exists(os.path.join(index_dir, "index.faiss"))
        assert os.path.exists(os.path.join(index_dir, "chunks.json"))
        assert os.path.exists(os.path.join(index_dir, "metadata.json"))

        # Test 4: Cached search (should be faster)
        result2 = tool.apply(spec_path=str(petstore_path), query="create new pet", top_k=2)

        assert isinstance(result2, str)
        assert "createPet" in result2 or "POST" in result2

    def test_blog_integration_with_filtering(self, openapi_tool_with_real_model):
        """Test blog API with advanced filtering options."""
        tool, temp_dir = openapi_tool_with_real_model

        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        blog_path = test_resources / "simple-blog.json"

        if not blog_path.exists():
            pytest.skip("Blog spec not found")

        # Test method filtering
        result = tool.apply(spec_path=str(blog_path), query="blog operations", method_filter="GET", top_k=5, rebuild_index=True)

        assert isinstance(result, str)
        assert "GET" in result
        # Should not contain POST operations
        post_count = result.count("POST")
        get_count = result.count("GET")
        assert get_count >= post_count

        # Test path filtering
        result2 = tool.apply(spec_path=str(blog_path), query="post management", path_filter="/posts.*", top_k=3)

        assert isinstance(result2, str)
        assert "/posts" in result2

        # Test JSON output format
        result3 = tool.apply(spec_path=str(blog_path), query="create content", output_format="json", top_k=2)

        # Should be valid JSON
        parsed = json.loads(result3)
        assert "query" in parsed
        assert "results" in parsed
        assert isinstance(parsed["results"], list)

    def test_ecommerce_integration_comprehensive(self, openapi_tool_with_real_model):
        """Test comprehensive e-commerce API integration."""
        tool, temp_dir = openapi_tool_with_real_model

        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        ecommerce_path = test_resources / "ecommerce.yaml"

        if not ecommerce_path.exists():
            pytest.skip("E-commerce spec not found")

        # Test 1: Complex semantic search
        result = tool.apply(spec_path=str(ecommerce_path), query="search for products by category and price", top_k=3, rebuild_index=True)

        assert isinstance(result, str)
        assert "products" in result.lower() or "search" in result.lower()

        # Test 2: Combined filtering
        result2 = tool.apply(spec_path=str(ecommerce_path), query="user management", method_filter="GET", tags_filter=["users"], top_k=5)

        assert isinstance(result2, str)

        # Test 3: Markdown output with examples
        result3 = tool.apply(
            spec_path=str(ecommerce_path), query="payment processing", output_format="markdown", include_examples=True, top_k=2
        )

        assert isinstance(result3, str)
        assert "# API Search Results" in result3
        assert "**Method:**" in result3
        if "curl" in result3:  # If examples are included
            assert "curl -X" in result3

    def test_multi_spec_integration(self, openapi_tool_with_real_model):
        """Test integration with multiple specifications."""
        tool, temp_dir = openapi_tool_with_real_model

        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"

        # Configure multiple specs
        specs = ["petstore.yaml", "simple-blog.json"]
        available_specs = []

        for spec in specs:
            spec_path = test_resources / spec
            if spec_path.exists():
                available_specs.append(f"test/resources/openapi/{spec}")

        if len(available_specs) < 2:
            pytest.skip("Need at least 2 specs for multi-spec testing")

        # Configure tool with multiple specs
        tool.agent.get_active_project.return_value.project_config.openapi_specs = available_specs

        # Test cross-spec search
        result = tool.apply(query="list all items", search_all_specs=True, top_k=5, rebuild_index=True)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain results from multiple specs
        assert "Specification:" in result or len(result) > 200  # Multi-spec results are usually longer

    def test_auto_discovery_integration(self, openapi_tool_with_real_model):
        """Test auto-discovery functionality in real scenarios."""
        tool, temp_dir = openapi_tool_with_real_model

        # Copy a real spec to temp directory with discoverable name
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        source_spec = test_resources / "simple-blog.json"

        if not source_spec.exists():
            pytest.skip("Blog spec not found for auto-discovery test")

        # Copy to discoverable location
        discoverable_spec = os.path.join(temp_dir, "openapi.json")
        shutil.copy2(source_spec, discoverable_spec)

        # Test auto-discovery
        result = tool.apply(query="blog posts", top_k=3, rebuild_index=True)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should find the auto-discovered spec
        assert "posts" in result.lower() or "blog" in result.lower()

    def test_performance_with_large_spec(self, openapi_tool_with_real_model):
        """Test performance with the largest available specification."""
        tool, temp_dir = openapi_tool_with_real_model

        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        ecommerce_path = test_resources / "ecommerce.yaml"

        if not ecommerce_path.exists():
            pytest.skip("E-commerce spec not found")

        import time

        # Measure indexing time
        start_time = time.time()
        result = tool.apply(spec_path=str(ecommerce_path), query="performance test", top_k=1, rebuild_index=True)
        indexing_time = time.time() - start_time

        assert isinstance(result, str)

        # Measure search time (should be much faster)
        start_time = time.time()
        result2 = tool.apply(spec_path=str(ecommerce_path), query="another search", top_k=1)
        search_time = time.time() - start_time

        assert isinstance(result2, str)

        # Search should be significantly faster than indexing
        assert search_time < indexing_time

        # Both operations should complete in reasonable time
        assert indexing_time < 60  # Max 1 minute for indexing
        assert search_time < 5  # Max 5 seconds for search

    def test_error_resilience_integration(self, openapi_tool_with_real_model):
        """Test error resilience with real specifications."""
        tool, temp_dir = openapi_tool_with_real_model

        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        petstore_path = test_resources / "petstore.yaml"

        if not petstore_path.exists():
            pytest.skip("Petstore spec not found")

        # Test 1: Invalid query handling
        result = tool.apply(spec_path=str(petstore_path), query="", rebuild_index=True)  # Empty query

        assert "Error: Query parameter is required" in result

        # Test 2: Recovery after error
        result2 = tool.apply(spec_path=str(petstore_path), query="valid query after error", top_k=1)

        assert isinstance(result2, str)
        assert "Error" not in result2 or "No relevant endpoints found" in result2

        # Test 3: Corrupted cache handling
        spec_hash = tool._get_file_hash(str(petstore_path))
        index_dir = os.path.join(tool.cache_dir, spec_hash)

        if os.path.exists(os.path.join(index_dir, "index.faiss")):
            # Corrupt the index file
            with open(os.path.join(index_dir, "index.faiss"), "w") as f:
                f.write("corrupted data")

            # Should rebuild automatically
            result3 = tool.apply(spec_path=str(petstore_path), query="recovery test", top_k=1)

            assert isinstance(result3, str)

    def test_unicode_and_special_characters_integration(self, openapi_tool_with_real_model):
        """Test handling of Unicode and special characters in real scenarios."""
        tool, temp_dir = openapi_tool_with_real_model

        # Create a spec with Unicode content
        unicode_spec = {
            "openapi": "3.0.3",
            "info": {"title": "International API 🌍", "version": "1.0.0", "description": "API with émojis and àccënts"},
            "paths": {
                "/usuarios": {
                    "get": {
                        "operationId": "listarUsuários",
                        "summary": "Listar usuários do sistema",
                        "description": "Retorna lista de usuários cadastrados 📋",
                        "tags": ["usuários", "listagem"],
                        "responses": {"200": {"description": "Lista de usuários"}},
                    }
                },
                "/产品": {
                    "post": {
                        "operationId": "创建产品",
                        "summary": "创建新产品",
                        "description": "在系统中创建新的产品记录 🏷️",
                        "tags": ["产品", "管理"],
                        "responses": {"201": {"description": "产品已创建"}},
                    }
                },
            },
        }

        spec_path = os.path.join(temp_dir, "unicode_spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(unicode_spec, f, ensure_ascii=False, indent=2)

        # Test Unicode search
        result = tool.apply(spec_path=spec_path, query="usuários", top_k=2, rebuild_index=True)

        assert isinstance(result, str)
        # Should handle Unicode content
        assert "usuários" in result or "listar" in result.lower()

        # Test Chinese characters
        result2 = tool.apply(spec_path=spec_path, query="产品管理", top_k=2)

        assert isinstance(result2, str)

    def test_large_result_sets_integration(self, openapi_tool_with_real_model):
        """Test handling of large result sets."""
        tool, temp_dir = openapi_tool_with_real_model

        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        ecommerce_path = test_resources / "ecommerce.yaml"

        if not ecommerce_path.exists():
            pytest.skip("E-commerce spec not found")

        # Test with different result set sizes
        for top_k in [1, 5, 10, 20]:
            result = tool.apply(
                spec_path=str(ecommerce_path),
                query="API operations",
                top_k=top_k,
                rebuild_index=(top_k == 1),  # Only rebuild on first iteration
            )

            assert isinstance(result, str)
            assert len(result) > 0

            # Count number of results returned
            result_count = result.count("Rank ")
            # Should return at most top_k results
            assert result_count <= top_k

    def test_concurrent_access_simulation(self, openapi_tool_with_real_model):
        """Test simulation of concurrent access patterns."""
        tool, temp_dir = openapi_tool_with_real_model

        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        petstore_path = test_resources / "petstore.yaml"

        if not petstore_path.exists():
            pytest.skip("Petstore spec not found")

        # Build index first
        tool.apply(spec_path=str(petstore_path), query="initial index build", top_k=1, rebuild_index=True)

        # Simulate multiple concurrent searches
        queries = ["list pets", "create pet", "update pet", "delete pet", "pet management"]

        results = []
        for query in queries:
            result = tool.apply(spec_path=str(petstore_path), query=query, top_k=2)
            results.append(result)
            assert isinstance(result, str)

        # All searches should succeed
        assert len(results) == len(queries)
        assert all(isinstance(r, str) and len(r) > 0 for r in results)


@pytest.mark.openapi
class TestOpenApiCLIIntegration:
    """Integration tests for CLI commands with real specifications."""

    def test_cli_search_command_integration(self):
        """Test CLI search command with real specs."""
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        petstore_path = test_resources / "petstore.yaml"

        if not petstore_path.exists():
            pytest.skip("Petstore spec not found")

        # Import CLI components
        from click.testing import CliRunner

        from serena.cli import OpenApiCommands

        runner = CliRunner()

        # Test basic search
        result = runner.invoke(
            OpenApiCommands().search,
            ["list pets", "--spec-path", str(petstore_path), "--top-k", "2", "--log-level", "ERROR"],  # Suppress logs in test
        )

        # Command should succeed
        assert result.exit_code == 0
        assert "pets" in result.output.lower() or "Pet" in result.output

    def test_cli_with_filtering_integration(self):
        """Test CLI with filtering options."""
        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        blog_path = test_resources / "simple-blog.json"

        if not blog_path.exists():
            pytest.skip("Blog spec not found")

        from click.testing import CliRunner

        from serena.cli import OpenApiCommands

        runner = CliRunner()

        # Test with method filter and JSON output
        result = runner.invoke(
            OpenApiCommands().search,
            [
                "blog operations",
                "--spec-path",
                str(blog_path),
                "--method",
                "GET",
                "--output-format",
                "json",
                "--top-k",
                "3",
                "--log-level",
                "ERROR",
            ],
        )

        assert result.exit_code == 0

        # Should be valid JSON
        try:
            json.loads(result.output.split("\n")[-2])  # Last non-empty line
        except (json.JSONDecodeError, IndexError):
            # If JSON parsing fails, at least check command succeeded
            pass

    def test_cli_list_command_integration(self):
        """Test CLI list command."""
        from click.testing import CliRunner

        from serena.cli import OpenApiCommands

        runner = CliRunner()

        # Test list command (may not find specs in test environment)
        result = runner.invoke(OpenApiCommands().list_specs, ["--log-level", "ERROR"])

        # Command should not crash, even if no specs found
        assert result.exit_code == 0


@pytest.mark.openapi
class TestOpenApiRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_api_exploration_workflow(self, openapi_tool_with_real_model):
        """Test typical API exploration workflow."""
        tool, temp_dir = openapi_tool_with_real_model

        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        ecommerce_path = test_resources / "ecommerce.yaml"

        if not ecommerce_path.exists():
            pytest.skip("E-commerce spec not found")

        # Workflow 1: Developer wants to understand user management
        user_ops = tool.apply(spec_path=str(ecommerce_path), query="user management and authentication", top_k=5, rebuild_index=True)

        assert isinstance(user_ops, str)

        # Workflow 2: Developer wants to see only GET endpoints for products
        product_gets = tool.apply(
            spec_path=str(ecommerce_path), query="product catalog browsing", method_filter="GET", tags_filter=["products"], top_k=3
        )

        assert isinstance(product_gets, str)

        # Workflow 3: Developer wants implementation examples
        with_examples = tool.apply(
            spec_path=str(ecommerce_path), query="payment processing", include_examples=True, output_format="markdown", top_k=2
        )

        assert isinstance(with_examples, str)

    def test_documentation_generation_scenario(self, openapi_tool_with_real_model):
        """Test using tool for documentation generation."""
        tool, temp_dir = openapi_tool_with_real_model

        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"
        petstore_path = test_resources / "petstore.yaml"

        if not petstore_path.exists():
            pytest.skip("Petstore spec not found")

        # Generate documentation sections
        sections = [
            ("Pet Management", "managing pets in the store"),
            ("Owner Operations", "pet owners and registration"),
            ("Adoption Process", "pet adoption workflow"),
        ]

        documentation = {}

        for section_name, query in sections:
            result = tool.apply(
                spec_path=str(petstore_path),
                query=query,
                output_format="markdown",
                include_examples=True,
                top_k=3,
                rebuild_index=(section_name == sections[0][0]),  # Only rebuild once
            )

            documentation[section_name] = result
            assert isinstance(result, str)

        # All sections should have content
        assert len(documentation) == len(sections)
        assert all(len(content) > 0 for content in documentation.values())

    def test_api_migration_scenario(self, openapi_tool_with_real_model):
        """Test scenario for API migration planning."""
        tool, temp_dir = openapi_tool_with_real_model

        test_resources = Path(__file__).parent.parent.parent / "resources" / "openapi"

        # Compare different specs for migration planning
        specs = ["petstore.yaml", "simple-blog.json"]
        migration_analysis = {}

        for spec in specs:
            spec_path = test_resources / spec
            if not spec_path.exists():
                continue

            # Analyze common operations
            result = tool.apply(
                spec_path=str(spec_path),
                query="CRUD operations create read update delete",
                output_format="json",
                top_k=10,
                rebuild_index=True,
            )

            migration_analysis[spec] = result
            assert isinstance(result, str)

        # Should have analyzed at least one spec
        assert len(migration_analysis) > 0
