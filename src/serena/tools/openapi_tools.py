"""
OpenAPI specification processing and semantic search tools.
Enables natural language queries to find relevant API endpoints.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np
from sensai.util import logging
from sentence_transformers import SentenceTransformer

from serena.tools import Tool, ToolMarkerDoesNotRequireActiveProject

if TYPE_CHECKING:
    from serena.agent import SerenaAgent

log = logging.getLogger(__name__)


class OpenApiTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """
    Tool for semantic search over OpenAPI specifications.
    Enables natural language queries to find relevant API endpoints.
    """

    def __init__(self, agent: "SerenaAgent"):
        super().__init__(agent)
        self._model: SentenceTransformer | None = None
        self._cache_dir: str | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            model_name = self.agent.serena_config.openapi.embedding_model
            log.info(f"Loading sentence transformer model: {model_name}")
            self._model = SentenceTransformer(model_name)
        return self._model

    @property
    def cache_dir(self) -> str:
        """Get the cache directory for OpenAPI indices."""
        if self._cache_dir is None:
            try:
                project_root = self.get_project_root()
                cache_dir = self.agent.serena_config.openapi.index_cache_dir
                self._cache_dir = os.path.join(project_root, cache_dir)
            except ValueError:
                # No active project, use temp directory
                self._cache_dir = os.path.join(tempfile.gettempdir(), "serena_openapi_cache")

        os.makedirs(self._cache_dir, exist_ok=True)
        return self._cache_dir

    def apply(
        self,
        spec_path: str | None = None,
        query: str = "",
        top_k: int = 3,
        rebuild_index: bool = False,
        search_all_specs: bool = False,
    ) -> str:
        """
        Search OpenAPI specification using natural language query.

        Args:
            spec_path: Path to OpenAPI spec file (relative to project root or absolute).
                      If None, will auto-discover or use project-configured specs.
            query: Natural language query describing desired API functionality
            top_k: Number of top results to return
            rebuild_index: Whether to rebuild the search index
            search_all_specs: If True, search across all project OpenAPI specs

        Returns:
            Formatted text with relevant API endpoints and usage instructions

        """
        try:
            # Determine which specs to search
            specs_to_search = self._get_specs_to_search(spec_path, search_all_specs)

            if not specs_to_search:
                return "Error: No OpenAPI specifications found. Please specify a spec path or configure specs in your project."

            if not query:
                return "Error: Query parameter is required for semantic search."

            all_results = []

            # Search each specification
            for spec_file in specs_to_search:
                try:
                    # Get or build the index for this spec
                    spec_hash = self._get_file_hash(spec_file)
                    index_dir = os.path.join(self.cache_dir, spec_hash)

                    if rebuild_index or not self._is_index_valid(index_dir, spec_file):
                        log.info(f"Building/rebuilding index for {spec_file}")
                        self._build_index(spec_file, index_dir)

                    # Perform semantic search
                    results = self._semantic_search(query, index_dir, top_k)

                    # Add spec path to results metadata
                    for result in results:
                        result["spec_file"] = spec_file

                    all_results.extend(results)

                except Exception as e:
                    log.warning(f"Error searching spec {spec_file}: {e}")
                    continue

            if not all_results:
                return f"No relevant endpoints found for query: {query}"

            # Sort all results by score and take top_k
            all_results.sort(key=lambda x: x["score"], reverse=True)
            top_results = all_results[:top_k]

            # Re-rank the top results
            for i, result in enumerate(top_results):
                result["rank"] = i + 1

            # Format results for LLM consumption
            return self._format_results(top_results, query, len(specs_to_search) > 1)

        except Exception as e:
            log.exception(f"Error in OpenAPI semantic search: {e}")
            return f"Error processing OpenAPI specification: {e!s}"

    def _get_specs_to_search(self, spec_path: str | None, search_all_specs: bool) -> list[str]:
        """Determine which OpenAPI specifications to search."""
        specs = []

        try:
            project_root = self.get_project_root()

            # If specific spec path provided, use it
            if spec_path:
                if not os.path.isabs(spec_path):
                    spec_path = os.path.join(project_root, spec_path)

                if os.path.exists(spec_path):
                    specs.append(spec_path)
                else:
                    log.warning(f"Specified OpenAPI spec not found: {spec_path}")

                return specs

            # If search_all_specs is True, get all configured specs
            if search_all_specs:
                try:
                    project_config = self.agent.get_active_project().project_config
                    for spec_rel_path in project_config.openapi_specs:
                        spec_abs_path = os.path.join(project_root, spec_rel_path)
                        if os.path.exists(spec_abs_path):
                            specs.append(spec_abs_path)
                        else:
                            log.warning(f"Configured OpenAPI spec not found: {spec_abs_path}")
                except Exception as e:
                    log.warning(f"Could not load project configuration: {e}")

            # If no specs found yet, try auto-discovery
            if not specs:
                specs.extend(self._auto_discover_specs(project_root))

        except ValueError:
            # No active project, try current directory auto-discovery
            specs.extend(self._auto_discover_specs(os.getcwd()))

        return specs

    def _auto_discover_specs(self, directory: str) -> list[str]:
        """Auto-discover OpenAPI specifications in a directory."""
        common_patterns = [
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
            "docs/openapi.yml",
            "docs/openapi.json",
            "docs/swagger.yaml",
            "docs/swagger.yml",
            "docs/swagger.json",
            "api/openapi.yaml",
            "api/openapi.yml",
            "api/openapi.json",
        ]

        found_specs = []
        for pattern in common_patterns:
            spec_path = os.path.join(directory, pattern)
            if os.path.exists(spec_path):
                found_specs.append(spec_path)

        return found_specs

    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for the file content and modification time."""
        with open(file_path, "rb") as f:
            content = f.read()

        stat = os.stat(file_path)
        content_hash = hashlib.md5(content).hexdigest()
        time_hash = hashlib.md5(str(stat.st_mtime).encode()).hexdigest()

        return hashlib.md5(f"{content_hash}_{time_hash}".encode()).hexdigest()

    def _is_index_valid(self, index_dir: str, spec_path: str) -> bool:
        """Check if the cached index is valid and up to date."""
        if not os.path.exists(index_dir):
            return False

        required_files = ["index.faiss", "chunks.json", "metadata.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(index_dir, file)):
                return False

        # Check if spec file is newer than the index
        metadata_path = os.path.join(index_dir, "metadata.json")
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            spec_mtime = os.path.getmtime(spec_path)
            index_mtime = metadata.get("created_at", 0)

            return spec_mtime <= index_mtime
        except (json.JSONDecodeError, FileNotFoundError):
            return False

    def _preprocess_openapi_spec(self, filepath: str) -> dict[str, Any]:
        """Preprocess OpenAPI spec with optional Redocly CLI validation."""
        temp_path = None

        try:
            # Try to use Redocly CLI if available and enabled
            if self.agent.serena_config.openapi.use_redocly_validation and self._is_redocly_available():
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
                    temp_path = temp_file.name

                # Bundle the spec using Redocly
                bundle_command = ["redocly", "bundle", filepath, "--output", temp_path]
                timeout = self.agent.serena_config.openapi.redocly_timeout
                result = subprocess.run(bundle_command, check=True, capture_output=True, text=True, timeout=timeout)

                if result.returncode != 0:
                    log.warning(f"Redocly bundle warning: {result.stderr}")

                # Validate the bundled spec
                lint_command = ["redocly", "lint", temp_path]
                result = subprocess.run(lint_command, check=False, capture_output=True, text=True, timeout=timeout)

                if result.returncode != 0:
                    log.warning(f"Redocly lint warning: {result.stderr}")

                # Load the processed spec
                with open(temp_path) as f:
                    spec = json.load(f)

                log.info("OpenAPI spec processed with Redocly CLI")
                return spec
            else:
                log.info("Redocly CLI not available, using direct parsing")

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning(f"Error during Redocly processing: {e}, falling back to direct parsing")
        except Exception as e:
            log.warning(f"Unexpected error with Redocly: {e}, falling back to direct parsing")
        finally:
            # Clean up temp file
            if temp_path:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

        # Fallback: load the original file directly
        try:
            with open(filepath, encoding="utf-8") as f:
                if filepath.endswith((".yaml", ".yml")):
                    try:
                        import yaml

                        return yaml.safe_load(f)
                    except ImportError:
                        raise ValueError("PyYAML is required to parse YAML files. Please install it or convert to JSON.")
                else:
                    return json.load(f)
        except Exception as e:
            log.error(f"Error loading OpenAPI spec: {e}")
            raise

    def _is_redocly_available(self) -> bool:
        """Check if Redocly CLI is available."""
        try:
            result = subprocess.run(["redocly", "--version"], check=False, capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _create_semantic_chunks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract searchable text chunks from OpenAPI specification."""
        chunks = []

        # Process API paths and operations
        for path, path_item in spec.get("paths", {}).items():
            if not isinstance(path_item, dict):
                continue

            for method, operation in path_item.items():
                if method.startswith("x-") or not isinstance(operation, dict):
                    continue

                operation_id = operation.get("operationId", f"{method}_{path}")
                summary = operation.get("summary", "")
                description = operation.get("description", "")
                tags = operation.get("tags", [])

                # Build searchable text description
                text_parts = [
                    f"API Endpoint: {operation_id}",
                    f"HTTP Method: {method.upper()}",
                    f"Path: {path}",
                ]

                if summary:
                    text_parts.append(f"Summary: {summary}")

                if description:
                    text_parts.append(f"Description: {description}")

                if tags:
                    text_parts.append(f"Tags: {', '.join(tags)}")

                # Add parameter information
                parameters = operation.get("parameters", [])
                if parameters:
                    param_descriptions = []
                    for param in parameters:
                        param_name = param.get("name", "unknown")
                        param_type = param.get("schema", {}).get("type", "unknown")
                        param_location = param.get("in", "unknown")
                        param_required = param.get("required", False)

                        param_desc = f"'{param_name}' ({param_type}) in {param_location}"
                        if param_required:
                            param_desc += " [required]"
                        param_descriptions.append(param_desc)

                    text_parts.append(f"Parameters: {', '.join(param_descriptions)}")

                # Add request body information
                request_body = operation.get("requestBody")
                if request_body:
                    content_types = list(request_body.get("content", {}).keys())
                    if content_types:
                        text_parts.append(f"Request Content Types: {', '.join(content_types)}")

                # Add response information
                responses = operation.get("responses", {})
                if responses:
                    response_codes = [code for code in responses.keys() if code != "default"]
                    if response_codes:
                        text_parts.append(f"Response Codes: {', '.join(response_codes)}")

                text_description = "\\n".join(text_parts)

                chunks.append(
                    {
                        "text": text_description,
                        "metadata": {
                            "operationId": operation_id,
                            "path": path,
                            "method": method.upper(),
                            "summary": summary,
                            "description": description,
                            "tags": tags,
                        },
                    }
                )

        return chunks

    def _build_index(self, spec_path: str, index_dir: str) -> None:
        """Build FAISS index for the OpenAPI specification."""
        os.makedirs(index_dir, exist_ok=True)

        # Preprocess the spec
        spec = self._preprocess_openapi_spec(spec_path)

        # Create semantic chunks
        chunks = self._create_semantic_chunks(spec)

        if not chunks:
            raise ValueError("No API endpoints found in the OpenAPI specification")

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        log.info(f"Generating embeddings for {len(texts)} API endpoints...")
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
        index.add(embeddings)

        # Save index and metadata
        faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump(chunks, f, indent=2)

        metadata = {
            "spec_path": spec_path,
            "num_chunks": len(chunks),
            "embedding_model": self.agent.serena_config.openapi.embedding_model,
            "created_at": os.path.getmtime(spec_path),
        }

        with open(os.path.join(index_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(f"Built search index with {len(chunks)} chunks at {index_dir}")

    def _semantic_search(self, query: str, index_dir: str, k: int) -> list[dict[str, Any]]:
        """Perform semantic search using the cached index."""
        # Load index and chunks
        index = faiss.read_index(os.path.join(index_dir, "index.faiss"))

        with open(os.path.join(index_dir, "chunks.json")) as f:
            chunks = json.load(f)

        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        # Search
        distances, indices = index.search(query_embedding, min(k, len(chunks)))

        # Return results with scores
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0], strict=False)):
            if idx < len(chunks):  # Valid index
                result = chunks[idx].copy()
                result["score"] = float(distance)
                result["rank"] = i + 1
                results.append(result)

        return results

    def _format_results(self, results: list[dict[str, Any]], query: str, multi_spec: bool = False) -> str:
        """Format search results for LLM consumption."""
        output = [f"Found {len(results)} relevant API endpoints for query: '{query}'\\n"]

        for result in results:
            metadata = result["metadata"]
            output.append(f"--- Rank {result['rank']} (Score: {result['score']:.3f}) ---")
            output.append(f"Endpoint: {metadata['operationId']}")
            output.append(f"Method: {metadata['method']} {metadata['path']}")

            # Include spec file information if searching multiple specs
            if multi_spec and "spec_file" in result:
                spec_name = os.path.basename(result["spec_file"])
                output.append(f"Specification: {spec_name}")

            if metadata.get("summary"):
                output.append(f"Summary: {metadata['summary']}")

            if metadata.get("description"):
                output.append(f"Description: {metadata['description']}")

            if metadata.get("tags"):
                output.append(f"Tags: {', '.join(metadata['tags'])}")

            output.append("")  # Empty line between results

        output.append(
            "To use these endpoints, refer to the OpenAPI specification for detailed parameter requirements, request/response schemas, and authentication."
        )

        return "\\n".join(output)

    def list_project_specs(self) -> list[str]:
        """List all OpenAPI specifications configured for the current project."""
        try:
            project = self.agent.get_active_project()
            project_root = project.project_root

            # Get configured specs
            configured_specs = []
            for spec_rel_path in project.project_config.openapi_specs:
                spec_abs_path = os.path.join(project_root, spec_rel_path)
                if os.path.exists(spec_abs_path):
                    configured_specs.append(spec_abs_path)

            # If no configured specs, return auto-discovered ones
            if not configured_specs:
                return self._auto_discover_specs(project_root)

            return configured_specs

        except ValueError:
            # No active project
            return self._auto_discover_specs(os.getcwd())

    def add_spec_to_project(self, spec_path: str) -> str:
        """Add an OpenAPI specification to the current project configuration."""
        try:
            project = self.agent.get_active_project()
            project_root = project.project_root

            # Convert to relative path
            if os.path.isabs(spec_path):
                try:
                    spec_rel_path = os.path.relpath(spec_path, project_root)
                except ValueError:
                    return f"Error: Specification path is outside project root: {spec_path}"
            else:
                spec_rel_path = spec_path
                spec_path = os.path.join(project_root, spec_rel_path)

            # Check if file exists
            if not os.path.exists(spec_path):
                return f"Error: OpenAPI specification file not found: {spec_path}"

            # Add to project configuration if not already present
            if spec_rel_path not in project.project_config.openapi_specs:
                project.project_config.openapi_specs.append(spec_rel_path)
                # Note: In a real implementation, you'd save the project config to disk here
                return f"Added OpenAPI specification to project: {spec_rel_path}"
            else:
                return f"OpenAPI specification already configured: {spec_rel_path}"

        except ValueError:
            return "Error: No active project. Cannot add specification to project configuration."
