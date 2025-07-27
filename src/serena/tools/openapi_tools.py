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

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False

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

    @model.setter
    def model(self, value: SentenceTransformer | None) -> None:
        """Set the sentence transformer model."""
        self._model = value

    @model.deleter
    def model(self) -> None:
        """Delete the sentence transformer model."""
        self._model = None

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

        try:
            os.makedirs(self._cache_dir, exist_ok=True)
        except PermissionError:
            # Fall back to temp directory if we can't create the cache dir
            log.warning(f"Cannot create cache directory {self._cache_dir}, falling back to temp directory")
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
        output_format: str = "human",
        include_examples: bool = False,
        method_filter: str | None = None,
        path_filter: str | None = None,
        tags_filter: list[str] | None = None,
        section_filter: str | None = None,
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
            output_format: Output format - "human", "json", or "markdown"
            include_examples: Whether to include code examples in output
            method_filter: Filter by HTTP method (GET, POST, PUT, DELETE, etc.)
            path_filter: Filter by path pattern (supports regex)
            tags_filter: Filter by tags (endpoints must have at least one matching tag)
            section_filter: Filter by API section type ("operations", "webhooks", "security_schemes")

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

                    # Perform semantic search with filters
                    results = self._semantic_search(
                        query,
                        index_dir,
                        top_k,
                        method_filter=method_filter,
                        path_filter=path_filter,
                        tags_filter=tags_filter,
                        section_filter=section_filter,
                    )

                    # Add spec path to results metadata
                    for result in results:
                        result["spec_file"] = spec_file

                    all_results.extend(results)

                except Exception as e:
                    log.warning(f"Error searching spec {spec_file}: {e}")
                    # If we're only searching one spec and it fails, return error
                    if len(specs_to_search) == 1:
                        raise
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
            return self._format_results(top_results, query, len(specs_to_search) > 1, output_format, include_examples)

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
                    project = self.agent.get_active_project()
                    if project is not None:
                        project_config = project.project_config
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
        """Generate a hash for the file content and modification time, including dependencies."""
        # Get all files that this spec depends on
        dependency_files = self._get_spec_dependencies(file_path)

        # Include main file in dependencies
        all_files = [file_path] + dependency_files

        # Create combined hash from all files
        combined_hash_parts = []

        for dep_file in sorted(all_files):  # Sort for consistent hashing
            try:
                with open(dep_file, "rb") as f:
                    content = f.read()

                stat = os.stat(dep_file)
                content_hash = hashlib.md5(content).hexdigest()
                time_hash = hashlib.md5(str(stat.st_mtime).encode()).hexdigest()
                file_hash = f"{dep_file}:{content_hash}:{time_hash}"
                combined_hash_parts.append(file_hash)
            except (FileNotFoundError, OSError) as e:
                log.warning(f"Could not hash dependency file {dep_file}: {e}")
                # Include file path and error in hash to ensure cache invalidation
                combined_hash_parts.append(f"{dep_file}:missing")

        # Create final hash from all file hashes
        combined_string = "|".join(combined_hash_parts)
        return hashlib.md5(combined_string.encode()).hexdigest()

    def _get_spec_dependencies(self, file_path: str, visited: set[str] | None = None) -> list[str]:
        """Get list of files that this OpenAPI spec depends on via $ref."""
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        abs_path = os.path.abspath(file_path)
        if abs_path in visited:
            return []
        visited.add(abs_path)

        dependencies: set[str] = set()

        try:
            # Load the spec to analyze dependencies
            with open(file_path, encoding="utf-8") as f:
                if file_path.endswith((".yaml", ".yml")):
                    if not YAML_AVAILABLE:
                        return []  # Can't analyze YAML dependencies without PyYAML
                    spec = yaml.safe_load(f)  # type: ignore[attr-defined]
                else:
                    spec = json.load(f)

            # Find all external $ref references
            base_dir = os.path.dirname(os.path.abspath(file_path))
            self._collect_external_refs(spec, base_dir, dependencies, visited)

        except Exception as e:
            log.debug(f"Could not analyze dependencies for {file_path}: {e}")
            return []

        return sorted(dependencies)

    def _collect_external_refs(self, obj: Any, base_dir: str, dependencies: set[str], visited: set[str]) -> None:
        """Recursively collect external $ref file paths."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "$ref" and isinstance(value, str):
                    # Check if it's an external reference (not internal #/...)
                    if not value.startswith("#/"):
                        # Resolve relative path
                        if "://" not in value:  # Skip URLs
                            ref_path = os.path.normpath(os.path.join(base_dir, value.split("#")[0]))
                            if os.path.exists(ref_path):
                                dependencies.add(ref_path)
                                # Recursively check dependencies of this file
                                sub_deps = self._get_spec_dependencies(ref_path, visited)
                                dependencies.update(sub_deps)
                elif isinstance(value, (dict, list)):
                    self._collect_external_refs(value, base_dir, dependencies, visited)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._collect_external_refs(item, base_dir, dependencies, visited)

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
        """Preprocess OpenAPI spec with multi-file $ref resolution."""
        # First, try Redocly bundle for automatic $ref resolution
        bundled_spec = self._try_redocly_bundle(filepath)
        if bundled_spec:
            return bundled_spec

        # Fallback: try jsonref for $ref resolution
        resolved_spec = self._try_jsonref_resolution(filepath)
        if resolved_spec:
            return resolved_spec

        # Final fallback: load file directly (single-file specs)
        return self._load_spec_direct(filepath)

    def _try_redocly_bundle(self, filepath: str) -> dict[str, Any] | None:
        """Try to use Redocly CLI bundle command to resolve $ref references."""
        if not self._is_redocly_available():
            log.debug("Redocly CLI not available for $ref resolution")
            return None

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
                temp_path = temp_file.name

            # Bundle the spec using Redocly - this resolves all $ref automatically
            bundle_command = ["redocly", "bundle", filepath, "--output", temp_path, "--format", "json"]
            timeout = self.agent.serena_config.openapi.redocly_timeout

            result = subprocess.run(bundle_command, check=False, capture_output=True, text=True, timeout=timeout)

            if result.returncode == 0:
                # Load the bundled spec with resolved references
                with open(temp_path) as f:
                    spec = json.load(f)

                # Optionally validate if validation is enabled
                if self.agent.serena_config.openapi.use_redocly_validation:
                    self._validate_with_redocly(temp_path)

                log.info("OpenAPI spec bundled with Redocly CLI (resolved $ref references)")
                return spec
            else:
                log.warning(f"Redocly bundle failed: {result.stderr}")
                return None

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning(f"Error during Redocly bundling: {e}")
            return None
        except Exception as e:
            log.warning(f"Unexpected error with Redocly bundling: {e}")
            return None
        finally:
            # Clean up temp file
            if temp_path:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    def _validate_with_redocly(self, spec_path: str) -> None:
        """Validate OpenAPI spec with Redocly CLI."""
        try:
            lint_command = ["redocly", "lint", spec_path]
            timeout = self.agent.serena_config.openapi.redocly_timeout
            result = subprocess.run(lint_command, check=False, capture_output=True, text=True, timeout=timeout)

            if result.returncode != 0:
                log.warning(f"Redocly validation warnings: {result.stderr}")
        except Exception as e:
            log.debug(f"Redocly validation failed: {e}")

    def _try_jsonref_resolution(self, filepath: str) -> dict[str, Any] | None:
        """Try to resolve $ref references using jsonref library."""
        try:
            import jsonref
        except ImportError:
            log.debug("jsonref library not available for $ref resolution")
            return None

        try:
            # Load the base spec
            with open(filepath, encoding="utf-8") as f:
                if filepath.endswith((".yaml", ".yml")):
                    if not YAML_AVAILABLE:
                        log.warning("PyYAML required for YAML $ref resolution")
                        return None
                    base_spec = yaml.safe_load(f)  # type: ignore[attr-defined]
                else:
                    base_spec = json.load(f)

            # Check if the spec contains any $ref references
            if not self._has_external_refs(base_spec):
                log.debug("No external $ref references found, using direct loading")
                return base_spec

            # Set up base URI for relative references
            base_uri = f"file://{os.path.dirname(os.path.abspath(filepath))}/"

            # Resolve all $ref references
            resolved_spec = jsonref.replace_refs(base_spec, base_uri=base_uri)

            # Convert back to regular dict (jsonref returns a proxy object)
            resolved_dict = json.loads(json.dumps(resolved_spec))

            log.info("OpenAPI spec processed with jsonref ($ref references resolved)")
            return resolved_dict

        except Exception as e:
            log.warning(f"Error during jsonref resolution: {e}")
            return None

    def _has_external_refs(self, spec: dict) -> bool:
        """Check if spec contains external $ref references."""

        def check_refs_recursive(obj: Any) -> bool:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "$ref" and isinstance(value, str):
                        # Check if it's an external reference (not internal #/...)
                        if not value.startswith("#/"):
                            return True
                    elif isinstance(value, (dict, list)):
                        if check_refs_recursive(value):
                            return True
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        if check_refs_recursive(item):
                            return True
            return False

        return check_refs_recursive(spec)

    def _load_spec_direct(self, filepath: str) -> dict[str, Any]:
        """Load OpenAPI spec directly without $ref resolution."""
        try:
            with open(filepath, encoding="utf-8") as f:
                if filepath.endswith((".yaml", ".yml")):
                    if not YAML_AVAILABLE:
                        raise ValueError("PyYAML is required to parse YAML files. Please install it or convert to JSON.")
                    return yaml.safe_load(f)  # type: ignore[attr-defined]
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

                # Add OpenAPI 3.x specific features

                # Add callbacks information
                callbacks = operation.get("callbacks", {})
                if callbacks:
                    callback_descriptions = []
                    for callback_name, callback_def in callbacks.items():
                        callback_descriptions.append(f"'{callback_name}' callback")
                        # Process callback operations
                        for expression, path_item in callback_def.items():
                            if isinstance(path_item, dict):
                                for cb_method, cb_operation in path_item.items():
                                    if isinstance(cb_operation, dict):
                                        cb_summary = cb_operation.get("summary", "")
                                        if cb_summary:
                                            callback_descriptions.append(f"  {cb_method.upper()} {expression}: {cb_summary}")

                    if callback_descriptions:
                        text_parts.append(f"Callbacks: {', '.join(callback_descriptions)}")

                # Add security requirements
                security = operation.get("security", [])
                if security:
                    security_schemes = []
                    for sec_req in security:
                        for scheme_name in sec_req.keys():
                            security_schemes.append(scheme_name)
                    if security_schemes:
                        text_parts.append(f"Security: {', '.join(security_schemes)}")

                # Add external documentation
                external_docs = operation.get("externalDocs")
                if external_docs:
                    ext_desc = external_docs.get("description", "")
                    ext_url = external_docs.get("url", "")
                    if ext_desc or ext_url:
                        text_parts.append(f"External Docs: {ext_desc} {ext_url}".strip())

                text_description = "\n".join(text_parts)

                chunk_metadata = {
                    "operationId": operation_id,
                    "path": path,
                    "method": method.upper(),
                    "summary": summary,
                    "description": description,
                    "tags": tags,
                    "type": "operation",
                }

                # Add OpenAPI 3.x metadata
                if callbacks:
                    chunk_metadata["callbacks"] = list(callbacks.keys())
                if security:
                    chunk_metadata["security"] = [list(req.keys()) for req in security]

                chunks.append(
                    {
                        "text": text_description,
                        "metadata": chunk_metadata,
                    }
                )

        # Process OpenAPI 3.x webhooks (OpenAPI 3.1+)
        webhooks = spec.get("webhooks", {})
        for webhook_name, webhook_def in webhooks.items():
            if not isinstance(webhook_def, dict):
                continue

            for method, operation in webhook_def.items():
                if method.startswith("x-") or not isinstance(operation, dict):
                    continue

                operation_id = operation.get("operationId", f"webhook_{webhook_name}_{method}")
                summary = operation.get("summary", "")
                description = operation.get("description", "")
                tags = operation.get("tags", [])

                text_parts = [
                    f"Webhook: {operation_id}",
                    f"Webhook Name: {webhook_name}",
                    f"HTTP Method: {method.upper()}",
                    "Type: Webhook/Event Notification",
                ]

                if summary:
                    text_parts.append(f"Summary: {summary}")

                if description:
                    text_parts.append(f"Description: {description}")

                if tags:
                    text_parts.append(f"Tags: {', '.join(tags)}")

                # Add webhook-specific information
                request_body = operation.get("requestBody")
                if request_body:
                    content_types = list(request_body.get("content", {}).keys())
                    if content_types:
                        text_parts.append(f"Webhook Payload Types: {', '.join(content_types)}")

                text_description = "\n".join(text_parts)

                chunks.append(
                    {
                        "text": text_description,
                        "metadata": {
                            "operationId": operation_id,
                            "webhookName": webhook_name,
                            "method": method.upper(),
                            "summary": summary,
                            "description": description,
                            "tags": tags,
                            "type": "webhook",
                        },
                    }
                )

        # Process components for additional context
        components = spec.get("components", {})

        # Process security schemes
        security_schemes = components.get("securitySchemes", {})
        for scheme_name, scheme_def in security_schemes.items():
            if not isinstance(scheme_def, dict):
                continue

            scheme_type = scheme_def.get("type", "")
            scheme_desc = scheme_def.get("description", "")

            text_parts = [
                f"Security Scheme: {scheme_name}",
                f"Type: {scheme_type}",
            ]

            if scheme_desc:
                text_parts.append(f"Description: {scheme_desc}")

            # Add scheme-specific details
            if scheme_type == "oauth2":
                flows = scheme_def.get("flows", {})
                if flows:
                    flow_types = list(flows.keys())
                    text_parts.append(f"OAuth2 Flows: {', '.join(flow_types)}")
            elif scheme_type == "apiKey":
                key_name = scheme_def.get("name", "")
                key_location = scheme_def.get("in", "")
                if key_name and key_location:
                    text_parts.append(f"API Key: {key_name} in {key_location}")

            text_description = "\\n".join(text_parts)

            chunks.append(
                {
                    "text": text_description,
                    "metadata": {
                        "name": scheme_name,
                        "type": "security_scheme",
                        "schemeType": scheme_type,
                        "description": scheme_desc,
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

    def _semantic_search(
        self,
        query: str,
        index_dir: str,
        k: int,
        method_filter: str | None = None,
        path_filter: str | None = None,
        tags_filter: list[str] | None = None,
        section_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform semantic search using the cached index."""
        import re

        # Load index and chunks
        index = faiss.read_index(os.path.join(index_dir, "index.faiss"))

        with open(os.path.join(index_dir, "chunks.json")) as f:
            chunks = json.load(f)

        # Apply filters to chunks before search
        if method_filter or path_filter or tags_filter or section_filter:
            filtered_indices = []
            for idx, chunk in enumerate(chunks):
                metadata = chunk["metadata"]

                # Section filter (filter by type of API element)
                if section_filter:
                    chunk_type = metadata.get("type", "operation")
                    if (section_filter == "operations" and chunk_type != "operation") or (
                        section_filter == "webhooks" and chunk_type != "webhook"
                    ):
                        continue
                    if section_filter == "security_schemes" and chunk_type != "security_scheme":
                        continue
                    if section_filter not in ["operations", "webhooks", "security_schemes"]:
                        # Custom section filter, check if it matches type or other metadata
                        if chunk_type != section_filter:
                            continue

                # Method filter (only applies to operations and webhooks)
                if method_filter and metadata.get("method"):
                    if metadata["method"].upper() != method_filter.upper():
                        continue

                # Path filter (regex support, only applies to operations)
                if path_filter and metadata.get("path"):
                    try:
                        if not re.search(path_filter, metadata["path"]):
                            continue
                    except re.error:
                        # Fallback to simple string matching if regex is invalid
                        if path_filter not in metadata["path"]:
                            continue

                # Tags filter (at least one tag must match)
                if tags_filter:
                    endpoint_tags = [tag.lower() for tag in metadata.get("tags", [])]
                    filter_tags = [tag.lower() for tag in tags_filter]
                    if not any(tag in endpoint_tags for tag in filter_tags):
                        continue

                filtered_indices.append(idx)

            if not filtered_indices:
                return []

            # Create filtered index if we have filters
            if len(filtered_indices) < len(chunks):
                # Get embeddings for filtered chunks
                all_embeddings = []
                for idx in range(len(chunks)):
                    embedding = index.reconstruct(idx)
                    all_embeddings.append(embedding)

                filtered_embeddings = np.array([all_embeddings[idx] for idx in filtered_indices]).astype("float32")

                # Create new index with filtered embeddings
                filtered_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
                filtered_index.add(filtered_embeddings)

                # Search filtered index
                query_embedding = self.model.encode([query], convert_to_tensor=False)
                query_embedding = np.array(query_embedding).astype("float32")
                faiss.normalize_L2(query_embedding)

                distances, indices = filtered_index.search(query_embedding, min(k, len(filtered_indices)))

                # Map back to original indices and return results
                results = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0], strict=False)):
                    if idx < len(filtered_indices):
                        original_idx = filtered_indices[idx]
                        result = chunks[original_idx].copy()
                        result["score"] = float(distance)
                        result["rank"] = i + 1
                        results.append(result)

                return results

        # No filters, use original search
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

    def _format_results(
        self,
        results: list[dict[str, Any]],
        query: str,
        multi_spec: bool = False,
        output_format: str = "human",
        include_examples: bool = False,
    ) -> str:
        """Format search results for LLM consumption."""
        if output_format == "json":
            return self._format_results_json(results, query, multi_spec, include_examples)
        elif output_format == "markdown":
            return self._format_results_markdown(results, query, multi_spec, include_examples)
        else:
            return self._format_results_human(results, query, multi_spec, include_examples)

    def _format_results_human(
        self, results: list[dict[str, Any]], query: str, multi_spec: bool = False, include_examples: bool = False
    ) -> str:
        """Format search results in human-readable format."""
        output = [f"Found {len(results)} relevant API endpoints for query: '{query}'\\n"]

        for result in results:
            metadata = result["metadata"]
            output.append(f"--- Rank {result['rank']} (Score: {result['score']:.3f}) ---")

            # Handle different types of results
            result_type = metadata.get("type", "operation")
            if result_type == "operation":
                output.append(f"Endpoint: {metadata['operationId']}")
                output.append(f"Method: {metadata['method']} {metadata['path']}")
            elif result_type == "webhook":
                output.append(f"Webhook: {metadata['operationId']}")
                output.append(f"Method: {metadata['method']} (Webhook: {metadata.get('webhookName', 'Unknown')})")
            elif result_type == "security_scheme":
                output.append(f"Security Scheme: {metadata.get('name', 'Unknown')}")
                output.append(f"Type: {metadata.get('schemeType', 'Unknown')}")
            else:
                # Fallback for unknown types
                output.append(f"Item: {metadata.get('operationId', metadata.get('name', 'Unknown'))}")
                if metadata.get("method") and metadata.get("path"):
                    output.append(f"Method: {metadata['method']} {metadata['path']}")
                elif metadata.get("method"):
                    output.append(f"Method: {metadata['method']}")
                elif metadata.get("schemeType"):
                    output.append(f"Type: {metadata['schemeType']}")

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

            if include_examples:
                example = self._generate_code_example(metadata)
                if example:
                    output.append(f"Example:\\n{example}")

            output.append("")  # Empty line between results

        output.append(
            "To use these endpoints, refer to the OpenAPI specification for detailed parameter requirements, request/response schemas, and authentication."
        )

        return "\\n".join(output)

    def _format_results_json(
        self, results: list[dict[str, Any]], query: str, multi_spec: bool = False, include_examples: bool = False
    ) -> str:
        """Format search results in JSON format."""
        formatted_results = []

        for result in results:
            metadata = result["metadata"]
            result_type = metadata.get("type", "operation")

            formatted_result = {
                "rank": result["rank"],
                "score": result["score"],
                "type": result_type,
                "summary": metadata.get("summary", ""),
                "description": metadata.get("description", ""),
                "tags": metadata.get("tags", []),
            }

            # Add type-specific fields
            if result_type == "operation":
                formatted_result.update(
                    {
                        "operationId": metadata["operationId"],
                        "method": metadata["method"],
                        "path": metadata["path"],
                    }
                )
            elif result_type == "webhook":
                formatted_result.update(
                    {
                        "operationId": metadata["operationId"],
                        "method": metadata["method"],
                        "webhookName": metadata.get("webhookName", ""),
                    }
                )
            elif result_type == "security_scheme":
                formatted_result.update(
                    {
                        "name": metadata.get("name", ""),
                        "schemeType": metadata.get("schemeType", ""),
                    }
                )
            else:
                # Fallback - include all available metadata
                formatted_result.update(
                    {
                        "operationId": metadata.get("operationId", ""),
                        "name": metadata.get("name", ""),
                        "method": metadata.get("method", ""),
                        "path": metadata.get("path", ""),
                    }
                )

            if multi_spec and "spec_file" in result:
                formatted_result["specification"] = os.path.basename(result["spec_file"])

            if include_examples:
                example = self._generate_code_example(metadata)
                if example:
                    formatted_result["example"] = example

            formatted_results.append(formatted_result)

        return json.dumps({"query": query, "total_results": len(results), "results": formatted_results}, indent=2)

    def _format_results_markdown(
        self, results: list[dict[str, Any]], query: str, multi_spec: bool = False, include_examples: bool = False
    ) -> str:
        """Format search results in Markdown format."""
        output = ["# API Search Results\\n", f"**Query:** {query}\\n", f"**Results:** {len(results)} endpoints found\\n"]

        for result in results:
            metadata = result["metadata"]
            result_type = metadata.get("type", "operation")

            if result_type == "operation":
                output.append(f"## {result['rank']}. {metadata['operationId']}")
                output.append(f"**Score:** {result['score']:.3f}\\n")
                output.append(f"**Method:** `{metadata['method']}`")
                output.append(f"**Path:** `{metadata['path']}`\\n")
            elif result_type == "webhook":
                output.append(f"## {result['rank']}. {metadata['operationId']} (Webhook)")
                output.append(f"**Score:** {result['score']:.3f}\\n")
                output.append(f"**Method:** `{metadata['method']}`")
                output.append(f"**Webhook:** `{metadata.get('webhookName', 'Unknown')}`\\n")
            elif result_type == "security_scheme":
                output.append(f"## {result['rank']}. {metadata.get('name', 'Unknown')} (Security Scheme)")
                output.append(f"**Score:** {result['score']:.3f}\\n")
                output.append(f"**Type:** `{metadata.get('schemeType', 'Unknown')}`\\n")
            else:
                # Fallback
                title = metadata.get("operationId", metadata.get("name", "Unknown"))
                output.append(f"## {result['rank']}. {title}")
                output.append(f"**Score:** {result['score']:.3f}\\n")
                if metadata.get("method") and metadata.get("path"):
                    output.append(f"**Method:** `{metadata['method']}`")
                    output.append(f"**Path:** `{metadata['path']}`\\n")
                elif metadata.get("schemeType"):
                    output.append(f"**Type:** `{metadata['schemeType']}`\\n")

            if multi_spec and "spec_file" in result:
                spec_name = os.path.basename(result["spec_file"])
                output.append(f"**Specification:** {spec_name}\\n")

            if metadata.get("summary"):
                output.append(f"**Summary:** {metadata['summary']}\\n")

            if metadata.get("description"):
                output.append(f"**Description:** {metadata['description']}\\n")

            if metadata.get("tags"):
                tags_formatted = ", ".join([f"`{tag}`" for tag in metadata["tags"]])
                output.append(f"**Tags:** {tags_formatted}\\n")

            if include_examples:
                example = self._generate_code_example(metadata)
                if example:
                    output.append(f"**Example:**\\n```bash\\n{example}\\n```\\n")

            output.append("---\\n")

        return "\\n".join(output)

    def _generate_code_example(self, metadata: dict[str, Any]) -> str:
        """Generate a code example for the API endpoint."""
        result_type = metadata.get("type", "operation")

        if result_type == "security_scheme":
            # No code example for security schemes
            return ""

        method = metadata.get("method", "")
        path = metadata.get("path", "")

        if not method:
            return ""

        if result_type == "webhook":
            # For webhooks, show example webhook payload
            return f"# Webhook endpoint expects {method.upper()} requests\n# Example webhook payload structure"

        if not path:
            return f"# {method.upper()} request"

        # Simple curl example for operations
        if method.upper() == "GET":
            return f"curl -X {method.upper()} '{path}'"
        elif method.upper() in ["POST", "PUT", "PATCH"]:
            return f"curl -X {method.upper()} '{path}' \\\\\n  -H 'Content-Type: application/json' \\\\\n  -d '{{\"data\": \"example\"}}'"
        elif method.upper() == "DELETE":
            return f"curl -X {method.upper()} '{path}'"
        else:
            return f"curl -X {method.upper()} '{path}'"

    def list_project_specs(self) -> list[str]:
        """List all OpenAPI specifications configured for the current project."""
        try:
            project = self.agent.get_active_project()
            if project is None:
                return self._auto_discover_specs(os.getcwd())

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
            if project is None:
                return "Error: No active project. Cannot add specification to project configuration."

            project_root = project.project_root

            # Convert to relative path
            if os.path.isabs(spec_path):
                try:
                    spec_rel_path = os.path.relpath(spec_path, project_root)
                    # Check if the relative path goes outside project root
                    if spec_rel_path.startswith("../") or spec_rel_path == "..":
                        return f"Error: Specification path is outside project root: {spec_path}"
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
