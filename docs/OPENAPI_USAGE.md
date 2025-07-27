# OpenAPI Integration Usage Guide

This guide provides comprehensive documentation for using Serena's OpenAPI integration features to search, explore, and understand API specifications.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [CLI Commands](#cli-commands)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

Serena's OpenAPI integration provides semantic search capabilities over OpenAPI specifications, enabling developers to:

- **Semantic Search**: Find relevant API endpoints using natural language queries
- **Multi-File Support**: Full support for OpenAPI specifications split across multiple files with `$ref` resolution
- **Section Filtering**: Search within specific API sections (operations, webhooks, security schemes)
- **Advanced Filtering**: Filter by HTTP methods, paths, tags, and more
- **Multiple Formats**: Output results in human-readable, JSON, or Markdown formats
- **Multi-Spec Support**: Search across multiple OpenAPI specifications
- **Auto-Discovery**: Automatically find OpenAPI specs in your project

## Quick Start

### 1. Basic Setup

Add OpenAPI specifications to your project configuration:

```yaml
# .serena/project.yml
openapi_specs:
  - "api/petstore.yaml"
  - "api/blog.json"
  - "docs/openapi.yaml"
```

### 2. First Search

```bash
# Search for pet-related endpoints
serena openapi search "list all pets in the store" --spec-path api/petstore.yaml

# Search across all configured specs
serena openapi search-all "user authentication"
```

### 3. Interactive Mode

```python
# In Serena agent mode
openapi_search("create new blog post", top_k=3)
```

## Configuration

### Global Configuration

Configure OpenAPI settings in your Serena configuration:

```yaml
# ~/.serena/serena_config.yml
openapi:
  embedding_model: "all-MiniLM-L6-v2"  # Sentence transformer model
  index_cache_dir: ".serena/openapi_cache"  # Cache directory
  use_redocly_validation: false  # Enable Redocly CLI validation
  redocly_timeout: 30  # Validation timeout in seconds
```

### Project Configuration

Configure specs per project:

```yaml
# .serena/project.yml
openapi_specs:
  - "specs/api-v1.yaml"
  - "specs/api-v2.yaml"
  - "docs/webhooks.json"
```

### Auto-Discovery

Serena automatically discovers OpenAPI specs with these patterns:
- `openapi.{yaml,yml,json}`
- `swagger.{yaml,yml,json}`
- `api.{yaml,yml,json}`
- Files in `docs/`, `specs/`, `api/` directories matching `*.{yaml,yml,json}`

### Multi-File Specifications

Serena fully supports OpenAPI specifications split across multiple files using `$ref` references:

```
project/
├── openapi.yaml          # Main specification
├── components/
│   ├── schemas.yaml       # Referenced schemas
│   ├── parameters.yaml    # Common parameters  
│   └── responses.yaml     # Shared responses
└── paths/
    ├── users.yaml         # User-related endpoints
    └── products.yaml      # Product-related endpoints
```

**Automatic Reference Resolution**: Serena automatically resolves `$ref` references using:
1. **Redocly CLI** (preferred): Uses `redocly bundle` for professional-grade resolution
2. **jsonref library** (fallback): Python-based reference resolution
3. **Direct loading** (final fallback): Single-file specifications

**Smart Caching**: Cache invalidation tracks all referenced files - when any dependency changes, the cache is automatically rebuilt.

## Basic Usage

### Semantic Search

Find endpoints using natural language:

```bash
# Find user management endpoints
serena openapi search "manage user accounts and profiles"

# Find payment processing
serena openapi search "process payments and handle billing"

# Find data retrieval operations
serena openapi search "get list of items from database"
```

### Method Filtering

Filter by HTTP methods:

```bash
# Only GET endpoints
serena openapi search "user data" --method GET

# Only POST and PUT endpoints
serena openapi search "create or update" --method POST --method PUT
```

### Path Filtering

Filter by URL path patterns:

```bash
# Endpoints under /api/v1/users
serena openapi search "user operations" --path "/api/v1/users.*"

# All admin endpoints
serena openapi search "administration" --path ".*/admin/.*"
```

### Tag Filtering

Filter by OpenAPI tags:

```bash
# Only user-related endpoints
serena openapi search "data management" --tags users --tags profiles

# Authentication endpoints
serena openapi search "login" --tags auth --tags security
```

## Advanced Features

### Section-Based Search

Search within specific API sections:

```bash
# Search only in regular operations
serena openapi search "user management" --section operations

# Search only in webhooks
serena openapi search "event notifications" --section webhooks

# Search only in security schemes
serena openapi search "authentication methods" --section security_schemes
```

### Output Formats

#### Human-Readable (default)
```bash
serena openapi search "create user" --output-format human
```

#### JSON Format
```bash
serena openapi search "create user" --output-format json
```

#### Markdown Format
```bash
serena openapi search "create user" --output-format markdown
```

### Including Examples

Generate curl examples for endpoints:

```bash
serena openapi search "create user" --include-examples --output-format markdown
```

### Multi-Specification Search

Search across all configured specifications:

```bash
# Search all specs in project
serena openapi search-all "authentication flows"

# Rebuild all indices
serena openapi search-all "user data" --rebuild-index
```

## CLI Commands

### `serena openapi search`

Search a single OpenAPI specification:

```bash
serena openapi search QUERY [OPTIONS]
```

**Options:**
- `--spec-path PATH`: Path to OpenAPI specification
- `--top-k INTEGER`: Number of results to return (default: 3)
- `--method METHOD`: Filter by HTTP method
- `--path PATTERN`: Filter by path regex pattern
- `--tags TAG`: Filter by tags (can be used multiple times)
- `--section SECTION`: Filter by API section (operations/webhooks/security_schemes)
- `--output-format FORMAT`: Output format (human/json/markdown)
- `--include-examples`: Include curl examples
- `--rebuild-index`: Force rebuild of search index

### `serena openapi search-all`

Search across all configured specifications:

```bash
serena openapi search-all QUERY [OPTIONS]
```

**Options:** Same as `search` command (except `--spec-path`)

### `serena openapi list`

List available OpenAPI specifications:

```bash
serena openapi list [OPTIONS]
```

**Options:**
- `--show-endpoints`: Show endpoint counts for each spec

## API Reference

### OpenAPI Tool Methods

#### `apply()`

Main search method:

```python
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
```

**Parameters:**
- `spec_path`: Path to specific OpenAPI spec (optional if using search_all_specs)
- `query`: Search query string
- `top_k`: Maximum number of results to return
- `rebuild_index`: Force rebuild of search index
- `search_all_specs`: Search across all configured specs
- `output_format`: Output format ("human", "json", "markdown")
- `include_examples`: Include curl examples in results
- `method_filter`: Filter by HTTP method
- `path_filter`: Filter by path regex pattern
- `tags_filter`: Filter by OpenAPI tags
- `section_filter`: Filter by API section type

## Examples

### Example 1: Basic API Exploration

```bash
# Discover what endpoints are available for user management
serena openapi search "user management operations" --top-k 5

# Find all GET endpoints for retrieving data
serena openapi search "retrieve data" --method GET --top-k 10
```

### Example 2: Documentation Generation

```bash
# Generate markdown documentation for authentication
serena openapi search "authentication and authorization" \
  --output-format markdown \
  --include-examples \
  --section security_schemes \
  --top-k 5

# Generate JSON for integration with other tools
serena openapi search "payment processing" \
  --output-format json \
  --tags payments \
  --top-k 3
```

### Example 3: Multi-File Specification Usage

```bash
# Search in multi-file OpenAPI specification
serena openapi search "user authentication endpoints" \
  --spec-path api/openapi.yaml \
  --section operations

# The tool automatically resolves all $ref references across:
# - api/components/schemas.yaml
# - api/components/security.yaml  
# - api/paths/users.yaml
# - And any other referenced files
```

### Example 4: API Migration Planning

```bash
# Compare CRUD operations across specs
serena openapi search-all "create read update delete operations" \
  --output-format json \
  --top-k 20

# Find all webhook configurations
serena openapi search-all "webhook events and callbacks" \
  --section webhooks \
  --output-format markdown
```

### Example 5: Development Workflow

```bash
# Find specific endpoint for implementation
serena openapi search "upload file to storage" \
  --method POST \
  --include-examples

# Explore API capabilities by domain
serena openapi search "shopping cart and checkout" \
  --tags cart \
  --tags checkout \
  --output-format markdown
```

### Example 6: Debugging and Troubleshooting

```bash
# Find error handling patterns
serena openapi search "error responses and status codes" \
  --top-k 10

# Look for rate limiting information
serena openapi search "rate limiting and throttling" \
  --section security_schemes
```

## Troubleshooting

### Common Issues

#### 1. No Specifications Found

**Problem:** `Error: No OpenAPI specifications found`

**Solutions:**
- Verify spec file paths in `.serena/project.yml`
- Check that OpenAPI files exist and are readable
- Use `--spec-path` to specify exact file location
- Run `serena openapi list` to see discovered specs

#### 2. Empty Search Results

**Problem:** `No relevant endpoints found for query`

**Solutions:**
- Try broader search terms
- Remove restrictive filters (method, path, tags)
- Check if the spec contains the functionality you're looking for
- Use `--rebuild-index` to refresh the search index

#### 3. Slow Performance

**Problem:** Search takes too long

**Solutions:**
- Ensure indices are built (first search will be slower)
- Use more specific queries to reduce result processing
- Consider using smaller `top_k` values
- Check available system memory

#### 4. Index Corruption

**Problem:** Search returns errors or unexpected results

**Solutions:**
- Use `--rebuild-index` to recreate search indices
- Clear cache directory: `rm -rf .serena/openapi_cache`
- Verify OpenAPI spec syntax with external tools

#### 5. Multi-File Reference Issues

**Problem:** `$ref` references not resolving correctly

**Solutions:**
- Verify all referenced files exist and are accessible
- Check relative path accuracy in `$ref` statements
- Install Redocly CLI for best resolution: `npm install -g @redocly/cli`
- Enable debug logging to see resolution process: `--log-level DEBUG`
- Verify file permissions on all referenced files

#### 6. Unicode/Encoding Issues

**Problem:** Special characters not displaying correctly

**Solutions:**
- Ensure OpenAPI specs are saved with UTF-8 encoding
- Use proper terminal encoding settings
- Check that embedding model supports the language

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
serena openapi search "query" --log-level DEBUG
```

### Performance Tips

1. **Index Management**: Indices are cached automatically. Use `--rebuild-index` only when specs change.

2. **Query Optimization**: More specific queries return better results faster.

3. **Filtering**: Use filters to narrow results and improve relevance.

4. **Batch Operations**: When exploring multiple aspects, use `search-all` for better performance.

### Getting Help

- Check the implementation plan: `docs/IMPLEMENTATION_PLAN.md`
- Review test examples in `test/serena/openapi/`
- Use `serena openapi --help` for command reference
- Enable debug logging to understand processing flow

## Advanced Configuration

### Custom Embedding Models

Configure different sentence transformer models:

```yaml
openapi:
  embedding_model: "all-mpnet-base-v2"  # Higher quality, slower
  # or
  embedding_model: "all-MiniLM-L12-v2"  # Balanced performance
```

### Cache Management

Control index caching behavior:

```yaml
openapi:
  index_cache_dir: "/custom/cache/path"  # Custom cache location
```

Clear cache manually:
```bash
rm -rf .serena/openapi_cache
```

### Redocly Integration

Enable OpenAPI validation with Redocly CLI:

```yaml
openapi:
  use_redocly_validation: true
  redocly_timeout: 60
```

Requires Redocly CLI installation:
```bash
npm install -g @redocly/cli
```