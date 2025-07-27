# OpenAPI Integration with Serena MCP for Monorepos

This guide explains how to set up and use Serena's enhanced OpenAPI functionality with Claude Code for large monorepos containing multi-file API specifications.

## Overview

The enhanced Serena MCP server provides powerful semantic search capabilities over OpenAPI specifications, allowing Claude Code to:

- Perform natural language searches across API endpoints
- Find relevant schemas, parameters, and responses
- Support multi-file OpenAPI specifications with `$ref` resolution
- Filter by HTTP methods, paths, tags, and sections
- Generate contextual code examples and documentation

## 1. Installation and Setup

### Install Enhanced Serena

```bash
# In your monorepo root
cd /path/to/your/monorepo

# Install the enhanced Serena (from source)
pip install -e /path/to/serena-openapi

# Or from PyPI (when published)
pip install serena-mcp-server
```

### Verify Installation

```bash
# Check that OpenAPI tools are available
serena-mcp-server --help
```

## 2. Project Configuration

### Create Serena Configuration

Create a `.serena/project.yml` file in your monorepo root:

```yaml
# .serena/project.yml
project_name: "your-monorepo"

# Language server configuration
language_servers:
  python: true
  typescript: true
  java: true
  # Add other languages as needed

# OpenAPI specifications (relative to project root)
openapi_specs:
  - "openapi/openapi.yaml"              # Main multi-file spec
  - "backend/user-service/api.yaml"     # Service-specific specs
  - "backend/payment-service/api.yaml"
  - "backend/order-service/api.yaml"

# OpenAPI-specific configuration
openapi:
  # Embedding model for semantic search
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  
  # Cache directory for search indices
  index_cache_dir: ".serena/openapi_cache"
  
  # Enable Redocly for spec validation and bundling
  use_redocly_validation: true
  
  # Auto-discovery patterns for finding specs
  auto_discovery_patterns:
    - "*/openapi.yaml"
    - "*/api-spec.yaml"
    - "services/*/openapi/*.yaml"
    - "api/*/openapi.yaml"
```

### Directory Structure Example

```
your-monorepo/
├── .serena/
│   ├── project.yml          # Main configuration
│   └── openapi_cache/       # Auto-generated search indices
├── openapi/                 # Main API specification
│   ├── openapi.yaml         # Root spec file
│   ├── components/
│   │   ├── schemas/
│   │   │   ├── user.yaml
│   │   │   ├── order.yaml
│   │   │   └── common.yaml
│   │   ├── responses/
│   │   │   ├── errors.yaml
│   │   │   └── success.yaml
│   │   ├── parameters/
│   │   │   └── common.yaml
│   │   └── security/
│   │       └── schemes.yaml
│   └── paths/
│       ├── users.yaml
│       ├── orders.yaml
│       ├── products.yaml
│       └── payments.yaml
├── backend/
│   ├── user-service/
│   │   ├── src/...
│   │   └── api.yaml         # Service-specific API
│   ├── payment-service/
│   │   ├── src/...
│   │   └── api.yaml
│   └── order-service/
│       ├── src/...
│       └── api.yaml
├── frontend/
│   ├── src/...
│   └── package.json
└── README.md
```

## 3. Starting the MCP Server

### Basic Setup

```bash
# Start from your monorepo root
cd /path/to/your/monorepo
serena-mcp-server
```

### Advanced Options

```bash
# With custom configuration
serena-mcp-server --project-root /path/to/your/monorepo --host localhost --port 8080

# With debug logging
serena-mcp-server --log-level DEBUG

# Force rebuild of search indices
serena-mcp-server --rebuild-indices

# List all discovered OpenAPI specs
serena-mcp-server --list-specs
```

## 4. Claude Code Integration

### MCP Configuration

Add this to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "serena": {
      "command": "serena-mcp-server",
      "args": [
        "--project-root", "/path/to/your/monorepo",
        "--log-level", "INFO"
      ],
      "env": {
        "PYTHONPATH": "/path/to/your/monorepo"
      }
    }
  }
}
```

### Alternative: Direct Connection

```bash
# Start server with specific port
serena-mcp-server --port 3001

# Connect Claude Code to localhost:3001
```

## 5. Usage Examples

Once connected, you can use natural language queries with Claude Code:

### API Discovery and Exploration

```
"Show me all endpoints that handle user authentication"
```

```
"Find endpoints related to order processing and payment"
```

```
"What are all the available user management operations?"
```

### Schema and Data Model Exploration

```
"Show me the schema for creating a new user"
```

```
"Find all endpoints that return paginated results"
```

```
"What are the required fields for the Order object?"
```

### Implementation Assistance

```
"Find the endpoint for updating user profiles and show me the request format"
```

```
"Show me all POST endpoints with their request bodies and examples"
```

```
"What authentication methods are supported by the payment endpoints?"
```

### Filtering and Advanced Search

```
"Find all GET endpoints in the users path with examples"
```

```
"Show me endpoints tagged with 'admin' that require authentication"
```

```
"Find all endpoints that accept file uploads"
```

### Multi-Specification Search

```
"Search across all services for endpoints that handle notifications"
```

```
"Find webhook definitions across all API specifications"
```

## 6. Available Tools and Parameters

### openapi_search Tool

Primary tool for semantic search with these parameters:

- `spec_path`: Specific OpenAPI file (optional, auto-discovers if not provided)
- `query`: Natural language query (required)
- `top_k`: Number of results to return (default: 3)
- `rebuild_index`: Force rebuild search index (default: false)
- `search_all_specs`: Search across all configured specs (default: false)
- `output_format`: "human", "json", or "markdown" (default: "human")
- `include_examples`: Include code examples (default: false)
- `method_filter`: Filter by HTTP method (GET, POST, etc.)
- `path_filter`: Filter by path pattern (supports regex)
- `tags_filter`: Filter by endpoint tags
- `section_filter`: Filter by section type ("operations", "webhooks", "security_schemes")

### add_spec_to_project Tool

Add new OpenAPI specifications to your project:

```
"Add the new analytics API spec at backend/analytics/api.yaml to the project"
```

### list_openapi_specs Tool

List all configured OpenAPI specifications:

```
"Show me all the API specifications configured in this project"
```

## 7. Advanced Configuration

### High-Performance Setup

For large monorepos with extensive API specifications:

```yaml
# .serena/project.yml
openapi:
  # Use more powerful embedding model for better search quality
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
  
  # Optimize cache settings
  index_cache_dir: ".serena/openapi_cache"
  
  # Enable Redocly for better multi-file handling
  use_redocly_validation: true
```

### Custom Discovery Patterns

```yaml
openapi:
  auto_discovery_patterns:
    - "*/openapi.yaml"
    - "*/api-spec.yaml"
    - "services/*/docs/api.yaml"
    - "microservices/*/openapi/*.yaml"
    - "api/v*/openapi.yaml"
```

### Environment-Specific Configuration

```yaml
# .serena/project.yml
contexts:
  development:
    openapi_specs:
      - "openapi/dev/openapi.yaml"
  
  production:
    openapi_specs:
      - "openapi/prod/openapi.yaml"
```

## 8. Best Practices

### Specification Organization

1. **Use consistent file naming**: `openapi.yaml` or `api.yaml`
2. **Organize by domain**: Group related endpoints in separate files
3. **Use meaningful tags**: Tag endpoints for better filtering
4. **Include examples**: Provide request/response examples
5. **Document thoroughly**: Use descriptions for all operations

### Performance Optimization

1. **Cache management**: Indices are automatically cached and updated
2. **Incremental updates**: Only changed specs trigger index rebuilds
3. **Selective searching**: Use filters to narrow search scope
4. **Batch operations**: Configure multiple specs in project.yml

### Multi-Team Workflows

1. **Service ownership**: Each service maintains its own OpenAPI spec
2. **Central discovery**: Main project.yml references all service specs
3. **Consistent standards**: Use shared schemas and components
4. **Version management**: Include API versions in file paths

## 9. Troubleshooting

### Common Issues

**Specs not found:**
```bash
# Check discovery patterns
serena-mcp-server --list-specs

# Verify file paths are relative to project root
ls openapi/openapi.yaml
```

**Search returns no results:**
```bash
# Rebuild search indices
serena-mcp-server --rebuild-indices

# Check spec validity
# Install redocly: npm install -g @redocly/cli
redocly lint openapi/openapi.yaml
```

**Performance issues:**
```bash
# Check index status
ls -la .serena/openapi_cache/

# Monitor memory usage
serena-mcp-server --log-level DEBUG
```

### Debug Commands

```bash
# Enable debug logging
export SERENA_LOG_LEVEL=DEBUG
serena-mcp-server

# Check MCP connection
curl -X POST http://localhost:3001/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/list"}'

# Validate OpenAPI specs
redocly lint openapi/**/*.yaml
```

## 10. Migration from Existing Setup

### From Swagger/OpenAPI 2.0

```bash
# Convert specs to OpenAPI 3.0+
npx swagger2openapi swagger.yaml -o openapi.yaml
```

### From Postman Collections

```bash
# Convert Postman to OpenAPI
npx p2o collection.json -f openapi.yaml
```

### From API Documentation Tools

Most API documentation tools can export OpenAPI 3.0 specifications. Place exported files in your project structure and configure them in `.serena/project.yml`.

## 11. Integration Examples

### With CI/CD Pipelines

```yaml
# .github/workflows/api-validation.yml
name: API Validation
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Validate OpenAPI specs
        run: |
          npm install -g @redocly/cli
          redocly lint openapi/**/*.yaml
      - name: Test Serena integration
        run: |
          pip install -e .
          serena-mcp-server --list-specs --project-root .
```

### With Development Environments

```yaml
# docker-compose.yml
version: '3.8'
services:
  serena-mcp:
    build: .
    ports:
      - "3001:3001"
    volumes:
      - .:/workspace
    command: serena-mcp-server --project-root /workspace
    environment:
      - SERENA_LOG_LEVEL=INFO
```

This setup provides Claude Code with powerful capabilities to understand, explore, and work with your API specifications in the context of your entire monorepo, making API development and integration much more efficient.