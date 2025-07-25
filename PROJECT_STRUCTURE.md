# Enhanced RAG System - Project Structure

## Essential Files and Directories

### Core Application Code
```
enhanced_rag/                      # Main application package
├── __init__.py                    # Package initialization
├── api/                          # API layer
│   ├── __init__.py
│   ├── cli.py                    # Command-line interface
│   ├── rest.py                   # REST API endpoints
│   └── schemas.py                # Pydantic data models
├── core/                         # Core functionality
│   ├── __init__.py
│   └── chunking.py               # Text chunking algorithms
├── interfaces/                   # Abstract interfaces
│   ├── __init__.py
│   ├── document_loader.py        # Document loading interface
│   ├── embedder.py               # Embedding interface
│   ├── llm_client.py             # LLM client interface
│   └── vector_store.py           # Vector store interface
├── pipeline.py                   # Main RAG pipeline
└── utils/                        # Utility modules
    ├── __init__.py
    ├── config.py                 # Configuration management
    ├── correlation.py            # Request correlation
    ├── logging.py                # Logging utilities
    ├── metrics.py                # Metrics collection
    └── retry.py                  # Retry logic
```

### Configuration
```
config/                           # Configuration files
├── config.yaml                   # Main configuration
├── domains.yaml                  # Domain-specific settings
├── local.yaml                    # Local development config
├── production.yaml               # Production config
├── prompts.yaml                  # LLM prompts
└── environments/                 # Environment-specific configs
```

### Testing
```
tests/                            # Test suite (79.76% coverage)
└── unit/                         # Unit tests
    ├── __init__.py
    ├── test_chunking.py          # Core chunking tests
    ├── test_cli_basic.py         # CLI tests
    ├── test_config.py            # Configuration tests
    ├── test_correlation.py       # Correlation tests
    ├── test_interfaces.py        # Interface tests
    ├── test_logging.py           # Logging tests
    ├── test_metrics.py           # Metrics tests
    ├── test_pipeline.py          # Pipeline tests
    ├── test_rest_api.py          # REST API tests
    ├── test_retry.py             # Retry logic tests
    └── test_schemas.py           # Schema validation tests
```

### Deployment
```
docker/                          # Docker configuration
├── Dockerfile                   # Container definition
└── docker-compose.yml          # Multi-service orchestration
```

### Project Management
```
pyproject.toml                   # Python project configuration
requirements.txt                 # Python dependencies
README.md                        # Project documentation
CLAUDE.md                        # Development notes
.gitignore                       # Git ignore rules
```

### Sample Data
```
data/                            # Data directory
└── samples/                     # Sample documents for testing
    ├── sample1.txt
    ├── sample2.txt
    ├── sample3.txt
    └── test_document.txt
```

### Scripts
```
scripts/                         # Utility scripts
├── setup.py                     # Setup script
└── test_setup.py               # Test setup
```

## Removed Files (Cleaned Up)

The following files were removed during cleanup:
- `htmlcov/` - HTML coverage reports (can be regenerated)
- `enhanced_rag.egg-info/` - Build artifacts (auto-generated)
- `.pytest_cache/` - Pytest cache (auto-generated)
- `logs/` - Empty log directory (created at runtime)
- `data/cache/` - Empty cache directory (created at runtime)
- `data/vector_index/` - Empty vector index directory (created at runtime)
- `tmp/` - Temporary files
- `current_config.yaml` - Temporary config file
- `tests/unit/test_abstract_coverage.py` - Temporary test file

## Key Features

- **79.76% Test Coverage** with 309 comprehensive tests
- **Production-ready** with Docker support
- **Modular architecture** with clean interfaces
- **Configuration management** for different environments
- **Comprehensive logging and metrics**
- **CLI and REST API** interfaces
- **Type-safe** with Pydantic schemas

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `pytest tests/`
3. Start API: `python -m enhanced_rag.api.rest`
4. Use CLI: `python -m enhanced_rag.api.cli --help`