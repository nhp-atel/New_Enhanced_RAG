#!/usr/bin/env python3
"""Test script to verify Enhanced RAG system setup."""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import enhanced_rag
        print("  ✅ enhanced_rag module")
        
        from enhanced_rag.utils.config import ConfigManager
        print("  ✅ Configuration manager")
        
        from enhanced_rag.core.chunking import ChunkerFactory, ChunkingStrategy
        print("  ✅ Text chunking")
        
        from enhanced_rag.interfaces import Embedder, VectorStore, LLMClient
        print("  ✅ Abstract interfaces")
        
        from enhanced_rag.api.cli import cli
        print("  ✅ CLI interface")
        
        from enhanced_rag.utils.logging import setup_logging
        print("  ✅ Logging utilities")
        
        from enhanced_rag.utils.metrics import RAGMetrics
        print("  ✅ Metrics collection")
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\n⚙️ Testing configuration...")
    
    try:
        from enhanced_rag.utils.config import ConfigManager
        
        config_manager = ConfigManager("./config")
        config = config_manager.load_config("local")
        
        print(f"  ✅ Environment: {config.environment}")
        print(f"  ✅ Chunk size: {config.chunking.chunk_size}")
        print(f"  ✅ LLM provider: {config.llm.provider}")
        print(f"  ✅ Embedding provider: {config.embedding.provider}")
        
        print("✅ Configuration loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_text_chunking():
    """Test text chunking functionality."""
    print("\n📝 Testing text chunking...")
    
    try:
        from enhanced_rag.core.chunking import chunk_text, ChunkingStrategy
        
        sample_text = """
        This is a sample document for testing the Enhanced RAG system.
        It contains multiple sentences and paragraphs to test the chunking functionality.
        
        The system should be able to split this text into appropriate chunks
        while maintaining semantic coherence and respecting the configured overlap.
        
        This is the final paragraph of our test document.
        """
        
        chunks = chunk_text(
            sample_text,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=200,
            overlap=50
        )
        
        print(f"  ✅ Created {len(chunks)} chunks")
        print(f"  ✅ First chunk: {chunks[0].content[:50]}...")
        print(f"  ✅ Chunk strategy: {chunks[0].metadata['strategy']}")
        
        print("✅ Text chunking successful!")
        return True
        
    except Exception as e:
        print(f"❌ Chunking error: {e}")
        return False


def test_environment_variables():
    """Test environment variable setup."""
    print("\n🔑 Testing environment variables...")
    
    api_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
    }
    
    found_keys = []
    for key, provider in api_keys.items():
        if os.getenv(key):
            print(f"  ✅ {provider} API key found")
            found_keys.append(provider)
        else:
            print(f"  ⚠️ {provider} API key not set")
    
    if found_keys:
        print(f"✅ Found API keys for: {', '.join(found_keys)}")
        return True
    else:
        print("⚠️ No API keys found. Set at least one in .env file")
        return False


def test_directories():
    """Test that required directories exist."""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "data",
        "data/vector_index",
        "tmp",
        "logs",
        "config"
    ]
    
    all_exist = True
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ (missing)")
            all_exist = False
    
    if all_exist:
        print("✅ All required directories exist!")
    else:
        print("⚠️ Some directories are missing. Run 'python scripts/setup.py'")
    
    return all_exist


def test_cli_commands():
    """Test CLI command availability."""
    print("\n🖥️ Testing CLI commands...")
    
    try:
        import subprocess
        
        # Test CLI help command
        result = subprocess.run(
            [sys.executable, "-m", "enhanced_rag.api.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("  ✅ CLI help command works")
            
            # Check for expected commands
            if "ingest" in result.stdout and "query" in result.stdout:
                print("  ✅ Core commands available (ingest, query)")
            else:
                print("  ⚠️ Some commands may be missing")
                
            print("✅ CLI interface working!")
            return True
        else:
            print(f"  ❌ CLI error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False


def create_sample_document():
    """Create a sample document for testing."""
    sample_dir = Path("data/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = sample_dir / "test_document.txt"
    
    if not sample_file.exists():
        sample_content = """
Enhanced RAG System Test Document

This is a sample document created for testing the Enhanced RAG system.

Introduction
The Enhanced RAG (Retrieval-Augmented Generation) system is designed to help you 
build intelligent document search and question-answering applications.

Key Features
- Modular architecture with pluggable components
- Configuration-driven design with no hardcoded values
- Production-ready with comprehensive monitoring
- Support for multiple document formats
- REST API and CLI interfaces

Getting Started
To get started with the Enhanced RAG system:
1. Install the dependencies
2. Configure your API keys
3. Load your documents
4. Start asking questions

Technical Details
The system uses vector embeddings to understand document content and retrieve
relevant information to answer user questions. It supports various embedding
providers and LLM models.

Conclusion
This Enhanced RAG system provides a robust foundation for building AI-powered
document analysis and question-answering applications.
"""
        
        with open(sample_file, 'w') as f:
            f.write(sample_content.strip())
        
        print(f"  ✅ Created sample document: {sample_file}")
        return str(sample_file)
    else:
        print(f"  ✅ Sample document already exists: {sample_file}")
        return str(sample_file)


def main():
    """Run all setup tests."""
    print("🚀 Enhanced RAG System Setup Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Text Chunking", test_text_chunking),
        ("Environment Variables", test_environment_variables),
        ("Directory Structure", test_directories),
        ("CLI Commands", test_cli_commands),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Create sample document
    print("\n📄 Creating sample document...")
    sample_file = create_sample_document()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Enhanced RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Set up your API keys in .env file")
        print("2. Try ingesting the sample document:")
        print(f"   enhanced-rag ingest {sample_file}")
        print("3. Ask a question:")
        print("   enhanced-rag query 'What is the Enhanced RAG system?'")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")
        print("Run 'python scripts/setup.py' to fix common issues.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)