#!/usr/bin/env python3
"""Setup script for Enhanced RAG System."""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 11):
        logger.error("Python 3.11 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")


def install_dependencies():
    """Install Python dependencies."""
    logger.info("Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True)
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        
        # Install in development mode
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], 
                      check=True)
        
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit(1)


def create_directories():
    """Create necessary directories."""
    logger.info("Creating directories...")
    
    directories = [
        "data",
        "data/vector_index", 
        "data/cache",
        "tmp",
        "logs",
        "config/environments"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def setup_configuration():
    """Setup configuration files."""
    logger.info("Setting up configuration...")
    
    # Create environment-specific config files if they don't exist
    base_config_path = Path("config/config.yaml")
    
    if not base_config_path.exists():
        logger.error("Base configuration file not found: config/config.yaml")
        sys.exit(1)
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create local config with development settings
    local_config = base_config.copy()
    local_config['environment'] = 'local'
    local_config['observability']['log_level'] = 'DEBUG'
    local_config['caching']['redis_url'] = None  # Use in-memory cache
    
    local_config_path = Path("config/local.yaml")
    if not local_config_path.exists():
        with open(local_config_path, 'w') as f:
            yaml.dump(local_config, f, default_flow_style=False, indent=2)
        logger.info("Created local configuration: config/local.yaml")


def setup_git_hooks():
    """Setup git pre-commit hooks."""
    logger.info("Setting up git hooks...")
    
    if not Path(".git").exists():
        logger.warning("Not a git repository, skipping git hooks")
        return
    
    try:
        subprocess.run(["pre-commit", "install"], check=True)
        logger.info("Git hooks installed successfully")
    except subprocess.CalledProcessError:
        logger.warning("Failed to install git hooks")
    except FileNotFoundError:
        logger.warning("pre-commit not found, skipping git hooks")


def check_optional_dependencies():
    """Check for optional dependencies and provide installation instructions."""
    logger.info("Checking optional dependencies...")
    
    optional_deps = {
        "faiss-gpu": {
            "package": "faiss-gpu",
            "description": "GPU-accelerated FAISS for faster vector operations",
            "install": "pip install faiss-gpu"
        },
        "torch": {
            "package": "torch",
            "description": "PyTorch for advanced ML models",
            "install": "pip install torch torchvision torchaudio"
        },
        "transformers": {
            "package": "transformers",
            "description": "Hugging Face transformers for local embeddings",
            "install": "pip install transformers"
        }
    }
    
    for name, info in optional_deps.items():
        try:
            __import__(info["package"])
            logger.info(f"✓ {name} is available")
        except ImportError:
            logger.info(f"○ {name} not installed - {info['description']}")
            logger.info(f"  Install with: {info['install']}")


def create_sample_data():
    """Create sample data for testing."""
    logger.info("Creating sample data...")
    
    sample_dir = Path("data/samples")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample documents
    sample_docs = [
        ("sample1.txt", "This is a sample document about artificial intelligence and machine learning."),
        ("sample2.txt", "Natural language processing is a key area of AI research."),
        ("sample3.txt", "Vector databases enable efficient similarity search for RAG systems.")
    ]
    
    for filename, content in sample_docs:
        sample_file = sample_dir / filename
        if not sample_file.exists():
            with open(sample_file, 'w') as f:
                f.write(content)
            logger.info(f"Created sample document: {sample_file}")


def verify_installation():
    """Verify that the installation was successful."""
    logger.info("Verifying installation...")
    
    try:
        # Try importing the main module
        import enhanced_rag
        logger.info("✓ Enhanced RAG module can be imported")
        
        # Check CLI command
        result = subprocess.run([sys.executable, "-m", "enhanced_rag.api.cli", "--help"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✓ CLI command is working")
        else:
            logger.warning("△ CLI command may have issues")
        
        # Check configuration loading
        from enhanced_rag.utils.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config("local")
        logger.info("✓ Configuration loading works")
        
    except Exception as e:
        logger.error(f"Installation verification failed: {e}")
        return False
    
    return True


def main():
    """Main setup function."""
    logger.info("Starting Enhanced RAG System setup...")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Setup configuration
    setup_configuration()
    
    # Setup git hooks
    setup_git_hooks()
    
    # Create sample data
    create_sample_data()
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Verify installation
    if verify_installation():
        logger.info("✅ Setup completed successfully!")
        
        print("\n" + "="*50)
        print("Enhanced RAG System Setup Complete!")
        print("="*50)
        print("\nNext steps:")
        print("1. Set up your API keys in environment variables or config files")
        print("2. Run tests: pytest tests/")
        print("3. Start the API server: python -m enhanced_rag.api.rest")
        print("4. Try the CLI: enhanced-rag --help")
        print("\nFor more information, see README.md")
        
    else:
        logger.error("❌ Setup completed with errors. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()