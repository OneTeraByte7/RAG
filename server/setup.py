# setup.py - Installation and setup script
"""
Setup script for Multimodal RAG System
Run: python setup.py
"""
import subprocess
import sys
from pathlib import Path
from loguru import logger


def install_requirements():
    """Install required packages"""
    logger.info("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def download_models():
    """Download required models"""
    logger.info("Downloading models...")
    
    from huggingface_hub import snapshot_download
    from config.settings import settings
    
    models = [
        ("nomic-ai/nomic-embed-text-v1.5", "embedding"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "llm"),
    ]
    
    for model_name, model_type in models:
        logger.info(f"Downloading {model_type} model: {model_name}")
        snapshot_download(
            repo_id=model_name,
            cache_dir=str(settings.MODELS_DIR),
            local_dir=str(settings.MODELS_DIR / model_type)
        )


def setup_databases():
    """Initialize databases"""
    logger.info("Setting up databases...")
    
    from database.vector_store import VectorStore
    from database.metadata_store import MetadataStore
    
    # Initialize vector store
    vector_store = VectorStore()
    logger.info("Vector store initialized")
    
    # Initialize metadata store
    metadata_store = MetadataStore()
    logger.info("Metadata store initialized")


def main():
    """Main setup function"""
    logger.info("Starting Multimodal RAG System setup...")
    
    try:
        # Install requirements
        install_requirements()
        
        # Download models
        download_models()
        
        # Setup databases
        setup_databases()
        
        logger.info("✅ Setup completed successfully!")
        logger.info("Run 'python run.py' to start the system")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
