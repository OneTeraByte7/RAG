"""
Configuration settings for the Multimodal RAG System
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Multimodal RAG System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Paths - CHANGED TO F: DRIVE
    BASE_DIR: Path = Path("F:/Smart India Hackathon_2025/SIH25231")
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    MODELS_DIR: Path = BASE_DIR / "models"  # Models will be stored here
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"
    
    # Model Settings
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DIMENSION: int = 768
    LLM_MODEL: str = "microsoft/phi-1_5"
    WHISPER_MODEL: str = "large-v3"
    CLIP_MODEL: str = "ViT-B/32"
    
    # Processing Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_CHUNKS_PER_DOC: int = 1000
    
    # Retrieval Settings
    TOP_K_RETRIEVAL: int = 20
    RERANK_TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Vector Database
    CHROMA_COLLECTION_NAME: str = "multimodal_rag"
    CHROMA_DISTANCE_FUNCTION: str = "cosine"
    
    # PostgreSQL
    POSTGRES_USER: str = "rag_user"
    POSTGRES_PASSWORD: str = "secure_password"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "rag_metadata"
    
    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_TTL: int = 3600  # 1 hour
    
    # Celery
    CELERY_BROKER_URL: str = "amqp://guest:guest@localhost:5672//"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    
    # GPU Settings
    USE_GPU: bool = True
    GPU_DEVICE: int = 0
    BATCH_SIZE: int = 8
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = BASE_DIR / "logs" / "app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def database_url(self) -> str:
        """PostgreSQL connection URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    def create_directories(self):
        """Create necessary directories"""
        for dir_path in [self.DATA_DIR, self.UPLOAD_DIR, self.MODELS_DIR, 
                         self.VECTOR_DB_DIR, self.LOG_FILE.parent]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_cache_env(self):
        """Set environment variables for model cache"""
        os.environ['HF_HOME'] = str(self.MODELS_DIR)
        os.environ['TRANSFORMERS_CACHE'] = str(self.MODELS_DIR)
        os.environ['HF_DATASETS_CACHE'] = str(self.MODELS_DIR)


# Global settings instance
settings = Settings()
settings.create_directories()
settings.setup_cache_env()  # Set cache location