"""
Configuration settings for the Multimodal RAG System
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator
from pathlib import Path
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    # Application
    APP_NAME: str = "Multimodal RAG System"
    VERSION: str = "1.0.0"
    DEBUG_INPUT: bool | str | None = Field(
        default=None,
        validation_alias="DEBUG",
        exclude=True,
    )
    DEBUG_ENABLED: bool = False
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    API_RELOAD: bool = False
    
    # Paths - CHANGED TO F: DRIVE
    BASE_DIR: Path = Path("F:/Smart India Hackathon_2025/SIH25231")
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    MODELS_DIR: Path = BASE_DIR / "models"  # Models will be stored here
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"
    
    # Model Settings
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DIMENSION: int = 768
    IMAGE_EMBEDDING_DIMENSION: int = 512
    AUDIO_EMBEDDING_DIMENSION: int = 768
    LLM_MODEL: str = "microsoft/phi-1_5"
    WHISPER_MODEL: str = "large-v3"
    CLIP_MODEL: str = "ViT-B/32"
    
    # Processing Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_CHUNKS_PER_DOC: int = 1000
    ENABLE_PARALLEL_PROCESSING: bool = True
    MAX_WORKERS: int = 4
    BATCH_INSERT_SIZE: int = 100  # Optimal batch size for ChromaDB
    
    # Retrieval Settings
    TOP_K_RETRIEVAL: int = 20
    RERANK_TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    ENABLE_FAST_SUMMARY: bool = True
    ENABLE_QUERY_CACHE: bool = True
    
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
    
    @model_validator(mode="after")
    def _normalise_debug(self):
        """Normalise DEBUG inputs and keep LOG_LEVEL aligned."""
        raw_value = getattr(self, "DEBUG_INPUT", None)

        # Short-circuit when no explicit DEBUG value was provided
        if raw_value is None:
            return self

        truthy = {"1", "true", "yes", "on", "t", "y"}
        falsy = {"0", "false", "no", "off", "f", "n"}
        level_map = {
            "trace": "TRACE",
            "debug": "DEBUG",
            "info": "INFO",
            "warn": "WARNING",
            "warning": "WARNING",
            "error": "ERROR",
            "critical": "CRITICAL",
            "fatal": "CRITICAL",
        }

        level = None

        if isinstance(raw_value, bool):
            self.DEBUG_ENABLED = raw_value
            level = "DEBUG" if raw_value else None
        elif isinstance(raw_value, str):
            lowered = raw_value.strip().lower()

            if lowered in truthy:
                self.DEBUG_ENABLED = True
                level = "DEBUG"
            elif lowered in falsy:
                self.DEBUG_ENABLED = False
            elif lowered in level_map:
                level = level_map[lowered]
                self.DEBUG_ENABLED = level in {"TRACE", "DEBUG"}
            else:
                raise ValueError(
                    "DEBUG must be a boolean value or a recognised log level (trace/debug/info/warn/error/critical)."
                )
        else:
            raise ValueError("DEBUG must be provided as a boolean or string.")

        if level and "LOG_LEVEL" not in self.model_fields_set:
            self.LOG_LEVEL = level

        if "API_RELOAD" not in self.model_fields_set:
            object.__setattr__(self, "API_RELOAD", bool(self.DEBUG_ENABLED))

        return self

    @property
    def DEBUG(self) -> bool:
        return self.DEBUG_ENABLED
    
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