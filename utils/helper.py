"""
Utility functions for the RAG system
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
from loguru import logger


def generate_id(content: str) -> str:
    """Generate unique ID from content"""
    return hashlib.md5(content.encode()).hexdigest()[:16]


def save_json(data: Dict, file_path: str):
    """Save dictionary to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {file_path}")


def load_json(file_path: str) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return Path(file_path).stat().st_size / (1024 * 1024)


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def chunk_list(lst: list, chunk_size: int):
    """Split list into chunks"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def get_supported_file_types() -> Dict[str, list]:
    """Get supported file types"""
    return {
        "documents": [".pdf", ".docx", ".doc", ".txt"],
        "images": [".jpg", ".jpeg", ".png", ".bmp", ".gif"],
        "audio": [".mp3", ".wav", ".m4a", ".ogg", ".flac"]
    }


def validate_file_type(file_path: str, category: str = None) -> bool:
    """Validate if file type is supported"""
    supported = get_supported_file_types()
    file_ext = Path(file_path).suffix.lower()
    
    if category:
        return file_ext in supported.get(category, [])
    
    # Check all categories
    for types in supported.values():
        if file_ext in types:
            return True
    return False


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        return self
        
    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {elapsed:.2f} seconds")