"""
Embedding models for text and multimodal content
"""
import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from loguru import logger
import clip
from PIL import Image

import sys
import os
config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)
try:
    from config.settings import settings
except ImportError:
    try:
        from config.settings import settings
    except ImportError as e:
        raise ImportError("Could not import 'settings'. Check your config directory and PYTHONPATH.") from e


class TextEmbedder:
    """Generate embeddings for text using Nomic-Embed"""
    
    def __init__(self):
        self.device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
        logger.info(f"Loading text embedding model on {self.device}")
        
        self.model = SentenceTransformer(
    settings.EMBEDDING_MODEL,
    device=self.device,
    trust_remote_code=True  # Add this line
)
        self.dimension = settings.EMBEDDING_DIMENSION
        
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=settings.BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        return self.embed_text(query)[0]


class ImageEmbedder:
    """Generate embeddings for images using CLIP"""
    
    def __init__(self):
        self.device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model on {self.device}")
        
        self.model, self.preprocess = clip.load(
            settings.CLIP_MODEL,
            device=self.device
        )
        
    def embed_image(self, image: Union[str, Image.Image, List]) -> np.ndarray:
        """
        Generate embeddings for image(s)
        
        Args:
            image: Image path, PIL Image, or list of either
            
        Returns:
            numpy array of embeddings
        """
        if not isinstance(image, list):
            image = [image]
        
        # Load and preprocess images
        processed_images = []
        for img in image:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            processed_images.append(self.preprocess(img))
        
        # Stack into batch
        image_batch = torch.stack(processed_images).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model.encode_image(image_batch)
            embeddings = embeddings.cpu().numpy()
            # Normalize
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def embed_text_for_image_search(self, text: str) -> np.ndarray:
        """
        Generate text embedding compatible with image embeddings
        For cross-modal search
        """
        text_tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features.cpu().numpy()
            text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        
        return text_features[0]


class MultimodalEmbedder:
    """Unified interface for all embedding types"""
    
    def __init__(self):
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        logger.info("Multimodal embedder initialized")
    
    def embed(self, content: Union[str, Image.Image], content_type: str = "text") -> np.ndarray:
        """
        Generate embedding based on content type
        
        Args:
            content: Text string or PIL Image
            content_type: "text", "image", or "query"
            
        Returns:
            Embedding vector
        """
        if content_type == "text":
            return self.text_embedder.embed_query(content)
        elif content_type == "image":
            return self.image_embedder.embed_image(content)[0]
        elif content_type == "cross_modal_text":
            # Text query for searching images
            return self.image_embedder.embed_text_for_image_search(content)
        else:
            raise ValueError(f"Unknown content type: {content_type}")
    
    def batch_embed(self, contents: List, content_type: str = "text") -> np.ndarray:
        """Batch embedding generation"""
        if content_type == "text":
            return self.text_embedder.embed_text(contents)
        elif content_type == "image":
            return self.image_embedder.embed_image(contents)
        else:
            raise ValueError(f"Unknown content type: {content_type}")