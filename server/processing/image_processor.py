"""
Image processing with OCR and visual embeddings
"""
import easyocr
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import hashlib
from datetime import datetime
import numpy as np

from config.settings import settings


class OCREngine:
    """Extract text from images using EasyOCR"""
    
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ['en']
        logger.info(f"Initializing OCR engine for languages: {self.languages}")
        
        self.reader = easyocr.Reader(
            self.languages,
            gpu=settings.USE_GPU
        )
        
    def extract_text(self, image_path: str) -> Dict:
        """
        Extract text from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with extracted text and bounding boxes
        """
        logger.info(f"Extracting text from image: {image_path}")
        
        try:
            # Perform OCR
            results = self.reader.readtext(image_path)
            
            # Parse results
            text_blocks = []
            full_text = []
            
            for bbox, text, confidence in results:
                text_blocks.append({
                    "text": text,
                    "bbox": bbox,
                    "confidence": confidence
                })
                full_text.append(text)
            
            return {
                "full_text": " ".join(full_text),
                "text_blocks": text_blocks,
                "num_blocks": len(text_blocks)
            }
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return {
                "full_text": "",
                "text_blocks": [],
                "num_blocks": 0
            }


class ImageProcessor:
    """Process images with OCR and metadata extraction"""
    
    def __init__(self):
        self.ocr = OCREngine()
        
    def process_image(self, image_path: str, extract_text: bool = True) -> Dict:
        """
        Process image file
        
        Args:
            image_path: Path to image
            extract_text: Whether to perform OCR
            
        Returns:
            Dict with image data and metadata
        """
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Load image
            img = Image.open(image_path)
            
            # Generate image ID
            image_id = self._generate_image_id(image_path)
            
            # Extract basic metadata
            metadata = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "file_size": Path(image_path).stat().st_size,
                "processed_at": datetime.now().isoformat()
            }
            
            result = {
                "image_id": image_id,
                "source": Path(image_path).name,
                "file_path": str(image_path),
                "metadata": metadata,
                "ocr_text": "",
                "has_text": False
            }
            
            # Perform OCR if requested
            if extract_text:
                ocr_result = self.ocr.extract_text(image_path)
                result["ocr_text"] = ocr_result["full_text"]
                result["text_blocks"] = ocr_result["text_blocks"]
                result["has_text"] = len(ocr_result["full_text"]) > 0
            
            logger.info(f"Image processed: {img.width}x{img.height}, "
                       f"text found: {result['has_text']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def process_and_prepare(self, image_path: str) -> Dict:
        """
        Process image and prepare for indexing
        
        Returns:
            Dict ready for embedding and storage
        """
        # Process image
        image_data = self.process_image(image_path)
        
        # Prepare for indexing
        index_data = {
            "id": image_data["image_id"],
            "type": "image",
            "source": image_data["source"],
            "file_path": image_data["file_path"],
            "text": image_data["ocr_text"],
            "has_text": image_data["has_text"],
            "metadata": image_data["metadata"]
        }
        
        return index_data
    
    def batch_process(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of processed image data
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.process_and_prepare(img_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
        
        logger.info(f"Batch processed {len(results)}/{len(image_paths)} images")
        return results
    
    def _generate_image_id(self, file_path: str) -> str:
        """Generate unique image ID"""
        return hashlib.md5(str(file_path).encode()).hexdigest()[:16]


class ImageAnalyzer:
    """Advanced image analysis (optional enhancements)"""
    
    def __init__(self):
        pass
    
    def detect_document_type(self, image: Image.Image) -> str:
        """
        Classify image as screenshot, photo, chart, etc.
        Simplified version - can be enhanced with ML
        """
        # Basic heuristics
        width, height = image.size
        aspect_ratio = width / height
        
        # Common screenshot aspect ratios
        if 1.3 < aspect_ratio < 1.8:
            return "screenshot"
        elif aspect_ratio > 2 or aspect_ratio < 0.5:
            return "chart_or_diagram"
        else:
            return "photo"
    
    def extract_colors(self, image: Image.Image) -> Dict:
        """Extract dominant colors from image"""
        # Convert to RGB
        img_rgb = image.convert('RGB')
        
        # Get pixel data
        pixels = np.array(img_rgb)
        pixels = pixels.reshape(-1, 3)
        
        # Simple color analysis
        mean_color = pixels.mean(axis=0).tolist()
        
        return {
            "mean_color_rgb": mean_color,
            "is_grayscale": len(set(mean_color)) == 1
        }