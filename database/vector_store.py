"""
Vector database for semantic search using ChromaDB
"""
import chromadb
from typing import List, Dict, Optional
import numpy as np
from loguru import logger
from pathlib import Path

from config.settings import settings


class VectorStore:
    """ChromaDB vector store for multimodal embeddings"""

    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or str(settings.VECTOR_DB_DIR)
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB at {self.persist_directory}")

        # Use new Chroma PersistentClient API
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Create or get collections
        self.text_collection = self._get_or_create_collection("text_chunks")
        self.image_collection = self._get_or_create_collection("image_embeddings")
        self.audio_collection = self._get_or_create_collection("audio_transcripts")

        logger.info("Vector store initialized")
    
    def _get_or_create_collection(self, name: str):
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(name)
            logger.info(f"Loaded existing collection: {name}")
        except:
            # Set metadata based on collection type
            metadata = {"hnsw:space": settings.CHROMA_DISTANCE_FUNCTION}
            
            collection = self.client.create_collection(
                name=name,
                metadata=metadata
            )
            logger.info(f"Created new collection: {name}")
        
        return collection
    
    def _delete_text_chunks_by_doc_id(self, doc_id: str) -> int:
        """Remove existing text chunks for a document before re-inserting."""
        if not doc_id or doc_id == "unknown":
            return 0

        try:
            deleted = self.text_collection.delete(where={"doc_id": doc_id})
            if isinstance(deleted, dict):
                count = len(deleted.get("ids", []))
            else:
                count = deleted if isinstance(deleted, int) else 0

            if count:
                logger.info(
                    f"Removed {count} existing text chunks for document {doc_id}"
                )
            return count
        except Exception as e:
            logger.warning(
                f"Failed to prune existing text chunks for {doc_id}: {e}"
            )

        return 0

    def add_text_chunks(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray
    ) -> None:
        """
        Add text chunks to vector store
        
        Args:
            chunks: List of chunk dicts with text and metadata
            embeddings: Corresponding embeddings
        """
        if len(chunks) == 0:
            return
        
        # Ensure we do not insert duplicate IDs for re-processed documents
        doc_ids = set()
        for chunk in chunks:
            metadata = chunk.setdefault("metadata", {})
            doc_id = metadata.get("doc_id")
            if doc_id:
                doc_ids.add(doc_id)

        if not doc_ids:
            logger.warning("No doc_id metadata found in chunks; skipping pre-delete step")
        else:
            for doc_id in doc_ids:
                self._delete_text_chunks_by_doc_id(doc_id)

        ids = [
            f"text_{chunk['metadata'].get('doc_id', 'unknown')}_{chunk['chunk_id']}"
            for chunk in chunks
        ]
        
        texts = [chunk["text"] for chunk in chunks]
        
        metadatas = [
            {
                "type": "text",
                "source": chunk["metadata"].get("source", "unknown"),
                "doc_id": chunk["metadata"].get("doc_id", "unknown"),
                "page": str(chunk["metadata"].get("page", "")),
                "chunk_id": str(chunk["chunk_id"])
            }
            for chunk in chunks
        ]
        
        self.text_collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} text chunks to vector store")
        self._persist_if_supported()
    
    def add_images(
        self,
        images: List[Dict],
        embeddings: np.ndarray
    ) -> None:
        """
        Add image embeddings to vector store
        
        Args:
            images: List of image data dicts
            embeddings: Corresponding visual embeddings
        """
        if len(images) == 0:
            return
        
        ids = [f"image_{img['id']}" for img in images]
        
        # Use OCR text as document
        documents = [img.get("text", "") for img in images]
        
        metadatas = [
            {
                "type": "image",
                "source": img["source"],
                "image_id": img["id"],
                "file_path": img["file_path"],
                "has_text": str(img.get("has_text", False))
            }
            for img in images
        ]
        
        self.image_collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(images)} images to vector store")
        self._persist_if_supported()
    
    def add_audio_chunks(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        audio_metadata: Dict
    ) -> None:
        """
        Add audio transcript chunks to vector store
        
        Args:
            chunks: List of audio chunk dicts
            embeddings: Corresponding embeddings
            audio_metadata: Metadata about the audio file
        """
        if len(chunks) == 0:
            return
        
        audio_id = audio_metadata.get("audio_id", "unknown")
        ids = [
            f"audio_{audio_id}_{chunk.get('chunk_id', idx)}"
            for idx, chunk in enumerate(chunks)
        ]
        
        documents = [chunk["text"] for chunk in chunks]

        metadatas = []
        for chunk in chunks:
            chunk_meta = {
                "type": "audio",
                "source": audio_metadata.get("source", "unknown"),
                "audio_id": audio_id,
                "doc_id": audio_metadata.get("audio_id", audio_id),
                "chunk_id": str(chunk.get("chunk_id")),
                "start": str(chunk.get("start")),
                "end": str(chunk.get("end"))
            }

            raw_metadata = chunk.get("metadata") or {}

            if raw_metadata:
                chunk_meta["type"] = raw_metadata.get("type", chunk_meta["type"])
                chunk_meta["source"] = raw_metadata.get("source", chunk_meta["source"])
                chunk_meta["doc_id"] = raw_metadata.get("doc_id", chunk_meta["doc_id"])
                chunk_meta["start"] = str(raw_metadata.get("start", chunk.get("start")))
                chunk_meta["end"] = str(raw_metadata.get("end", chunk.get("end")))
                if raw_metadata.get("segment_ids") is not None:
                    chunk_meta["segment_ids"] = ",".join(
                        [str(seg) for seg in raw_metadata.get("segment_ids", [])]
                    )

            metadatas.append(chunk_meta)
        
        self.audio_collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} audio chunks to vector store")
        self._persist_if_supported()
    
    def search(
        self,
        query_embedding: np.ndarray,
        collection_type: str = "text",
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Search vector store
        
        Args:
            query_embedding: Query embedding vector
            collection_type: "text", "image", or "audio"
            n_results: Number of results to return
            where: Metadata filters
            
        Returns:
            Search results
        """
        # Select collection
        if collection_type == "text":
            collection = self.text_collection
        elif collection_type == "image":
            collection = self.image_collection
        elif collection_type == "audio":
            collection = self.audio_collection
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")
        
        # Perform search
        if collection.count() == 0:
            logger.debug(f"No vectors in {collection_type} collection; returning empty result.")
            return self._empty_query_result()
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where
            )
        except RuntimeError as exc:
            if "Nothing found on disk" in str(exc):
                logger.warning(
                    f"HNSW index missing on disk for collection {collection.name}; returning empty result."
                )
                return self._empty_query_result()
            raise
        
        return results
    
    def search_all(
        self,
        query_embedding: np.ndarray,
        n_results_per_type: int = 10
    ) -> Dict:
        """
        Search across all collections
        
        Args:
            query_embedding: Query embedding
            n_results_per_type: Results per collection
            
        Returns:
            Combined results from all collections
        """
        results = {
            "text": self.search(query_embedding, "text", n_results_per_type),
            "image": self.search(query_embedding, "image", n_results_per_type),
            "audio": self.search(query_embedding, "audio", n_results_per_type)
        }
        
        return results
    
    def delete_by_source(self, source: str, collection_type: str = None) -> int:
        """Delete all entries from a specific source and return the count removed."""
        collections = []
        total_deleted = 0
        
        if collection_type:
            if collection_type == "text":
                collections = [self.text_collection]
            elif collection_type == "image":
                collections = [self.image_collection]
            elif collection_type == "audio":
                collections = [self.audio_collection]
        else:
            collections = [self.text_collection, self.image_collection, self.audio_collection]
        
        for collection in collections:
            try:
                # Get all IDs with this source first
                results = collection.get(where={"source": source})
                if results['ids']:
                    total_deleted += len(results['ids'])
                    # Delete by IDs instead of where clause
                    collection.delete(ids=results['ids'])
                    logger.info(f"Deleted {len(results['ids'])} entries from {source} in {collection.name}")
            except Exception as e:
                logger.error(f"Error deleting from {collection.name}: {e}")

        return total_deleted
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "text_count": self.text_collection.count(),
            "image_count": self.image_collection.count(),
            "audio_count": self.audio_collection.count(),
            "total_count": (
                self.text_collection.count() +
                self.image_collection.count() +
                self.audio_collection.count()
            )
        }
    
    def persist(self):
        """Persist the database to disk"""
        if hasattr(self.client, "persist"):
            self.client.persist()
            logger.info("Vector store persisted to disk")
        else:
            logger.debug("Persistent client handles writes automatically; no manual persist needed")

    def _persist_if_supported(self):
        if hasattr(self.client, "persist"):
            try:
                self.client.persist()
            except Exception as exc:
                logger.warning(f"Failed to persist vector store: {exc}")

    @staticmethod
    def _empty_query_result() -> Dict:
        return {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]]
        }