"""
Hybrid search combining semantic (vector) and keyword (BM25) search
"""
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
import math
from loguru import logger

from database.vector_store import VectorStore
from models.embeddings import MultimodalEmbedder
from config.settings import settings


class BM25:
    """BM25 keyword search implementation"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.corpus_ids = []
    
    def fit(self, corpus: List[str], corpus_ids: List[str]):
        """
        Build BM25 index
        
        Args:
            corpus: List of documents (text strings)
            corpus_ids: List of document IDs
        """
        self.corpus = corpus
        self.corpus_ids = corpus_ids
        
        # Tokenize documents
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        # Calculate document frequencies
        df = defaultdict(int)
        for doc in tokenized_corpus:
            for word in set(doc):
                df[word] += 1
        
        # Calculate IDF
        num_docs = len(corpus)
        for word, freq in df.items():
            self.idf[word] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)
        
        # Store document lengths
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        logger.info(f"BM25 index built with {num_docs} documents")
    
    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Search using BM25
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of results with scores
        """
        query_tokens = query.lower().split()
        scores = np.zeros(len(self.corpus))
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            idf_score = self.idf[token]
            
            for idx, doc in enumerate(self.corpus):
                doc_tokens = doc.lower().split()
                term_freq = doc_tokens.count(token)
                
                if term_freq == 0:
                    continue
                
                doc_length = self.doc_len[idx]
                norm_tf = (term_freq * (self.k1 + 1)) / (
                    term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
                )
                
                scores[idx] += idf_score * norm_tf
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            {
                "id": self.corpus_ids[idx],
                "text": self.corpus[idx],
                "score": float(scores[idx])
            }
            for idx in top_indices if scores[idx] > 0
        ]
        
        return results


class HybridSearchEngine:
    """Hybrid search combining semantic and keyword search"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedder = MultimodalEmbedder()
        self.bm25_indices = {
            "text": None,
            "image": None,
            "audio": None
        }
        logger.info("Hybrid search engine initialized")
    
    def build_bm25_index(self, collection_type: str = "text"):
        """Build BM25 index for a collection"""
        logger.info(f"Building BM25 index for {collection_type}")
        
        # Get collection
        if collection_type == "text":
            collection = self.vector_store.text_collection
        elif collection_type == "image":
            collection = self.vector_store.image_collection
        elif collection_type == "audio":
            collection = self.vector_store.audio_collection
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")
        
        # Get all documents
        results = collection.get()
        
        if not results['ids']:
            logger.warning(f"No documents found in {collection_type} collection")
            self.bm25_indices[collection_type] = None  # Set to None explicitly
            return
        
        # Build BM25 index
        bm25 = BM25()
        bm25.fit(results['documents'], results['ids'])
        self.bm25_indices[collection_type] = bm25
        
        logger.info(f"BM25 index built for {collection_type}: {len(results['ids'])} docs")
        
    def semantic_search(
        self,
        query: str,
        collection_type: str = "text",
        top_k: int = 20
    ) -> List[Dict]:
        """
        Semantic search using embeddings
        
        Args:
            query: Query string
            collection_type: Collection to search
            top_k: Number of results
            
        Returns:
            List of results
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query, content_type="text")
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            collection_type=collection_type,
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "score": 1 - results['distances'][0][i],  # Convert distance to similarity
                "metadata": results['metadatas'][0][i]
            })
        
        return formatted_results
    
    def keyword_search(
        self,
        query: str,
        collection_type: str = "text",
        top_k: int = 20
    ) -> List[Dict]:
        """
        Keyword search using BM25
        """
        bm25 = self.bm25_indices.get(collection_type)
        
        if bm25 is None:
            logger.warning(f"BM25 index not built for {collection_type}, building now...")
            self.build_bm25_index(collection_type)
            bm25 = self.bm25_indices[collection_type]
        
        # Add this safety check
        if bm25 is None:
            logger.warning(f"No documents in {collection_type} collection for keyword search")
            return []
        
        return bm25.search(query, top_k)
    
    def hybrid_search(
        self,
        query: str,
        collection_type: str = "text",
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict]:
        """
        Hybrid search with RRF (Reciprocal Rank Fusion)
        
        Args:
            query: Query string
            collection_type: Collection to search
            top_k: Final number of results
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            
        Returns:
            Fused results
        """
        # Get results from both methods
        semantic_results = self.semantic_search(query, collection_type, top_k * 2)
        keyword_results = self.keyword_search(query, collection_type, top_k * 2)
        
        # Reciprocal Rank Fusion
        rrf_scores = defaultdict(float)
        k = 60  # RRF constant
        
        # Add semantic scores
        for rank, result in enumerate(semantic_results, start=1):
            rrf_scores[result['id']] += semantic_weight / (k + rank)
        
        # Add keyword scores
        for rank, result in enumerate(keyword_results, start=1):
            rrf_scores[result['id']] += keyword_weight / (k + rank)
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top-k results with full metadata
        final_results = []
        id_to_result = {r['id']: r for r in semantic_results + keyword_results}
        
        for doc_id, score in sorted_ids[:top_k]:
            if doc_id in id_to_result:
                result = id_to_result[doc_id].copy()
                result['score'] = float(score)
                final_results.append(result)
        
        return final_results
    
    def cross_modal_search(
        self,
        query: str,
        top_k_per_type: int = 5
    ) -> Dict:
        """
        Search across all modalities
        
        Args:
            query: Query string
            top_k_per_type: Results per modality
            
        Returns:
            Results from all modalities
        """
        results = {
            "text": self.hybrid_search(query, "text", top_k_per_type),
            "image": self.hybrid_search(query, "image", top_k_per_type),
            "audio": self.hybrid_search(query, "audio", top_k_per_type)
        }
        
        return results