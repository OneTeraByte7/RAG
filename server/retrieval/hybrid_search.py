"""
Hybrid search combining semantic (vector) and keyword (BM25) search
"""
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict, Counter
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
        self.idf = {}
        self.corpus_ids = []
        self.doc_len = []
        self.avgdl = 0
        self.tokenized_corpus = []
        self.term_freqs = []
        self.inverted_index = defaultdict(list)
        self.norm_factors = []
    
    def fit(self, corpus: List[str], corpus_ids: List[str]):
        """
        Build BM25 index
        
        Args:
            corpus: List of documents (text strings)
            corpus_ids: List of document IDs
        """
        self.corpus = corpus
        self.corpus_ids = corpus_ids

        # Tokenize documents once
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.tokenized_corpus = tokenized_corpus

        # Calculate document frequencies and build inverted index
        df = defaultdict(int)
        self.inverted_index = defaultdict(list)
        self.term_freqs = []

        for idx, doc_tokens in enumerate(tokenized_corpus):
            term_freq = Counter(doc_tokens)
            self.term_freqs.append(term_freq)

            for word in term_freq.keys():
                df[word] += 1
                self.inverted_index[word].append(idx)

        # Calculate IDF
        num_docs = len(corpus)
        for word, freq in df.items():
            self.idf[word] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)

        # Store document lengths and normalization factors
        self.doc_len = [len(doc_tokens) for doc_tokens in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        if self.doc_len:
            avgdl = self.avgdl if self.avgdl > 0 else 1.0
            self.norm_factors = [
                self.k1 * (1 - self.b + self.b * dl / avgdl)
                for dl in self.doc_len
            ]
        else:
            self.norm_factors = []
        
        logger.info(f"BM25 index built with {num_docs} documents")
    
    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Search using BM25 with optimized scoring
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of results with scores
        """
        query_tokens = query.lower().split()
        num_docs = len(self.corpus)
        if num_docs == 0:
            return []

        # Filter query tokens to only those in vocabulary
        query_tokens = [token for token in query_tokens if token in self.idf]
        
        if not query_tokens:
            return []

        scores = np.zeros(num_docs, dtype=np.float32)

        # Optimized scoring loop
        for token in query_tokens:
            idf_score = self.idf[token]
            doc_indices = self.inverted_index[token]
            
            for idx in doc_indices:
                term_freq = self.term_freqs[idx][token]
                norm_denominator = term_freq + self.norm_factors[idx] if self.norm_factors else term_freq + self.k1
                norm_tf = (term_freq * (self.k1 + 1)) / norm_denominator
                scores[idx] += idf_score * norm_tf
        
        # Get top-k results efficiently
        if top_k >= num_docs:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        results = [
            {
                "id": self.corpus_ids[idx],
                "text": self.corpus[idx],
                "score": float(scores[idx])
            }
            for idx in top_indices if scores[idx] > 0
        ]
        
        return results[:top_k]
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
        # Pick the right embedding head for each modality
        embedding_type = "text"
        if collection_type == "image":
            embedding_type = "cross_modal_text"
        elif collection_type == "audio":
            embedding_type = "text"  # audio search is performed over transcripts

        # Generate query embedding
        query_embedding = self.embedder.embed(query, content_type=embedding_type)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            collection_type=collection_type,
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if not results['ids'][0]:
            return formatted_results

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
        candidate_pool = max(top_k, 5)
        rerank_cap = getattr(settings, "RERANK_TOP_K", 10)
        candidate_pool = min(candidate_pool * 2, max(rerank_cap, candidate_pool))

        # Get results from both methods, limiting to a smaller candidate pool for speed
        semantic_results = self.semantic_search(query, collection_type, candidate_pool)
        keyword_results = self.keyword_search(query, collection_type, candidate_pool)
        
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