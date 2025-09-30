"""
FastAPI application for Multimodal RAG System
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import time
from pathlib import Path
import shutil
from loguru import logger
import numpy as np

from config.settings import settings
from models.embeddings import MultimodalEmbedder
from models.llm import LLMGenerator
from models.audio import AudioProcessor
from processing.document_processor import DocumentProcessor
from processing.image_processor import ImageProcessor
from database.vector_store import VectorStore
from database.metadata_store import MetadataStore
from retrieval.hybrid_search import HybridSearchEngine
from retrieval.query_router import QueryRouter

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Offline Multimodal RAG System for Government Centers"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
embedder = None
llm = None
vector_store = None
metadata_store = None
search_engine = None
query_router = None


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    search_type: Optional[str] = "hybrid"  # hybrid, semantic, keyword
    filters: Optional[Dict] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict]
    num_sources: int
    processing_time_ms: float
    search_results: List[Dict]


class DocumentUploadResponse(BaseModel):
    success: bool
    doc_id: str
    source: str
    num_chunks: int
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize models and databases on startup"""
    global embedder, llm, vector_store, metadata_store, search_engine, query_router
    
    logger.info("Starting Multimodal RAG System...")
    
    try:
        # Initialize components
        embedder = MultimodalEmbedder()
        llm = LLMGenerator()
        vector_store = VectorStore()
        metadata_store = MetadataStore()
        search_engine = HybridSearchEngine()
        query_router = QueryRouter()
        
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "version": settings.VERSION
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    vector_stats = vector_store.get_stats()
    metadata_stats = metadata_store.get_stats()
    
    return {
        "vector_store": vector_stats,
        "metadata_store": metadata_stats
    }


@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF or DOCX)
    """
    start_time = time.time()
    
    try:
        # Save uploaded file
        file_path = settings.UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing document: {file.filename}")
        
        # Process document
        doc_processor = DocumentProcessor()
        chunks = doc_processor.process_document(str(file_path))
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from document")
        
        # Generate embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.text_embedder.embed_text(chunk_texts)
        
        # Store in vector database
        vector_store.add_text_chunks(chunks, embeddings)
        
        # Store metadata
        doc_id = chunks[0]["metadata"]["doc_id"]
        metadata_store.add_document({
            "doc_id": doc_id,
            "source": file.filename,
            "file_path": str(file_path),
            "type": "pdf" if file.filename.endswith('.pdf') else "docx",
            "metadata": {"num_chunks": len(chunks)}
        })
        metadata_store.add_chunks(chunks, doc_id)
        metadata_store.update_document_indexed(doc_id, True)
        
        # Rebuild BM25 index
        search_engine.build_bm25_index("text")
        
        processing_time = (time.time() - start_time) * 1000
        
        return DocumentUploadResponse(
            success=True,
            doc_id=doc_id,
            source=file.filename,
            num_chunks=len(chunks),
            message=f"Document processed successfully in {processing_time:.2f}ms"
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and process an image
    """
    start_time = time.time()
    
    try:
        # Save uploaded file
        file_path = settings.UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing image: {file.filename}")
        
        # Process image
        img_processor = ImageProcessor()
        image_data = img_processor.process_and_prepare(str(file_path))
        
        # Generate embeddings
        # Visual embedding
        visual_embedding = embedder.image_embedder.embed_image(str(file_path))[0]
        
        # Store in vector database
        vector_store.add_images([image_data], np.array([visual_embedding]))
        
        # Store metadata
        metadata_store.add_document({
            "doc_id": image_data["id"],
            "source": file.filename,
            "file_path": str(file_path),
            "type": "image",
            "metadata": image_data["metadata"]
        })
        metadata_store.update_document_indexed(image_data["id"], True)
        
        # Rebuild BM25 index
        search_engine.build_bm25_index("image")
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "image_id": image_data["id"],
            "source": file.filename,
            "has_text": image_data["has_text"],
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload and process an audio file
    """
    start_time = time.time()
    
    try:
        # Save uploaded file
        file_path = settings.UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing audio: {file.filename}")
        
        # Process audio
        audio_processor = AudioProcessor()
        audio_data = audio_processor.process_audio(str(file_path))
        
        # Generate audio ID
        import hashlib
        audio_id = hashlib.md5(file.filename.encode()).hexdigest()[:16]
        
        # Generate embeddings for chunks
        chunk_texts = [chunk["text"] for chunk in audio_data["chunks"]]
        embeddings = embedder.text_embedder.embed_text(chunk_texts)
        
        # Store in vector database
        audio_metadata = {
            "audio_id": audio_id,
            "source": file.filename,
            "duration": audio_data["duration"]
        }
        vector_store.add_audio_chunks(audio_data["chunks"], embeddings, audio_metadata)
        
        # Store metadata
        metadata_store.add_document({
            "doc_id": audio_id,
            "source": file.filename,
            "file_path": str(file_path),
            "type": "audio",
            "duration": audio_data["duration"],
            "metadata": {
                "language": audio_data["language"],
                "num_segments": len(audio_data["segments"])
            }
        })
        metadata_store.update_document_indexed(audio_id, True)
        
        # Rebuild BM25 index
        search_engine.build_bm25_index("audio")
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "audio_id": audio_id,
            "source": file.filename,
            "duration": audio_data["duration"],
            "num_segments": len(audio_data["segments"]),
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error uploading audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """
    Query the RAG system
    """
    start_time = time.time()
    
    try:
        # Analyze query
        query_analysis = query_router.analyze_query(request.query)
        
        # Perform search based on analysis
        if query_analysis["search_strategy"] == "cross_modal":
            search_results = search_engine.cross_modal_search(
                request.query,
                top_k_per_type=request.top_k // 3
            )
            # Flatten results
            all_results = (
                search_results["text"] +
                search_results["image"] +
                search_results["audio"]
            )
        else:
            # Single modality search
            target_modality = query_analysis["target_modalities"][0] if query_analysis["target_modalities"] else "text"
            all_results = search_engine.hybrid_search(
                request.query,
                collection_type=target_modality,
                top_k=request.top_k
            )
        
        # Prepare contexts for LLM
        contexts = []
        for result in all_results[:10]:  # Top 10 for LLM
            contexts.append({
                "id": result["id"],
                "text": result["text"],
                "type": result.get("metadata", {}).get("type", "unknown"),
                "source": result.get("metadata", {}).get("source", "unknown"),
                "page": result.get("metadata", {}).get("page", ""),
                "score": result["score"]
            })
        
        # Generate answer
        llm_result = llm.generate_answer(request.query, contexts)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log query
        metadata_store.log_query({
            "query": request.query,
            "type": query_analysis["search_strategy"],
            "num_results": len(all_results),
            "response_time_ms": processing_time
        })
        
        return QueryResponse(
            query=request.query,
            answer=llm_result["answer"],
            citations=llm_result["citations"],
            num_sources=llm_result["num_sources_used"],
            processing_time_ms=processing_time,
            search_results=[{
                "id": r["id"],
                "text": r["text"][:200] + "...",
                "score": r["score"],
                "metadata": r.get("metadata", {})
            } for r in all_results[:10]]
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents(doc_type: Optional[str] = Query(None)):
    """List all indexed documents"""
    try:
        documents = metadata_store.get_all_documents(doc_type)
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks"""
    try:
        # Get document info
        doc = metadata_store.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from vector store
        vector_store.delete_by_source(doc["source"])
        
        # TODO: Delete from metadata store (add method)
        
        return {"success": True, "message": f"Document {doc_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )