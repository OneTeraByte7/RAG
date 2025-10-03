"""
FastAPI application for Multimodal RAG System
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
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
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None)
):
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
        
        # Process document with custom document_id if provided
        doc_processor = DocumentProcessor()
        doc_id_override = document_id.strip() if document_id and document_id.strip() else None
        chunks = doc_processor.process_document(
            str(file_path),
            doc_id=doc_id_override
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from document")

        doc_id = chunks[0]["metadata"].get("doc_id") if chunks[0].get("metadata") else None
        if not doc_id:
            raise HTTPException(status_code=500, detail="Document processor returned chunks without doc_id metadata")

        if doc_id_override and doc_id != doc_id_override:
            logger.warning(
                "Document processor returned doc_id '{}' different from override '{}'",
                doc_id,
                doc_id_override
            )
            doc_id = doc_id_override

        # Check if document with this ID already exists
        existing_doc = metadata_store.get_document(doc_id)

        filename_lower = file.filename.lower()
        doc_type = "pdf" if filename_lower.endswith('.pdf') else "docx"
        
        # Generate embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.text_embedder.embed_text(chunk_texts)
        
        # Store in vector database
        vector_store.add_text_chunks(chunks, embeddings)
        
        # Store metadata
        doc_metadata = {
            "num_chunks": len(chunks),
            "file_size": file_path.stat().st_size if file_path.exists() else None,
            "doc_type": doc_type
        }
        metadata_store.add_document({
            "doc_id": doc_id,
            "source": file.filename,
            "file_path": str(file_path),
            "type": doc_type,
            "metadata": doc_metadata
        })
        metadata_store.add_chunks(chunks, doc_id)
        metadata_store.update_document_indexed(doc_id, True)
        
        # Rebuild BM25 index
        search_engine.build_bm25_index("text")
        
        processing_time = (time.time() - start_time) * 1000
        action = "updated" if existing_doc else "processed"
        
        return DocumentUploadResponse(
            success=True,
            doc_id=doc_id,
            source=file.filename,
            num_chunks=len(chunks),
            message=f"Document {action} successfully in {processing_time:.2f}ms"
        )
        
    except HTTPException:
        raise
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
        
        # Check if image already exists
        existing_doc = metadata_store.get_document(image_data["id"])
        if existing_doc:
            logger.warning(f"Image {image_data['id']} already exists. Deleting old version first.")
            try:
                await delete_document_by_id(image_data["id"])
            except Exception as e:
                logger.error(f"Failed to delete existing image: {e}")
        
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
        import hashlib
        audio_id = hashlib.md5(file.filename.encode()).hexdigest()[:16]
        audio_data = audio_processor.process_audio(
            str(file_path),
            doc_id=audio_id,
            source=file.filename
        )
        
        # Check if audio already exists
        existing_doc = metadata_store.get_document(audio_id)
        if existing_doc:
            logger.warning(f"Audio {audio_id} already exists. Deleting old version first.")
            try:
                await delete_document_by_id(audio_id)
            except Exception as e:
                logger.error(f"Failed to delete existing audio: {e}")
        
        # Generate embeddings for chunks
        chunk_texts = [chunk["text"] for chunk in audio_data["chunks"] if chunk.get("text")]
        if not chunk_texts:
            raise HTTPException(status_code=400, detail="No transcription content extracted from audio")
        embeddings = embedder.text_embedder.embed_text(chunk_texts)
        
        # Store in vector database
        audio_metadata = {
            "audio_id": audio_id,
            "source": file.filename,
            "duration": audio_data["duration"]
        }
        vector_store.add_audio_chunks(audio_data["chunks"], embeddings, audio_metadata)
        
        # Store metadata
        doc_metadata = {
            "file_size": file_path.stat().st_size if file_path.exists() else None,
            "language": audio_data.get("language"),
            "num_segments": len(audio_data.get("segments", [])),
            "num_chunks": len(audio_data.get("chunks", []))
        }
        metadata_store.add_document({
            "doc_id": audio_id,
            "source": file.filename,
            "file_path": str(file_path),
            "type": "audio",
            "duration": audio_data["duration"],
            "metadata": doc_metadata
        })
        metadata_store.add_chunks(audio_data["chunks"], audio_id)
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
        # Perform search based on analysis
        if query_analysis["search_strategy"] == "cross_modal":
            # For cross-modal, search text collection only for now
            # (images need visual embeddings, not text embeddings)
            all_results = search_engine.hybrid_search(
                request.query,
                collection_type="text",  # Force text only
                top_k=request.top_k
            )
        else:
            # Single modality search
            target_modality = query_analysis["target_modalities"][0] if query_analysis["target_modalities"] else "text"
            
            # Only search text collection for text queries
            if target_modality in ["image", "audio"]:
                target_modality = "text"  # Fallback to text
            
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


@app.get("/documents/{doc_id}")
async def get_document_details(doc_id: str):
    """Get details of a specific document"""
    try:
        doc = metadata_store.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_type = doc.get("doc_type", "text")
        chunk_count = 0

        try:
            if doc_type in {"pdf", "docx", "text"}:
                results = vector_store.text_collection.get(where={"doc_id": doc_id})
                chunk_count = len(results.get("ids", [])) if results else 0
            elif doc_type == "image":
                results = vector_store.image_collection.get(where={"image_id": doc_id})
                chunk_count = len(results.get("ids", [])) if results else 0
            elif doc_type == "audio":
                results = vector_store.audio_collection.get(where={"audio_id": doc_id})
                chunk_count = len(results.get("ids", [])) if results else 0
        except Exception as e:
            logger.warning(f"Unable to fetch chunk count for document {doc_id}: {e}")

        doc["chunk_count"] = chunk_count
        
        return doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def delete_document_by_id(doc_id: str, doc: Optional[Dict] = None) -> Dict[str, int]:
    """Helper function to delete a document by ID."""
    doc_info = doc or metadata_store.get_document(doc_id)
    if not doc_info:
        return {"vectors": 0, "chunks": 0}

    try:
        vectors_deleted = vector_store.delete_by_source(doc_info["source"])
    except Exception as e:
        logger.error(f"Error deleting vectors for document {doc_id}: {e}")
        raise

    metadata_stats = {"documents": 0, "chunks": 0}
    try:
        metadata_stats = metadata_store.delete_document(doc_id)
    except Exception as e:
        logger.warning(f"Could not delete metadata for document {doc_id}: {e}")
    
    try:
        vector_store.persist()
    except Exception as e:
        logger.warning(f"Failed to persist vector store after deleting {doc_id}: {e}")

    return {
        "vectors": vectors_deleted,
        "chunks": metadata_stats.get("chunks", 0)
    }


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its chunks"""
    try:
        # Get document info first
        doc = metadata_store.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        # Delete the document
        delete_stats = await delete_document_by_id(doc_id, doc)
        vectors_deleted = delete_stats.get("vectors", 0)
        metadata_chunks_deleted = delete_stats.get("chunks", 0)
        
        # Rebuild BM25 index
        doc_type = doc.get("doc_type", "text")
        if doc_type in ["pdf", "docx", "text"]:
            search_engine.build_bm25_index("text")
        elif doc_type == "image":
            search_engine.build_bm25_index("image")
        elif doc_type == "audio":
            search_engine.build_bm25_index("audio")
        
        return {
            "success": True,
            "message": f"Document {doc_id} deleted successfully",
            "vector_entries_deleted": vectors_deleted,
            "metadata_chunks_deleted": metadata_chunks_deleted
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def delete_all_documents():
    """Delete all documents from the system (use with caution!)"""
    try:
        # Get all documents
        all_docs = metadata_store.get_all_documents()
        
        if not all_docs:
            return {
                "success": True,
                "message": "No documents to delete",
                "documents_deleted": 0
            }
        
        total_vectors = 0
        total_metadata_chunks = 0
        doc_types_to_reindex = set()
        # Delete each document
        for doc in all_docs:
            try:
                delete_stats = await delete_document_by_id(doc["id"], doc)
                total_vectors += delete_stats.get("vectors", 0)
                total_metadata_chunks += delete_stats.get("chunks", 0)
                doc_types_to_reindex.add(doc.get("doc_type", "text"))
            except Exception as e:
                logger.error(f"Error deleting document {doc['id']}: {e}")
        
        # Rebuild all BM25 indices
        try:
            if any(dt in {"pdf", "docx", "text"} for dt in doc_types_to_reindex):
                search_engine.build_bm25_index("text")
            if "image" in doc_types_to_reindex:
                search_engine.build_bm25_index("image")
            if "audio" in doc_types_to_reindex:
                search_engine.build_bm25_index("audio")
        except Exception as e:
            logger.warning(f"Error rebuilding BM25 indices: {e}")
        
        return {
            "success": True,
            "message": "All documents deleted successfully",
            "documents_deleted": len(all_docs),
            "vector_entries_deleted": total_vectors,
            "metadata_chunks_deleted": total_metadata_chunks
        }
        
    except Exception as e:
        logger.error(f"Error deleting all documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )