"""
FastAPI application for Multimodal RAG System
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import re
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


SMALL_TALK_RESPONSES = [
    {
        "triggers": {"hi", "hi!", "hello", "hello!", "hey", "hey!", "hey there", "hola"},
        "response": "Hi there! 👋 I'm ready when you are—ask about any uploaded document or drop a new file to begin."
    },
    {
        "triggers": {"thanks", "thank you", "thank you!", "thanks!", "thanks a lot"},
        "response": "Any time! If you need a quick recap or want to dig into specific sections, just let me know."
    },
    {
        "triggers": {"good morning", "good afternoon", "good evening"},
        "response": "Hello! Hope you're having a great day. I'm here to summarize documents, answer questions, or help you find key points."
    }
]


def match_small_talk(query_text: str) -> Optional[str]:
    if not query_text:
        return None

    normalized = " ".join(query_text.lower().strip().split())
    if not normalized:
        return None

    for item in SMALL_TALK_RESPONSES:
        if normalized in item["triggers"]:
            return item["response"]

    if any(normalized.startswith(prefix) for prefix in ("hi ", "hello ", "hey ")):
        return SMALL_TALK_RESPONSES[0]["response"]

    return None


def normalize_display_value(value: Optional[str], fallback: str = "Not specified") -> str:
    if value is None:
        return fallback

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in {"unknown", "n/a", "na", "none", "null"}:
            return fallback
        return cleaned

    return str(value)


SANITIZE_TRANSLATION = str.maketrans({
    "#": "",
    "_": "",
    "*": "",
    ".": ""
})


def sanitize_answer_text(text: Optional[str]) -> str:
    if not text:
        return ""

    sanitized = text.translate(SANITIZE_TRANSLATION)
    sanitized = re.sub(r"\bunknown\b", "Not specified", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r" {2,}", " ", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


AADHAAR_NUMBER_REGEX = re.compile(r"(?:(?:\d{4}\s\d{4}\s\d{4})|\d{12})")
LEGIT_QUERY_KEYWORDS = {
    "legit",
    "legitimate",
    "authentic",
    "genuine",
    "real",
    "fake",
    "valid",
    "validity",
    "authenticity",
    "original",
    "fraud"
}


def _format_aadhaar_number(raw: str) -> str:
    digits_only = re.sub(r"\D", "", raw or "")
    if len(digits_only) != 12:
        return raw.strip()
    return f"{digits_only[0:4]} {digits_only[4:8]} {digits_only[8:12]}"


def _collect_aadhaar_numbers(contexts: List[Dict]) -> List[Dict]:
    matches = []
    for idx, ctx in enumerate(contexts):
        text_blob = " ".join([
            ctx.get("raw_text", ""),
            ctx.get("text", "")
        ])
        if not text_blob:
            continue
        raw_numbers = AADHAAR_NUMBER_REGEX.findall(text_blob)
        formatted_numbers = []
        for number in raw_numbers:
            formatted = _format_aadhaar_number(number)
            if formatted not in formatted_numbers:
                formatted_numbers.append(formatted)
        if formatted_numbers:
            matches.append({
                "index": idx,
                "numbers": formatted_numbers
            })
    return matches


def _collect_legitimacy_features(contexts: List[Dict]) -> Dict[int, List[str]]:
    features_by_index: Dict[int, List[str]] = {}
    for idx, ctx in enumerate(contexts):
        text_blob = " ".join([
            ctx.get("raw_text", ""),
            ctx.get("text", "")
        ]).lower()
        if not text_blob:
            continue

        features = []
        if "secure qr" in text_blob or "offline verification" in text_blob:
            features.append("mentions Secure QR code verification guidance")
        if "uidai" in text_blob or "unique identification authority" in text_blob:
            features.append("references UIDAI oversight")
        if "aadhaar" in text_blob:
            features.append("includes Aadhaar branding")
        if "proof of identity" in text_blob:
            features.append("states it is proof of identity")

        if features:
            features_by_index[idx] = features

    return features_by_index


def _build_citation_from_context(ctx: Dict, source_number: int) -> Dict:
    snippet_source = ctx.get("raw_text") or ctx.get("text") or ""
    return {
        "source_id": ctx.get("id"),
        "doc_id": normalize_display_value(ctx.get("doc_id"), ""),
        "source_type": normalize_display_value(ctx.get("type")),
        "source_name": normalize_display_value(ctx.get("source")),
        "location": normalize_display_value(ctx.get("page")),
        "text_snippet": sanitize_answer_text(snippet_source)[:200],
        "source_number": source_number
    }


def _build_special_answer(query: str, contexts: List[Dict]) -> Optional[Dict]:
    if not contexts:
        return None

    lowered_query = query.lower()
    needs_legitimacy = any(keyword in lowered_query for keyword in LEGIT_QUERY_KEYWORDS)
    needs_aadhaar_number = ("aadhaar" in lowered_query and "number" in lowered_query) or (
        "12" in lowered_query and "digit" in lowered_query
    )

    aadhaar_matches = _collect_aadhaar_numbers(contexts)
    legitimacy_features = _collect_legitimacy_features(contexts)

    # Skip if no special handling is required or no supporting context
    if not needs_legitimacy and not (needs_aadhaar_number and aadhaar_matches):
        return None

    citation_order: List[int] = []
    for item in aadhaar_matches:
        idx = item["index"]
        if idx not in citation_order:
            citation_order.append(idx)
    for idx in legitimacy_features.keys():
        if idx not in citation_order:
            citation_order.append(idx)

    if not citation_order:
        return None

    citations: List[Dict] = []
    index_to_source_number: Dict[int, int] = {}
    for idx in citation_order:
        ctx = contexts[idx]
        citation = _build_citation_from_context(ctx, len(citations) + 1)
        citations.append(citation)
        index_to_source_number[idx] = citation["source_number"]

    answer_parts: List[str] = []

    if needs_legitimacy:
        feature_sentences = []
        for idx, features in legitimacy_features.items():
            source_number = index_to_source_number.get(idx)
            if not source_number:
                continue
            described = "; ".join(sorted(set(features)))
            if described:
                feature_sentences.append(f"{described} [Source {source_number}]")

        if feature_sentences:
            feature_summary = "; ".join(feature_sentences)
            answer_parts.append(
                "I cannot guarantee the document's authenticity without scanning the Secure QR code, but the captured text "
                f"{feature_summary}."
            )
        else:
            # Reference the first available source for the disclaimer
            first_source = index_to_source_number.get(citation_order[0])
            answer_parts.append(
                "I cannot confirm authenticity without offline verification of the Secure QR code [Source "
                f"{first_source}]"
            )

    if needs_aadhaar_number and aadhaar_matches:
        first_match = aadhaar_matches[0]
        primary_number = first_match["numbers"][0]
        source_number = index_to_source_number.get(first_match["index"], citations[0]["source_number"])
        answer_parts.append(
            f"The Aadhaar number visible in the image is {primary_number} [Source {source_number}]"
        )

    if not answer_parts:
        return None

    combined_answer = " ".join(answer_parts)

    return {
        "answer": combined_answer,
        "citations": citations,
        "num_sources_used": len(citations),
        "already_formatted": True
    }


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    search_type: Optional[str] = "hybrid"  # hybrid, semantic, keyword, cross_modal
    filters: Optional[Dict] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict]
    num_sources: int
    processing_time_ms: float
    search_results: List[Dict]
    stage_timings: Optional[Dict[str, float]] = None


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
    timings: Dict[str, float] = {}
    normalized_query = request.query.strip()
    small_talk_reply = match_small_talk(normalized_query)

    if small_talk_reply:
        processing_time = (time.time() - start_time) * 1000
        if metadata_store:
            metadata_store.log_query({
                "query": normalized_query,
                "type": "small_talk",
                "num_results": 0,
                "response_time_ms": processing_time,
                "stage_timings": {"total_ms": processing_time}
            })

        return QueryResponse(
            query=normalized_query,
            answer=small_talk_reply,
            citations=[],
            num_sources=0,
            processing_time_ms=processing_time,
            search_results=[],
            stage_timings={"total_ms": processing_time}
        )
    
    try:
        # Analyze query
        analysis_start = time.time()
        query_analysis = query_router.analyze_query(normalized_query)
        explicit_modalities = query_analysis.get("explicit_modalities", [])
        default_modality_order = ["text", "image", "audio"]
        modality_order = []
        for modality in explicit_modalities:
            if modality not in modality_order:
                modality_order.append(modality)
        for modality in default_modality_order:
            if modality not in modality_order:
                modality_order.append(modality)
        timings["analysis_ms"] = (time.time() - analysis_start) * 1000
        
        # Perform search based on analysis
        search_start = time.time()
        cross_modal_payload: Dict[str, List[Dict]] = {}
        if query_analysis["search_strategy"] == "cross_modal" or request.search_type == "cross_modal":
            per_type_k = max(2, min(5, request.top_k))
            cross_modal_payload = search_engine.cross_modal_search(
                normalized_query,
                top_k_per_type=per_type_k
            )
            all_results = []
            for modality in modality_order:
                modality_results = cross_modal_payload.get(modality, [])
                for item in modality_results:
                    enriched = item.copy()
                    enriched.setdefault("metadata", {})
                    enriched["modality"] = modality
                    all_results.append(enriched)
        else:
            # Single modality search
            target_candidates = query_analysis["target_modalities"] or ["text"]
            target_modality = explicit_modalities[0] if explicit_modalities else target_candidates[0]

            hybrid_results = search_engine.hybrid_search(
                normalized_query,
                collection_type=target_modality,
                top_k=request.top_k
            )

            if not hybrid_results and target_modality != "text":
                target_modality = "text"
                hybrid_results = search_engine.hybrid_search(
                    normalized_query,
                    collection_type=target_modality,
                    top_k=request.top_k
                )
            all_results = []
            for item in hybrid_results:
                enriched = item.copy()
                enriched.setdefault("metadata", {})
                enriched["modality"] = target_modality
                all_results.append(enriched)
        timings["retrieval_ms"] = (time.time() - search_start) * 1000

        priority_lookup = {modality: index for index, modality in enumerate(modality_order)}

        def priority_tuple(result: Dict) -> tuple:
            modality = result.get("modality", "text")
            score = result.get("score") or 0.0
            return (
                priority_lookup.get(modality, len(priority_lookup)),
                -float(score)
            )

        all_results.sort(key=priority_tuple)
        
        # Prepare contexts for LLM
        context_start = time.time()
        contexts = []
        seen_keys = set()
        context_cap = min(4, getattr(settings, "RERANK_TOP_K", 10))

        def summarize_result_text(modality: str, raw_text: str, meta: Dict) -> str:
            text_block = raw_text or ""
            if modality == "image":
                source_name = normalize_display_value(meta.get("source"))
                if text_block:
                    return f"Image '{source_name}' OCR text: {text_block}"
                return f"Image '{source_name}' with no readable text extracted."
            if modality == "audio":
                source_name = normalize_display_value(meta.get("source"))
                start_ts = meta.get("start") or meta.get("timestamp")
                if start_ts:
                    return f"Audio clip '{source_name}' around {start_ts}s: {text_block}"
                return f"Audio clip '{source_name}': {text_block}" if text_block else f"Audio clip '{source_name}' with limited transcript."
            return text_block

        for result in all_results:
            metadata = result.get("metadata", {}) or {}
            doc_identifier = (
                metadata.get("doc_id")
                or metadata.get("image_id")
                or metadata.get("audio_id")
                or metadata.get("source")
                or result.get("id")
            )

            if isinstance(doc_identifier, str):
                candidate = doc_identifier.strip()
                if candidate.lower() in {"", "unknown", "not specified"}:
                    doc_identifier = result.get("id")
                else:
                    doc_identifier = candidate
            else:
                doc_identifier = result.get("id")

            doc_key = doc_identifier or result["id"]

            if doc_key in seen_keys:
                continue

            text_for_context = summarize_result_text(result.get("modality", "text"), result.get("text", ""), metadata)
            if not text_for_context:
                continue

            seen_keys.add(doc_key)
            contexts.append({
                "id": result["id"],
                "text": text_for_context,
                "type": normalize_display_value(metadata.get("type"), result.get("modality", "text")),
                "source": normalize_display_value(metadata.get("source")),
                "page": normalize_display_value(metadata.get("page") or metadata.get("location") or metadata.get("start"), ""),
                "doc_id": normalize_display_value(doc_identifier, ""),
                "score": result.get("score"),
                "raw_text": result.get("text", "")
            })

            if len(contexts) >= context_cap:
                break
        timings["context_build_ms"] = (time.time() - context_start) * 1000
        
        # Generate answer (with Aadhaar-specific shortcut when applicable)
        special_result = _build_special_answer(normalized_query, contexts)
        if special_result:
            llm_result = special_result
            timings["generation_ms"] = 0.0
        else:
            llm_start = time.time()
            llm_result = llm.generate_answer(normalized_query, contexts)
            timings["generation_ms"] = (time.time() - llm_start) * 1000

        sanitized_answer = sanitize_answer_text(llm_result.get("answer"))

        cleaned_citations = []
        for citation in llm_result.get("citations", []):
            cleaned_citations.append({
                "source_id": citation.get("source_id"),
                "doc_id": normalize_display_value(citation.get("doc_id"), ""),
                "source_type": normalize_display_value(citation.get("source_type")),
                "source_name": normalize_display_value(citation.get("source_name")),
                "location": normalize_display_value(citation.get("location")),
                "text_snippet": sanitize_answer_text(citation.get("text_snippet", "")),
                "source_number": citation.get("source_number")
            })

        if (not sanitized_answer) and contexts:
            fallback_summary = llm._build_extractive_summary(normalized_query, contexts)
            sanitized_answer = sanitize_answer_text(fallback_summary.get("answer"))
            cleaned_citations = []
            for citation in fallback_summary.get("citations", []):
                cleaned_citations.append({
                    "source_id": citation.get("source_id"),
                    "doc_id": normalize_display_value(citation.get("doc_id"), ""),
                    "source_type": normalize_display_value(citation.get("source_type")),
                    "source_name": normalize_display_value(citation.get("source_name")),
                    "location": normalize_display_value(citation.get("location")),
                    "text_snippet": sanitize_answer_text(citation.get("text_snippet", "")),
                    "source_number": citation.get("source_number")
                })

        if not sanitized_answer:
            fallback_text = ""
            fallback_source = None

            if cleaned_citations:
                primary_citation = cleaned_citations[0]
                fallback_text = primary_citation.get("text_snippet", "")
                fallback_source = primary_citation.get("source_number")

            if not fallback_text and contexts:
                first_context = contexts[0]
                fallback_text = sanitize_answer_text(first_context.get("raw_text") or first_context.get("text", ""))
                fallback_source = 1 if contexts else None

            if fallback_text:
                sanitized_answer = fallback_text
                if fallback_source and f"[Source {fallback_source}]" not in sanitized_answer:
                    sanitized_answer = f"{sanitized_answer} [Source {fallback_source}]"

            if not sanitized_answer:
                sanitized_answer = "I could not assemble a readable summary from the available context."
        
        processing_time = (time.time() - start_time) * 1000
        timings["total_ms"] = processing_time
        
        # Log query
        metadata_store.log_query({
            "query": normalized_query,
            "type": query_analysis["search_strategy"],
            "num_results": len(all_results),
            "response_time_ms": processing_time,
            "stage_timings": timings
        })
        
        return QueryResponse(
            query=normalized_query,
            answer=sanitized_answer,
            citations=cleaned_citations,
            num_sources=len(cleaned_citations),
            processing_time_ms=processing_time,
            search_results=[{
                "id": r["id"],
                "text": (r.get("text") or "")[:200] + ("..." if (r.get("text") or "").strip() else ""),
                "score": r.get("score"),
                "metadata": r.get("metadata", {}),
                "modality": r.get("modality", "text")
            } for r in all_results[:10]],
            stage_timings=timings
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


@app.get("/documents/{doc_id}/download")
async def download_document(doc_id: str):
    """Stream the original document or media file to the client"""
    doc = metadata_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(doc.get("file_path", ""))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not available on server")

    return FileResponse(file_path, filename=file_path.name)


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