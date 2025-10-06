import argparse
import shlex
import uvicorn
from loguru import logger
from pathlib import Path
import sys
import os
import hashlib
from typing import Optional, Dict, Callable

os.environ['HF_HOME'] = 'F:/Smart India Hackathon_2025/SIH25231/models_cache'
os.environ['TRANSFORMERS_CACHE'] = 'F:/Smart India Hackathon_2025/SIH25231/models_cache/transformers'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'F:/Smart India Hackathon_2025/SIH25231/models_cache/hub'
os.environ['TORCH_HOME'] = 'F:/Smart India Hackathon_2025/SIH25231/models_cache/torch'


from config.settings import settings


def setup_logging():
    """Configure logging"""
    logger.add(
        settings.LOG_FILE,
        rotation="500 MB",
        retention="10 days",
        level=settings.LOG_LEVEL
    )


def run_api_server():
    """Run the FastAPI server"""
    logger.info(f"Starting API server on {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )


def run_batch_processing(directory: str):
    """Process all files in a directory"""
    from processing.document_processor import DocumentProcessor
    from processing.image_processor import ImageProcessor
    from models.embeddings import MultimodalEmbedder
    from database.vector_store import VectorStore
    from database.metadata_store import MetadataStore
    
    logger.info(f"Batch processing directory: {directory}")
    
    # Initialize components
    embedder = MultimodalEmbedder()
    vector_store = VectorStore()
    metadata_store = MetadataStore()
    doc_processor = DocumentProcessor()
    img_processor = ImageProcessor()
    
    # Process all files
    directory_path = Path(directory)
    
    # Process documents
    for ext in ['.pdf', '.docx']:
        for file_path in directory_path.glob(f"*{ext}"):
            try:
                logger.info(f"Processing {file_path}")
                chunks = doc_processor.process_document(str(file_path))
                
                # Generate embeddings
                texts = [c["text"] for c in chunks]
                embeddings = embedder.text_embedder.embed_text(texts)
                
                # Store
                vector_store.add_text_chunks(chunks, embeddings)
                
                doc_id = chunks[0]["metadata"]["doc_id"]
                metadata_store.add_document({
                    "doc_id": doc_id,
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "type": ext.replace('.', ''),
                    "metadata": {}
                })
                metadata_store.add_chunks(chunks, doc_id)
                metadata_store.update_document_indexed(doc_id, True)
                
                logger.info(f"✅ Processed {file_path.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
    
    # Process images
    for ext in ['.jpg', '.png', '.jpeg']:
        for file_path in directory_path.glob(f"*{ext}"):
            try:
                logger.info(f"Processing {file_path}")
                image_data = img_processor.process_and_prepare(str(file_path))
                
                # Generate embedding
                embedding = embedder.image_embedder.embed_image(str(file_path))[0]
                
                # Store
                vector_store.add_images([image_data], embedding.reshape(1, -1))
                
                metadata_store.add_document({
                    "doc_id": image_data["id"],
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "type": "image",
                    "metadata": image_data["metadata"]
                })
                metadata_store.update_document_indexed(image_data["id"], True)
                
                logger.info(f"✅ Processed {file_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
    
    vector_store.persist()
    logger.info("✅ Batch processing completed")


def ingest_document_cli(
    file_path: str,
    doc_processor,
    embedder,
    vector_store,
    metadata_store,
    rebuilders: Optional[Dict[str, Callable[[], None]]] = None,
    doc_id_override: Optional[str] = None
):
    """Process and store a single document from the CLI."""
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    text_exts = {".pdf", ".docx", ".doc"}
    audio_exts = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".webm"}

    rebuilders = rebuilders or {}
    doc_type = None
    chunk_count = 0
    doc_id = None

    if suffix in text_exts:
        if suffix == ".pdf":
            doc_type = "pdf"
        elif suffix == ".doc":
            doc_type = "doc"
        else:
            doc_type = "docx"
        chunks = doc_processor.process_document(str(path), doc_id_override)
        if not chunks:
            raise ValueError("No content extracted from document")

        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.text_embedder.embed_text(chunk_texts)

        vector_store.add_text_chunks(chunks, embeddings)

        doc_id = chunks[0]["metadata"]["doc_id"]
        doc_metadata = {
            "num_chunks": len(chunks),
            "file_size": path.stat().st_size,
            "doc_type": doc_type
        }

        metadata_store.add_document({
            "doc_id": doc_id,
            "source": path.name,
            "file_path": str(path),
            "type": doc_type,
            "metadata": doc_metadata
        })
        metadata_store.add_chunks(chunks, doc_id)
        metadata_store.update_document_indexed(doc_id, True)

        rebuild_cb = rebuilders.get("text")
        if rebuild_cb:
            rebuild_cb()

        chunk_count = len(chunks)

    elif suffix in audio_exts:
        from models.audio import AudioProcessor

        doc_type = "audio"
        audio_processor = AudioProcessor()

        override = doc_id_override.strip() if doc_id_override and doc_id_override.strip() else None
        doc_id = override or hashlib.md5(str(path).encode()).hexdigest()[:16]

        audio_data = audio_processor.process_audio(
            str(path),
            doc_id=doc_id,
            source=path.name
        )

        chunk_records = [chunk for chunk in audio_data.get("chunks", []) if chunk.get("text")]
        if not chunk_records:
            raise ValueError("No transcription content extracted from audio")

        chunk_texts = [chunk["text"] for chunk in chunk_records]
        embeddings = embedder.text_embedder.embed_text(chunk_texts)

        vector_store.add_audio_chunks(
            chunk_records,
            embeddings,
            {
                "audio_id": doc_id,
                "source": path.name,
                "duration": audio_data.get("duration")
            }
        )

        metadata_store.add_document({
            "doc_id": doc_id,
            "source": path.name,
            "file_path": str(path),
            "type": "audio",
            "duration": audio_data.get("duration"),
            "metadata": {
                "file_size": path.stat().st_size,
                "language": audio_data.get("language"),
                "num_segments": len(audio_data.get("segments", [])),
                "num_chunks": len(chunk_records)
            }
        })
        metadata_store.add_chunks(chunk_records, doc_id)
        metadata_store.update_document_indexed(doc_id, True)

        rebuild_cb = rebuilders.get("audio")
        if rebuild_cb:
            rebuild_cb()

        chunk_count = len(chunk_records)

    else:
        raise ValueError("Only PDF, DOCX, and common audio files are supported")

    return doc_id, chunk_count, doc_type


def _print_cli_help():
    print("\nCommands:")
    print("  /add <path> [--id <custom_id>]  ➜ Ingest a PDF, DOCX, or audio file into the knowledge base")
    print("  /list                           ➜ List indexed documents")
    print("  /reindex                        ➜ Rebuild BM25 indexes for all modalities")
    print("  /help                           ➜ Show this help message")
    print("  /exit                           ➜ Quit the CLI")
    print("\nType any other text to run a query against indexed documents.\n")


def run_cli_query():
    """Run interactive CLI query interface"""
    from models.embeddings import MultimodalEmbedder
    from models.llm import LLMGenerator
    from retrieval.hybrid_search import HybridSearchEngine
    from retrieval.query_router import QueryRouter
    from processing.document_processor import DocumentProcessor
    from database.metadata_store import MetadataStore
    
    logger.info("Starting CLI query interface")

    embedder = MultimodalEmbedder()
    llm = LLMGenerator()
    search_engine = HybridSearchEngine()
    query_router = QueryRouter()
    metadata_store = MetadataStore()
    doc_processor = DocumentProcessor()

    print("\n" + "=" * 60)
    print("Multimodal RAG System - CLI Interface")
    print("=" * 60)
    _print_cli_help()

    while True:
        try:
            user_input = input("\n🗨️  Input (/help for commands): ").strip()

            if not user_input:
                continue

            if user_input.startswith(('/', ':')):
                try:
                    parts = shlex.split(user_input[1:])
                except ValueError as exc:
                    print(f"❌ Unable to parse command: {exc}")
                    continue

                if not parts:
                    continue

                command = parts[0].lower()
                args = parts[1:]

                if command in {"exit", "quit", "q"}:
                    print("Goodbye!")
                    break
                if command in {"help", "h"}:
                    _print_cli_help()
                    continue
                if command == "list":
                    docs = metadata_store.get_all_documents()
                    if not docs:
                        print("No documents indexed yet.")
                    else:
                        print("\nIndexed documents:")
                        for doc in docs:
                            status = "indexed" if doc.get("is_indexed") else "pending"
                            print(f"  • {doc['id']} ({doc['source']}) - {doc['doc_type']} [{status}]")
                    continue
                if command == "reindex":
                    search_engine.build_bm25_index("text")
                    search_engine.build_bm25_index("image")
                    search_engine.build_bm25_index("audio")
                    print("✅ Rebuilt BM25 index for text, image, and audio collections")
                    continue
                if command == "add":
                    if not args:
                        print("Usage: /add <path> [--id <custom_id>]")
                        continue

                    doc_path = None
                    doc_id_override = None
                    idx = 0
                    while idx < len(args):
                        token = args[idx]
                        if token in {"--id", "-i"}:
                            if idx + 1 >= len(args):
                                print("❌ Missing value for --id")
                                doc_path = None
                                break
                            doc_id_override = args[idx + 1]
                            idx += 2
                        elif token.startswith("--id="):
                            doc_id_override = token.split("=", 1)[1]
                            idx += 1
                        elif doc_path is None:
                            doc_path = token
                            idx += 1
                        else:
                            # Additional positional tokens are unexpected
                            print("❌ Unexpected extra argument. Wrap paths with spaces in quotes.")
                            doc_path = None
                            break

                    if not doc_path:
                        continue

                    try:
                        doc_id, chunk_count, doc_type = ingest_document_cli(
                            doc_path,
                            doc_processor,
                            embedder,
                            search_engine.vector_store,
                            metadata_store,
                            rebuilders={
                                "text": lambda: search_engine.build_bm25_index("text"),
                                "image": lambda: search_engine.build_bm25_index("image"),
                                "audio": lambda: search_engine.build_bm25_index("audio")
                            },
                            doc_id_override=doc_id_override
                        )
                        if doc_type == "audio":
                            label = "Audio"
                        elif doc_type == "pdf":
                            label = "PDF"
                        elif doc_type == "docx":
                            label = "DOCX"
                        elif doc_type == "doc":
                            label = "DOC"
                        else:
                            label = doc_type.upper() if doc_type else "Document"

                        print(f"✅ {label} '{doc_id}' indexed with {chunk_count} chunks")
                    except Exception as exc:
                        logger.error(f"CLI ingestion failed: {exc}")
                        print(f"❌ Failed to index document: {exc}")
                    continue

                print(f"❌ Unknown command: {command}. Type /help for options.")
                continue

            if user_input.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            query = user_input

            analysis = query_router.analyze_query(query)
            print(f"\n📊 Strategy: {analysis['search_strategy']}")

            detected_modalities = analysis.get("target_modalities") or ["text"]
            modality_order = ["text", "image", "audio"]
            modalities = [m for m in modality_order if m in detected_modalities]
            modalities.extend([m for m in detected_modalities if m not in modalities])
            modalities = modalities or ["text"]
            print(f"🎯 Modalities: {', '.join(modalities)}")

            results = []
            if analysis["search_strategy"] == "cross_modal" or len(modalities) > 1:
                cross_results = search_engine.cross_modal_search(query, top_k_per_type=5)
                for modality in modalities:
                    results.extend(cross_results.get(modality, []))
            else:
                target_modality = modalities[0]
                results = search_engine.hybrid_search(query, target_modality, top_k=10)

            if not results:
                print("❌ No results found")
                continue

            def _to_float(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            results.sort(key=lambda r: r.get("score", 0), reverse=True)

            found_modalities = sorted({
                (r.get("metadata", {}) or {}).get("type", modalities[0])
                for r in results
            })
            print(f"\n📚 Found {len(results)} results across {', '.join(found_modalities)}")

            for idx, result in enumerate(results[:5], start=1):
                meta = result.get("metadata", {})
                result_type = meta.get("type", modalities[0])
                source = meta.get("source", meta.get("doc_id", "unknown"))

                location_bits = []
                if result_type in {"pdf", "docx", "text"}:
                    page = meta.get("page")
                    if page:
                        location_bits.append(f"page {page}")
                if result_type == "audio":
                    start_val = _to_float(meta.get("start"))
                    end_val = _to_float(meta.get("end"))
                    if start_val is not None:
                        if end_val is not None and end_val != start_val:
                            location_bits.append(f"{start_val:.1f}s–{end_val:.1f}s")
                        else:
                            location_bits.append(f"{start_val:.1f}s")

                location = f" ({', '.join(location_bits)})" if location_bits else ""
                display_type = {
                    "text": "TEXT",
                    "audio": "AUDIO",
                    "image": "IMAGE",
                    "pdf": "PDF",
                    "docx": "DOCX",
                    "doc": "DOC"
                }.get(result_type, str(result_type).upper())
                snippet = (result.get("text", "")[:200] or "").replace("\n", " ")
                print(f"  {idx}. [{display_type}] {source}{location}: {snippet}...")

            contexts = []
            for r in results[:5]:
                meta = r.get("metadata", {}) or {}
                result_type = meta.get("type", modalities[0])
                source = meta.get("source", meta.get("doc_id", "unknown"))
                context_entry = {
                    "id": r.get("id"),
                    "text": r.get("text"),
                    "type": result_type,
                    "source": source,
                    "page": meta.get("page", "")
                }

                if result_type == "audio":
                    start_val = _to_float(meta.get("start"))
                    end_val = _to_float(meta.get("end"))
                    if start_val is not None:
                        if end_val is not None and end_val != start_val:
                            timestamp = f"{start_val:.1f}s – {end_val:.1f}s"
                        else:
                            timestamp = f"{start_val:.1f}s"
                        context_entry["timestamp"] = timestamp
                        context_entry["page"] = timestamp

                contexts.append(context_entry)

            print("\n🤖 Generating answer...")
            answer = llm.generate_answer(query, contexts)

            print("\n" + "=" * 60)
            print("ANSWER:")
            print("=" * 60)
            print(answer["answer"])

            if answer["citations"]:
                print("\n📖 CITATIONS:")
                for i, citation in enumerate(answer["citations"], 1):
                    print(f"\n[{i}] {citation['source_name']} ({citation['location']})")
                    print(f"    {citation['text_snippet']}")

            print("\n" + "=" * 60)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"❌ Error: {e}")


def main():
    """Main entry point"""
    setup_logging()

    if len(sys.argv) == 1:
        logger.info("No CLI arguments supplied; starting API server in simple mode.")
        print(f"\nServer running at http://{settings.API_HOST}:{settings.API_PORT} (Ctrl+C to stop)\n")
        run_api_server()
        return
    
    parser = argparse.ArgumentParser(description="Multimodal RAG System")
    parser.add_argument(
        "--mode",
        choices=["api", "batch", "cli"],
        default="api",
        help="Run mode: api (server), batch (process files), cli (interactive)"
    )
    parser.add_argument(
        "--directory",
        help="Directory for batch processing"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "api":
            run_api_server()
        elif args.mode == "batch":
            if not args.directory:
                logger.error("--directory required for batch mode")
                sys.exit(1)
            run_batch_processing(args.directory)
        elif args.mode == "cli":
            run_cli_query()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()