import argparse
import uvicorn
from loguru import logger
from pathlib import Path
import sys
import os

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
        reload=settings.DEBUG,
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


def run_cli_query():
    """Run interactive CLI query interface"""
    from models.embeddings import MultimodalEmbedder
    from models.llm import LLMGenerator
    from retrieval.hybrid_search import HybridSearchEngine
    from retrieval.query_router import QueryRouter
    
    logger.info("Starting CLI query interface")
    
    # Initialize components
    embedder = MultimodalEmbedder()
    llm = LLMGenerator()
    search_engine = HybridSearchEngine()
    query_router = QueryRouter()
    
    print("\n" + "="*60)
    print("Multimodal RAG System - CLI Interface")
    print("="*60)
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Analyze query
            analysis = query_router.analyze_query(query)
            print(f"\n📊 Strategy: {analysis['search_strategy']}")
            print(f"🎯 Modalities: {', '.join(analysis['target_modalities'])}")
            
            # Search
            results = search_engine.hybrid_search(query, "text", top_k=10)
            
            if not results:
                print("❌ No results found")
                continue
            
            print(f"\n📚 Found {len(results)} results")
            
            # Generate answer
            contexts = [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "type": r.get("metadata", {}).get("type", "text"),
                    "source": r.get("metadata", {}).get("source", "unknown"),
                    "page": r.get("metadata", {}).get("page", "")
                }
                for r in results[:5]
            ]
            
            print("\n🤖 Generating answer...")
            answer = llm.generate_answer(query, contexts)
            
            print("\n" + "="*60)
            print("ANSWER:")
            print("="*60)
            print(answer["answer"])
            
            if answer["citations"]:
                print("\n📖 CITATIONS:")
                for i, citation in enumerate(answer["citations"], 1):
                    print(f"\n[{i}] {citation['source_name']} ({citation['location']})")
                    print(f"    {citation['text_snippet']}")
            
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"❌ Error: {e}")


def main():
    """Main entry point"""
    setup_logging()
    
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