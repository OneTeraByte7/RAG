"""
PostgreSQL metadata store for document tracking and relationships
"""
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, Text, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger

from config.settings import settings

Base = declarative_base()


class Document(Base):
    """Document metadata table"""
    __tablename__ = "documents"
    
    id = Column(String(16), primary_key=True)
    source = Column(String(255), nullable=False, index=True)
    file_path = Column(String(512), nullable=False)
    doc_type = Column(String(50), nullable=False)  # pdf, docx, image, audio
    file_size = Column(Integer)
    num_pages = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)  # For audio
    processed_at = Column(DateTime, default=datetime.now)
    metadata_json = Column(JSON)
    is_indexed = Column(Boolean, default=False)


class Chunk(Base):
    """Chunk metadata table"""
    __tablename__ = "chunks"
    
    id = Column(String(64), primary_key=True)
    doc_id = Column(String(16), nullable=False, index=True)
    chunk_id = Column(Integer, nullable=False)
    chunk_type = Column(String(50), nullable=False)  # text, image, audio
    text = Column(Text)
    page_num = Column(Integer, nullable=True)
    start_time = Column(Float, nullable=True)  # For audio
    end_time = Column(Float, nullable=True)
    metadata_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)


class Query(Base):
    """Query history table"""
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))  # semantic, keyword, hybrid
    num_results = Column(Integer)
    response_time_ms = Column(Float)
    user_id = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    metadata_json = Column(JSON)


class MetadataStore:
    """PostgreSQL metadata store"""
    
    def __init__(self):
        logger.info("Initializing metadata store")
        
        # Use SQLite instead of PostgreSQL
        db_path = settings.DATA_DIR / "metadata.db"
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=settings.DEBUG
        )
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info("Metadata store initialized")
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def add_document(self, doc_data: Dict) -> None:
        """
        Add document metadata
        
        Args:
            doc_data: Document information dict
        """
        session = self.get_session()
        try:
            now = datetime.now()
            existing = session.query(Document).filter(Document.id == doc_data["doc_id"]).first()

            if existing:
                existing.source = doc_data["source"]
                existing.file_path = doc_data.get("file_path", "")
                existing.doc_type = doc_data["type"]
                existing.file_size = doc_data.get("metadata", {}).get("file_size")
                existing.num_pages = (
                    doc_data.get("total_pages") or
                    doc_data.get("metadata", {}).get("num_paragraphs")
                )
                existing.duration = doc_data.get("duration")
                existing.metadata_json = doc_data.get("metadata", {})
                existing.processed_at = now
                existing.is_indexed = False
                logger.info(f"Updated existing document metadata: {doc_data['doc_id']}")
            else:
                doc = Document(
                    id=doc_data["doc_id"],
                    source=doc_data["source"],
                    file_path=doc_data.get("file_path", ""),
                    doc_type=doc_data["type"],
                    file_size=doc_data.get("metadata", {}).get("file_size"),
                    num_pages=doc_data.get("total_pages") or doc_data.get("metadata", {}).get("num_paragraphs"),
                    duration=doc_data.get("duration"),
                    metadata_json=doc_data.get("metadata", {}),
                    processed_at=now,
                    is_indexed=False
                )
                
                session.add(doc)
            session.commit()
            logger.info(f"Document metadata stored: {doc_data['doc_id']}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding document metadata: {e}")
            raise
        finally:
            session.close()
    
    def update_document_indexed(self, doc_id: str, indexed: bool = True) -> None:
        """Mark document as indexed"""
        session = self.get_session()
        try:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.is_indexed = indexed
                session.commit()
                logger.info(f"Updated document {doc_id} indexed status: {indexed}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating document: {e}")
        finally:
            session.close()
    
    def add_chunks(self, chunks: List[Dict], doc_id: str) -> None:
        """
        Add chunk metadata
        
        Args:
            chunks: List of chunk dicts
            doc_id: Parent document ID
        """
        session = self.get_session()
        try:
            deleted = session.query(Chunk).filter(Chunk.doc_id == doc_id).delete(synchronize_session=False)
            if deleted:
                logger.info(f"Removed {deleted} existing chunks for document {doc_id}")

            chunk_objs = []
            for chunk in chunks:
                chunk_obj = Chunk(
                    id=f"{doc_id}_{chunk['chunk_id']}",
                    doc_id=doc_id,
                    chunk_id=chunk["chunk_id"],
                    chunk_type=chunk.get("metadata", {}).get("type", "text"),
                    text=chunk.get("text", ""),
                    page_num=chunk.get("metadata", {}).get("page"),
                    start_time=chunk.get("start"),
                    end_time=chunk.get("end"),
                    metadata_json=chunk.get("metadata", {})
                )
                chunk_objs.append(chunk_obj)
            
            session.bulk_save_objects(chunk_objs)
            session.commit()
            logger.info(f"Added {len(chunk_objs)} chunks for document {doc_id}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding chunks: {e}")
            raise
        finally:
            session.close()

    def delete_document(self, doc_id: str) -> Dict[str, int]:
        """Delete a document and its chunks from the metadata store."""
        session = self.get_session()
        try:
            chunk_count = session.query(Chunk).filter(Chunk.doc_id == doc_id).delete(synchronize_session=False)
            doc_count = session.query(Document).filter(Document.id == doc_id).delete(synchronize_session=False)
            session.commit()
            logger.info(f"Deleted document {doc_id} (doc rows: {doc_count}, chunks: {chunk_count})")
            return {"documents": doc_count, "chunks": chunk_count}
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting document {doc_id}: {e}")
            raise
        finally:
            session.close()
    
    def log_query(self, query_data: Dict) -> None:
        """
        Log query for analytics
        
        Args:
            query_data: Query information
        """
        session = self.get_session()
        try:
            query = Query(
                query_text=query_data["query"],
                query_type=query_data.get("type", "semantic"),
                num_results=query_data.get("num_results", 0),
                response_time_ms=query_data.get("response_time_ms", 0),
                user_id=query_data.get("user_id"),
                metadata_json=query_data.get("metadata", {})
            )
            
            session.add(query)
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging query: {e}")
        finally:
            session.close()
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document metadata by ID"""
        session = self.get_session()
        try:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                return {
                    "id": doc.id,
                    "source": doc.source,
                    "file_path": doc.file_path,
                    "doc_type": doc.doc_type,
                    "file_size": doc.file_size,
                    "num_pages": doc.num_pages,
                    "duration": doc.duration,
                    "processed_at": doc.processed_at.isoformat(),
                    "is_indexed": doc.is_indexed,
                    "metadata": doc.metadata_json
                }
            return None
        finally:
            session.close()
    
    def get_all_documents(self, doc_type: Optional[str] = None) -> List[Dict]:
        """Get all documents, optionally filtered by type"""
        session = self.get_session()
        try:
            query = session.query(Document)
            if doc_type:
                query = query.filter(Document.doc_type == doc_type)
            
            docs = query.all()
            return [
                {
                    "id": doc.id,
                    "source": doc.source,
                    "doc_type": doc.doc_type,
                    "processed_at": doc.processed_at.isoformat(),
                    "is_indexed": doc.is_indexed
                }
                for doc in docs
            ]
        finally:
            session.close()
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """Get chunk by ID"""
        session = self.get_session()
        try:
            chunk = session.query(Chunk).filter(Chunk.id == chunk_id).first()
            if chunk:
                return {
                    "id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "page_num": chunk.page_num,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "metadata": chunk.metadata_json
                }
            return None
        finally:
            session.close()
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        session = self.get_session()
        try:
            total_docs = session.query(Document).count()
            indexed_docs = session.query(Document).filter(Document.is_indexed == True).count()
            total_chunks = session.query(Chunk).count()
            total_queries = session.query(Query).count()
            
            return {
                "total_documents": total_docs,
                "indexed_documents": indexed_docs,
                "total_chunks": total_chunks,
                "total_queries": total_queries
            }
        finally:
            session.close()