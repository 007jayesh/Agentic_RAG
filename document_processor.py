"""Document processing module for Agentic RAG System"""
import os
import re
import json
import logging
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from config import Config

logger = logging.getLogger(__name__)


class EnhancedDocumentProcessor:
    """Production document processing with advanced chunking"""

    def __init__(self, embeddings_model, config: Config):
        self.embeddings = embeddings_model
        self.config = config

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\d+\n', '\n', text)
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        return text.strip()

    def hybrid_chunking(self, documents: List[Document]) -> List[Document]:
        """Enhanced hybrid chunking with error handling"""
        logger.info("🔄 Starting hybrid chunking process...")

        try:
            # Clean documents
            cleaned_documents = []
            for doc in documents:
                cleaned_text = self.clean_text(doc.page_content)
                if len(cleaned_text.strip()) > 100:
                    doc.page_content = cleaned_text
                    cleaned_documents.append(doc)

            logger.info(f"📄 Cleaned {len(cleaned_documents)} documents")

            # Recursive chunking
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE * 2,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                length_function=len,
            )

            initial_chunks = recursive_splitter.split_documents(cleaned_documents)
            logger.info(f"🔨 Created {len(initial_chunks)} initial chunks")

            # Apply semantic chunking with fallback
            final_chunks = []
            try:
                semantic_chunker = SemanticChunker(
                    self.embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=80
                )

                for i, chunk in enumerate(initial_chunks):
                    if len(chunk.page_content) > 200:
                        try:
                            semantic_chunks = semantic_chunker.create_documents([chunk.page_content])
                            for j, semantic_chunk in enumerate(semantic_chunks):
                                semantic_chunk.metadata.update(chunk.metadata)
                                semantic_chunk.metadata['chunk_id'] = f"{i}_{j}"
                                semantic_chunk.metadata['chunk_type'] = 'hybrid_semantic'
                            final_chunks.extend(semantic_chunks)
                        except Exception as e:
                            logger.warning(f"⚠️ Semantic chunking failed for chunk {i}: {e}")
                            chunk.metadata['chunk_type'] = 'recursive_fallback'
                            final_chunks.append(chunk)
                    else:
                        chunk.metadata['chunk_type'] = 'recursive_only'
                        final_chunks.append(chunk)

            except Exception as e:
                logger.error(f"❌ Semantic chunking completely failed: {e}")
                final_chunks = initial_chunks
                for chunk in final_chunks:
                    chunk.metadata['chunk_type'] = 'recursive_only'

            logger.info(f"🎯 Created {len(final_chunks)} final chunks")
            return final_chunks

        except Exception as e:
            logger.error(f"❌ Document chunking failed: {e}")
            raise

    def add_contextual_metadata(self, chunks: List[Document]) -> List[Document]:
        """Add contextual information for better retrieval"""
        logger.info("🏷️ Adding contextual metadata...")

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'position': i,
                'total_chunks': len(chunks),
                'char_count': len(chunk.page_content),
                'word_count': len(chunk.page_content.split()),
                'chunk_hash': hash(chunk.page_content[:100])  # For deduplication
            })

            # Add neighboring context previews
            if i > 0:
                chunk.metadata['prev_chunk_preview'] = chunks[i-1].page_content[:100] + "..."
            if i < len(chunks) - 1:
                chunk.metadata['next_chunk_preview'] = chunks[i+1].page_content[:100] + "..."

        return chunks


def process_and_create_vectorstore(pdf_path: str, vectorstore_path: Path, config: Config):
    """Process PDF and create vectorstore with OpenAI embeddings"""
    logger.info(f"📖 Processing document: {pdf_path}")

    try:
        # Load documents
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"📄 Loaded {len(documents)} pages")

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model=config.EMBEDDING_MODEL,
            dimensions=config.EMBEDDING_DIMENSIONS
        )

        # Process documents
        processor = EnhancedDocumentProcessor(embeddings, config)
        chunks = processor.hybrid_chunking(documents)
        chunks = processor.add_contextual_metadata(chunks)

        logger.info(f"✨ Created {len(chunks)} chunks")

        # Create vector store
        logger.info("🗂️ Creating FAISS vector store...")
        db = FAISS.from_documents(chunks, embeddings)

        # Save vector store
        index_path = vectorstore_path / "openai_embeddings_index"
        db.save_local(str(index_path))
        logger.info(f"💾 Vector store saved to: {index_path}")

        # Save metadata
        metadata_path = vectorstore_path / "processing_metadata.json"
        metadata = {
            "total_pages": len(documents),
            "total_chunks": len(chunks),
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "processing_timestamp": pd.Timestamp.now().isoformat(),
            "pdf_path": pdf_path
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("✅ Document processing completed successfully!")
        return len(chunks), metadata

    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")
        raise