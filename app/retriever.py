"""
FAISS-based retriever for clinical documents.
Handles vector search and document retrieval for RAG pipeline.
"""

import os
import logging
from typing import List, Optional, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class ClinicalRetriever:
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.initialized = False

    async def initialize(self):
        try:
            logger.info("Initializing clinical retriever...")

            self.embeddings = HuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v1",
                model_kwargs={
                    'device': 'cuda' if self._cuda_available() else 'cpu',
                    'trust_remote_code': True  # Required to allow execution of model-specific code
                },
                encode_kwargs={'normalize_embeddings': True}
            )

            if os.path.exists(self.index_path):
                await self._load_index()
            else:
                logger.warning(f"No existing index found at {self.index_path}")
                await self._create_empty_index()

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            self.initialized = True
            logger.info("Clinical retriever initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise

    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    async def _load_index(self):
        try:
            logger.info(f"Loading FAISS index from {self.index_path}")
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            logger.info("Creating new empty index...")
            await self._create_empty_index()

    async def _create_empty_index(self):
        try:
            dummy_doc = Document(
                page_content="Dummy document for initialization",
                metadata={"source": "initialization", "type": "dummy"}
            )
            self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            self.vectorstore.save_local(self.index_path)
            logger.info("Empty FAISS index created")
        except Exception as e:
            logger.error(f"Error creating empty index: {e}")
            raise

    def get_retriever(self):
        if not self.initialized:
            raise RuntimeError("Retriever not initialized")
        return self.retriever

    async def add_documents(self, documents: List[Document]):
        try:
            if not self.initialized:
                raise RuntimeError("Retriever not initialized")

            logger.info(f"Adding {len(documents)} documents to index")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            split_docs = text_splitter.split_documents(documents)
            self.vectorstore.add_documents(split_docs)
            self.vectorstore.save_local(self.index_path)

            logger.info(f"Added {len(split_docs)} document chunks to index")

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def search(self, query: str, k: int = 5) -> List[Document]:
        try:
            if not self.initialized:
                raise RuntimeError("Retriever not initialized")

            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc for doc in docs if doc.metadata.get("type") != "dummy"]

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    async def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        try:
            if not self.initialized:
                raise RuntimeError("Retriever not initialized")

            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            return [(doc, score) for doc, score in docs_with_scores if doc.metadata.get("type") != "dummy"]

        except Exception as e:
            logger.error(f"Error searching with scores: {e}")
            return []

    async def get_stats(self) -> Dict:
        try:
            if not self.initialized:
                return {"status": "not_initialized"}

            all_docs = self.vectorstore.similarity_search("", k=1000)
            real_docs = [doc for doc in all_docs if doc.metadata.get("type") != "dummy"]

            return {
                "status": "initialized",
                "total_documents": len(real_docs),
                "index_path": self.index_path,
                "embedding_model": "nomic-ai/nomic-embed-text-v1"
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}

def setup_retriever(index_path: str = "faiss_index") -> ClinicalRetriever:
    retriever = ClinicalRetriever(index_path=index_path)
    return retriever
