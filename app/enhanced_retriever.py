"""
Enhanced Medical Document Retriever with intelligent chunking,
semantic search, and clinical context awareness.
UPDATED: Integrated with Vector Store Manager and Patient Data Embeddings.
"""

import os
import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import re
import json

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PDFPlumberLoader,
    UnstructuredWordDocumentLoader
)
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

@dataclass
class ClinicalDocument:
    """Enhanced document class with clinical metadata."""
    content: str
    metadata: Dict[str, Any]
    clinical_domain: str
    evidence_level: str  # A, B, C, D based on medical evidence
    last_updated: datetime
    relevance_score: float = 0.0

class MedicalTextSplitter(RecursiveCharacterTextSplitter):
    """Specialized text splitter for medical documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        # Medical-specific separators
        medical_separators = [
            "\n\n# ",        # Markdown headers
            "\n\n## ",       # Subheaders
            "\n\n### ",      # Sub-subheaders
            "\n\nDiagnosis:", # Clinical sections
            "\n\nTreatment:",
            "\n\nProcedure:",
            "\n\nContraindications:",
            "\n\nSide Effects:",
            "\n\nMonitoring:",
            "\n\n",          # Double newlines
            "\n",            # Single newlines
            ". ",            # Sentences
            " ",             # Words
            ""               # Characters
        ]
        
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=medical_separators,
            **kwargs
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text with medical context preservation."""
        # Preprocess to identify clinical sections
        text = self._preserve_clinical_context(text)
        
        # Use parent's split_text
        chunks = super().split_text(text)
        
        # Post-process to ensure clinical coherence
        processed_chunks = []
        for chunk in chunks:
            processed_chunk = self._enhance_chunk_context(chunk)
            processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def _preserve_clinical_context(self, text: str) -> str:
        """Preserve important clinical context during splitting."""
        # Mark important clinical patterns that shouldn't be split
        clinical_patterns = [
            r'(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)(?:/(?:day|dose|kg))?)',  # Dosages
            r'(ICD-?10?[:\-\s]*[A-Z]\d{2}(?:\.\d+)?)',                        # ICD codes
            r'(CPT[:\-\s]*\d{5})',                                            # CPT codes
            r'(\b(?:contraindicated?|adverse|side effect|warning|caution)\b)', # Safety info
        ]
        
        for pattern in clinical_patterns:
            # Replace spaces in matches with special markers to prevent splitting
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                original = match.group(0)
                preserved = original.replace(' ', '§SPACE§')
                text = text.replace(original, preserved)
        
        return text
    
    def _enhance_chunk_context(self, chunk: str) -> str:
        """Enhance chunk with preserved context."""
        # Restore preserved spaces
        chunk = chunk.replace('§SPACE§', ' ')
        
        # Add context clues if chunk seems to start mid-sentence
        if chunk and not chunk[0].isupper() and not chunk.startswith(('•', '-', '1.', '2.')):
            chunk = "[...continued] " + chunk
        
        return chunk

class ClinicalEmbeddingStrategy:
    """Strategy for creating embeddings optimized for clinical content."""
    
    def __init__(self, base_embeddings: Embeddings):
        self.base_embeddings = base_embeddings
        self.clinical_terms = self._load_clinical_vocabulary()
    
    def _load_clinical_vocabulary(self) -> Dict[str, List[str]]:
        """Load clinical vocabulary for better embedding."""
        return {
            'cardiovascular': [
                'myocardial infarction', 'coronary artery disease', 'hypertension',
                'atrial fibrillation', 'heart failure', 'angina', 'arrhythmia'
            ],
            'endocrine': [
                'diabetes mellitus', 'thyroid', 'insulin', 'glucose', 'hormone',
                'endocrinology', 'metabolic', 'hypoglycemia', 'hyperglycemia'
            ],
            'respiratory': [
                'asthma', 'COPD', 'pneumonia', 'bronchitis', 'dyspnea',
                'respiratory failure', 'pulmonary embolism'
            ],
            'neurological': [
                'stroke', 'seizure', 'dementia', 'parkinson', 'multiple sclerosis',
                'neuropathy', 'headache', 'migraine'
            ],
            'psychiatric': [
                'depression', 'anxiety', 'bipolar', 'schizophrenia', 'PTSD',
                'psychiatric', 'mental health', 'mood disorder'
            ]
        }
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings with clinical context enhancement."""
        # Enhance texts with clinical context
        enhanced_texts = []
        for text in texts:
            enhanced_text = self._enhance_clinical_context(text)
            enhanced_texts.append(enhanced_text)
        
        # Generate embeddings
        embeddings = await self.base_embeddings.aembed_documents(enhanced_texts)
        return embeddings
    
    async def embed_query(self, text: str) -> List[float]:
        """Create query embedding with clinical context."""
        enhanced_text = self._enhance_clinical_context(text)
        embedding = await self.base_embeddings.aembed_query(enhanced_text)
        return embedding
    
    def _enhance_clinical_context(self, text: str) -> str:
        """Enhance text with clinical context for better embeddings."""
        text_lower = text.lower()
        context_additions = []
        
        # Add domain context if clinical terms are found
        for domain, terms in self.clinical_terms.items():
            for term in terms:
                if term in text_lower:
                    context_additions.append(f"clinical_domain_{domain}")
                    break
        
        # Add medical context markers
        if any(word in text_lower for word in ['diagnosis', 'treatment', 'medication', 'procedure']):
            context_additions.append("medical_content")
        
        if any(word in text_lower for word in ['contraindication', 'adverse', 'side effect']):
            context_additions.append("safety_information")
        
        # Combine original text with context
        if context_additions:
            enhanced_text = text + " " + " ".join(context_additions)
        else:
            enhanced_text = text
        
        return enhanced_text

class AdvancedClinicalRetriever:
    """
    Advanced retriever with clinical intelligence and semantic search.
    UPDATED: Works with Vector Store Manager and Patient Embeddings.
    """
    
    def __init__(
        self, 
        index_path: str = "faiss_index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vectorstore = None
        self.clinical_embeddings = None
        self.text_splitter = MedicalTextSplitter()
        self.document_cache = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize the advanced clinical retriever."""
        try:
            logger.info("Initializing advanced clinical retriever...")
            
            # Setup embeddings
            await self._setup_embeddings()
            
            # Setup vector store
            await self._setup_vectorstore()
            
            self.initialized = True
            logger.info("Advanced clinical retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced clinical retriever: {e}")
            raise
    
    async def _setup_embeddings(self):
        """Setup clinical-aware embeddings."""
        try:
            # Try new langchain-huggingface first
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            
            base_embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={
                    'device': 'cuda' if self._cuda_available() else 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Wrap with clinical enhancement
            self.clinical_embeddings = ClinicalEmbeddingStrategy(base_embeddings)
            self.embeddings = base_embeddings  # Keep reference to base for compatibility
            
            logger.info("Clinical embeddings setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up clinical embeddings: {e}")
            raise
    
    async def _setup_vectorstore(self):
        """Setup vector store with existing index or create new one."""
        try:
            from langchain_community.vectorstores import FAISS
            
            if os.path.exists(self.index_path):
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Existing FAISS index loaded successfully")
            else:
                logger.info("Creating new FAISS index")
                await self._create_empty_index()
            
        except Exception as e:
            logger.error(f"Error setting up vectorstore: {e}")
            raise
    
    async def _create_empty_index(self):
        """Create empty FAISS index."""
        try:
            from langchain_community.vectorstores import FAISS
            
            dummy_doc = Document(
                page_content="Clinical AI system initialization document",
                metadata={"source": "system", "type": "initialization"}
            )
            
            self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            self.vectorstore.save_local(self.index_path)
            
            logger.info("Empty FAISS index created")
            
        except Exception as e:
            logger.error(f"Error creating empty index: {e}")
            raise
    
    def _cuda_available(self) -> bool:
        """Check CUDA availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def add_clinical_documents(
        self, 
        documents: List[Document],
        document_type: str = "clinical_guideline"
    ) -> Dict[str, Any]:
        """Add clinical documents with enhanced processing."""
        try:
            logger.info(f"Adding {len(documents)} clinical documents")
            
            # Enhance documents with clinical metadata
            enhanced_docs = []
            for doc in documents:
                enhanced_doc = await self._enhance_document_metadata(doc, document_type)
                enhanced_docs.append(enhanced_doc)
            
            # Split documents using clinical-aware splitter
            split_docs = []
            for doc in enhanced_docs:
                chunks = self.text_splitter.split_documents([doc])
                split_docs.extend(chunks)
            
            # Add clinical context to each chunk
            contextual_docs = []
            for doc in split_docs:
                contextual_doc = self._add_clinical_context(doc)
                contextual_docs.append(contextual_doc)
            
            # Add to vector store
            self.vectorstore.add_documents(contextual_docs)
            self.vectorstore.save_local(self.index_path)
            
            stats = {
                "original_documents": len(documents),
                "total_chunks": len(contextual_docs),
                "average_chunk_size": sum(len(doc.page_content) for doc in contextual_docs) // len(contextual_docs),
                "document_types": list(set(doc.metadata.get("document_type", "unknown") for doc in contextual_docs))
            }
            
            logger.info(f"Successfully added documents: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error adding clinical documents: {e}")
            raise
    
    async def _enhance_document_metadata(
        self, 
        document: Document, 
        document_type: str
    ) -> Document:
        """Enhance document with clinical metadata."""
        enhanced_metadata = document.metadata.copy()
        
        # Add document type
        enhanced_metadata["document_type"] = document_type
        
        # Extract clinical domain
        clinical_domain = self._classify_clinical_domain(document.page_content)
        enhanced_metadata["clinical_domain"] = clinical_domain
        
        # Determine evidence level (simplified heuristic)
        evidence_level = self._determine_evidence_level(document.page_content)
        enhanced_metadata["evidence_level"] = evidence_level
        
        # Add processing timestamp
        enhanced_metadata["processed_at"] = datetime.now().isoformat()
        
        # Extract key clinical terms
        clinical_terms = self._extract_clinical_terms(document.page_content)
        enhanced_metadata["clinical_terms"] = clinical_terms
        
        return Document(
            page_content=document.page_content,
            metadata=enhanced_metadata
        )
    
    def _classify_clinical_domain(self, content: str) -> str:
        """Classify document into clinical domain."""
        content_lower = content.lower()
        
        domain_keywords = {
            'cardiovascular': ['heart', 'cardiac', 'coronary', 'hypertension', 'blood pressure'],
            'endocrine': ['diabetes', 'thyroid', 'hormone', 'insulin', 'glucose'],
            'respiratory': ['lung', 'asthma', 'copd', 'respiratory', 'pneumonia'],
            'neurological': ['brain', 'neuro', 'stroke', 'seizure', 'dementia'],
            'psychiatric': ['mental', 'depression', 'anxiety', 'psychiatric', 'mood'],
            'oncology': ['cancer', 'tumor', 'oncology', 'malignant', 'chemotherapy'],
            'infectious': ['infection', 'antibiotic', 'bacterial', 'viral', 'sepsis']
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return 'general'
    
    def _determine_evidence_level(self, content: str) -> str:
        """Determine evidence level of clinical content."""
        content_lower = content.lower()
        
        # Level A (High): Randomized controlled trials, meta-analyses
        if any(term in content_lower for term in ['randomized controlled trial', 'meta-analysis', 'rct', 'systematic review']):
            return 'A'
        
        # Level B (Moderate): Cohort studies, case-control studies
        elif any(term in content_lower for term in ['cohort study', 'case-control', 'observational study']):
            return 'B'
        
        # Level C (Low): Case series, case reports
        elif any(term in content_lower for term in ['case series', 'case report', 'expert opinion']):
            return 'C'
        
        # Level D (Very Low): Expert opinion only
        else:
            return 'D'
    
    def _extract_clinical_terms(self, content: str) -> List[str]:
        """Extract key clinical terms from content."""
        # Medical terminology patterns
        patterns = [
            r'\b[A-Z]{3,}\b',                    # Medical abbreviations
            r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b',  # Dosages
            r'\bICD-?10?[:\-\s]*[A-Z]\d{2}(?:\.\d+)?\b',    # ICD codes
            r'\bCPT[:\-\s]*\d{5}\b',             # CPT codes
        ]
        
        terms = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            terms.extend(matches)
        
        # Remove duplicates and return
        return list(set(terms))
    
    def _add_clinical_context(self, document: Document) -> Document:
        """Add clinical context to document chunk."""
        content = document.page_content
        metadata = document.metadata.copy()
        
        # Add context based on clinical domain
        clinical_domain = metadata.get('clinical_domain', 'general')
        content_with_context = f"[Clinical Domain: {clinical_domain}] {content}"
        
        # Add evidence level context
        evidence_level = metadata.get('evidence_level', 'D')
        content_with_context = f"[Evidence Level: {evidence_level}] {content_with_context}"
        
        return Document(
            page_content=content_with_context,
            metadata=metadata
        )
    
    async def search_clinical_context(
        self, 
        query: str, 
        clinical_context: Dict[str, Any] = None,
        k: int = 5,
        filter_domain: str = None
    ) -> List[Document]:
        """
        Advanced search with clinical context awareness.
        UPDATED: Enhanced for integration with patient embeddings.
        """
        try:
            if not self.initialized:
                raise RuntimeError("Retriever not initialized")
            
            # Enhance query with clinical context
            enhanced_query = await self._enhance_query_with_context(query, clinical_context)
            
            # Perform similarity search
            if filter_domain:
                # Use metadata filtering if supported
                try:
                    docs = self.vectorstore.similarity_search(
                        enhanced_query, 
                        k=k,
                        filter={"clinical_domain": filter_domain}
                    )
                except:
                    # Fallback if filtering not supported
                    docs = self.vectorstore.similarity_search(enhanced_query, k=k*2)
                    docs = [doc for doc in docs if doc.metadata.get("clinical_domain") == filter_domain][:k]
            else:
                docs = self.vectorstore.similarity_search(enhanced_query, k=k)
            
            # Post-process and rank results
            ranked_docs = await self._rank_by_clinical_relevance(docs, query, clinical_context)
            
            # Remove system documents
            filtered_docs = [
                doc for doc in ranked_docs 
                if doc.metadata.get("source") != "system"
            ]
            
            logger.info(f"Clinical search returned {len(filtered_docs)} documents for query: {query[:50]}...")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error in clinical context search: {e}")
            return []
    
    async def _enhance_query_with_context(
        self, 
        query: str, 
        clinical_context: Dict[str, Any] = None
    ) -> str:
        """Enhance search query with clinical context."""
        enhanced_query = query
        
        if clinical_context:
            # Add active diagnoses to query
            if clinical_context.get('active_diagnoses'):
                diagnoses = clinical_context['active_diagnoses'][:3]  # Top 3
                enhanced_query += " " + " ".join(diagnoses)
            
            # Add primary medical domain
            if clinical_context.get('primary_domain'):
                enhanced_query += f" clinical_domain_{clinical_context['primary_domain']}"
            
            # Add medication context
            if clinical_context.get('current_medications'):
                meds = clinical_context['current_medications'][:2]  # Top 2
                enhanced_query += " " + " ".join(meds)
        
        return enhanced_query
    
    async def _rank_by_clinical_relevance(
        self, 
        documents: List[Document], 
        query: str,
        clinical_context: Dict[str, Any] = None
    ) -> List[Document]:
        """Rank documents by clinical relevance."""
        if not documents:
            return documents
        
        # Calculate relevance scores
        scored_docs = []
        for doc in documents:
            score = self._calculate_clinical_relevance_score(doc, query, clinical_context)
            scored_docs.append((doc, score))
        
        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs]
    
    def _calculate_clinical_relevance_score(
        self, 
        document: Document, 
        query: str,
        clinical_context: Dict[str, Any] = None
    ) -> float:
        """Calculate clinical relevance score for a document."""
        score = 0.0
        
        # Base similarity score (placeholder - would use actual embedding similarity)
        score += 1.0
        
        # Evidence level bonus
        evidence_level = document.metadata.get('evidence_level', 'D')
        evidence_scores = {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.4}
        score += evidence_scores.get(evidence_level, 0.4)
        
        # Clinical domain match
        if clinical_context and clinical_context.get('primary_domain'):
            doc_domain = document.metadata.get('clinical_domain', 'general')
            if doc_domain == clinical_context['primary_domain']:
                score += 0.5
        
        # Recency bonus (newer documents are more relevant)
        processed_at = document.metadata.get('processed_at')
        if processed_at:
            try:
                processed_date = datetime.fromisoformat(processed_at)
                days_old = (datetime.now() - processed_date).days
                if days_old < 365:  # Less than a year old
                    score += 0.3 * (365 - days_old) / 365
            except:
                pass
        
        return score
    
    async def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever and its index."""
        try:
            if not self.initialized:
                return {"status": "not_initialized"}
            
            # Get all documents (limit for performance)
            all_docs = self.vectorstore.similarity_search("", k=1000)
            real_docs = [doc for doc in all_docs if doc.metadata.get("source") != "system"]
            
            # Calculate statistics
            domain_counts = {}
            evidence_counts = {}
            doc_type_counts = {}
            
            for doc in real_docs:
                # Domain statistics
                domain = doc.metadata.get('clinical_domain', 'unknown')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
                # Evidence level statistics
                evidence = doc.metadata.get('evidence_level', 'unknown')
                evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1
                
                # Document type statistics
                doc_type = doc.metadata.get('document_type', 'unknown')
                doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
            
            return {
                "status": "initialized",
                "total_documents": len(real_docs),
                "index_path": self.index_path,
                "embedding_model": self.embedding_model,
                "domain_distribution": domain_counts,
                "evidence_level_distribution": evidence_counts,
                "document_type_distribution": doc_type_counts,
                "cache_size": len(self.document_cache)
            }
            
        except Exception as e:
            logger.error(f"Error getting retriever stats: {e}")
            return {"status": "error", "error": str(e)}
    
    # Legacy compatibility methods
    async def search(self, query: str, k: int = 5) -> List[Document]:
        """Legacy search method for compatibility."""
        return await self.search_clinical_context(query, k=k)
    
    async def add_documents(self, documents: List[Document]):
        """Legacy add documents method for compatibility."""
        await self.add_clinical_documents(documents)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Legacy stats method for compatibility."""
        return await self.get_retriever_stats()