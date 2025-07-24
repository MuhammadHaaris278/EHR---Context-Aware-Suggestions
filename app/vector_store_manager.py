"""
Vector Store Manager - Coordinates multiple vector stores for comprehensive medical search.
Manages patient data, clinical literature, and specialized medical knowledge bases.
Optimized for massive EHR datasets with intelligent search routing.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
import json

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from .patient_data_embedder import PatientDataEmbedder
from .enhanced_retriever import AdvancedClinicalRetriever

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Enhanced search result with source tracking and relevance scoring."""
    document: Document
    score: float
    source_type: str  # "patient_data", "clinical_literature", "drug_interactions", etc.
    relevance_rank: int
    search_context: Dict[str, Any]

@dataclass
class MultiSearchRequest:
    """Multi-vector search request with context and filters."""
    query: str
    patient_id: Optional[str] = None
    search_scope: List[str] = None  # ["patient_data", "clinical_literature", "drug_interactions"]
    max_results_per_source: int = 5
    clinical_context: Optional[Dict[str, Any]] = None
    temporal_filter: Optional[Dict[str, Any]] = None  # {"days_back": 90}
    priority_filter: Optional[List[int]] = None  # [1, 2] for high priority only

class VectorStoreManager:
    """
    Centralized manager for all vector stores in the medical AI system.
    Provides unified search interface across patient data, clinical literature, and specialized knowledge.
    """
    
    def __init__(
        self, 
        base_path: str = "vector_stores",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.base_path = base_path
        self.embedding_model = embedding_model
        
        # Vector store components
        self.patient_embedder: Optional[PatientDataEmbedder] = None
        self.clinical_retriever: Optional[AdvancedClinicalRetriever] = None
        self.specialized_stores: Dict[str, FAISS] = {}
        
        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "patient_searches": 0,
            "clinical_searches": 0,
            "multi_source_searches": 0,
            "average_search_time": 0.0,
            "cache_hits": 0
        }
        
        # Search cache for frequently accessed queries
        self.search_cache = {}
        self.cache_expiry_hours = 1
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize all vector store components."""
        try:
            logger.info("Initializing Vector Store Manager...")
            
            # Ensure base directory exists
            os.makedirs(self.base_path, exist_ok=True)
            
            # Initialize patient data embedder
            patient_index_path = os.path.join(self.base_path, "patient_embeddings_faiss")
            self.patient_embedder = PatientDataEmbedder(
                index_path=patient_index_path,
                embedding_model=self.embedding_model
            )
            await self.patient_embedder.initialize()
            
            # Initialize clinical literature retriever
            clinical_index_path = os.path.join(self.base_path, "clinical_literature_faiss")
            self.clinical_retriever = AdvancedClinicalRetriever(
                index_path=clinical_index_path,
                embedding_model=self.embedding_model
            )
            await self.clinical_retriever.initialize()
            
            # Initialize specialized stores
            await self._initialize_specialized_stores()
            
            # Load search statistics
            await self._load_search_stats()
            
            self.initialized = True
            logger.info("✅ Vector Store Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store Manager: {e}")
            raise
    
    async def _initialize_specialized_stores(self):
        """Initialize specialized medical knowledge vector stores."""
        try:
            specialized_configs = {
                "drug_interactions": {
                    "description": "Drug-drug interactions and contraindications",
                    "priority": True
                },
                "lab_references": {
                    "description": "Laboratory reference ranges and interpretations",
                    "priority": True
                },
                "medical_procedures": {
                    "description": "Medical procedures and protocols",
                    "priority": False
                },
                "diagnostic_criteria": {
                    "description": "Diagnostic criteria and decision trees",
                    "priority": True
                }
            }
            
            for store_name, config in specialized_configs.items():
                store_path = os.path.join(self.base_path, f"{store_name}_faiss")
                
                if os.path.exists(store_path):
                    logger.info(f"Loading specialized store: {store_name}")
                    try:
                        # Load existing specialized store
                        store = FAISS.load_local(
                            store_path,
                            self.clinical_retriever.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        self.specialized_stores[store_name] = store
                        logger.info(f"✅ Loaded {store_name} vector store")
                    except Exception as e:
                        logger.warning(f"Could not load {store_name}: {e}")
                else:
                    # Create empty specialized store for future use
                    logger.info(f"Creating empty specialized store: {store_name}")
                    await self._create_empty_specialized_store(store_name, store_path)
            
        except Exception as e:
            logger.error(f"Error initializing specialized stores: {e}")
    
    async def _create_empty_specialized_store(self, store_name: str, store_path: str):
        """Create empty specialized vector store."""
        try:
            dummy_doc = Document(
                page_content=f"{store_name.replace('_', ' ').title()} knowledge base initialization.",
                metadata={
                    "store_type": store_name,
                    "is_dummy": True,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            store = FAISS.from_documents([dummy_doc], self.clinical_retriever.embeddings)
            store.save_local(store_path)
            self.specialized_stores[store_name] = store
            
            logger.info(f"Created empty {store_name} vector store")
            
        except Exception as e:
            logger.error(f"Error creating {store_name} store: {e}")
    
    async def comprehensive_search(
        self,
        request: MultiSearchRequest
    ) -> List[SearchResult]:
        """
        Comprehensive search across all relevant vector stores.
        Intelligently routes queries and combines results.
        """
        try:
            if not self.initialized:
                raise RuntimeError("Vector Store Manager not initialized")
            
            start_time = datetime.now()
            
            # Check cache first
            cache_key = self._create_cache_key(request)
            if cache_key in self.search_cache:
                cache_entry = self.search_cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    self.search_stats["cache_hits"] += 1
                    logger.info(f"Cache hit for query: {request.query[:50]}...")
                    return cache_entry["results"]
            
            logger.info(f"Comprehensive search: '{request.query[:100]}...'")
            
            # Determine search scope
            search_scope = request.search_scope or self._determine_optimal_search_scope(request)
            
            # Execute searches in parallel
            search_tasks = []
            
            # Patient data search
            if "patient_data" in search_scope and request.patient_id:
                search_tasks.append(
                    self._search_patient_data(request)
                )
            
            # Clinical literature search
            if "clinical_literature" in search_scope:
                search_tasks.append(
                    self._search_clinical_literature(request)
                )
            
            # Specialized store searches
            for specialized_store in ["drug_interactions", "lab_references", "diagnostic_criteria"]:
                if specialized_store in search_scope and specialized_store in self.specialized_stores:
                    search_tasks.append(
                        self._search_specialized_store(request, specialized_store)
                    )
            
            # Execute all searches
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and rank results
            combined_results = []
            for i, results in enumerate(search_results):
                if isinstance(results, Exception):
                    logger.error(f"Search task {i} failed: {results}")
                    continue
                combined_results.extend(results)
            
            # Intelligent ranking and deduplication
            final_results = self._rank_and_deduplicate_results(
                combined_results, 
                request
            )
            
            # Update statistics
            search_time = (datetime.now() - start_time).total_seconds()
            self._update_search_stats(request, search_time)
            
            # Cache results
            self.search_cache[cache_key] = {
                "results": final_results,
                "timestamp": datetime.now(),
                "query": request.query
            }
            
            logger.info(f"Comprehensive search completed: {len(final_results)} results in {search_time:.2f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive search: {e}")
            return []
    
    async def _search_patient_data(self, request: MultiSearchRequest) -> List[SearchResult]:
        """Search patient-specific data."""
        try:
            if not request.patient_id or not self.patient_embedder.is_patient_embedded(request.patient_id):
                return []
            
            # Determine search parameters
            k = request.max_results_per_source
            
            # Apply temporal filter if specified
            if request.temporal_filter:
                days_back = request.temporal_filter.get("days_back", 90)
                patient_docs = await self.patient_embedder.get_patient_timeline_context(
                    request.patient_id,
                    days_back=days_back,
                    k=k
                )
            else:
                # Semantic search
                patient_docs = await self.patient_embedder.search_patient_context(
                    patient_id=request.patient_id,
                    query=request.query,
                    k=k,
                    filter_chunk_types=request.clinical_context.get("chunk_types") if request.clinical_context else None,
                    filter_clinical_domains=request.clinical_context.get("clinical_domains") if request.clinical_context else None
                )
            
            # Convert to SearchResult objects
            results = []
            for i, doc in enumerate(patient_docs):
                # Calculate relevance score based on priority and content match
                score = self._calculate_patient_relevance_score(doc, request.query)
                
                result = SearchResult(
                    document=doc,
                    score=score,
                    source_type="patient_data",
                    relevance_rank=i + 1,
                    search_context={
                        "patient_id": request.patient_id,
                        "chunk_type": doc.metadata.get("chunk_type"),
                        "clinical_domain": doc.metadata.get("clinical_domain"),
                        "priority": doc.metadata.get("priority")
                    }
                )
                results.append(result)
            
            self.search_stats["patient_searches"] += 1
            return results
            
        except Exception as e:
            logger.error(f"Error searching patient data: {e}")
            return []
    
    async def _search_clinical_literature(self, request: MultiSearchRequest) -> List[SearchResult]:
        """Search clinical literature and guidelines."""
        try:
            # Prepare clinical context
            clinical_context = request.clinical_context or {}
            
            # Use clinical retriever's advanced search
            clinical_docs = await self.clinical_retriever.search_clinical_context(
                query=request.query,
                clinical_context=clinical_context,
                k=request.max_results_per_source
            )
            
            # Convert to SearchResult objects
            results = []
            for i, doc in enumerate(clinical_docs):
                score = self._calculate_clinical_relevance_score(doc, request.query)
                
                result = SearchResult(
                    document=doc,
                    score=score,
                    source_type="clinical_literature",
                    relevance_rank=i + 1,
                    search_context={
                        "evidence_level": doc.metadata.get("evidence_level"),
                        "clinical_domain": doc.metadata.get("clinical_domain"),
                        "document_type": doc.metadata.get("document_type")
                    }
                )
                results.append(result)
            
            self.search_stats["clinical_searches"] += 1
            return results
            
        except Exception as e:
            logger.error(f"Error searching clinical literature: {e}")
            return []
    
    async def _search_specialized_store(
        self, 
        request: MultiSearchRequest, 
        store_name: str
    ) -> List[SearchResult]:
        """Search specialized medical knowledge store."""
        try:
            if store_name not in self.specialized_stores:
                return []
            
            store = self.specialized_stores[store_name]
            
            # Search the specialized store
            specialized_docs = store.similarity_search(
                request.query,
                k=request.max_results_per_source
            )
            
            # Filter out dummy documents
            real_docs = [
                doc for doc in specialized_docs 
                if not doc.metadata.get("is_dummy", False)
            ]
            
            # Convert to SearchResult objects
            results = []
            for i, doc in enumerate(real_docs):
                score = self._calculate_specialized_relevance_score(doc, request.query, store_name)
                
                result = SearchResult(
                    document=doc,
                    score=score,
                    source_type=store_name,
                    relevance_rank=i + 1,
                    search_context={
                        "specialized_store": store_name,
                        "store_metadata": doc.metadata
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching {store_name}: {e}")
            return []
    
    def _determine_optimal_search_scope(self, request: MultiSearchRequest) -> List[str]:
        """Intelligently determine which vector stores to search."""
        scope = []
        
        # Always search clinical literature for medical queries
        scope.append("clinical_literature")
        
        # Search patient data if patient ID provided
        if request.patient_id:
            scope.append("patient_data")
        
        # Determine specialized stores based on query content
        query_lower = request.query.lower()
        
        # Drug-related queries
        if any(term in query_lower for term in [
            "medication", "drug", "interaction", "contraindication", 
            "side effect", "adverse", "prescription"
        ]):
            scope.append("drug_interactions")
        
        # Lab-related queries
        if any(term in query_lower for term in [
            "lab", "laboratory", "test", "result", "reference", "range",
            "blood", "urine", "glucose", "hemoglobin", "creatinine"
        ]):
            scope.append("lab_references")
        
        # Diagnostic queries
        if any(term in query_lower for term in [
            "diagnosis", "diagnostic", "criteria", "differential",
            "symptoms", "signs", "workup"
        ]):
            scope.append("diagnostic_criteria")
        
        return scope
    
    def _calculate_patient_relevance_score(self, doc: Document, query: str) -> float:
        """Calculate relevance score for patient data."""
        score = 0.5  # Base score
        
        # Priority bonus (lower priority number = higher score)
        priority = doc.metadata.get("priority", 3)
        priority_bonus = (4 - priority) * 0.2  # 0.6 for priority 1, 0.4 for priority 2, 0.2 for priority 3
        score += priority_bonus
        
        # Recent data bonus
        timestamp_str = doc.metadata.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                days_old = (datetime.now() - timestamp).days
                if days_old < 30:
                    score += 0.3
                elif days_old < 90:
                    score += 0.2
                elif days_old < 365:
                    score += 0.1
            except:
                pass
        
        # Content relevance (simple keyword matching)
        query_terms = query.lower().split()
        content_lower = doc.page_content.lower()
        matching_terms = sum(1 for term in query_terms if term in content_lower)
        if query_terms:
            content_score = matching_terms / len(query_terms) * 0.3
            score += content_score
        
        return min(score, 1.0)
    
    def _calculate_clinical_relevance_score(self, doc: Document, query: str) -> float:
        """Calculate relevance score for clinical literature."""
        score = 0.4  # Base score for clinical literature
        
        # Evidence level bonus
        evidence_level = doc.metadata.get("evidence_level", "D")
        evidence_scores = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
        score += evidence_scores.get(evidence_level, 0.1)
        
        # Document type bonus
        doc_type = doc.metadata.get("document_type", "")
        if doc_type == "clinical_guideline":
            score += 0.2
        elif doc_type == "clinical_protocol":
            score += 0.15
        
        # Content relevance
        query_terms = query.lower().split()
        content_lower = doc.page_content.lower()
        matching_terms = sum(1 for term in query_terms if term in content_lower)
        if query_terms:
            content_score = matching_terms / len(query_terms) * 0.2
            score += content_score
        
        return min(score, 1.0)
    
    def _calculate_specialized_relevance_score(
        self, 
        doc: Document, 
        query: str, 
        store_name: str
    ) -> float:
        """Calculate relevance score for specialized stores."""
        score = 0.3  # Base score for specialized content
        
        # Store-specific bonuses
        store_bonuses = {
            "drug_interactions": 0.3,
            "diagnostic_criteria": 0.25,
            "lab_references": 0.2,
            "medical_procedures": 0.15
        }
        score += store_bonuses.get(store_name, 0.1)
        
        # Content relevance
        query_terms = query.lower().split()
        content_lower = doc.page_content.lower()
        matching_terms = sum(1 for term in query_terms if term in content_lower)
        if query_terms:
            content_score = matching_terms / len(query_terms) * 0.25
            score += content_score
        
        return min(score, 1.0)
    
    def _rank_and_deduplicate_results(
        self, 
        results: List[SearchResult], 
        request: MultiSearchRequest
    ) -> List[SearchResult]:
        """Rank results and remove duplicates."""
        if not results:
            return []
        
        # Remove duplicates based on content similarity
        unique_results = []
        seen_content_hashes = set()
        
        for result in results:
            # Create content hash for deduplication
            content_hash = hash(result.document.page_content[:200])
            
            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                unique_results.append(result)
        
        # Sort by relevance score (descending)
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply priority filter if specified
        if request.priority_filter:
            unique_results = [
                result for result in unique_results
                if result.search_context.get("priority") in request.priority_filter
            ]
        
        # Limit total results
        max_total_results = 20  # Reasonable limit for LLM context
        return unique_results[:max_total_results]
    
    def _create_cache_key(self, request: MultiSearchRequest) -> str:
        """Create cache key for search request."""
        key_components = [
            request.query,
            request.patient_id or "no_patient",
            str(sorted(request.search_scope or [])),
            str(request.max_results_per_source),
            str(request.clinical_context or {}),
            str(request.temporal_filter or {}),
            str(request.priority_filter or [])
        ]
        return hash("|".join(key_components))
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        timestamp = cache_entry.get("timestamp")
        if not timestamp:
            return False
        
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        return age_hours < self.cache_expiry_hours
    
    def _update_search_stats(self, request: MultiSearchRequest, search_time: float):
        """Update search statistics."""
        self.search_stats["total_searches"] += 1
        
        if request.patient_id:
            self.search_stats["patient_searches"] += 1
        
        if not request.search_scope or len(request.search_scope) > 1:
            self.search_stats["multi_source_searches"] += 1
        
        # Update average search time
        total_time = (
            self.search_stats["average_search_time"] * 
            (self.search_stats["total_searches"] - 1) + 
            search_time
        )
        self.search_stats["average_search_time"] = total_time / self.search_stats["total_searches"]
    
    async def _load_search_stats(self):
        """Load search statistics from disk."""
        try:
            stats_path = os.path.join(self.base_path, "search_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    saved_stats = json.load(f)
                    self.search_stats.update(saved_stats)
                logger.info("Search statistics loaded")
        except Exception as e:
            logger.warning(f"Could not load search stats: {e}")
    
    async def save_search_stats(self):
        """Save search statistics to disk."""
        try:
            stats_path = os.path.join(self.base_path, "search_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(self.search_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving search stats: {e}")
    
    # Public methods for managing vector stores
    
    async def embed_patient(
        self, 
        patient_id: str, 
        patient_data: Dict,
        force_reembed: bool = False
    ) -> Dict[str, Any]:
        """Embed patient data into the system."""
        if not self.patient_embedder:
            raise RuntimeError("Patient embedder not initialized")
        
        return await self.patient_embedder.embed_patient_data(
            patient_id, 
            patient_data, 
            force_reembed
        )
    
    async def add_clinical_documents(
        self, 
        documents: List[Document],
        document_type: str = "clinical_reference"
    ) -> Dict[str, Any]:
        """Add clinical documents to the literature store."""
        if not self.clinical_retriever:
            raise RuntimeError("Clinical retriever not initialized")
        
        return await self.clinical_retriever.add_clinical_documents(
            documents, 
            document_type
        )
    
    async def add_specialized_content(
        self,
        store_name: str,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """Add content to specialized vector store."""
        try:
            if store_name not in self.specialized_stores:
                raise ValueError(f"Unknown specialized store: {store_name}")
            
            store = self.specialized_stores[store_name]
            
            # Add documents to the store
            store.add_documents(documents)
            
            # Save the updated store
            store_path = os.path.join(self.base_path, f"{store_name}_faiss")
            store.save_local(store_path)
            
            return {
                "store_name": store_name,
                "documents_added": len(documents),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error adding to {store_name}: {e}")
            return {
                "store_name": store_name,
                "status": "error",
                "error": str(e)
            }
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all vector stores."""
        try:
            stats = {
                "manager_info": {
                    "initialized": self.initialized,
                    "base_path": self.base_path,
                    "embedding_model": self.embedding_model
                },
                "search_performance": self.search_stats,
                "patient_embeddings": {},
                "clinical_literature": {},
                "specialized_stores": {}
            }
            
            # Patient embedding stats
            if self.patient_embedder:
                stats["patient_embeddings"] = await self.patient_embedder.get_embedding_stats()
            
            # Clinical literature stats
            if self.clinical_retriever:
                stats["clinical_literature"] = await self.clinical_retriever.get_retriever_stats()
            
            # Specialized store stats
            for store_name, store in self.specialized_stores.items():
                try:
                    # Get basic info about specialized stores
                    stats["specialized_stores"][store_name] = {
                        "available": True,
                        "store_path": os.path.join(self.base_path, f"{store_name}_faiss")
                    }
                except Exception as e:
                    stats["specialized_stores"][store_name] = {
                        "available": False,
                        "error": str(e)
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting comprehensive stats: {e}")
            return {"error": str(e)}
    
    async def clear_all_caches(self):
        """Clear all caches across the system."""
        self.search_cache.clear()
        
        if self.patient_embedder:
            # Patient embedder might have its own cache
            pass
        
        if self.clinical_retriever:
            # Clinical retriever might have document cache
            if hasattr(self.clinical_retriever, 'document_cache'):
                self.clinical_retriever.document_cache.clear()
        
        logger.info("All caches cleared")
    
    # Quick search methods for convenience
    
    async def quick_patient_search(
        self,
        patient_id: str,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """Quick search within patient data only."""
        if not self.patient_embedder or not self.patient_embedder.is_patient_embedded(patient_id):
            return []
        
        return await self.patient_embedder.search_patient_context(
            patient_id=patient_id,
            query=query,
            k=k
        )
    
    async def quick_clinical_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """Quick search within clinical literature only."""
        if not self.clinical_retriever:
            return []
        
        return await self.clinical_retriever.search_clinical_context(
            query=query,
            k=k
        )
    
    async def hybrid_patient_clinical_search(
        self,
        patient_id: str,
        query: str,
        patient_results: int = 3,
        clinical_results: int = 2
    ) -> Dict[str, List[Document]]:
        """Hybrid search combining patient data and clinical literature."""
        results = {
            "patient_context": [],
            "clinical_guidance": []
        }
        
        # Search patient data
        if self.patient_embedder and self.patient_embedder.is_patient_embedded(patient_id):
            results["patient_context"] = await self.patient_embedder.search_patient_context(
                patient_id=patient_id,
                query=query,
                k=patient_results
            )
        
        # Search clinical literature
        if self.clinical_retriever:
            results["clinical_guidance"] = await self.clinical_retriever.search_clinical_context(
                query=query,
                k=clinical_results
            )
        
        return results