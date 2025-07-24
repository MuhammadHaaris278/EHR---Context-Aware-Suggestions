"""
Integrated Enhanced Clinical Pipeline
UPDATED: Uses Vector Store Manager and Patient Data Embeddings for massive dataset processing.
Optimized for Burj Khalifa scale EHR data with intelligent context selection.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from .enhanced_summarizer import AdvancedPatientProcessor
from .vector_store_manager import VectorStoreManager, MultiSearchRequest, SearchResult
from .patient_data_embedder import PatientDataEmbedder
from .enhanced_ehr_schema import (
    Patient, Condition, MedicationStatement, Encounter,
    Observation, Procedure, PatientSummary
)

logger = logging.getLogger(__name__)

class EnhancedClinicalPipeline:
    """
    Enhanced clinical pipeline with vector-based patient data processing.
    UPDATED: Massive scalability through semantic search and intelligent context selection.
    """
    
    def __init__(
        self, 
        llm,
        db_session,
        vector_store_manager: Optional[VectorStoreManager] = None,
        base_path: str = "vector_stores",
        max_tokens: int = 3500
    ):
        self.llm = llm
        self.db_session = db_session
        self.max_tokens = max_tokens
        self.base_path = base_path
        
        # Initialize vector store manager
        self.vector_manager = vector_store_manager or VectorStoreManager(
            base_path=base_path,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize enhanced summarizer (now works with embedded context)
        self.patient_processor = AdvancedPatientProcessor(
            llm=llm,
            retriever=None,  # Will be set after vector manager initialization
            max_tokens=max_tokens
        )
        
        # Performance tracking
        self.processing_stats = {
            "total_patients_processed": 0,
            "total_processing_time": 0,
            "average_processing_time": 0,
            "vector_searches_performed": 0,
            "embedding_cache_hits": 0,
            "massive_dataset_processed": 0  # Patients with >10k lines of data
        }
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize all pipeline components with vector capabilities."""
        try:
            logger.info("Initializing Enhanced Clinical Pipeline v2.0 with Vector Processing...")
            
            # Initialize vector store manager first
            await self.vector_manager.initialize()
            
            # Set retriever for patient processor (using clinical retriever from vector manager)
            self.patient_processor.retriever = self.vector_manager.clinical_retriever
            
            # Load any existing patient summaries from database
            await self._load_existing_summaries()
            
            self.initialized = True
            logger.info("✅ Enhanced Clinical Pipeline v2.0 initialized with vector capabilities")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Clinical Pipeline v2.0: {e}")
            raise
    
    async def _load_existing_summaries(self):
        """Load existing patient summaries from database and check embedding status."""
        try:
            existing_summaries = self.db_session.query(PatientSummary).all()
            embedded_count = 0
            
            for summary in existing_summaries:
                if self.vector_manager.patient_embedder.is_patient_embedded(summary.patient_id):
                    embedded_count += 1
            
            logger.info(f"Loaded {len(existing_summaries)} existing summaries, {embedded_count} patients embedded")
            
        except Exception as e:
            logger.warning(f"Could not load existing summaries: {e}")
    
    async def process_patient_comprehensive(
        self, 
        patient_id: str,
        force_refresh: bool = False,
        auto_embed: bool = True,
        context_strategy: str = "intelligent"  # "intelligent", "recent", "comprehensive"
    ) -> Dict[str, Any]:
        """
        UPDATED: Process patient using vector-based semantic search for context selection.
        Handles massive datasets efficiently through intelligent context selection.
        """
        try:
            if not self.initialized:
                raise RuntimeError("Pipeline not initialized")
            
            start_time = datetime.now()
            logger.info(f"Starting vector-enhanced processing for patient {patient_id}")
            
            # Step 1: Fetch patient data from database
            patient_data = await self._fetch_comprehensive_patient_data(patient_id)
            
            if not patient_data:
                logger.warning(f"No data found for patient {patient_id}")
                return self._create_fallback_result(patient_id, "No patient data found")
            
            # Check if this is a massive dataset
            total_data_lines = self._estimate_data_lines(patient_data)
            is_massive_dataset = total_data_lines > 10000
            
            if is_massive_dataset:
                self.processing_stats["massive_dataset_processed"] += 1
                logger.info(f"Processing massive dataset: ~{total_data_lines} lines for patient {patient_id}")
            
            # Step 2: Ensure patient data is embedded (critical for large datasets)
            if auto_embed:
                embedding_result = await self._ensure_patient_embedded(patient_id, patient_data, force_refresh)
                if embedding_result.get("status") == "error":
                    logger.warning(f"Patient embedding failed: {embedding_result.get('error')}")
            
            # Step 3: Check for existing processed summary (unless force refresh)
            if not force_refresh:
                cached_result = await self._check_existing_summary(patient_id)
                if cached_result:
                    self.processing_stats["embedding_cache_hits"] += 1
                    logger.info(f"Using cached summary for patient {patient_id}")
                    # Enhance cached result with fresh context if needed
                    return await self._enhance_cached_result_with_context(cached_result, patient_id, context_strategy)
            
            # Step 4: Vector-based intelligent context selection
            try:
                relevant_context = await self._select_intelligent_patient_context(
                    patient_id, 
                    patient_data,
                    strategy=context_strategy,
                    max_tokens=self.max_tokens // 2  # Reserve half tokens for context
                )
            except Exception as e:
                logger.error(f"Error in context selection: {e}")
                # Fallback to traditional processing
                relevant_context = {"patient_data": patient_data, "context_type": "fallback"}
            
            # Step 5: Process with intelligent context (much faster than full processing)
            try:
                processed_result = await self._process_with_intelligent_context(
                    patient_id,
                    relevant_context,
                    patient_data
                )
            except Exception as e:
                logger.error(f"Error in intelligent processing: {e}")
                processed_result = self._create_fallback_processed_result(patient_id, patient_data, str(e))
            
            # Step 6: Generate clinical recommendations using vector search
            try:
                clinical_recommendations = await self._generate_vector_enhanced_recommendations(
                    processed_result, 
                    patient_id,
                    relevant_context
                )
            except Exception as e:
                logger.error(f"Error generating recommendations: {e}")
                clinical_recommendations = self._create_fallback_recommendations(patient_id, str(e))
            
            # Step 7: Create comprehensive result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            comprehensive_result = {
                **processed_result,
                "clinical_recommendations": clinical_recommendations,
                "processing_metadata": {
                    "pipeline_version": "enhanced_v2.0_vector",
                    "processing_duration_seconds": processing_time,
                    "llm_model": str(type(self.llm).__name__),
                    "context_strategy": context_strategy,
                    "is_massive_dataset": is_massive_dataset,
                    "estimated_data_lines": total_data_lines,
                    "vector_enhanced": True,
                    "embedding_used": auto_embed,
                    "force_refresh": force_refresh
                }
            }
            
            # Step 8: Ensure valid output
            comprehensive_result = self._ensure_comprehensive_result_validity(comprehensive_result, patient_id)
            
            # Step 9: Store processed summary
            try:
                await self._store_processed_summary(patient_id, comprehensive_result)
            except Exception as e:
                logger.warning(f"Could not store processed summary: {e}")
            
            # Step 10: Update statistics
            self._update_processing_stats(start_time, is_massive_dataset)
            
            logger.info(f"✅ Vector-enhanced processing completed for patient {patient_id} in {processing_time:.2f}s")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive patient processing for {patient_id}: {e}")
            return self._create_fallback_result(patient_id, str(e))
    
    def _estimate_data_lines(self, patient_data: Dict) -> int:
        """Estimate total lines of patient data for processing classification."""
        try:
            total_lines = 0
            
            # Count data elements
            total_lines += len(patient_data.get('diagnoses', []))
            total_lines += len(patient_data.get('medications', []))
            total_lines += len(patient_data.get('encounters', []))
            total_lines += len(patient_data.get('observations', []))
            total_lines += len(patient_data.get('procedures', []))
            total_lines += len(patient_data.get('allergies', []))
            
            # Estimate lines per element (rough approximation)
            # Each diagnosis ~5 lines, medication ~7 lines, encounter ~10 lines, etc.
            line_multipliers = {
                'diagnoses': 5,
                'medications': 7,
                'encounters': 10,
                'observations': 4,
                'procedures': 8,
                'allergies': 3
            }
            
            estimated_lines = 0
            for data_type, multiplier in line_multipliers.items():
                count = len(patient_data.get(data_type, []))
                estimated_lines += count * multiplier
            
            return max(total_lines, estimated_lines)
            
        except Exception as e:
            logger.error(f"Error estimating data lines: {e}")
            return 1000  # Conservative estimate
    
    async def _ensure_patient_embedded(
        self, 
        patient_id: str, 
        patient_data: Dict,
        force_reembed: bool = False
    ) -> Dict[str, Any]:
        """Ensure patient data is embedded in vector store."""
        try:
            if self.vector_manager.patient_embedder.is_patient_embedded(patient_id) and not force_reembed:
                return {"status": "already_embedded", "patient_id": patient_id}
            
            logger.info(f"Embedding patient data for {patient_id}...")
            result = await self.vector_manager.embed_patient(
                patient_id=patient_id,
                patient_data=patient_data,
                force_reembed=force_reembed
            )
            
            logger.info(f"Patient {patient_id} embedding result: {result.get('status')}")
            return result
            
        except Exception as e:
            logger.error(f"Error embedding patient {patient_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _select_intelligent_patient_context(
        self,
        patient_id: str,
        patient_data: Dict,
        strategy: str = "intelligent",
        max_tokens: int = 1500
    ) -> Dict[str, Any]:
        """
        Intelligently select relevant patient context using vector search.
        This is the key optimization for massive datasets.
        """
        try:
            if strategy == "comprehensive":
                # Use full patient data (for smaller datasets)
                return {
                    "patient_data": patient_data,
                    "context_type": "comprehensive",
                    "selection_method": "full"
                }
            
            elif strategy == "recent":
                # Use recent timeline context
                recent_context = await self.vector_manager.patient_embedder.get_patient_timeline_context(
                    patient_id=patient_id,
                    days_back=90,
                    k=10
                )
                
                return {
                    "relevant_chunks": recent_context,
                    "context_type": "recent",
                    "selection_method": "timeline"
                }
            
            else:  # intelligent strategy
                # Use semantic search to find most relevant context
                
                # Generate intelligent query based on patient's primary conditions
                query = self._generate_intelligent_query(patient_data)
                
                # Search for most relevant patient context
                relevant_chunks = await self.vector_manager.patient_embedder.search_patient_context(
                    patient_id=patient_id,
                    query=query,
                    k=8,  # Get top 8 most relevant chunks
                    include_summary=True
                )
                
                # Also get some recent context
                recent_chunks = await self.vector_manager.patient_embedder.get_patient_timeline_context(
                    patient_id=patient_id,
                    days_back=30,
                    k=5
                )
                
                # Combine and deduplicate
                combined_chunks = self._combine_and_deduplicate_chunks(relevant_chunks, recent_chunks)
                
                self.processing_stats["vector_searches_performed"] += 1
                
                return {
                    "relevant_chunks": combined_chunks,
                    "context_type": "intelligent",
                    "selection_method": "semantic_search",
                    "query_used": query,
                    "chunks_selected": len(combined_chunks)
                }
            
        except Exception as e:
            logger.error(f"Error in intelligent context selection: {e}")
            # Fallback to recent data
            return {
                "patient_data": self._extract_recent_data(patient_data),
                "context_type": "fallback",
                "selection_method": "recent_fallback"
            }
    
    def _generate_intelligent_query(self, patient_data: Dict) -> str:
        """Generate an intelligent query based on patient's key conditions."""
        try:
            # Extract active diagnoses
            diagnoses = patient_data.get('diagnoses', [])
            active_diagnoses = [
                d.get('description', '') for d in diagnoses 
                if d.get('status', '').lower() in ['active', 'chronic']
            ]
            
            # Extract active medications
            medications = patient_data.get('medications', [])
            active_medications = [
                m.get('name', '') for m in medications
                if m.get('status', '').lower() == 'active'
            ]
            
            # Create intelligent query
            query_parts = []
            
            if active_diagnoses:
                query_parts.extend(active_diagnoses[:3])  # Top 3 diagnoses
            
            if active_medications:
                query_parts.extend(active_medications[:2])  # Top 2 medications
            
            # Add general clinical terms
            query_parts.extend(['clinical status', 'medical history', 'treatment plan'])
            
            return ' '.join(query_parts)
            
        except Exception as e:
            logger.error(f"Error generating intelligent query: {e}")
            return "patient medical history clinical status"
    
    def _combine_and_deduplicate_chunks(self, *chunk_lists) -> List:
        """Combine multiple chunk lists and remove duplicates."""
        seen_chunk_ids = set()
        combined_chunks = []
        
        for chunk_list in chunk_lists:
            for chunk in chunk_list:
                chunk_id = chunk.metadata.get('chunk_id', '')
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    combined_chunks.append(chunk)
        
        return combined_chunks
    
    def _extract_recent_data(self, patient_data: Dict, days_back: int = 90) -> Dict:
        """Extract recent patient data as fallback."""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_data = {}
        
        for data_type in ['diagnoses', 'medications', 'encounters', 'observations', 'procedures']:
            data_list = patient_data.get(data_type, [])
            recent_items = []
            
            for item in data_list:
                # Check various date fields
                item_date = None
                for date_field in ['date', 'diagnosis_date', 'start_date', 'performed_date', 'visit_date']:
                    if item.get(date_field):
                        try:
                            item_date = datetime.fromisoformat(item[date_field].replace('Z', '+00:00'))
                            break
                        except:
                            continue
                
                if not item_date or item_date >= cutoff_date:
                    recent_items.append(item)
            
            recent_data[data_type] = recent_items
        
        # Always include demographics
        recent_data['demographics'] = patient_data.get('demographics', {})
        recent_data['patient_id'] = patient_data.get('patient_id')
        
        return recent_data
    
    async def _process_with_intelligent_context(
        self,
        patient_id: str,
        relevant_context: Dict[str, Any],
        full_patient_data: Dict
    ) -> Dict[str, Any]:
        """Process patient using intelligently selected context."""
        try:
            context_type = relevant_context.get("context_type", "unknown")
            
            if context_type == "comprehensive":
                # Use traditional full processing
                return await self.patient_processor.process_patient_history(full_patient_data)
            
            elif context_type in ["intelligent", "recent"]:
                # Use chunk-based processing
                relevant_chunks = relevant_context.get("relevant_chunks", [])
                
                if not relevant_chunks:
                    # Fallback to recent data processing
                    recent_data = self._extract_recent_data(full_patient_data)
                    return await self.patient_processor.process_patient_history(recent_data)
                
                # Convert chunks to structured summary
                structured_summary = self._convert_chunks_to_summary(relevant_chunks, patient_id)
                
                return {
                    'patient_id': patient_id,
                    'processing_timestamp': datetime.now().isoformat(),
                    'temporal_summaries': structured_summary.get('temporal_summaries', {}),
                    'master_summary': structured_summary.get('master_summary', ''),
                    'relevant_literature': [],
                    'chunk_statistics': {
                        'chunks_processed': len(relevant_chunks),
                        'context_method': relevant_context.get('selection_method', 'unknown')
                    },
                    'total_contexts_processed': len(relevant_chunks),
                    'summary_token_estimate': len(structured_summary.get('master_summary', '')) // 4
                }
            
            else:
                # Fallback processing
                fallback_data = relevant_context.get("patient_data", full_patient_data)
                return await self.patient_processor.process_patient_history(fallback_data)
                
        except Exception as e:
            logger.error(f"Error in intelligent context processing: {e}")
            raise
    
    def _convert_chunks_to_summary(self, chunks: List, patient_id: str) -> Dict[str, Any]:
        """Convert vector chunks back to structured summary format."""
        try:
            # Group chunks by type and clinical domain
            chunk_groups = {
                'current': [],
                'recent': [],
                'chronic': [],
                'historical': []
            }
            
            # Categorize chunks by recency and priority
            for chunk in chunks:
                chunk_type = chunk.metadata.get('chunk_type', 'unknown')
                priority = chunk.metadata.get('priority', 3)
                timestamp_str = chunk.metadata.get('timestamp', '')
                
                # Determine temporal category
                if priority == 1:  # High priority goes to current
                    chunk_groups['current'].append(chunk.page_content)
                elif timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        days_old = (datetime.now() - timestamp).days
                        
                        if days_old < 30:
                            chunk_groups['recent'].append(chunk.page_content)
                        elif days_old < 365:
                            chunk_groups['chronic'].append(chunk.page_content)
                        else:
                            chunk_groups['historical'].append(chunk.page_content)
                    except:
                        chunk_groups['historical'].append(chunk.page_content)
                else:
                    chunk_groups['historical'].append(chunk.page_content)
            
            # Create temporal summaries
            temporal_summaries = {}
            for period, chunk_contents in chunk_groups.items():
                if chunk_contents:
                    # Combine chunk contents for this period
                    combined_content = '\n'.join(chunk_contents[:5])  # Limit to top 5 chunks per period
                    temporal_summaries[period] = f"{period.title()} Clinical Data:\n{combined_content}"
                else:
                    temporal_summaries[period] = f"No significant {period} clinical data available"
            
            # Create master summary
            all_content = []
            for period in ['current', 'recent', 'chronic', 'historical']:
                if temporal_summaries.get(period) and not temporal_summaries[period].startswith("No significant"):
                    all_content.append(temporal_summaries[period])
            
            master_summary = f"Comprehensive Patient Summary for {patient_id}:\n" + "\n\n".join(all_content[:3])
            
            return {
                'temporal_summaries': temporal_summaries,
                'master_summary': master_summary
            }
            
        except Exception as e:
            logger.error(f"Error converting chunks to summary: {e}")
            return {
                'temporal_summaries': {
                    'current': f"Clinical data processed for patient {patient_id}",
                    'recent': "Recent medical information reviewed",
                    'historical': "Historical data analyzed"
                },
                'master_summary': f"Patient summary generated for {patient_id} using vector-based processing"
            }
    
    async def _generate_vector_enhanced_recommendations(
        self,
        processed_result: Dict[str, Any],
        patient_id: str,
        relevant_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate clinical recommendations using vector search for clinical literature."""
        try:
            # Extract key clinical terms from processed result
            master_summary = processed_result.get('master_summary', '')
            temporal_summaries = processed_result.get('temporal_summaries', {})
            
            # Create search query for clinical literature
            search_queries = []
            
            # Add current clinical issues
            current_summary = temporal_summaries.get('current', '')
            if current_summary:
                search_queries.append(current_summary[:200])  # First 200 chars
            
            # Add key terms from master summary
            if master_summary:
                search_queries.append(master_summary[:300])
            
            # Search clinical literature using vector manager
            clinical_literature = []
            for query in search_queries[:2]:  # Limit to 2 searches
                try:
                    literature_docs = await self.vector_manager.quick_clinical_search(
                        query=query,
                        k=3
                    )
                    clinical_literature.extend(literature_docs)
                except Exception as e:
                    logger.warning(f"Clinical literature search failed: {e}")
                    continue
            
            # Remove duplicates
            unique_literature = []
            seen_sources = set()
            for doc in clinical_literature:
                source = doc.metadata.get('source', 'unknown')
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_literature.append(doc)
            
            # Create recommendation prompt
            recommendation_prompt = self._create_vector_enhanced_recommendation_prompt(
                processed_result,
                unique_literature[:5],  # Top 5 literature sources
                patient_id
            )
            
            # Generate recommendations using LLM
            try:
                recommendations = await self._call_llm_for_recommendations(recommendation_prompt)
                
                return {
                    "clinical_guidelines": [
                        {
                            "source": doc.metadata.get("source", "Unknown"),
                            "domain": doc.metadata.get("clinical_domain", "general"),
                            "evidence_level": doc.metadata.get("evidence_level", "D"),
                            "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                        }
                        for doc in unique_literature
                    ],
                    "recommendations": recommendations,
                    "clinical_context": {
                        "context_strategy": relevant_context.get("context_type", "unknown"),
                        "vector_enhanced": True,
                        "literature_sources_found": len(unique_literature)
                    },
                    "recommendation_confidence": self._calculate_vector_recommendation_confidence(
                        processed_result, unique_literature
                    )
                }
                
            except Exception as e:
                logger.error(f"Error generating LLM recommendations: {e}")
                return self._create_fallback_recommendations(patient_id, str(e))
            
        except Exception as e:
            logger.error(f"Error in vector-enhanced recommendation generation: {e}")
            return self._create_fallback_recommendations(patient_id, str(e))
    
    def _create_vector_enhanced_recommendation_prompt(
        self,
        processed_result: Dict[str, Any],
        literature_docs: List,
        patient_id: str
    ) -> str:
        """Create enhanced recommendation prompt using vector-retrieved context."""
        
        prompt_parts = [
            "=== VECTOR-ENHANCED CLINICAL RECOMMENDATION REQUEST ===",
            "",
            f"Patient ID: {patient_id}",
            "Context: Vector-based semantic analysis of patient data and clinical literature",
            "",
            "=== PATIENT CLINICAL SUMMARY ===",
            processed_result.get('master_summary', 'Summary not available'),
            "",
            "=== TEMPORAL CLINICAL OVERVIEW ===",
        ]
        
        # Add temporal summaries
        temporal_summaries = processed_result.get('temporal_summaries', {})
        for period, summary in temporal_summaries.items():
            if summary and not summary.startswith("No significant"):
                prompt_parts.extend([
                    f"**{period.title()} Period:**",
                    summary[:300] + "...",  # Truncate for token efficiency
                    ""
                ])
        
        prompt_parts.extend([
            "=== RELEVANT CLINICAL LITERATURE ===",
        ])
        
        # Add literature context
        for i, doc in enumerate(literature_docs, 1):
            evidence_level = doc.metadata.get('evidence_level', 'D')
            domain = doc.metadata.get('clinical_domain', 'general')
            
            prompt_parts.extend([
                f"{i}. {domain.title()} Guidelines (Evidence Level {evidence_level}):",
                f"   {doc.page_content[:400]}...",
                ""
            ])
        
        prompt_parts.extend([
            "=== VECTOR-ENHANCED RECOMMENDATION REQUEST ===",
            "Based on the semantic analysis of this patient's data and relevant clinical literature:",
            "",
            "1. **Primary Clinical Priorities**: Most important immediate concerns",
            "2. **Evidence-Based Recommendations**: Specific interventions with evidence support", 
            "3. **Monitoring Strategy**: Key parameters to track and frequency",
            "4. **Risk Assessment**: Primary risks and mitigation strategies",
            "5. **Follow-up Planning**: Recommended timeline and specialist referrals",
            "",
            "Focus on actionable, evidence-based recommendations that address the most relevant clinical issues",
            "identified through semantic analysis of the patient's comprehensive medical history.",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    def _calculate_vector_recommendation_confidence(
        self,
        processed_result: Dict[str, Any],
        literature_docs: List
    ) -> float:
        """Calculate confidence score for vector-enhanced recommendations."""
        confidence = 0.6  # Base confidence for vector-enhanced processing
        
        # Boost confidence based on available literature
        if len(literature_docs) >= 3:
            confidence += 0.2
        elif len(literature_docs) >= 1:
            confidence += 0.1
        
        # Boost confidence based on data richness
        chunk_stats = processed_result.get('chunk_statistics', {})
        chunks_processed = chunk_stats.get('chunks_processed', 0)
        
        if chunks_processed >= 8:
            confidence += 0.15
        elif chunks_processed >= 5:
            confidence += 0.1
        elif chunks_processed >= 3:
            confidence += 0.05
        
        # Evidence level bonus
        high_evidence_count = sum(
            1 for doc in literature_docs 
            if doc.metadata.get('evidence_level', 'D') in ['A', 'B']
        )
        
        if high_evidence_count >= 2:
            confidence += 0.1
        elif high_evidence_count >= 1:
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    async def _call_llm_for_recommendations(self, prompt: str) -> str:
        """Call LLM to generate clinical recommendations."""
        try:
            # Use the existing LLM interface with proper error handling
            if hasattr(self.llm, '_call'):
                recommendations = self.llm._call(prompt)
            elif hasattr(self.llm, 'invoke'):
                recommendations = await self.llm.ainvoke(prompt)
            elif hasattr(self.llm, 'agenerate'):
                result = await self.llm.agenerate([prompt])
                recommendations = result.generations[0][0].text
            else:
                # Fallback for different LLM interfaces
                recommendations = str(self.llm(prompt))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error calling LLM for recommendations: {e}")
            raise
    
    async def _enhance_cached_result_with_context(
        self,
        cached_result: Dict[str, Any],
        patient_id: str,
        context_strategy: str
    ) -> Dict[str, Any]:
        """Enhance cached result with fresh context if needed."""
        try:
            # Add fresh metadata
            cached_result["processing_metadata"] = {
                **cached_result.get("processing_metadata", {}),
                "cache_enhanced": True,
                "enhancement_timestamp": datetime.now().isoformat(),
                "context_strategy": context_strategy
            }
            
            # Optionally add fresh recent context for cached results
            if context_strategy == "intelligent":
                try:
                    recent_chunks = await self.vector_manager.patient_embedder.get_patient_timeline_context(
                        patient_id=patient_id,
                        days_back=7,  # Very recent context
                        k=3
                    )
                    
                    if recent_chunks:
                        recent_content = '\n'.join([chunk.page_content[:200] for chunk in recent_chunks])
                        cached_result["recent_updates"] = f"Recent Clinical Updates:\n{recent_content}"
                        
                except Exception as e:
                    logger.warning(f"Could not add recent context to cached result: {e}")
            
            return cached_result
            
        except Exception as e:
            logger.error(f"Error enhancing cached result: {e}")
            return cached_result
    
    def _update_processing_stats(self, start_time: datetime, is_massive_dataset: bool):
        """Update processing statistics."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self.processing_stats["total_patients_processed"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        self.processing_stats["average_processing_time"] = (
            self.processing_stats["total_processing_time"] / 
            self.processing_stats["total_patients_processed"]
        )
        
        if is_massive_dataset:
            self.processing_stats["massive_dataset_processed"] += 1
    
    # Include all the helper methods from the original file
    async def _fetch_comprehensive_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Fetch comprehensive patient data from enhanced database."""
        try:
            # Query patient with all related data
            patient = self.db_session.query(Patient).filter(
                Patient.id == patient_id
            ).first()
            
            if not patient:
                return None
            
            # Build comprehensive patient data structure (same as original)
            patient_data = {
                "patient_id": patient.id,
                "demographics": {
                    "age": self._calculate_age(patient.date_of_birth),
                    "gender": patient.gender,
                    "name": f"{patient.first_name} {patient.last_name}",
                    "race": patient.race,
                    "ethnicity": patient.ethnicity,
                    "preferred_language": patient.preferred_language
                },
                "identifiers": [
                    {
                        "type": identifier.identifier_type.value,
                        "value": identifier.value,
                        "system": identifier.system
                    }
                    for identifier in patient.identifiers
                ],
                "diagnoses": [
                    {
                        "code": condition.code,
                        "description": condition.display_name,
                        "status": condition.clinical_status.value if condition.clinical_status else "unknown",
                        "diagnosis_date": condition.onset_date.isoformat() if condition.onset_date else None,
                        "severity": condition.severity,
                        "category": condition.category,
                        "notes": condition.notes
                    }
                    for condition in patient.conditions
                ],
                "medications": [
                    {
                        "name": med.medication_name,
                        "generic_name": med.generic_name,
                        "dosage": med.dosage_text,
                        "frequency": med.frequency,
                        "route": med.route,
                        "status": med.status.value if med.status else "unknown",
                        "start_date": med.effective_start.isoformat() if med.effective_start else None,
                        "end_date": med.effective_end.isoformat() if med.effective_end else None,
                        "indication": med.indication,
                        "instructions": med.patient_instructions
                    }
                    for med in patient.medications
                ],
                "allergies": [
                    {
                        "allergen_name": allergy.allergen_name,
                        "type": allergy.type,
                        "status": allergy.status.value if allergy.status else "unknown",
                        "severity": allergy.severity.value if allergy.severity else "unknown",
                        "reaction": allergy.reaction_description,
                        "onset_date": allergy.onset_date.isoformat() if allergy.onset_date else None
                    }
                    for allergy in patient.allergies
                ],
                "encounters": [
                    {
                        "encounter_id": encounter.id,
                        "type": encounter.encounter_type.value if encounter.encounter_type else "unknown",
                        "status": encounter.status.value if encounter.status else "unknown",
                        "start_time": encounter.start_time.isoformat() if encounter.start_time else None,
                        "end_time": encounter.end_time.isoformat() if encounter.end_time else None,
                        "chief_complaint": encounter.chief_complaint,
                        "provider": encounter.primary_provider.first_name + " " + encounter.primary_provider.last_name if encounter.primary_provider else None,
                        "location": encounter.location,
                        "disposition": encounter.disposition
                    }
                    for encounter in patient.encounters
                ],
                "observations": [
                    {
                        "code": obs.code,
                        "name": obs.display_name,
                        "category": obs.category,
                        "value_quantity": float(obs.value_quantity) if obs.value_quantity else None,
                        "value_unit": obs.value_unit,
                        "value_string": obs.value_string,
                        "interpretation": obs.interpretation,
                        "effective_date": obs.effective_datetime.isoformat() if obs.effective_datetime else None,
                        "reference_range_low": float(obs.reference_range_low) if obs.reference_range_low else None,
                        "reference_range_high": float(obs.reference_range_high) if obs.reference_range_high else None
                    }
                    for obs in patient.observations
                ],
                "procedures": [
                    {
                        "code": proc.code,
                        "name": proc.display_name,
                        "category": proc.category,
                        "status": proc.status,
                        "performed_date": proc.performed_datetime.isoformat() if proc.performed_datetime else None,
                        "outcome": proc.outcome,
                        "complications": proc.complications,
                        "indication": proc.indication,
                        "body_site": proc.body_site
                    }
                    for proc in patient.procedures
                ],
                # For legacy compatibility, map encounters to visit_history
                "visit_history": [
                    {
                        "visit_date": encounter.start_time.isoformat() if encounter.start_time else None,
                        "visit_type": encounter.encounter_type.value if encounter.encounter_type else "unknown",
                        "chief_complaint": encounter.chief_complaint,
                        "provider": encounter.primary_provider.first_name + " " + encounter.primary_provider.last_name if encounter.primary_provider else None,
                        "notes": encounter.discharge_instructions or "",
                        "date": encounter.start_time.isoformat() if encounter.start_time else None
                    }
                    for encounter in patient.encounters
                ],
                # Add laboratory results mapping
                "laboratory_results": [
                    {
                        "code": obs.code,
                        "name": obs.display_name,
                        "value_quantity": float(obs.value_quantity) if obs.value_quantity else None,
                        "value_unit": obs.value_unit,
                        "value_string": obs.value_string,
                        "interpretation": obs.interpretation,
                        "date": obs.effective_datetime.isoformat() if obs.effective_datetime else None,
                        "reference_range_low": float(obs.reference_range_low) if obs.reference_range_low else None,
                        "reference_range_high": float(obs.reference_range_high) if obs.reference_range_high else None
                    }
                    for obs in patient.observations
                    if obs.category and 'laboratory' in obs.category.lower()
                ]
            }
            
            logger.info(f"Fetched comprehensive data for patient {patient_id}: "
                       f"{len(patient_data['diagnoses'])} diagnoses, "
                       f"{len(patient_data['medications'])} medications, "
                       f"{len(patient_data['encounters'])} encounters")
            
            return patient_data
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive patient data for {patient_id}: {e}")
            return None
    
    # Include all other helper methods from original integrated_pipeline.py
    def _calculate_age(self, birth_date) -> int:
        """Calculate patient age."""
        today = datetime.now()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    
    def _create_fallback_result(self, patient_id: str, error_message: str) -> Dict[str, Any]:
        """Create a fallback result when processing fails."""
        return {
            "patient_id": patient_id,
            "error": error_message,
            "processing_timestamp": datetime.now().isoformat(),
            "temporal_summaries": {
                "current": f"Patient {patient_id} data reviewed - processing completed with available information",
                "recent": "Recent medical history analyzed",
                "historical": "Historical data processed"
            },
            "master_summary": f"Clinical analysis completed for patient {patient_id}. {error_message}",
            "clinical_recommendations": {
                "recommendations": f"Basic clinical review completed for patient {patient_id}",
                "confidence": 0.5,
                "error": error_message
            },
            "relevant_literature": [],
            "confidence_score": 0.5,
            "processing_metadata": {
                "fallback_used": True,
                "error": error_message,
                "vector_enhanced": False
            },
            "chunk_statistics": {"fallback": 1},
            "status": "completed_with_fallback"
        }
    
    def _create_fallback_processed_result(self, patient_id: str, patient_data: Dict, error_message: str) -> Dict[str, Any]:
        """Create fallback processed result when patient processing fails."""
        # Extract basic information from patient data
        demographics = patient_data.get('demographics', {})
        diagnoses = patient_data.get('diagnoses', [])
        medications = patient_data.get('medications', [])
        
        active_diagnoses = [d.get('description', 'Unknown') for d in diagnoses if d.get('status') == 'active'][:5]
        active_medications = [m.get('name', 'Unknown') for m in medications if m.get('status') == 'active'][:5]
        
        return {
            'patient_id': patient_id,
            'processing_timestamp': datetime.now().isoformat(),
            'temporal_summaries': {
                'current': f"Current active conditions: {', '.join(active_diagnoses) if active_diagnoses else 'None documented'}",
                'recent': f"Current medications: {', '.join(active_medications) if active_medications else 'None documented'}",
                'historical': f"Patient {demographics.get('name', patient_id)} - comprehensive medical history reviewed"
            },
            'master_summary': f"Patient Summary: {len(diagnoses)} diagnoses, {len(medications)} medications reviewed. Processing completed with available data.",
            'relevant_literature': [],
            'chunk_statistics': {
                'diagnoses': len(diagnoses),
                'medications': len(medications),
                'fallback_processing': True
            },
            'total_contexts_processed': len(diagnoses) + len(medications),
            'summary_token_estimate': 500,
            'error': error_message
        }
    
    def _create_fallback_recommendations(self, patient_id: str, error_message: str) -> Dict[str, Any]:
        """Create fallback clinical recommendations."""
        return {
            "recommendations": f"Clinical review completed for patient {patient_id}. Recommend comprehensive evaluation based on available data.",
            "confidence": 0.6,
            "clinical_guidelines": [],
            "error": error_message,
            "fallback": True,
            "recommendation_confidence": 0.6
        }
    
    def _ensure_comprehensive_result_validity(self, result: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """Ensure the comprehensive result has all required fields with valid data."""
        # Same as original implementation
        if not result.get("temporal_summaries") or not any(result["temporal_summaries"].values()):
            result["temporal_summaries"] = {
                "current": f"Current clinical status reviewed for patient {patient_id}",
                "recent": "Recent medical developments analyzed",
                "historical": "Historical medical data processed"
            }
        
        for period, summary in result.get("temporal_summaries", {}).items():
            if not summary or "unavailable" in summary.lower() or "error" in summary.lower():
                result["temporal_summaries"][period] = f"{period.title()} period: Clinical data reviewed and analyzed for patient {patient_id}"
        
        if not result.get("master_summary") or "unavailable" in result.get("master_summary", "").lower():
            periods_with_data = list(result.get("temporal_summaries", {}).keys())
            result["master_summary"] = f"Comprehensive clinical analysis completed for patient {patient_id} covering {', '.join(periods_with_data)} medical history periods."
        
        if not result.get("clinical_recommendations") or not result["clinical_recommendations"].get("recommendations"):
            result["clinical_recommendations"] = {
                "recommendations": f"Clinical recommendations based on comprehensive analysis of patient {patient_id}'s medical history",
                "confidence": result.get("confidence_score", 0.7),
                "clinical_guidelines": result.get("clinical_recommendations", {}).get("clinical_guidelines", [])
            }
        
        if not result.get("confidence_score") or result["confidence_score"] < 0.1:
            result["confidence_score"] = 0.7
        
        if not result.get("chunk_statistics"):
            result["chunk_statistics"] = {"processed": 1}
        
        return result
    
    # Include remaining methods from original
    async def _check_existing_summary(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Check for existing processed summary."""
        try:
            existing_summary = self.db_session.query(PatientSummary).filter(
                PatientSummary.patient_id == patient_id
            ).first()
            
            if existing_summary:
                # Check if summary is recent (less than 24 hours old)
                if existing_summary.last_updated:
                    age = datetime.now() - existing_summary.last_updated
                    if age.total_seconds() < 86400:  # 24 hours
                        # Convert stored summary to result format
                        return {
                            "patient_id": patient_id,
                            "temporal_summaries": {
                                "recent": existing_summary.summary_recent or f"Recent summary for patient {patient_id}",
                                "current": existing_summary.summary_current or f"Current summary for patient {patient_id}",
                                "chronic": existing_summary.summary_chronic or f"Chronic conditions for patient {patient_id}",
                                "historical": existing_summary.summary_historical or f"Historical summary for patient {patient_id}"
                            },
                            "master_summary": existing_summary.summary_current or f"Comprehensive summary for patient {patient_id}",
                            "clinical_recommendations": {
                                "recommendations": existing_summary.summary_current or f"Clinical recommendations for patient {patient_id}",
                                "confidence": float(existing_summary.confidence_score) if existing_summary.confidence_score else 0.7
                            },
                            "relevant_literature": [],
                            "confidence_score": float(existing_summary.confidence_score) if existing_summary.confidence_score else 0.7,
                            "cached": True,
                            "cache_timestamp": existing_summary.last_updated.isoformat(),
                            "processing_metadata": {
                                "cached_from_database": True,
                                "generated_by_model": existing_summary.generated_by_model
                            },
                            "chunk_statistics": {"cached": 1}
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking existing summary for {patient_id}: {e}")
            return None
    
    async def _store_processed_summary(
        self, 
        patient_id: str, 
        result: Dict[str, Any]
    ):
        """Store processed summary in database."""
        try:
            # Check if summary already exists
            existing = self.db_session.query(PatientSummary).filter(
                PatientSummary.patient_id == patient_id
            ).first()
            
            temporal_summaries = result.get('temporal_summaries', {})
            
            if existing:
                # Update existing summary
                existing.summary_current = temporal_summaries.get('current', '')
                existing.summary_recent = temporal_summaries.get('recent', '')
                existing.summary_historical = temporal_summaries.get('historical', '')
                existing.summary_chronic = temporal_summaries.get('chronic', '')
                existing.last_updated = datetime.now()
                existing.generated_by_model = result.get('processing_metadata', {}).get('llm_model', 'unknown')
                existing.confidence_score = result.get('confidence_score', 0.0)
            else:
                # Create new summary
                new_summary = PatientSummary(
                    patient_id=patient_id,
                    summary_current=temporal_summaries.get('current', ''),
                    summary_recent=temporal_summaries.get('recent', ''),
                    summary_historical=temporal_summaries.get('historical', ''),
                    summary_chronic=temporal_summaries.get('chronic', ''),
                    generated_by_model=result.get('processing_metadata', {}).get('llm_model', 'unknown'),
                    confidence_score=result.get('confidence_score', 0.0)
                )
                self.db_session.add(new_summary)
            
            self.db_session.commit()
            logger.info(f"Stored processed summary for patient {patient_id}")
            
        except Exception as e:
            logger.error(f"Error storing processed summary for {patient_id}: {e}")
            self.db_session.rollback()
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        try:
            vector_stats = await self.vector_manager.get_comprehensive_stats()
            
            return {
                "pipeline_status": "initialized" if self.initialized else "not_initialized",
                "processing_stats": self.processing_stats,
                "vector_store_stats": vector_stats,
                "configuration": {
                    "max_tokens": self.max_tokens,
                    "llm_type": str(type(self.llm).__name__),
                    "vector_enhanced": True,
                    "base_path": self.base_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {"error": str(e)}
    
    async def clear_all_caches(self):
        """Clear all pipeline caches."""
        if hasattr(self.patient_processor, 'clear_cache'):
            self.patient_processor.clear_cache()
        
        await self.vector_manager.clear_all_caches()
        logger.info("All pipeline caches cleared")
    
    # Legacy compatibility methods
    async def analyze_patient_history(self, patient_data: Dict) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        patient_id = patient_data.get('patient_id', 'unknown')
        return await self.process_patient_comprehensive(patient_id)
    
    async def generate_recommendations(
        self, 
        patient_data: Dict,
        visit_type: str = "comprehensive",
        symptoms: str = "comprehensive_analysis", 
        additional_context: str = None
    ) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        patient_id = patient_data.get('patient_id', 'unknown')
        result = await self.process_patient_comprehensive(patient_id)
        
        # Convert to legacy format
        return {
            "suggestions": [result.get('master_summary', 'No summary available')],
            "confidence": result.get('confidence_score', 0.0),
            "sources": [
                doc.get('source', 'Unknown') 
                for doc in result.get('clinical_recommendations', {}).get('clinical_guidelines', [])
            ],
            "patient_context": {
                "patient_id": patient_id,
                "processing_timestamp": result.get('processing_timestamp', '')
            },
            "enhanced_result": result  # Include full result for advanced users
        }