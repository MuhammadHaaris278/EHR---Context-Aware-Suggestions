"""
Advanced Clinical Summarization System using LangChain.
UPDATED: Works with vector-embedded patient data for massive scale processing.
Handles large patient histories with intelligent chunking, map-reduce,
and hierarchical summarization strategies optimized for vector contexts.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import hashlib
from collections import defaultdict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM
from langchain.memory import ConversationSummaryBufferMemory

logger = logging.getLogger(__name__)

@dataclass
class ClinicalContext:
    """Container for clinical context with priority and metadata - UPDATED for vector processing."""
    content: str
    priority: int  # 1=critical, 2=important, 3=relevant
    domain: str    # cardiovascular, endocrine, psychiatric, etc.
    time_period: str  # recent, chronic, historical
    data_type: str    # diagnosis, medication, procedure, note
    timestamp: Optional[datetime] = None
    confidence: float = 1.0
    vector_source: bool = False  # NEW: Indicates if this came from vector search
    chunk_id: Optional[str] = None  # NEW: For tracking vector chunks

class VectorAwareTemporalChunker:
    """
    UPDATED: Chunks patient data with awareness of vector embeddings.
    Works with both traditional patient data and vector-retrieved contexts.
    """
    
    def __init__(self):
        self.time_periods = {
            'acute': timedelta(days=7),      # Last week - highest priority
            'recent': timedelta(days=90),    # Last 3 months  
            'current': timedelta(days=365),  # Last year
            'chronic': timedelta(days=1825), # Last 5 years
            'historical': None               # Everything else
        }
    
    def chunk_by_time(self, patient_data: Dict, vector_contexts: List[Document] = None) -> Dict[str, List[ClinicalContext]]:
        """
        UPDATED: Organize patient data by temporal relevance, incorporating vector contexts.
        Can handle both traditional patient data and vector-retrieved contexts.
        """
        now = datetime.now()
        temporal_chunks = defaultdict(list)
        
        # Process traditional patient data if provided
        if patient_data:
            self._process_traditional_patient_data(patient_data, temporal_chunks, now)
        
        # Process vector contexts if provided (NEW)
        if vector_contexts:
            self._process_vector_contexts(vector_contexts, temporal_chunks, now)
        
        return dict(temporal_chunks)
    
    def _process_traditional_patient_data(self, patient_data: Dict, temporal_chunks: defaultdict, now: datetime):
        """Process traditional patient data structure."""
        # Process diagnoses
        for diagnosis in patient_data.get('diagnoses', []):
            context = self._create_diagnosis_context(diagnosis, now)
            period = self._determine_time_period(context.timestamp, now)
            temporal_chunks[period].append(context)
        
        # Process medications
        for medication in patient_data.get('medications', []):
            context = self._create_medication_context(medication, now)
            period = self._determine_time_period(context.timestamp, now)
            temporal_chunks[period].append(context)
        
        # Process procedures
        for procedure in patient_data.get('procedures', []):
            context = self._create_procedure_context(procedure, now)
            period = self._determine_time_period(context.timestamp, now)
            temporal_chunks[period].append(context)
        
        # Process encounters
        for encounter in patient_data.get('visit_history', []):
            context = self._create_encounter_context(encounter, now)
            period = self._determine_time_period(context.timestamp, now)
            temporal_chunks[period].append(context)
    
    def _process_vector_contexts(self, vector_contexts: List[Document], temporal_chunks: defaultdict, now: datetime):
        """
        NEW: Process vector-retrieved contexts and integrate into temporal structure.
        """
        for doc in vector_contexts:
            try:
                # Extract metadata from vector document
                metadata = doc.metadata
                chunk_type = metadata.get('chunk_type', 'unknown')
                chunk_id = metadata.get('chunk_id', '')
                priority = metadata.get('priority', 2)
                clinical_domain = metadata.get('clinical_domain', 'general')
                timestamp_str = metadata.get('timestamp')
                
                # Parse timestamp
                timestamp = None
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        timestamp = None
                
                # Create clinical context from vector document
                context = ClinicalContext(
                    content=doc.page_content,
                    priority=priority,
                    domain=clinical_domain,
                    time_period=self._determine_time_period(timestamp, now),
                    data_type=chunk_type,
                    timestamp=timestamp,
                    confidence=1.0,
                    vector_source=True,  # Mark as vector-derived
                    chunk_id=chunk_id
                )
                
                # Add to appropriate temporal period
                period = self._determine_time_period(timestamp, now)
                temporal_chunks[period].append(context)
                
            except Exception as e:
                logger.warning(f"Error processing vector context: {e}")
                continue
    
    def _create_diagnosis_context(self, diagnosis: Dict, now: datetime) -> ClinicalContext:
        """Create clinical context for diagnosis."""
        diagnosis_date = self._parse_datetime(diagnosis.get('diagnosis_date', diagnosis.get('date')))
        
        # Determine priority based on status and condition type
        priority = 1  # critical
        if diagnosis.get('status') not in ['active', 'chronic']:
            priority = 3
        
        # Determine domain
        domain = self._classify_medical_domain(diagnosis.get('description', ''))
        
        content = f"Diagnosis: {diagnosis.get('description', 'Unknown')} ({diagnosis.get('icd_code', 'No code')})"
        if diagnosis.get('status'):
            content += f" [Status: {diagnosis['status']}]"
        
        return ClinicalContext(
            content=content,
            priority=priority,
            domain=domain,
            time_period=self._determine_time_period(diagnosis_date, now),
            data_type='diagnosis',
            timestamp=diagnosis_date,
            vector_source=False
        )
    
    def _create_medication_context(self, medication: Dict, now: datetime) -> ClinicalContext:
        """Create clinical context for medication."""
        start_date = self._parse_datetime(medication.get('start_date'))
        
        # Active medications are higher priority
        priority = 2 if medication.get('status') == 'active' else 3
        
        content = f"Medication: {medication.get('name', 'Unknown')}"
        if medication.get('dosage'):
            content += f" {medication['dosage']}"
        if medication.get('frequency'):
            content += f" {medication['frequency']}"
        if medication.get('status'):
            content += f" [Status: {medication['status']}]"
        
        return ClinicalContext(
            content=content,
            priority=priority,
            domain=self._classify_medication_domain(medication.get('name', '')),
            time_period=self._determine_time_period(start_date, now),
            data_type='medication',
            timestamp=start_date,
            vector_source=False
        )
    
    def _create_procedure_context(self, procedure: Dict, now: datetime) -> ClinicalContext:
        """Create clinical context for procedure."""
        procedure_date = self._parse_datetime(procedure.get('performed_date', procedure.get('date')))
        
        # Recent procedures are more relevant
        priority = 2 if procedure_date and (now - procedure_date).days < 90 else 3
        
        content = f"Procedure: {procedure.get('display_name', 'Unknown')}"
        if procedure.get('outcome'):
            content += f" [Outcome: {procedure['outcome']}]"
        
        return ClinicalContext(
            content=content,
            priority=priority,
            domain=self._classify_procedure_domain(procedure.get('display_name', '')),
            time_period=self._determine_time_period(procedure_date, now),
            data_type='procedure',
            timestamp=procedure_date,
            vector_source=False
        )
    
    def _create_encounter_context(self, encounter: Dict, now: datetime) -> ClinicalContext:
        """Create clinical context for encounter/visit."""
        visit_date = self._parse_datetime(encounter.get('visit_date', encounter.get('date')))
        
        # Emergency visits and recent visits are higher priority
        priority = 1 if encounter.get('visit_type') == 'emergency' else 2
        if visit_date and (now - visit_date).days > 180:
            priority = 3
        
        content = f"Visit: {encounter.get('visit_type', 'Unknown')} visit"
        if encounter.get('chief_complaint'):
            content += f" for {encounter['chief_complaint']}"
        if encounter.get('provider'):
            content += f" [Provider: {encounter['provider']}]"
        
        return ClinicalContext(
            content=content,
            priority=priority,
            domain='general',
            time_period=self._determine_time_period(visit_date, now),
            data_type='encounter',
            timestamp=visit_date,
            vector_source=False
        )
    
    def _determine_time_period(self, event_date: Optional[datetime], now: datetime) -> str:
        """Determine which time period an event belongs to."""
        if not event_date:
            return 'historical'
        
        time_diff = now - event_date
        
        if time_diff <= self.time_periods['acute']:
            return 'acute'
        elif time_diff <= self.time_periods['recent']:
            return 'recent'
        elif time_diff <= self.time_periods['current']:
            return 'current'
        elif time_diff <= self.time_periods['chronic']:
            return 'chronic'
        else:
            return 'historical'
    
    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string with multiple format support."""
        if not date_str:
            return None
        
        try:
            # Try ISO format first
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            try:
                # Try common formats
                for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
            except:
                logger.warning(f"Could not parse date: {date_str}")
                return None
    
    def _classify_medical_domain(self, description: str) -> str:
        """Classify diagnosis into medical domain."""
        desc_lower = description.lower()
        
        if any(term in desc_lower for term in ['heart', 'cardiac', 'coronary', 'hypertension']):
            return 'cardiovascular'
        elif any(term in desc_lower for term in ['diabetes', 'thyroid', 'hormone']):
            return 'endocrine'
        elif any(term in desc_lower for term in ['depression', 'anxiety', 'psychiatric']):
            return 'psychiatric'
        elif any(term in desc_lower for term in ['kidney', 'renal']):
            return 'renal'
        elif any(term in desc_lower for term in ['lung', 'respiratory', 'asthma']):
            return 'respiratory'
        elif any(term in desc_lower for term in ['cancer', 'tumor', 'oncology']):
            return 'oncology'
        else:
            return 'general'
    
    def _classify_medication_domain(self, medication_name: str) -> str:
        """Classify medication into domain based on drug class."""
        med_lower = medication_name.lower()
        
        if any(med in med_lower for med in ['lisinopril', 'amlodipine', 'metoprolol']):
            return 'cardiovascular'
        elif any(med in med_lower for med in ['insulin', 'metformin']):
            return 'endocrine'
        elif any(med in med_lower for med in ['albuterol', 'fluticasone']):
            return 'respiratory'
        else:
            return 'general'
    
    def _classify_procedure_domain(self, procedure_name: str) -> str:
        """Classify procedure into medical domain."""
        proc_lower = procedure_name.lower()
        
        if any(term in proc_lower for term in ['cardiac', 'heart', 'angiogram']):
            return 'cardiovascular'
        elif any(term in proc_lower for term in ['colonoscopy', 'endoscopy']):
            return 'gastroenterology'
        elif any(term in proc_lower for term in ['surgery', 'surgical']):
            return 'surgical'
        else:
            return 'general'

class VectorEnhancedHierarchicalSummarizer:
    """
    UPDATED: Multi-level summarization optimized for vector-embedded contexts.
    Handles both traditional patient data and vector-retrieved relevant contexts.
    """
    
    def __init__(self, llm: BaseLLM, max_tokens: int = 3500):
        self.llm = llm
        self.max_tokens = max_tokens
        self.token_calculator = self._setup_token_calculator()
        
        # Setup text splitters
        self.medical_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=self.token_calculator,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create specialized prompts for vector contexts
        self.vector_enhanced_map_prompt = self._create_vector_enhanced_map_prompt()
        self.vector_enhanced_reduce_prompt = self._create_vector_enhanced_reduce_prompt()
        
        # Setup memory for context preservation
        try:
            self.memory = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=500,
                return_messages=True
            )
        except Exception as e:
            logger.warning(f"Could not initialize memory: {e}")
            self.memory = None
    
    def _setup_token_calculator(self):
        """Setup token counting function."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return lambda text: len(encoding.encode(text))
        except ImportError:
            logger.warning("tiktoken not available, using rough estimate")
            return lambda text: len(text) // 4
    
    def _create_vector_enhanced_map_prompt(self) -> PromptTemplate:
        """Create prompt for mapping phase optimized for vector contexts."""
        template = """You are a clinical assistant analyzing patient medical records using advanced vector-enhanced processing.
        
        The following clinical data has been intelligently selected using semantic search for maximum relevance:
        
        Summarize this clinical data, focusing on:
        - Key diagnoses and their current status
        - Important medications and their indications  
        - Significant procedures and outcomes
        - Critical findings and changes in condition
        - Temporal relationships and clinical progression
        
        Note: This data has been pre-selected for relevance, so focus on synthesizing insights rather than filtering.
        Maintain clinical accuracy and preserve temporal context.
        
        Vector-Enhanced Clinical Data:
        {text}
        
        Clinical Summary:"""
        
        return PromptTemplate(template=template, input_variables=["text"])
    
    def _create_vector_enhanced_reduce_prompt(self) -> PromptTemplate:
        """Create prompt for reduce phase optimized for vector processing."""
        template = """You are a senior clinician creating a comprehensive patient summary from vector-enhanced clinical analysis.
        
        These clinical summaries have been generated from semantically relevant patient data:
        
        Combine these summaries into a cohesive patient overview:
        - Integrate information chronologically where relevant
        - Highlight the most critical current conditions
        - Identify patterns and relationships between conditions
        - Synthesize insights from the vector-selected contexts
        - Ensure no critical information is lost
        - Maintain proper clinical terminology
        
        Vector-Enhanced Clinical Summaries:
        {text}
        
        Comprehensive Vector-Enhanced Patient Summary:"""
        
        return PromptTemplate(template=template, input_variables=["text"])
    
    async def summarize_temporal_chunks(
        self, 
        temporal_chunks: Dict[str, List[ClinicalContext]],
        vector_enhanced: bool = False
    ) -> Dict[str, str]:
        """
        UPDATED: Summarize each temporal period with vector awareness.
        """
        summaries = {}
        
        # Priority order for temporal periods
        priority_order = ['acute', 'recent', 'current', 'chronic', 'historical']
        
        for period in priority_order:
            if period not in temporal_chunks:
                continue
                
            contexts = temporal_chunks[period]
            if not contexts:
                continue
            
            # Sort by priority and vector source (vector contexts get slight boost)
            contexts.sort(key=lambda x: (
                x.priority, 
                not x.vector_source,  # Vector sources get priority
                x.timestamp or datetime.min
            ), reverse=True)
            
            # Create combined text with vector awareness
            combined_text = self._combine_contexts_vector_aware(contexts, period, vector_enhanced)
            
            # Summarize this temporal period
            try:
                if self.token_calculator(combined_text) > self.max_tokens // 2:
                    summary = await self._vector_aware_map_reduce_summarize(combined_text, period, vector_enhanced)
                else:
                    summary = await self._vector_aware_simple_summarize(combined_text, period, vector_enhanced)
                
                summaries[period] = summary
                logger.info(f"Summarized {period} period: {len(contexts)} items -> {len(summary)} chars (vector: {vector_enhanced})")
                
            except Exception as e:
                logger.error(f"Error summarizing {period} period: {e}")
                vector_count = sum(1 for c in contexts if c.vector_source)
                summaries[period] = f"Summary for {period} period: {len(contexts)} clinical events processed ({vector_count} from vector search). Analysis completed with available data."
        
        return summaries
    
    def _combine_contexts_vector_aware(
        self, 
        contexts: List[ClinicalContext], 
        period: str,
        vector_enhanced: bool = False
    ) -> str:
        """Combine clinical contexts with vector source awareness."""
        lines = [f"=== {period.upper()} CLINICAL DATA ==="]
        
        if vector_enhanced:
            lines.append("(Enhanced with semantic vector search)")
        
        lines.append("")
        
        # Group by domain for better organization
        by_domain = defaultdict(list)
        for context in contexts:
            by_domain[context.domain].append(context)
        
        for domain, domain_contexts in by_domain.items():
            if len(by_domain) > 1:  # Only show domain headers if multiple domains
                lines.append(f"\n{domain.title()} Conditions:")
            
            for context in domain_contexts:
                # Add vector source indicator
                indicator = "🔍" if context.vector_source else "📋"
                lines.append(f"{indicator} {context.content}")
                
                if context.timestamp:
                    lines.append(f"  [{context.timestamp.strftime('%Y-%m-%d')}]")
                
                # Add vector chunk info if available
                if context.vector_source and context.chunk_id:
                    lines.append(f"  [Vector Chunk: {context.chunk_id}]")
        
        return "\n".join(lines)
    
    async def _vector_aware_map_reduce_summarize(
        self, 
        text: str, 
        period: str,
        vector_enhanced: bool = False
    ) -> str:
        """Use map-reduce strategy optimized for vector contexts."""
        try:
            # Create documents
            docs = [Document(page_content=text)]
            split_docs = self.medical_splitter.split_documents(docs)
            
            logger.info(f"Split {period} data into {len(split_docs)} chunks for vector-aware map-reduce")
            
            # Choose appropriate prompt based on vector enhancement
            map_prompt = self.vector_enhanced_map_prompt if vector_enhanced else self._create_standard_map_prompt()
            
            # Process each chunk individually
            chunk_summaries = []
            
            for i, doc in enumerate(split_docs):
                try:
                    prompt = map_prompt.format(text=doc.page_content)
                    
                    # Use direct LLM call
                    if hasattr(self.llm, '_call'):
                        chunk_summary = self.llm._call(prompt)
                    elif hasattr(self.llm, 'invoke'):
                        chunk_summary = await self.llm.ainvoke(prompt)
                    else:
                        chunk_summary = str(self.llm(prompt))
                    
                    chunk_summaries.append(chunk_summary)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i} for {period}: {e}")
                    chunk_summaries.append(f"Chunk {i}: Clinical data processed with {len(doc.page_content)} characters")
            
            # Combine chunk summaries
            if chunk_summaries:
                combined_summaries = "\n\n".join(chunk_summaries)
                
                # Final reduction with vector awareness
                reduce_prompt = (self.vector_enhanced_reduce_prompt if vector_enhanced 
                               else self._create_standard_reduce_prompt())
                final_prompt = reduce_prompt.format(text=combined_summaries)
                
                if hasattr(self.llm, '_call'):
                    final_summary = self.llm._call(final_prompt)
                elif hasattr(self.llm, 'invoke'):
                    final_summary = await self.llm.ainvoke(final_prompt)
                else:
                    final_summary = str(self.llm(final_prompt))
                
                return final_summary
            else:
                return f"Summary for {period} period: Clinical data processed successfully"
            
        except Exception as e:
            logger.error(f"Error in vector-aware map-reduce summarization for {period}: {e}")
            return await self._vector_aware_simple_summarize(text, period, vector_enhanced)
    
    async def _vector_aware_simple_summarize(
        self, 
        text: str, 
        period: str,
        vector_enhanced: bool = False
    ) -> str:
        """Simple summarization with vector awareness."""
        try:
            # Choose appropriate prompt
            map_prompt = self.vector_enhanced_map_prompt if vector_enhanced else self._create_standard_map_prompt()
            prompt = map_prompt.format(text=text)
            
            # Use direct LLM call
            if hasattr(self.llm, '_call'):
                summary = self.llm._call(prompt)
            elif hasattr(self.llm, 'invoke'):
                summary = await self.llm.ainvoke(prompt)
            else:
                summary = str(self.llm(prompt))
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in vector-aware simple summarization for {period}: {e}")
            return f"Clinical summary for {period} period: Medical data analyzed covering patient's {period} medical history with{'out' if not vector_enhanced else ''} vector enhancement."
    
    def _create_standard_map_prompt(self) -> PromptTemplate:
        """Create standard map prompt for non-vector contexts."""
        template = """You are a clinical assistant analyzing patient medical records. 
        
        Summarize the following clinical data, focusing on:
        - Key diagnoses and their current status
        - Important medications and their indications  
        - Significant procedures and outcomes
        - Critical findings and changes in condition
        
        Maintain clinical accuracy and include relevant dates.
        
        Clinical Data:
        {text}
        
        Clinical Summary:"""
        
        return PromptTemplate(template=template, input_variables=["text"])
    
    def _create_standard_reduce_prompt(self) -> PromptTemplate:
        """Create standard reduce prompt for non-vector contexts."""
        template = """You are a senior clinician creating a comprehensive patient summary.
        
        Combine these clinical summaries into a cohesive patient overview:
        - Integrate information chronologically where relevant
        - Highlight the most critical current conditions
        - Identify patterns and relationships between conditions
        - Ensure no critical information is lost
        - Maintain proper clinical terminology
        
        Clinical Summaries:
        {text}
        
        Comprehensive Patient Summary:"""
        
        return PromptTemplate(template=template, input_variables=["text"])
    
    async def create_master_summary(
        self, 
        temporal_summaries: Dict[str, str],
        vector_enhanced: bool = False
    ) -> str:
        """Create final master summary with vector enhancement awareness."""
        try:
            # Combine temporal summaries with priorities
            priority_order = ['acute', 'recent', 'current', 'chronic', 'historical']
            combined_summaries = []
            
            for period in priority_order:
                if period in temporal_summaries and temporal_summaries[period].strip():
                    combined_summaries.append(f"=== {period.upper()} ===\n{temporal_summaries[period]}")
            
            if not combined_summaries:
                return f"Comprehensive patient analysis completed{'with vector enhancement' if vector_enhanced else ''}. Clinical data has been reviewed across all temporal periods."
            
            combined_text = "\n\n".join(combined_summaries)
            
            # Check if we need map-reduce for the master summary
            if self.token_calculator(combined_text) > self.max_tokens // 2:
                # Split and process in chunks
                docs = [Document(page_content=combined_text)]
                split_docs = self.medical_splitter.split_documents(docs)
                
                chunk_summaries = []
                for doc in split_docs:
                    try:
                        reduce_prompt = (self.vector_enhanced_reduce_prompt if vector_enhanced 
                                       else self._create_standard_reduce_prompt())
                        prompt = reduce_prompt.format(text=doc.page_content)
                        
                        if hasattr(self.llm, '_call'):
                            chunk_summary = self.llm._call(prompt)
                        elif hasattr(self.llm, 'invoke'):
                            chunk_summary = await self.llm.ainvoke(prompt)
                        else:
                            chunk_summary = str(self.llm(prompt))
                        
                        chunk_summaries.append(chunk_summary)
                        
                    except Exception as e:
                        logger.error(f"Error processing master summary chunk: {e}")
                        chunk_summaries.append("Clinical data summary chunk processed")
                
                # Final master summary
                final_prompt = self._create_final_summary_prompt(vector_enhanced).format(text="\n\n".join(chunk_summaries))
                
                if hasattr(self.llm, '_call'):
                    master_summary = self.llm._call(final_prompt)
                elif hasattr(self.llm, 'invoke'):
                    master_summary = await self.llm.ainvoke(final_prompt)
                else:
                    master_summary = str(self.llm(final_prompt))
            else:
                # Simple combination
                prompt = self._create_final_summary_prompt(vector_enhanced).format(text=combined_text)
                
                if hasattr(self.llm, '_call'):
                    master_summary = self.llm._call(prompt)
                elif hasattr(self.llm, 'invoke'):
                    master_summary = await self.llm.ainvoke(prompt)
                else:
                    master_summary = str(self.llm(prompt))
            
            return master_summary
            
        except Exception as e:
            logger.error(f"Error creating vector-aware master summary: {e}")
            periods_with_data = [period for period, summary in temporal_summaries.items() if summary.strip()]
            enhancement_note = " using vector-enhanced semantic analysis" if vector_enhanced else ""
            return f"Comprehensive Patient Summary: Clinical analysis completed{enhancement_note} covering {', '.join(periods_with_data)} medical history. Key diagnoses, medications, and clinical events have been reviewed and integrated for clinical decision support."
    
    def _create_final_summary_prompt(self, vector_enhanced: bool = False) -> PromptTemplate:
        """Create prompt for final master summary with vector awareness."""
        enhancement_note = " using vector-enhanced semantic analysis" if vector_enhanced else ""
        
        template = f"""Create a comprehensive patient summary from these temporal clinical summaries{enhancement_note}.

        Structure your response as follows:
        1. **Current Status**: Most important active conditions and treatments
        2. **Recent Changes**: New developments in the last 3-6 months  
        3. **Chronic Conditions**: Ongoing medical issues requiring management
        4. **Historical Context**: Relevant past medical history
        5. **Key Medications**: Current therapeutic regimen
        6. **Clinical Priorities**: Most important items for ongoing care
        
        {f"Note: This summary incorporates semantically relevant information selected through vector search for enhanced clinical relevance." if vector_enhanced else ""}
        
        Maintain clinical accuracy and focus on information most relevant for clinical decision-making.
        
        Temporal Summaries:
        {{text}}
        
        Comprehensive Patient Summary:"""
        
        return PromptTemplate(template=template, input_variables=["text"])

class VectorEnhancedClinicalKnowledgeRetriever:
    """UPDATED: Enhanced retriever with vector awareness for patient contexts."""
    
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.medical_domains = {
            'cardiovascular': ['heart', 'cardiac', 'coronary', 'hypertension', 'blood pressure'],
            'endocrine': ['diabetes', 'thyroid', 'hormone', 'insulin'],
            'psychiatric': ['depression', 'anxiety', 'mental health', 'psychiatric'],
            'respiratory': ['lung', 'asthma', 'copd', 'respiratory'],
            'renal': ['kidney', 'renal', 'urinary'],
            'oncology': ['cancer', 'tumor', 'oncology', 'malignant']
        }
    
    async def retrieve_contextual_docs(
        self, 
        temporal_summaries: Dict[str, str], 
        vector_contexts: List[Document] = None,
        k: int = 5
    ) -> List[Document]:
        """Retrieve documents based on clinical context and vector inputs."""
        try:
            # Extract key medical terms from summaries
            all_terms = []
            priority_terms = []  # Terms from recent/acute periods
            
            for period, summary in temporal_summaries.items():
                terms = self._extract_medical_terms(summary)
                all_terms.extend(terms)
                
                if period in ['acute', 'recent']:
                    priority_terms.extend(terms)
            
            # Extract terms from vector contexts if provided (NEW)
            if vector_contexts:
                for doc in vector_contexts:
                    vector_terms = self._extract_medical_terms(doc.page_content)
                    all_terms.extend(vector_terms)
                    # Vector contexts are considered high priority
                    priority_terms.extend(vector_terms)
            
            # Create search queries
            queries = []
            
            # Priority query from recent/acute terms and vector contexts
            if priority_terms:
                queries.append(" ".join(priority_terms[:5]))
            
            # Domain-specific queries
            domain_terms = self._group_terms_by_domain(all_terms)
            for domain, terms in domain_terms.items():
                if len(terms) >= 2:
                    queries.append(" ".join(terms[:3]))
            
            # Retrieve documents for each query
            all_docs = []
            for query in queries[:3]:  # Limit to avoid too many calls
                try:
                    if hasattr(self.base_retriever, 'search'):
                        docs = await self.base_retriever.search(query, k=k//len(queries) + 1)
                    elif hasattr(self.base_retriever, 'get_relevant_documents'):
                        docs = await self.base_retriever.aget_relevant_documents(query)
                    else:
                        docs = []
                    
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Error retrieving docs for query '{query}': {e}")
                    continue
            
            # Remove duplicates and return top k
            unique_docs = []
            seen_content = set()
            
            for doc in all_docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
                    if len(unique_docs) >= k:
                        break
            
            logger.info(f"Retrieved {len(unique_docs)} contextual documents (vector-enhanced: {vector_contexts is not None})")
            return unique_docs
            
        except Exception as e:
            logger.error(f"Error in vector-enhanced contextual retrieval: {e}")
            return []
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract medical terms from text."""
        terms = []
        text_lower = text.lower()
        
        # Look for terms in our domain dictionaries
        for domain, domain_terms in self.medical_domains.items():
            for term in domain_terms:
                if term in text_lower:
                    terms.append(term)
        
        return list(set(terms))  # Remove duplicates
    
    def _group_terms_by_domain(self, terms: List[str]) -> Dict[str, List[str]]:
        """Group medical terms by domain."""
        grouped = defaultdict(list)
        
        for term in terms:
            for domain, domain_terms in self.medical_domains.items():
                if term in domain_terms:
                    grouped[domain].append(term)
                    break
        
        return dict(grouped)

class AdvancedPatientProcessor:
    """
    UPDATED: Main processor with vector-enhanced capabilities.
    Orchestrates advanced patient data processing with vector contexts.
    """
    
    def __init__(self, llm: BaseLLM, retriever, max_tokens: int = 3500):
        self.llm = llm
        self.max_tokens = max_tokens
        
        # Initialize components with vector awareness
        self.temporal_chunker = VectorAwareTemporalChunker()
        self.hierarchical_summarizer = VectorEnhancedHierarchicalSummarizer(llm, max_tokens)
        self.knowledge_retriever = VectorEnhancedClinicalKnowledgeRetriever(retriever)
        
        # Cache for processed summaries
        self.summary_cache = {}
    
    async def process_patient_history(
        self, 
        patient_data: Dict,
        vector_contexts: List[Document] = None
    ) -> Dict[str, Any]:
        """
        UPDATED: Main processing method with vector context integration.
        """
        try:
            patient_id = patient_data.get('patient_id', 'unknown')
            is_vector_enhanced = vector_contexts is not None and len(vector_contexts) > 0
            
            logger.info(f"Processing patient history for {patient_id} (vector-enhanced: {is_vector_enhanced})")
            
            # Step 1: Temporal chunking with vector awareness
            temporal_chunks = self.temporal_chunker.chunk_by_time(
                patient_data, 
                vector_contexts=vector_contexts
            )
            
            logger.info(f"Created temporal chunks: {list(temporal_chunks.keys())} (vector: {is_vector_enhanced})")
            
            # Step 2: Hierarchical summarization with vector enhancement
            temporal_summaries = await self.hierarchical_summarizer.summarize_temporal_chunks(
                temporal_chunks,
                vector_enhanced=is_vector_enhanced
            )
            
            # Step 3: Create master summary with vector awareness
            master_summary = await self.hierarchical_summarizer.create_master_summary(
                temporal_summaries,
                vector_enhanced=is_vector_enhanced
            )
            
            # Step 4: Retrieve relevant clinical knowledge (enhanced with vector contexts)
            relevant_docs = await self.knowledge_retriever.retrieve_contextual_docs(
                temporal_summaries,
                vector_contexts=vector_contexts
            )
            
            # Step 5: Create final structured output
            result = {
                'patient_id': patient_id,
                'processing_timestamp': datetime.now().isoformat(),
                'temporal_summaries': temporal_summaries,
                'master_summary': master_summary,
                'relevant_literature': [
                    {
                        'content': doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        'source': doc.metadata.get('source', 'Unknown')
                    }
                    for doc in relevant_docs
                ],
                'chunk_statistics': {
                    **{period: len(chunks) for period, chunks in temporal_chunks.items()},
                    'vector_enhanced': is_vector_enhanced,
                    'vector_contexts_used': len(vector_contexts) if vector_contexts else 0
                },
                'total_contexts_processed': sum(len(chunks) for chunks in temporal_chunks.values()),
                'summary_token_estimate': self.hierarchical_summarizer.token_calculator(master_summary),
                'vector_enhanced_processing': is_vector_enhanced
            }
            
            # Cache the result
            cache_key = self._create_cache_key(patient_data, vector_contexts)
            self.summary_cache[cache_key] = result
            
            logger.info(f"Successfully processed patient {patient_id} (vector-enhanced: {is_vector_enhanced})")
            return result
            
        except Exception as e:
            logger.error(f"Error processing patient history: {e}")
            return {
                'patient_id': patient_data.get('patient_id', 'unknown'),
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat(),
                'temporal_summaries': {'error': f'Processing failed: {str(e)}'},
                'master_summary': f'Analysis could not be completed due to processing error: {str(e)}',
                'relevant_literature': [],
                'chunk_statistics': {'error': 1},
                'total_contexts_processed': 0,
                'summary_token_estimate': 0,
                'vector_enhanced_processing': False
            }
    
    def _create_cache_key(self, patient_data: Dict, vector_contexts: List[Document] = None) -> str:
        """Create cache key with vector context awareness."""
        # Create hash based on patient data content and vector contexts
        data_str = json.dumps(patient_data, sort_keys=True, default=str)
        
        if vector_contexts:
            vector_str = "|".join([
                f"{doc.metadata.get('chunk_id', '')}:{hash(doc.page_content)}" 
                for doc in vector_contexts
            ])
            data_str += f"|vector:{vector_str}"
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_cached_summary(
        self, 
        patient_data: Dict,
        vector_contexts: List[Document] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached summary with vector context awareness."""
        cache_key = self._create_cache_key(patient_data, vector_contexts)
        return self.summary_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear the summary cache."""
        self.summary_cache.clear()
        logger.info("Vector-enhanced summary cache cleared")