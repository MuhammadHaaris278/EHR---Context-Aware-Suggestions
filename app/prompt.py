"""
Enhanced Clinical Prompt Templates for Vector-Enhanced Medical AI System.
UPDATED: Optimized for vector-based context selection and massive EHR datasets.
Includes prompts for GPT-4.1, Mistral AI, and vector-enhanced processing.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ClinicalPromptTemplate:
    """
    Enhanced clinical prompt templates optimized for vector-enhanced processing.
    UPDATED: Supports massive datasets with intelligent context selection.
    """
    
    def __init__(self):
        self.base_templates = self._initialize_base_templates()
        self.vector_enhanced_templates = self._initialize_vector_templates()
        self.model_specific_templates = self._initialize_model_specific_templates()
    
    def _initialize_base_templates(self) -> Dict[str, str]:
        """Initialize base clinical prompt templates."""
        return {
            "comprehensive_analysis": """
=== COMPREHENSIVE CLINICAL ANALYSIS REQUEST ===

You are an expert clinical diagnostician analyzing a patient's comprehensive medical history.
Provide a thorough diagnostic assessment based on the available clinical data.

Patient Information:
{patient_summary}

Clinical History Overview:
{clinical_timeline}

Current Clinical Status:
{current_status}

Recent Changes:
{recent_changes}

REQUEST: Provide a comprehensive clinical analysis including:
1. **Primary Clinical Assessment**: Most likely diagnoses with supporting evidence
2. **Diagnostic Reasoning**: Step-by-step clinical thought process
3. **Risk Stratification**: Immediate, short-term, and long-term risks
4. **Clinical Recommendations**: Evidence-based interventions and monitoring
5. **Follow-up Planning**: Recommended timeline and specialist referrals

Focus on actionable clinical insights based on the patient's specific medical history.
""",
            
            "diagnostic_assessment": """
=== DIAGNOSTIC ASSESSMENT REQUEST ===

Analyze this patient's clinical presentation and provide a structured diagnostic assessment.

Patient Clinical Data:
{clinical_data}

Relevant Medical Literature:
{literature_context}

Provide:
1. **Primary Diagnosis** with confidence level and supporting evidence
2. **Differential Diagnoses** ranked by likelihood with rationale
3. **Clinical Reasoning** explaining your diagnostic thought process
4. **Recommended Workup** specific tests and evaluations needed
5. **Risk Assessment** key clinical risks and monitoring requirements

Base your analysis on evidence-based medicine and clinical best practices.
""",
            
            "treatment_recommendations": """
=== TREATMENT RECOMMENDATION REQUEST ===

Based on this patient's clinical profile, provide evidence-based treatment recommendations.

Patient Clinical Profile:
{patient_profile}

Current Treatment Context:
{treatment_context}

Clinical Guidelines:
{clinical_guidelines}

Provide:
1. **Treatment Priorities** most important therapeutic interventions
2. **Medication Recommendations** specific drugs, dosages, and rationales
3. **Non-pharmacological Interventions** lifestyle and therapeutic modifications
4. **Monitoring Plan** key parameters to track and frequency
5. **Safety Considerations** contraindications and drug interactions

Ensure all recommendations are patient-specific and evidence-based.
"""
        }
    
    def _initialize_vector_templates(self) -> Dict[str, str]:
        """Initialize vector-enhanced prompt templates."""
        return {
            "vector_comprehensive_analysis": """
=== VECTOR-ENHANCED COMPREHENSIVE CLINICAL ANALYSIS ===

You are analyzing a patient using advanced semantic search technology that has intelligently selected
the most clinically relevant information from their comprehensive medical history.

🔍 SEMANTIC CONTEXT SELECTION:
The following clinical data has been selected using vector-based semantic search for maximum relevance:

Vector-Selected Patient Summary:
{vector_patient_summary}

Semantically Relevant Clinical Timeline:
{vector_clinical_timeline}

Intelligent Context Selection Results:
{context_selection_metadata}

📚 RELEVANT CLINICAL LITERATURE:
{vector_literature_context}

COMPREHENSIVE ANALYSIS REQUEST:
Using the semantically selected clinical data, provide:

1. **Primary Clinical Assessment**
   - Most likely diagnoses with confidence levels
   - Key supporting evidence from the selected contexts
   - Clinical significance of identified patterns

2. **Vector-Enhanced Diagnostic Reasoning**
   - How the semantic selection supports your clinical reasoning
   - Integration of multiple relevant clinical contexts
   - Temporal relationships and disease progression patterns

3. **Risk Stratification**
   - Immediate risks (0-48 hours) with specific monitoring
   - Short-term risks (days-weeks) with interventions
   - Long-term risks with preventive strategies

4. **Evidence-Based Clinical Recommendations**
   - Prioritized interventions based on vector-selected contexts
   - Specific monitoring parameters and frequencies
   - Therapeutic adjustments based on clinical patterns

5. **Intelligent Follow-up Planning**
   - Timeline for reassessment based on risk stratification
   - Specialist referrals with specific indications
   - Patient education priorities

FOCUS: Leverage the power of semantic context selection to provide highly relevant,
personalized clinical insights that address the most important aspects of this patient's care.
""",
            
            "massive_dataset_analysis": """
=== MASSIVE EHR DATASET ANALYSIS ===

You are analyzing a patient with an extensive medical history (Burj Khalifa scale data: {estimated_data_lines}+ lines).
Advanced vector embedding technology has intelligently selected the most relevant clinical contexts.

🏗️ MASSIVE DATASET PROCESSING:
- Total estimated data lines: {estimated_data_lines}
- Context selection strategy: {context_strategy}
- Semantic relevance filtering: ENABLED
- Temporal prioritization: ACTIVE

📊 INTELLIGENTLY SELECTED CONTEXTS:
{selected_contexts}

🎯 CLINICAL PRIORITY MATRIX:
{priority_contexts}

ANALYSIS REQUEST FOR MASSIVE DATASET:
Despite the enormous volume of clinical data, focus your analysis on the intelligently selected contexts:

1. **High-Priority Clinical Issues** (from semantic selection)
   - Most critical current conditions requiring immediate attention
   - Key patterns identified across the massive dataset
   - Clinically significant trends and changes

2. **Integrated Diagnostic Assessment** 
   - Primary diagnoses supported by multiple data points
   - How different contexts support or contradict each other
   - Confidence levels based on data completeness and consistency

3. **Massive Data Synthesis**
   - Key insights that emerge only from comprehensive data analysis
   - Long-term patterns visible in extensive medical history
   - Rare events or conditions identified through comprehensive review

4. **Scalable Clinical Recommendations**
   - Prioritized interventions based on massive data analysis
   - Monitoring strategies that account for data complexity
   - Long-term care planning informed by comprehensive history

5. **Data-Driven Risk Assessment**
   - Risk patterns identified across extensive timeframes
   - Predictive insights based on comprehensive data analysis
   - Prevention strategies informed by long-term data trends

ADVANTAGE: Use the power of massive dataset analysis to identify patterns and insights
that would be impossible with limited data, while maintaining clinical relevance through
intelligent context selection.
""",
            
            "semantic_patient_search": """
=== SEMANTIC PATIENT SEARCH ANALYSIS ===

You are analyzing results from a semantic search within a patient's embedded medical data.
This represents the most relevant clinical information based on the search query.

🔍 SEARCH CONTEXT:
Query: "{search_query}"
Patient ID: {patient_id}
Results Found: {results_count}
Search Scope: {search_scope}

📋 SEMANTICALLY RELEVANT RESULTS:
{search_results}

SEMANTIC ANALYSIS REQUEST:
Based on the semantic search results, provide:

1. **Query-Specific Clinical Insights**
   - Direct answers to the clinical query
   - Relevant findings from the semantic search
   - Clinical significance of discovered information

2. **Pattern Recognition**
   - Recurring themes across search results
   - Temporal patterns in the semantic matches
   - Clinical relationships identified through search

3. **Contextual Clinical Assessment**
   - How search results relate to overall patient care
   - Missing information that might be clinically relevant
   - Recommendations for additional semantic searches

4. **Actionable Clinical Guidance**
   - Specific recommendations based on search findings
   - Monitoring or follow-up suggested by results
   - Clinical decisions supported by semantic analysis

FOCUS: Extract maximum clinical value from the semantically selected information
to answer the specific clinical query while maintaining comprehensive patient context.
"""
        }
    
    def _initialize_model_specific_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize model-specific prompt templates."""
        return {
            "gpt4": {
                "system_prompt": """You are an expert clinical diagnostician with decades of experience in internal medicine and diagnostic reasoning. You have access to advanced vector-enhanced patient data analysis that provides the most clinically relevant information from comprehensive medical histories.

DIAGNOSTIC EXCELLENCE FRAMEWORK:
1. SYSTEMATIC ANALYSIS: Review vector-selected patient data systematically
2. PATTERN RECOGNITION: Identify clinical patterns from semantic context selection
3. EVIDENCE-BASED REASONING: Support conclusions with specific patient data
4. DIFFERENTIAL DIAGNOSIS: Generate ranked alternatives with clear likelihood estimates
5. RISK STRATIFICATION: Assess immediate vs long-term patient risks
6. CLINICAL INTEGRATION: Consider medications, comorbidities, and patient factors

VECTOR-ENHANCED CAPABILITIES:
- Access to semantically relevant clinical contexts from massive datasets
- Intelligent filtering of most important clinical information
- Temporal pattern recognition across comprehensive medical histories
- Integration of multiple clinical domains and specialties

FORMAT YOUR RESPONSE WITH CLEAR SECTION HEADERS:
Use structured output with clear sections for optimal clinical communication.""",
                
                "diagnostic_prompt": """
VECTOR-ENHANCED DIAGNOSTIC ANALYSIS for GPT-4.1

Patient Context (Semantically Selected):
{patient_context}

Clinical Literature (Vector-Retrieved):
{literature_context}

Using your advanced diagnostic capabilities and the vector-enhanced clinical data:

PRIMARY DIAGNOSIS
[State the single most likely diagnosis with confidence level and key supporting evidence]

DIFFERENTIAL DIAGNOSES  
[List 3-5 alternative diagnoses with likelihood percentages and supporting evidence]

CLINICAL REASONING
[Provide systematic diagnostic thought process integrating vector-selected contexts]

RISK STRATIFICATION
[Categorize risks by timeframe with specific patient risks and interventions]

RECOMMENDED DIAGNOSTIC WORKUP
[List specific tests with timing and clinical rationale]

IMMEDIATE MANAGEMENT PRIORITIES
[List urgent interventions with specific actions and monitoring]

FOLLOW-UP CARE PLAN
[Detailed follow-up recommendations with timeline and specialist referrals]

Base all conclusions on the vector-enhanced clinical data provided.
"""
            },
            
            "mistral": {
                "system_prompt": """You are an expert clinical diagnostician using advanced AI-powered medical analysis. You have access to vector-enhanced patient data that provides highly relevant clinical contexts from comprehensive medical histories.

CLINICAL ANALYSIS APPROACH:
1. Systematic review of vector-selected clinical data
2. Evidence-based diagnostic reasoning
3. Risk-stratified clinical recommendations
4. Patient-centered care planning

VECTOR ENHANCEMENT BENEFITS:
- Intelligent selection of most relevant clinical information
- Comprehensive pattern recognition across medical history
- Evidence-based literature integration
- Scalable analysis of massive medical datasets

Provide structured, actionable clinical recommendations based on the enhanced data analysis.""",
                
                "diagnostic_prompt": """
VECTOR-ENHANCED CLINICAL ANALYSIS for Mistral AI

Semantically Selected Patient Data:
{patient_context}

Relevant Clinical Evidence:
{literature_context}

Analysis Metadata:
{analysis_metadata}

Provide a comprehensive clinical analysis with the following structure:

**PRIMARY DIAGNOSIS**
[Most likely diagnosis with confidence assessment and key evidence]

**DIFFERENTIAL DIAGNOSES**
[Alternative diagnoses ranked by likelihood with supporting rationale]

**CLINICAL REASONING**
[Step-by-step diagnostic thought process using vector-enhanced data]

**RISK STRATIFICATION**
[Risk categories with specific patient risks and mitigation strategies]

**RECOMMENDED DIAGNOSTIC WORKUP**
[Specific tests and evaluations with clinical rationale]

**IMMEDIATE MANAGEMENT PRIORITIES**
[Urgent interventions and monitoring requirements]

**FOLLOW-UP CARE PLAN**
[Comprehensive follow-up recommendations with timeline]

Focus on actionable insights derived from the vector-enhanced clinical analysis.
"""
            }
        }
    
    def create_comprehensive_diagnostic_prompt(
        self,
        processed_data: Dict[str, Any],
        literature_context: Dict[str, Any],
        model_type: str = "general"
    ) -> str:
        """
        Create comprehensive diagnostic prompt optimized for vector-enhanced processing.
        """
        try:
            # Determine if this is vector-enhanced processing
            is_vector_enhanced = processed_data.get('vector_enhanced_processing', False)
            is_massive_dataset = processed_data.get('processing_metadata', {}).get('is_massive_dataset', False)
            
            # Select appropriate template
            if is_massive_dataset:
                template_key = "massive_dataset_analysis"
            elif is_vector_enhanced:
                template_key = "vector_comprehensive_analysis"
            else:
                template_key = "comprehensive_analysis"
            
            # Get base template
            if template_key in self.vector_enhanced_templates:
                base_template = self.vector_enhanced_templates[template_key]
            else:
                base_template = self.base_templates.get("comprehensive_analysis", "")
            
            # Prepare context data
            patient_summary = self._format_patient_summary(processed_data)
            clinical_timeline = self._format_clinical_timeline(processed_data)
            literature_content = self._format_literature_context(literature_context)
            
            # Format template with appropriate data
            if is_massive_dataset:
                return base_template.format(
                    estimated_data_lines=processed_data.get('processing_metadata', {}).get('estimated_data_lines', 'Unknown'),
                    context_strategy=processed_data.get('processing_metadata', {}).get('context_strategy', 'intelligent'),
                    selected_contexts=clinical_timeline,
                    priority_contexts=self._extract_priority_contexts(processed_data)
                )
            elif is_vector_enhanced:
                return base_template.format(
                    vector_patient_summary=patient_summary,
                    vector_clinical_timeline=clinical_timeline,
                    context_selection_metadata=self._format_context_metadata(processed_data),
                    vector_literature_context=literature_content
                )
            else:
                return base_template.format(
                    patient_summary=patient_summary,
                    clinical_timeline=clinical_timeline,
                    current_status=self._extract_current_status(processed_data),
                    recent_changes=self._extract_recent_changes(processed_data)
                )
            
        except Exception as e:
            logger.error(f"Error creating comprehensive diagnostic prompt: {e}")
            return self._create_fallback_prompt(processed_data, literature_context)
    
    def create_model_specific_prompt(
        self,
        model_type: str,
        processed_data: Dict[str, Any],
        literature_context: Dict[str, Any]
    ) -> str:
        """Create model-specific optimized prompt."""
        try:
            if model_type.lower() not in self.model_specific_templates:
                model_type = "gpt4"  # Default to GPT-4.1
            
            model_templates = self.model_specific_templates[model_type.lower()]
            diagnostic_template = model_templates.get("diagnostic_prompt", "")
            
            # Format with patient data
            patient_context = self._format_patient_context_for_model(processed_data, model_type)
            literature_content = self._format_literature_context(literature_context)
            analysis_metadata = self._format_analysis_metadata(processed_data)
            
            return diagnostic_template.format(
                patient_context=patient_context,
                literature_context=literature_content,
                analysis_metadata=analysis_metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating model-specific prompt for {model_type}: {e}")
            return self.create_comprehensive_diagnostic_prompt(processed_data, literature_context)
    
    def create_semantic_search_prompt(
        self,
        search_query: str,
        patient_id: str,
        search_results: List[Dict[str, Any]],
        search_metadata: Dict[str, Any]
    ) -> str:
        """Create prompt for analyzing semantic search results."""
        try:
            template = self.vector_enhanced_templates["semantic_patient_search"]
            
            formatted_results = self._format_search_results(search_results)
            
            return template.format(
                search_query=search_query,
                patient_id=patient_id,
                results_count=len(search_results),
                search_scope=", ".join(search_metadata.get("search_scope", ["patient_data"])),
                search_results=formatted_results
            )
            
        except Exception as e:
            logger.error(f"Error creating semantic search prompt: {e}")
            return f"Analyze the following search results for query '{search_query}': {search_results}"
    
    def get_system_prompt_for_model(self, model_type: str) -> str:
        """Get system prompt optimized for specific model."""
        try:
            if model_type.lower() in self.model_specific_templates:
                return self.model_specific_templates[model_type.lower()]["system_prompt"]
            else:
                # Return general system prompt
                return """You are an expert clinical diagnostician with access to vector-enhanced patient data analysis. 
                Provide comprehensive, evidence-based clinical assessments using the intelligently selected clinical contexts 
                from comprehensive medical histories. Focus on actionable clinical insights and patient-centered care recommendations."""
        except Exception as e:
            logger.error(f"Error getting system prompt for {model_type}: {e}")
            return "You are an expert clinical diagnostician. Provide comprehensive clinical analysis based on the provided patient data."
    
    # Helper methods for formatting context data
    
    def _format_patient_summary(self, processed_data: Dict[str, Any]) -> str:
        """Format patient summary for prompt inclusion."""
        try:
            patient_id = processed_data.get('patient_id', 'Unknown')
            temporal_summaries = processed_data.get('temporal_summaries', {})
            
            summary_parts = [f"Patient ID: {patient_id}"]
            
            # Add temporal summaries
            for period, summary in temporal_summaries.items():
                if summary and summary.strip():
                    summary_parts.append(f"\n{period.title()} Status:\n{summary}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error formatting patient summary: {e}")
            return f"Patient ID: {processed_data.get('patient_id', 'Unknown')}"
    
    def _format_clinical_timeline(self, processed_data: Dict[str, Any]) -> str:
        """Format clinical timeline for prompt inclusion."""
        try:
            temporal_summaries = processed_data.get('temporal_summaries', {})
            
            timeline_parts = []
            priority_order = ['acute', 'recent', 'current', 'chronic', 'historical']
            
            for period in priority_order:
                if period in temporal_summaries:
                    summary = temporal_summaries[period]
                    if summary and summary.strip():
                        timeline_parts.append(f"=== {period.upper()} PERIOD ===\n{summary}")
            
            return "\n\n".join(timeline_parts)
            
        except Exception as e:
            logger.error(f"Error formatting clinical timeline: {e}")
            return "Clinical timeline not available"
    
    def _format_literature_context(self, literature_context: Dict[str, Any]) -> str:
        """Format literature context for prompt inclusion."""
        try:
            if not literature_context:
                return "No relevant clinical literature found."
            
            literature_parts = []
            
            # Add literature content if available
            if literature_context.get('content'):
                literature_parts.append("Relevant Clinical Literature:")
                literature_parts.append(literature_context['content'])
            
            # Add sources if available
            if literature_context.get('sources'):
                literature_parts.append("\nLiterature Sources:")
                for i, source in enumerate(literature_context['sources'][:5], 1):
                    literature_parts.append(f"{i}. {source}")
            
            return "\n".join(literature_parts) if literature_parts else "No relevant clinical literature found."
            
        except Exception as e:
            logger.error(f"Error formatting literature context: {e}")
            return "Literature context formatting error"
    
    def _format_context_metadata(self, processed_data: Dict[str, Any]) -> str:
        """Format context selection metadata."""
        try:
            metadata = processed_data.get('processing_metadata', {})
            chunk_stats = processed_data.get('chunk_statistics', {})
            
            metadata_parts = []
            
            if metadata.get('context_strategy'):
                metadata_parts.append(f"Context Selection Strategy: {metadata['context_strategy']}")
            
            if metadata.get('is_massive_dataset'):
                metadata_parts.append(f"Massive Dataset Processing: {metadata.get('estimated_data_lines', 'Unknown')} lines")
            
            if chunk_stats.get('vector_contexts_used'):
                metadata_parts.append(f"Vector Contexts Used: {chunk_stats['vector_contexts_used']}")
            
            if chunk_stats.get('chunks_processed'):
                metadata_parts.append(f"Total Chunks Processed: {chunk_stats['chunks_processed']}")
            
            return "\n".join(metadata_parts) if metadata_parts else "Standard processing metadata"
            
        except Exception as e:
            logger.error(f"Error formatting context metadata: {e}")
            return "Metadata not available"
    
    def _extract_priority_contexts(self, processed_data: Dict[str, Any]) -> str:
        """Extract high-priority clinical contexts."""
        try:
            temporal_summaries = processed_data.get('temporal_summaries', {})
            
            priority_contexts = []
            
            # Prioritize acute and recent contexts
            for period in ['acute', 'recent']:
                if period in temporal_summaries:
                    summary = temporal_summaries[period]
                    if summary and summary.strip():
                        priority_contexts.append(f"HIGH PRIORITY - {period.upper()}:\n{summary}")
            
            return "\n\n".join(priority_contexts) if priority_contexts else "No high-priority contexts identified"
            
        except Exception as e:
            logger.error(f"Error extracting priority contexts: {e}")
            return "Priority context extraction error"
    
    def _extract_current_status(self, processed_data: Dict[str, Any]) -> str:
        """Extract current clinical status."""
        try:
            temporal_summaries = processed_data.get('temporal_summaries', {})
            current_summary = temporal_summaries.get('current', temporal_summaries.get('recent', ''))
            
            return current_summary if current_summary else "Current status not available"
            
        except Exception as e:
            logger.error(f"Error extracting current status: {e}")
            return "Current status extraction error"
    
    def _extract_recent_changes(self, processed_data: Dict[str, Any]) -> str:
        """Extract recent changes in patient condition."""
        try:
            temporal_summaries = processed_data.get('temporal_summaries', {})
            recent_summary = temporal_summaries.get('recent', temporal_summaries.get('acute', ''))
            
            return recent_summary if recent_summary else "No recent changes documented"
            
        except Exception as e:
            logger.error(f"Error extracting recent changes: {e}")
            return "Recent changes extraction error"
    
    def _format_patient_context_for_model(self, processed_data: Dict[str, Any], model_type: str) -> str:
        """Format patient context optimized for specific model."""
        try:
            if model_type.lower() == "gpt4":
                # GPT-4.1 can handle more detailed context
                return self._format_clinical_timeline(processed_data)
            elif model_type.lower() == "mistral":
                # Mistral prefers more structured context
                return self._format_patient_summary(processed_data)
            else:
                # General formatting
                return self._format_patient_summary(processed_data)
                
        except Exception as e:
            logger.error(f"Error formatting patient context for {model_type}: {e}")
            return self._format_patient_summary(processed_data)
    
    def _format_analysis_metadata(self, processed_data: Dict[str, Any]) -> str:
        """Format analysis metadata for model consumption."""
        try:
            metadata = processed_data.get('processing_metadata', {})
            
            metadata_parts = []
            metadata_parts.append(f"Processing Type: {metadata.get('pipeline_version', 'Unknown')}")
            metadata_parts.append(f"Vector Enhanced: {metadata.get('vector_enhanced', False)}")
            
            if metadata.get('context_strategy'):
                metadata_parts.append(f"Context Strategy: {metadata['context_strategy']}")
            
            if metadata.get('processing_duration_seconds'):
                metadata_parts.append(f"Processing Time: {metadata['processing_duration_seconds']:.2f}s")
            
            return " | ".join(metadata_parts)
            
        except Exception as e:
            logger.error(f"Error formatting analysis metadata: {e}")
            return "Analysis metadata not available"
    
    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results for prompt inclusion."""
        try:
            if not search_results:
                return "No search results found."
            
            result_parts = []
            
            for i, result in enumerate(search_results[:10], 1):  # Limit to top 10 results
                result_parts.append(f"\nResult {i}:")
                result_parts.append(f"Content: {result.get('content', 'No content')[:300]}...")
                result_parts.append(f"Source: {result.get('source_type', 'Unknown')}")
                result_parts.append(f"Relevance Score: {result.get('relevance_score', 0.0):.2f}")
                
                if result.get('metadata'):
                    metadata = result['metadata']
                    if metadata.get('clinical_domain'):
                        result_parts.append(f"Clinical Domain: {metadata['clinical_domain']}")
                    if metadata.get('chunk_type'):
                        result_parts.append(f"Data Type: {metadata['chunk_type']}")
            
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"Error formatting search results: {e}")
            return "Search results formatting error"
    
    def _create_fallback_prompt(self, processed_data: Dict[str, Any], literature_context: Dict[str, Any]) -> str:
        """Create fallback prompt when main template creation fails."""
        try:
            patient_id = processed_data.get('patient_id', 'Unknown')
            master_summary = processed_data.get('master_summary', 'Summary not available')
            
            return f"""
CLINICAL ANALYSIS REQUEST

Patient ID: {patient_id}

Clinical Summary:
{master_summary}

Literature Context:
{literature_context.get('content', 'No literature available')}

Please provide a comprehensive clinical analysis including:
1. Primary diagnostic considerations
2. Clinical reasoning and evidence
3. Risk assessment and management priorities
4. Recommended follow-up and monitoring

Base your analysis on the available clinical information.
"""
            
        except Exception as e:
            logger.error(f"Error creating fallback prompt: {e}")
            return "Please provide a clinical analysis based on the available patient data."