"""
Clinical prompt templates for the diagnostic system.
UPDATED VERSION - Focus on diagnostic reasoning from patient history analysis.
Handles comprehensive patient data and generates structured diagnostic assessments.
"""

from typing import Dict, List, Optional, Any
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ClinicalPromptTemplate:
    """
    Updated clinical prompt template manager for diagnostic reasoning.
    Focuses on comprehensive patient history analysis and structured diagnostic output.
    """
    
    def __init__(self):
        self.diagnostic_system_prompt = self._create_diagnostic_system_prompt()
        self.prompt_template = self._create_diagnostic_prompt_template()
    
    def _create_diagnostic_system_prompt(self) -> str:
        """Create the system prompt optimized for diagnostic reasoning."""
        return """You are an expert clinical diagnostician with decades of experience in internal medicine, emergency medicine, and differential diagnosis. Your role is to analyze comprehensive patient data and provide accurate, evidence-based diagnostic assessments.

CORE DIAGNOSTIC PRINCIPLES:
1. SYSTEMATIC ANALYSIS: Review patient data chronologically and by organ system
2. PATTERN RECOGNITION: Identify clinical patterns, trends, and correlations across time
3. EVIDENCE-BASED REASONING: Support all conclusions with specific patient data
4. DIFFERENTIAL DIAGNOSIS: Generate comprehensive differential diagnoses ranked by likelihood
5. RISK STRATIFICATION: Assess immediate vs long-term risks and complications
6. CLINICAL INTEGRATION: Consider medications, comorbidities, and patient factors holistically

DIAGNOSTIC METHODOLOGY:
- Analyze symptoms, vital signs, laboratory results, imaging, and clinical course together
- Consider epidemiology, patient demographics, and individual risk factors
- Identify red flags and conditions requiring immediate attention
- Account for medication effects, drug interactions, and iatrogenic causes
- Balance common diagnoses with important rare conditions that must not be missed
- Use temporal relationships and disease progression patterns

RESPONSE STRUCTURE REQUIREMENTS:
Your diagnostic assessment must be structured as follows:

1. **PRIMARY DIAGNOSIS**
   - State the most likely diagnosis based on available evidence
   - Provide confidence level (High 80-95% / Medium 60-80% / Low 40-60%)
   - List 3-5 key supporting evidence points from patient data

2. **DIFFERENTIAL DIAGNOSES**
   - List 3-5 alternative diagnoses in order of likelihood
   - For each: provide likelihood estimate and 2-3 supporting/contradicting factors
   - Include any "must not miss" diagnoses even if less likely

3. **CLINICAL REASONING**
   - Explain your diagnostic thought process step-by-step
   - Describe how you weighted different pieces of evidence
   - Address any conflicting or unusual findings

4. **RISK STRATIFICATION**
   - Immediate risks requiring urgent attention (next 24-48 hours)
   - Short-term risks (next weeks to months)
   - Long-term complications and monitoring needs

5. **RECOMMENDED DIAGNOSTIC WORKUP**
   - Laboratory tests needed to confirm/exclude diagnoses
   - Imaging studies indicated with specific modalities
   - Specialized testing or procedures required
   - Priority and timing of each test

6. **IMMEDIATE MANAGEMENT PRIORITIES**
   - Urgent interventions needed now
   - Monitoring requirements and parameters
   - Medication adjustments or new prescriptions
   - When to escalate care or seek specialist consultation

7. **FOLLOW-UP RECOMMENDATIONS**
   - Timeline for reassessment and next appointment
   - Specialist referrals with urgency level
   - Patient education priorities and warning signs
   - Long-term monitoring and prevention strategies

CRITICAL SAFETY REQUIREMENTS:
- Always consider life-threatening conditions first
- Account for patient's age, comorbidities, and medications in all recommendations
- Be explicit about uncertainty and when urgent evaluation is needed
- Never dismiss symptoms that could represent serious pathology
- Consider drug interactions and contraindications
- Base all recommendations on the specific patient data provided

FORMAT REQUIREMENTS:
- Use clear section headers as shown above
- Be specific and actionable in all recommendations
- Reference specific dates, test results, and clinical events from patient history
- Provide reasoning for high-risk or unusual recommendations
- Keep diagnostic assessment comprehensive but focused"""

    def _create_diagnostic_prompt_template(self) -> PromptTemplate:
        """Create the main diagnostic prompt template for comprehensive patient analysis."""
        template = """You are conducting a comprehensive diagnostic analysis of a complex patient case. Analyze all available patient data systematically and provide a structured diagnostic assessment.

PATIENT CLINICAL DATA AND MEDICAL LITERATURE:
{context}

COMPREHENSIVE PATIENT CASE FOR DIAGNOSTIC ANALYSIS:
{question}

DIAGNOSTIC ANALYSIS:

Based on the comprehensive patient data above and relevant medical literature, provide a complete diagnostic assessment following the structured format specified in your system instructions.

Focus on:
- Analyzing the complete patient timeline and identifying key diagnostic clues
- Integrating all available data (demographics, history, medications, visits, patterns)
- Considering the chronological progression of the patient's condition
- Identifying clinical patterns and their diagnostic significance
- Providing evidence-based diagnostic reasoning

IMPORTANT: Reference specific patient data points (dates, test results, medications, symptoms, visit notes) to support all diagnostic conclusions.

STRUCTURED DIAGNOSTIC ASSESSMENT:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def get_prompt(self) -> PromptTemplate:
        """Get the diagnostic prompt template."""
        return self.prompt_template
    
    def create_comprehensive_diagnostic_prompt(
        self, 
        processed_patient_data: Dict, 
        literature_context: Dict
    ) -> str:
        """
        Create a comprehensive diagnostic prompt from processed patient data.
        This is the main method used by the new diagnostic pipeline.
        """
        try:
            patient_summary = processed_patient_data.get("patient_summary", {})
            chronological_history = processed_patient_data.get("chronological_history", [])
            clinical_patterns = processed_patient_data.get("clinical_patterns", {})
            risk_profile = processed_patient_data.get("risk_profile", {})
            recent_changes = processed_patient_data.get("recent_changes", {})
            data_stats = processed_patient_data.get("data_statistics", {})
            
            # Build the comprehensive diagnostic prompt
            prompt_sections = [
                "=== COMPREHENSIVE PATIENT CASE FOR DIAGNOSTIC ANALYSIS ===",
                "",
                f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Patient ID: {patient_summary.get('patient_id', 'Unknown')}",
                "",
                "=== PATIENT DEMOGRAPHIC AND SUMMARY DATA ===",
                f"Demographics: {self._format_demographics(patient_summary.get('demographics', {}))}",
                f"Data Completeness: {data_stats.get('data_completeness', 0.0):.1%}",
                f"Total Medical History Span: {data_stats.get('total_diagnoses', 0)} diagnoses, {data_stats.get('total_medications', 0)} medications, {data_stats.get('total_visits', 0)} visits",
                "",
                f"Current Active Conditions ({patient_summary.get('active_conditions_count', 0)}):",
                self._format_list_with_bullets(patient_summary.get('primary_conditions', [])),
                "",
                f"Current Active Medications ({patient_summary.get('active_medications_count', 0)}):",
                self._format_list_with_bullets(patient_summary.get('key_medications', [])),
                "",
                "=== CHRONOLOGICAL MEDICAL HISTORY (Most Recent First) ===",
                "This timeline shows the patient's medical journey with dates and clinical events:",
                ""
            ]
            
            # Add chronological history with better formatting
            for i, event in enumerate(chronological_history[:25], 1):  # Limit to 25 most recent events
                event_date = event.get('date', 'Unknown date')
                event_type = event.get('type', 'unknown').replace('_', ' ').title()
                event_desc = event.get('description', 'No description')
                
                # Add additional context for visits
                if event.get('type') == 'visit':
                    provider = event.get('provider', '')
                    notes = event.get('notes', '')
                    if provider:
                        event_desc += f" [Provider: {provider}]"
                    if notes:
                        event_desc += f" [Notes: {notes[:150]}{'...' if len(notes) > 150 else ''}]"
                
                prompt_sections.append(f"{i:2d}. {event_date}: {event_type} - {event_desc}")
            
            prompt_sections.extend([
                "",
                "=== CLINICAL PATTERNS AND ANALYTICS ===",
                "",
                "RECURRING SYMPTOMS AND THEMES:",
                f"Most frequent symptoms/concerns: {', '.join(clinical_patterns.get('recurring_symptoms', [])[:8])}",
                "",
                "VISIT PATTERN ANALYSIS:",
                self._format_visit_patterns(clinical_patterns.get('visit_frequency', {})),
                "",
                "MEDICATION THERAPEUTIC CLASSES:",
                self._format_medication_patterns(clinical_patterns.get('medication_patterns', {})),
                "",
                "=== COMPREHENSIVE RISK ASSESSMENT ===",
                "",
                f"Age-Related Risk Level: {risk_profile.get('age_risk', 'Unknown').title()}",
                f"Overall Clinical Complexity: {risk_profile.get('overall_complexity', 'Unknown').title()}",
                "",
                "HIGH-RISK CONDITIONS IDENTIFIED:",
            ])
            
            # Add condition risks
            condition_risks = risk_profile.get('condition_risks', [])
            if condition_risks:
                for risk in condition_risks[:5]:
                    prompt_sections.append(f"  • {risk.get('condition', 'Unknown')}: {risk.get('risk_level', 'Unknown')} risk")
            else:
                prompt_sections.append("  • No specific high-risk conditions flagged")
            
            prompt_sections.append("")
            prompt_sections.append("HIGH-RISK MEDICATIONS IDENTIFIED:")
            
            # Add medication risks
            med_risks = risk_profile.get('medication_risks', [])
            if med_risks:
                for risk in med_risks[:5]:
                    prompt_sections.append(f"  • {risk.get('medication', 'Unknown')}: {risk.get('risk_type', 'Unknown')} risk")
            else:
                prompt_sections.append("  • No specific high-risk medications flagged")
            
            prompt_sections.extend([
                "",
                "=== RECENT CLINICAL CHANGES (Last 90 Days) ===",
                "",
                f"New Medications Started: {len(recent_changes.get('new_medications', []))}",
                f"Medications Discontinued: {len(recent_changes.get('stopped_medications', []))}",
                f"New Diagnoses Added: {len(recent_changes.get('new_diagnoses', []))}",
                f"Recent Healthcare Visits: {len(recent_changes.get('recent_visits', []))}",
                ""
            ])
            
            # Add details of recent changes if significant
            if recent_changes.get('new_diagnoses'):
                prompt_sections.append("RECENT NEW DIAGNOSES:")
                for dx in recent_changes['new_diagnoses'][:5]:
                    prompt_sections.append(f"  • {dx.get('description', 'Unknown')} (diagnosed: {dx.get('date', 'Unknown date')})")
                prompt_sections.append("")
            
            if recent_changes.get('new_medications'):
                prompt_sections.append("RECENTLY STARTED MEDICATIONS:")
                for med in recent_changes['new_medications'][:5]:
                    prompt_sections.append(f"  • {med.get('name', 'Unknown')} {med.get('dosage', '')} (started: {med.get('start_date', 'Unknown')})")
                prompt_sections.append("")
            
            if recent_changes.get('recent_visits'):
                prompt_sections.append("RECENT HEALTHCARE ENCOUNTERS:")
                for visit in recent_changes['recent_visits'][:3]:
                    visit_desc = f"  • {visit.get('date', 'Unknown')}: {visit.get('type', 'Visit').title()}"
                    if visit.get('complaint'):
                        visit_desc += f" - {visit['complaint']}"
                    if visit.get('provider'):
                        visit_desc += f" (with {visit['provider']})"
                    prompt_sections.append(visit_desc)
                prompt_sections.append("")
            
            # Add medical literature context if available
            if literature_context.get('content'):
                prompt_sections.extend([
                    "=== RELEVANT MEDICAL LITERATURE AND GUIDELINES ===",
                    "",
                    "Based on the patient's conditions and symptoms, here is relevant medical knowledge:",
                    "",
                    literature_context['content'][:2500],  # Limit to 2500 characters
                    "",
                    f"Literature Sources: {', '.join(literature_context.get('sources', [])[:5])}",
                    ""
                ])
            
            prompt_sections.extend([
                "=== DIAGNOSTIC ANALYSIS REQUEST ===",
                "",
                "Based on this comprehensive patient case data, provide a complete diagnostic assessment that:",
                "",
                "1. Analyzes the patient's clinical timeline and identifies key diagnostic patterns",
                "2. Considers the chronological progression and recent changes in the patient's condition", 
                "3. Integrates all available clinical data including demographics, medications, and visit history",
                "4. Accounts for the patient's risk factors, comorbidities, and clinical complexity",
                "5. References specific dates, events, and data points from the patient's history",
                "",
                "Provide your analysis in the structured format specified in your system instructions:",
                "- Primary Diagnosis (with confidence level and supporting evidence)",
                "- Differential Diagnoses (ranked with likelihood and reasoning)",
                "- Clinical Reasoning (step-by-step diagnostic thought process)",
                "- Risk Stratification (immediate, short-term, and long-term risks)",
                "- Recommended Diagnostic Workup (specific tests with priority/timing)",
                "- Immediate Management Priorities (urgent interventions and monitoring)",
                "- Follow-up Recommendations (timeline, referrals, patient education)",
                "",
                "CRITICAL: Base all diagnostic conclusions on the specific patient data provided above.",
                "Reference specific clinical events, dates, test results, and patterns from this patient's unique medical history."
            ])
            
            diagnostic_prompt = "\n".join(prompt_sections)
            
            logger.info(f"Comprehensive diagnostic prompt created:")
            logger.info(f"  - Total length: {len(diagnostic_prompt)} characters")
            logger.info(f"  - Chronological events: {len(chronological_history)}")
            logger.info(f"  - Clinical patterns: {len(clinical_patterns)}")
            logger.info(f"  - Literature sources: {len(literature_context.get('sources', []))}")
            
            return diagnostic_prompt
            
        except Exception as e:
            logger.error(f"Error creating comprehensive diagnostic prompt: {e}")
            return f"Error creating diagnostic prompt: {str(e)}"
    
    def _format_demographics(self, demographics: Dict) -> str:
        """Format demographics for clinical presentation."""
        if not demographics:
            return "Demographics not available"
        
        parts = []
        if demographics.get('age'):
            parts.append(f"{demographics['age']} years old")
        if demographics.get('gender'):
            parts.append(demographics['gender'])
        if demographics.get('name'):
            parts.append(f"({demographics['name']})")
        
        return ", ".join(parts) if parts else "Limited demographic information"
    
    def _format_list_with_bullets(self, items: List[str]) -> str:
        """Format a list with bullet points."""
        if not items:
            return "  • None documented"
        
        return "\n".join([f"  • {item}" for item in items[:10]])  # Limit to 10 items
    
    def _format_visit_patterns(self, visit_frequency: Dict) -> str:
        """Format visit frequency patterns."""
        if not visit_frequency:
            return "  • No clear visit patterns identified"
        
        patterns = []
        for visit_type, count in sorted(visit_frequency.items(), key=lambda x: x[1], reverse=True):
            patterns.append(f"  • {visit_type.title()}: {count} visits")
        
        return "\n".join(patterns[:5])  # Top 5 visit types
    
    def _format_medication_patterns(self, med_patterns: Dict) -> str:
        """Format medication class patterns."""
        if not med_patterns:
            return "  • No specific medication patterns identified"
        
        patterns = []
        for med_class, count in sorted(med_patterns.items(), key=lambda x: x[1], reverse=True):
            patterns.append(f"  • {med_class.title()}: {count} medications")
        
        return "\n".join(patterns[:5])  # Top 5 medication classes
    
    # Legacy methods for backward compatibility
    def format_with_patient_context(self, query: str, patient_data: Dict) -> str:
        """Legacy method - redirects to new diagnostic prompt creation."""
        logger.warning("Legacy format_with_patient_context called - consider using create_comprehensive_diagnostic_prompt")
        
        # Create a basic formatted query for backward compatibility
        demographics = patient_data.get("demographics", {})
        diagnoses = patient_data.get("diagnoses", [])
        medications = patient_data.get("medications", [])
        
        formatted_query = f"""
=== PATIENT CLINICAL PROFILE ===
Demographics: {self._format_demographics(demographics)}
Active Conditions: {', '.join([d['description'] for d in diagnoses if d.get('status') == 'active'][:5])}
Current Medications: {', '.join([m['name'] for m in medications if m.get('status') == 'active'][:5])}

=== CLINICAL CONSULTATION REQUEST ===
{query}
"""
        
        return formatted_query.strip()
    
    def create_enhanced_query(self, patient_data: Dict, visit_type: str, symptoms: str, additional_context: Optional[str] = None) -> str:
        """Legacy method - redirects to comprehensive diagnostic prompt."""
        logger.warning("Legacy create_enhanced_query called - consider using create_comprehensive_diagnostic_prompt")
        
        # For backward compatibility, create a simplified query
        query_parts = [
            f"Patient Visit Type: {visit_type}",
            f"Reported Symptoms: {symptoms}"
        ]
        
        if additional_context:
            query_parts.append(f"Additional Context: {additional_context}")
        
        query_parts.append("Please provide clinical recommendations based on patient history.")
        
        query = "\n".join(query_parts)
        return self.format_with_patient_context(query, patient_data)


def create_clinical_prompt_template() -> ClinicalPromptTemplate:
    """Factory function to create the updated clinical prompt template."""
    return ClinicalPromptTemplate()


# Utility functions for diagnostic prompt enhancement
def extract_diagnostic_keywords(patient_data: Dict) -> List[str]:
    """Extract key diagnostic terms from patient data."""
    keywords = []
    
    # Extract from diagnoses
    diagnoses = patient_data.get("diagnoses", [])
    for dx in diagnoses:
        if dx.get("status") == "active":
            keywords.append(dx["description"].lower())
    
    # Extract from medications (implies conditions)
    medications = patient_data.get("medications", [])
    medication_to_condition = {
        "insulin": "diabetes",
        "metformin": "diabetes", 
        "lisinopril": "hypertension",
        "atorvastatin": "dyslipidemia",
        "warfarin": "anticoagulation",
        "albuterol": "asthma"
    }
    
    for med in medications:
        if med.get("status") == "active":
            med_name = med["name"].lower()
            for med_keyword, condition in medication_to_condition.items():
                if med_keyword in med_name:
                    keywords.append(condition)
    
    # Extract from recent visit complaints
    visits = patient_data.get("visit_history", [])
    for visit in visits[:5]:  # Recent visits
        complaint = visit.get("chief_complaint", "")
        if complaint:
            keywords.extend(complaint.lower().split()[:3])  # First 3 words
    
    return list(set(keywords))  # Remove duplicates


def validate_diagnostic_prompt(prompt: str) -> Dict[str, Any]:
    """Validate that a diagnostic prompt contains necessary components."""
    validation_result = {
        "valid": True,
        "warnings": [],
        "stats": {
            "total_length": len(prompt),
            "estimated_tokens": len(prompt) // 4
        }
    }
    
    # Check for essential sections
    required_sections = [
        "patient",
        "diagnosis", 
        "medication",
        "history",
        "clinical"
    ]
    
    prompt_lower = prompt.lower()
    
    for section in required_sections:
        if section not in prompt_lower:
            validation_result["warnings"].append(f"Missing '{section}' content")
    
    # Check prompt length
    if len(prompt) < 500:
        validation_result["warnings"].append("Prompt may be too short for comprehensive analysis")
    elif len(prompt) > 8000:
        validation_result["warnings"].append("Prompt may be too long - consider optimization")
    
    # Set validity based on warnings
    if len(validation_result["warnings"]) > 3:
        validation_result["valid"] = False
    
    return validation_result