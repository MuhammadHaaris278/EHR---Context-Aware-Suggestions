"""
Clinical prompt templates for the recommendation system.
Handles prompt construction and formatting for medical AI.
"""

from typing import Dict, List, Optional
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)

class ClinicalPromptTemplate:
    """
    Clinical prompt template manager.
    Constructs context-aware prompts for medical recommendations.
    """
    
    def __init__(self):
        self.system_prompt = self._create_system_prompt()
        self.prompt_template = self._create_prompt_template()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for clinical recommendations."""
        return """You are a clinical decision support AI assistant. Your role is to provide evidence-based medical recommendations based on patient data and clinical guidelines.

IMPORTANT GUIDELINES:
1. Always base recommendations on established medical guidelines and best practices
2. Consider patient history, current medications, and contraindications
3. Provide specific, actionable recommendations
4. Include relevant diagnostic steps, monitoring requirements, and follow-up plans
5. Consider urgency level based on symptoms and visit type
6. Format recommendations as numbered, clear action items
7. Do not provide definitive diagnoses - focus on differential diagnosis and next steps
8. Always recommend appropriate specialist consultations when indicated

RESPONSE FORMAT:
Provide 3-6 specific, actionable recommendations numbered clearly. Each recommendation should be:
- Specific and actionable
- Medically appropriate for the patient context
- Evidence-based
- Consider patient safety and contraindications"""

    def _create_prompt_template(self) -> PromptTemplate:
        """Create the main prompt template compatible with RetrievalQA."""
        # This template is designed to work with RetrievalQA's context injection
        template = """You are a clinical decision support AI assistant. Your role is to provide evidence-based medical recommendations based on patient data and clinical guidelines.

IMPORTANT GUIDELINES:
1. Always base recommendations on established medical guidelines and best practices
2. Consider patient history, current medications, and contraindications
3. Provide specific, actionable recommendations
4. Include relevant diagnostic steps, monitoring requirements, and follow-up plans
5. Consider urgency level based on symptoms and visit type
6. Format recommendations as numbered, clear action items
7. Do not provide definitive diagnoses - focus on differential diagnosis and next steps
8. Always recommend appropriate specialist consultations when indicated

RESPONSE FORMAT:
Provide 3-6 specific, actionable recommendations numbered clearly. Each recommendation should be:
- Specific and actionable
- Medically appropriate for the patient context
- Evidence-based
- Consider patient safety and contraindications

Context: {context}

Question: {question}

Based on the above context and question, provide specific clinical recommendations:

RECOMMENDATIONS:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def get_prompt(self) -> PromptTemplate:
        """Get the configured prompt template."""
        return self.prompt_template
    
    def format_with_patient_context(self, query: str, patient_data: Dict) -> str:
        """Format query with patient context for retrieval."""
        patient_context = self._format_patient_context(patient_data)
        
        # Combine query with patient context for better retrieval
        formatted_query = f"""
Patient Information:
{patient_context}

Clinical Query: {query}

Find relevant medical guidelines and recommendations for this patient's current presentation.
"""
        
        return formatted_query.strip()
    
    def _format_patient_context(self, patient_data: Dict) -> str:
        """Format patient data into readable context."""
        try:
            context_parts = []
            
            # Demographics
            demographics = patient_data.get("demographics", {})
            if demographics:
                demo_text = f"Patient: {demographics.get('age', 'Unknown')} year old {demographics.get('gender', 'patient')}"
                context_parts.append(demo_text)
            
            # Active diagnoses
            diagnoses = patient_data.get("diagnoses", [])
            if diagnoses:
                active_dx = [d["description"] for d in diagnoses if d.get("status") == "active"]
                if active_dx:
                    context_parts.append(f"Active Diagnoses: {', '.join(active_dx[:5])}")
            
            # Current medications
            medications = patient_data.get("medications", [])
            if medications:
                active_meds = [f"{m['name']} {m.get('dosage', '')}" for m in medications if m.get("status") == "active"]
                if active_meds:
                    context_parts.append(f"Current Medications: {', '.join(active_meds[:8])}")
            
            # Recent visit history
            visits = patient_data.get("visit_history", [])
            if visits:
                recent_visits = []
                for visit in visits[:3]:  # Last 3 visits
                    visit_text = f"{visit.get('type', 'Visit')} - {visit.get('chief_complaint', 'No complaint recorded')}"
                    recent_visits.append(visit_text)
                
                if recent_visits:
                    context_parts.append(f"Recent Visits: {'; '.join(recent_visits)}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error formatting patient context: {e}")
            return f"Patient ID: {patient_data.get('patient_id', 'Unknown')}"
    
    def create_enhanced_query(self, patient_data: Dict, visit_type: str, symptoms: str, additional_context: Optional[str] = None) -> str:
        """Create enhanced query with patient context for better retrieval."""
        try:
            patient_context = self._format_patient_context(patient_data)
            
            query_parts = [
                f"PATIENT CONTEXT:\n{patient_context}",
                f"VISIT TYPE: {visit_type}",
                f"PRESENTING SYMPTOMS: {symptoms}"
            ]
            
            if additional_context:
                query_parts.append(f"ADDITIONAL CONTEXT: {additional_context}")
            
            query_parts.append("CLINICAL QUESTION: What are the appropriate next steps and recommendations for this patient?")
            
            return "\n\n".join(query_parts)
            
        except Exception as e:
            logger.error(f"Error creating enhanced query: {e}")
            return f"Patient symptoms: {symptoms}, Visit type: {visit_type}"
    
    def create_emergency_prompt(self, patient_data: Dict, symptoms: str) -> str:
        """Create specialized prompt for emergency situations."""
        patient_context = self._format_patient_context(patient_data)
        
        emergency_template = f"""EMERGENCY SITUATION - PRIORITIZE IMMEDIATE INTERVENTIONS

PATIENT CONTEXT:
{patient_context}

CHIEF COMPLAINT/SYMPTOMS:
{symptoms}

PRIORITY ACTIONS NEEDED:
Focus on immediate assessment, stabilization, and urgent interventions. Consider:
1. Immediate vital signs and monitoring
2. Emergency diagnostic tests
3. Immediate treatments or interventions
4. Specialist consultations if needed
5. Disposition planning

Provide urgent, prioritized recommendations for this emergency presentation."""
        
        return emergency_template
    
    def create_followup_prompt(self, patient_data: Dict, symptoms: str) -> str:
        """Create specialized prompt for follow-up visits."""
        patient_context = self._format_patient_context(patient_data)
        
        followup_template = f"""FOLLOW-UP VISIT CONTEXT

PATIENT CONTEXT:
{patient_context}

CURRENT PRESENTATION:
{symptoms}

FOLLOW-UP CONSIDERATIONS:
Focus on monitoring disease progression, medication effectiveness, and preventive care:
1. Review of current treatment effectiveness
2. Monitoring for complications or side effects
3. Preventive care recommendations
4. Lifestyle modifications
5. Next follow-up planning

Provide comprehensive follow-up recommendations for this patient."""
        
        return followup_template
    
    def create_routine_prompt(self, patient_data: Dict, symptoms: str) -> str:
        """Create specialized prompt for routine visits."""
        patient_context = self._format_patient_context(patient_data)
        
        routine_template = f"""ROUTINE VISIT CONTEXT

PATIENT CONTEXT:
{patient_context}

CURRENT CONCERNS:
{symptoms}

ROUTINE CARE CONSIDERATIONS:
Focus on preventive care, health maintenance, and comprehensive assessment:
1. Preventive screening recommendations
2. Health maintenance activities
3. Risk factor modification
4. Routine monitoring for chronic conditions
5. Patient education and counseling

Provide comprehensive routine care recommendations for this patient."""
        
        return routine_template
    
    def format_recommendations(self, recommendations: List[str]) -> str:
        """Format recommendations into structured output."""
        if not recommendations:
            return "No specific recommendations generated."
        
        formatted = []
        for i, rec in enumerate(recommendations, 1):
            # Clean up the recommendation
            clean_rec = rec.strip()
            if not clean_rec:
                continue
            
            # Ensure it starts with a number if not already
            if not clean_rec[0].isdigit():
                clean_rec = f"{i}. {clean_rec}"
            
            formatted.append(clean_rec)
        
        return "\n".join(formatted)
    
    def validate_recommendation(self, recommendation: str) -> bool:
        """Validate that a recommendation is appropriate and safe."""
        # Basic validation checks
        if not recommendation or len(recommendation.strip()) < 10:
            return False
        
        # Check for inappropriate content
        dangerous_keywords = [
            "discontinue all medications",
            "ignore symptoms",
            "no treatment needed",
            "definitely not",
            "100% certain"
        ]
        
        recommendation_lower = recommendation.lower()
        for keyword in dangerous_keywords:
            if keyword in recommendation_lower:
                logger.warning(f"Potentially dangerous recommendation detected: {recommendation}")
                return False
        
        return True
    
    def get_visit_type_prompt(self, visit_type: str, patient_data: Dict, symptoms: str) -> str:
        """Get appropriate prompt based on visit type."""
        visit_type_lower = visit_type.lower()
        
        if "emergency" in visit_type_lower or "urgent" in visit_type_lower:
            return self.create_emergency_prompt(patient_data, symptoms)
        elif "follow" in visit_type_lower:
            return self.create_followup_prompt(patient_data, symptoms)
        elif "routine" in visit_type_lower:
            return self.create_routine_prompt(patient_data, symptoms)
        else:
            # Default to enhanced query
            return self.create_enhanced_query(patient_data, visit_type, symptoms)

def create_clinical_prompt_template() -> ClinicalPromptTemplate:
    """Factory function to create clinical prompt template."""
    return ClinicalPromptTemplate()