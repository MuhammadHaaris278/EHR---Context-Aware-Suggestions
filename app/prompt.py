"""
Clinical prompt templates for the recommendation system.
Handles prompt construction and formatting for medical AI.
ENHANCED VERSION - Better patient context integration.
"""

from typing import Dict, List, Optional
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import logging
import json

logger = logging.getLogger(__name__)

class ClinicalPromptTemplate:
    """
    Enhanced clinical prompt template manager.
    Constructs context-aware prompts for medical recommendations with improved patient integration.
    """
    
    def __init__(self):
        self.system_prompt = self._create_system_prompt()
        self.prompt_template = self._create_prompt_template()
    
    def _create_system_prompt(self) -> str:
        """Create the enhanced system prompt for clinical recommendations."""
        return """You are an advanced clinical decision support AI assistant designed to provide evidence-based medical recommendations. Your primary role is to analyze patient data comprehensively and provide safe, appropriate clinical guidance.

CORE RESPONSIBILITIES:
1. Analyze complete patient clinical profile (demographics, medical history, current medications, risk factors)
2. Provide evidence-based recommendations following established medical guidelines
3. Consider patient-specific contraindications and drug interactions
4. Prioritize patient safety in all recommendations
5. Suggest appropriate diagnostic workup and monitoring
6. Recommend specialist referrals when indicated

CRITICAL SAFETY PROTOCOLS:
- Always consider patient allergies and contraindications
- Check for potential drug-drug interactions
- Account for age-specific considerations
- Consider comorbidities and their impact on treatment
- Never recommend discontinuing life-saving medications without clear alternatives
- Always suggest appropriate monitoring and follow-up

RESPONSE REQUIREMENTS:
- Provide 4-6 specific, actionable clinical recommendations
- Number each recommendation clearly (1., 2., 3., etc.)
- Be specific about dosages, timing, and monitoring when appropriate
- Include rationale for high-risk recommendations
- Consider both immediate and long-term patient needs
- Format for easy clinical implementation

CLINICAL FOCUS AREAS:
- Immediate diagnostic assessment
- Treatment optimization
- Safety monitoring and follow-up
- Preventive care opportunities
- Patient education priorities
- Care coordination needs"""

    def _create_prompt_template(self) -> PromptTemplate:
        """Create the enhanced main prompt template compatible with RetrievalQA."""
        template = """You are an advanced clinical decision support AI assistant. Analyze the patient's complete clinical profile and provide evidence-based recommendations.

CLINICAL DECISION SUPPORT GUIDELINES:
- Base all recommendations on established medical guidelines and evidence
- Consider the patient's complete medical history and current medications
- Account for contraindications, allergies, and drug interactions
- Provide specific, actionable recommendations with clear rationale
- Prioritize patient safety and appropriate monitoring
- Include diagnostic workup, treatment options, and follow-up planning

MEDICAL LITERATURE CONTEXT:
{context}

PATIENT CLINICAL QUERY:
{question}

CLINICAL ANALYSIS AND RECOMMENDATIONS:

Based on the medical literature above and the patient's clinical presentation, provide comprehensive clinical recommendations addressing:

1. IMMEDIATE ASSESSMENT: What diagnostic steps are most appropriate?
2. TREATMENT CONSIDERATIONS: What therapeutic interventions should be considered?
3. SAFETY & MONITORING: What safety measures and monitoring are required?
4. SPECIALIST REFERRALS: Are there indications for specialist consultation?
5. FOLLOW-UP PLANNING: What follow-up care is needed?
6. PATIENT EDUCATION: What key information should be provided to the patient?

Format your response as numbered recommendations (1., 2., 3., etc.) that are specific, actionable, and clinically appropriate for this patient's unique situation.

RECOMMENDATIONS:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def get_prompt(self) -> PromptTemplate:
        """Get the configured prompt template."""
        return self.prompt_template
    
    def format_with_patient_context(self, query: str, patient_data: Dict) -> str:
        """Format query with enhanced patient context for better retrieval."""
        try:
            patient_context = self._format_comprehensive_patient_context(patient_data)
            
            # Combine query with patient context for optimal retrieval
            formatted_query = f"""
=== PATIENT CLINICAL PROFILE ===
{patient_context}

=== CLINICAL CONSULTATION REQUEST ===
{query}

=== INFORMATION NEEDED ===
Please find relevant medical guidelines, evidence-based recommendations, and clinical protocols that address this patient's specific clinical situation, taking into account their medical history, current medications, and risk factors.
"""
            
            logger.info(f"Patient context formatted for retrieval: {len(patient_context)} characters")
            return formatted_query.strip()
            
        except Exception as e:
            logger.error(f"Error formatting patient context: {e}")
            return query
    
    def _format_comprehensive_patient_context(self, patient_data: Dict) -> str:
        """Format patient data into comprehensive clinical context."""
        try:
            context_sections = []
            
            # SECTION 1: Demographics and Basic Information
            demographics = patient_data.get("demographics", {})
            if demographics:
                demo_parts = []
                age = demographics.get('age', 'Unknown age')
                gender = demographics.get('gender', 'Unknown gender')
                demo_parts.append(f"Age: {age}, Gender: {gender}")
                
                if demographics.get('race'):
                    demo_parts.append(f"Race: {demographics['race']}")
                if demographics.get('ethnicity'):
                    demo_parts.append(f"Ethnicity: {demographics['ethnicity']}")
                if demographics.get('insurance'):
                    demo_parts.append(f"Insurance: {demographics['insurance']}")
                
                context_sections.append(f"DEMOGRAPHICS: {' | '.join(demo_parts)}")
            
            # SECTION 2: Active Medical Conditions
            diagnoses = patient_data.get("diagnoses", [])
            if diagnoses:
                active_conditions = []
                for dx in diagnoses:
                    if dx.get("status") == "active":
                        condition = dx["description"]
                        if dx.get("date_diagnosed"):
                            condition += f" (dx: {dx['date_diagnosed']})"
                        if dx.get("severity"):
                            condition += f" [{dx['severity']}]"
                        active_conditions.append(condition)
                
                if active_conditions:
                    context_sections.append(f"ACTIVE CONDITIONS: {' | '.join(active_conditions[:8])}")
            
            # SECTION 3: Current Medications (Enhanced)
            medications = patient_data.get("medications", [])
            if medications:
                active_meds = []
                for med in medications:
                    if med.get("status") == "active":
                        med_info = med["name"]
                        if med.get("dosage"):
                            med_info += f" {med['dosage']}"
                        if med.get("frequency"):
                            med_info += f" {med['frequency']}"
                        if med.get("route"):
                            med_info += f" ({med['route']})"
                        active_meds.append(med_info)
                
                if active_meds:
                    context_sections.append(f"CURRENT MEDICATIONS: {' | '.join(active_meds[:12])}")
            
            # SECTION 4: Allergies and Contraindications
            allergies = patient_data.get("allergies", [])
            if allergies:
                allergy_list = []
                for allergy in allergies:
                    allergy_info = allergy.get("allergen", "Unknown allergen")
                    if allergy.get("reaction"):
                        allergy_info += f" (causes {allergy['reaction']})"
                    if allergy.get("severity"):
                        allergy_info += f" [{allergy['severity']}]"
                    allergy_list.append(allergy_info)
                
                context_sections.append(f"ALLERGIES: {' | '.join(allergy_list)}")
            
            # SECTION 5: Recent Clinical History
            visits = patient_data.get("visit_history", [])
            if visits:
                recent_visits = []
                for visit in visits[:5]:  # Last 5 visits
                    visit_summary = f"{visit.get('type', 'Visit')}"
                    if visit.get('chief_complaint'):
                        visit_summary += f": {visit['chief_complaint']}"
                    if visit.get('date'):
                        visit_summary += f" ({visit['date']})"
                    recent_visits.append(visit_summary)
                
                context_sections.append(f"RECENT VISITS: {' | '.join(recent_visits)}")
            
            # SECTION 6: Laboratory and Vital Signs (if available)
            labs = patient_data.get("laboratory_results", [])
            if labs:
                recent_labs = []
                for lab in labs[:6]:  # Most recent labs
                    lab_info = f"{lab.get('test_name', 'Test')}: {lab.get('value', 'N/A')} {lab.get('unit', '')}"
                    if lab.get('date'):
                        lab_info += f" ({lab['date']})"
                    if lab.get('flag'):
                        lab_info += f" [{lab['flag']}]"
                    recent_labs.append(lab_info)
                
                context_sections.append(f"RECENT LABS: {' | '.join(recent_labs)}")
            
            # SECTION 7: Risk Factors and Clinical Notes
            risk_factors = self._extract_comprehensive_risk_factors(patient_data)
            if risk_factors:
                context_sections.append(f"RISK FACTORS: {' | '.join(risk_factors)}")
            
            # SECTION 8: Social History (if available)
            social_history = patient_data.get("social_history", {})
            if social_history:
                social_items = []
                if social_history.get("smoking_status"):
                    social_items.append(f"Smoking: {social_history['smoking_status']}")
                if social_history.get("alcohol_use"):
                    social_items.append(f"Alcohol: {social_history['alcohol_use']}")
                if social_history.get("exercise_frequency"):
                    social_items.append(f"Exercise: {social_history['exercise_frequency']}")
                
                if social_items:
                    context_sections.append(f"SOCIAL HISTORY: {' | '.join(social_items)}")
            
            return "\n".join(context_sections)
            
        except Exception as e:
            logger.error(f"Error formatting comprehensive patient context: {e}")
            return f"Patient ID: {patient_data.get('patient_id', 'Unknown')} - Error formatting context"
    
    def _extract_comprehensive_risk_factors(self, patient_data: Dict) -> List[str]:
        """Extract comprehensive risk factors from patient data."""
        risk_factors = []
        
        try:
            # Age-based risk assessment
            demographics = patient_data.get("demographics", {})
            age = demographics.get('age', 0)
            if isinstance(age, (int, float)):
                if age > 75:
                    risk_factors.append("Advanced age (>75)")
                elif age > 65:
                    risk_factors.append("Elderly (65-75)")
                elif age < 18:
                    risk_factors.append("Pediatric patient")
            
            # Diagnosis-based risk factors
            diagnoses = patient_data.get("diagnoses", [])
            dx_risk_map = {
                "diabetes": "Diabetes mellitus",
                "hypertension": "Hypertension", 
                "heart": "Cardiovascular disease",
                "cardiac": "Cardiovascular disease",
                "coronary": "Coronary artery disease",
                "copd": "Chronic obstructive pulmonary disease",
                "asthma": "Respiratory disease",
                "renal": "Chronic kidney disease",
                "kidney": "Chronic kidney disease",
                "cancer": "History of malignancy",
                "stroke": "Cerebrovascular disease",
                "depression": "Mental health condition",
                "anxiety": "Mental health condition"
            }
            
            for dx in diagnoses:
                dx_lower = dx.get("description", "").lower()
                for keyword, risk_factor in dx_risk_map.items():
                    if keyword in dx_lower and risk_factor not in risk_factors:
                        risk_factors.append(risk_factor)
            
            # Medication-based risk factors
            medications = patient_data.get("medications", [])
            med_risk_map = {
                "warfarin": "Anticoagulation therapy",
                "coumadin": "Anticoagulation therapy",
                "insulin": "Insulin-dependent diabetes",
                "metformin": "Type 2 diabetes",
                "atorvastatin": "Dyslipidemia",
                "lisinopril": "Hypertension/Heart failure",
                "metoprolol": "Cardiovascular disease",
                "prednisone": "Chronic steroid use",
                "immunosuppressive": "Immunocompromised"
            }
            
            for med in medications:
                med_lower = med.get("name", "").lower()
                for keyword, risk_factor in med_risk_map.items():
                    if keyword in med_lower and risk_factor not in risk_factors:
                        risk_factors.append(risk_factor)
            
            # Allergy-based considerations
            allergies = patient_data.get("allergies", [])
            if allergies:
                severe_allergies = [a for a in allergies if a.get("severity") in ["severe", "life-threatening"]]
                if severe_allergies:
                    risk_factors.append("Severe drug allergies")
            
            return risk_factors[:10]  # Limit to most important
            
        except Exception as e:
            logger.error(f"Error extracting risk factors: {e}")
            return []
    
    def create_enhanced_query(self, patient_data: Dict, visit_type: str, symptoms: str, additional_context: Optional[str] = None) -> str:
        """Create enhanced query with comprehensive patient context for optimal LLM performance."""
        try:
            # Get comprehensive patient context
            patient_context = self._format_comprehensive_patient_context(patient_data)
            
            # Create structured clinical query
            query_sections = [
                "=== CLINICAL DECISION SUPPORT REQUEST ===",
                "",
                "=== PATIENT CLINICAL PROFILE ===",
                patient_context,
                "",
                "=== CURRENT CLINICAL ENCOUNTER ===",
                f"Visit Type: {visit_type}",
                f"Chief Complaint/Symptoms: {symptoms}",
            ]
            
            if additional_context:
                query_sections.extend([
                    f"Additional Clinical Notes: {additional_context}",
                ])
            
            query_sections.extend([
                "",
                "=== CLINICAL CONSULTATION QUESTIONS ===",
                "Based on this patient's complete clinical profile and current presentation:",
                "",
                "1. What is the most appropriate immediate diagnostic approach?",
                "2. What treatment options should be considered given this patient's medical history?",
                "3. Are there any contraindications or drug interactions to consider?",
                "4. What monitoring and safety measures are indicated?",
                "5. Should any specialist consultations be arranged?",
                "6. What follow-up care and patient education are needed?",
                "",
                "Please provide specific, evidence-based recommendations that account for this patient's",
                "unique clinical context, current medications, and risk factors."
            ])
            
            enhanced_query = "\n".join(query_sections)
            
            # Log query characteristics for debugging
            logger.info(f"Enhanced query created:")
            logger.info(f"  - Total length: {len(enhanced_query)} characters")
            logger.info(f"  - Patient context: {len(patient_context)} characters")
            logger.info(f"  - Visit type: {visit_type}")
            logger.info(f"  - Symptoms included: {'Yes' if symptoms else 'No'}")
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error creating enhanced query: {e}")
            return f"Patient symptoms: {symptoms}, Visit type: {visit_type}, Error: {str(e)}"
    
    def create_emergency_prompt(self, patient_data: Dict, symptoms: str) -> str:
        """Create specialized prompt for emergency situations with enhanced patient awareness."""
        patient_context = self._format_comprehensive_patient_context(patient_data)
        
        emergency_template = f"""🚨 EMERGENCY CLINICAL SITUATION - IMMEDIATE INTERVENTION REQUIRED 🚨

=== PATIENT CLINICAL PROFILE ===
{patient_context}

=== EMERGENCY PRESENTATION ===
Chief Complaint/Symptoms: {symptoms}

=== EMERGENCY PROTOCOLS ===
IMMEDIATE PRIORITIES (within 5-15 minutes):
1. Vital signs assessment and stabilization
2. Primary survey (ABCDE approach)
3. Immediate diagnostic tests based on presentation
4. Emergency interventions if indicated

CRITICAL CONSIDERATIONS FOR THIS PATIENT:
- Review current medications for contraindications
- Consider patient's medical history and comorbidities
- Account for age-specific emergency protocols
- Assess for allergies before medication administration

Please provide URGENT, prioritized recommendations focusing on:
1. Immediate life-saving interventions
2. Critical diagnostic tests needed NOW
3. Emergency medications/treatments to consider
4. Specialist notifications required
5. Disposition planning (ICU, ward, discharge)

Format as numbered priority actions with specific timelines where appropriate."""
        
        return emergency_template
    
    def create_followup_prompt(self, patient_data: Dict, symptoms: str) -> str:
        """Create specialized prompt for follow-up visits with enhanced monitoring focus."""
        patient_context = self._format_comprehensive_patient_context(patient_data)
        
        followup_template = f"""📋 FOLLOW-UP VISIT - COMPREHENSIVE CARE REVIEW

=== PATIENT CLINICAL PROFILE ===
{patient_context}

=== CURRENT FOLLOW-UP CONCERNS ===
Current Presentation: {symptoms}

=== FOLLOW-UP CARE PRIORITIES ===
Focus on comprehensive review and optimization:

1. TREATMENT EFFECTIVENESS REVIEW
   - Assess response to current therapies
   - Review medication adherence and effectiveness
   - Evaluate achievement of treatment goals

2. MONITORING & SURVEILLANCE
   - Review required laboratory monitoring
   - Assess for medication side effects
   - Screen for disease complications

3. PREVENTIVE CARE OPPORTUNITIES
   - Age-appropriate screening recommendations
   - Vaccination status review
   - Health maintenance activities

4. CARE COORDINATION
   - Specialist follow-up requirements
   - Referral needs assessment
   - Care plan optimization

Provide comprehensive follow-up recommendations that build on this patient's established care plan while addressing current concerns and optimizing long-term outcomes."""
        
        return followup_template
    
    def create_routine_prompt(self, patient_data: Dict, symptoms: str) -> str:
        """Create specialized prompt for routine visits with preventive care focus."""
        patient_context = self._format_comprehensive_patient_context(patient_data)
        
        routine_template = f"""🔍 ROUTINE VISIT - COMPREHENSIVE HEALTH ASSESSMENT

=== PATIENT CLINICAL PROFILE ===
{patient_context}

=== CURRENT HEALTH CONCERNS ===
Patient Concerns: {symptoms}

=== ROUTINE CARE FRAMEWORK ===
Comprehensive approach to health maintenance:

1. HEALTH SCREENING & PREVENTION
   - Age-appropriate screening recommendations
   - Risk factor assessment and modification
   - Vaccination updates needed

2. CHRONIC DISEASE MANAGEMENT
   - Review of established conditions
   - Medication optimization opportunities
   - Complication prevention strategies

3. LIFESTYLE & WELLNESS
   - Diet and nutrition counseling
   - Exercise recommendations
   - Stress management and mental health

4. HEALTH MAINTENANCE
   - Specialist referral scheduling
   - Laboratory monitoring requirements
   - Patient education priorities

Provide comprehensive routine care recommendations that address immediate concerns while optimizing this patient's overall health and preventing future complications."""
        
        return routine_template
    
    def format_recommendations(self, recommendations: List[str]) -> str:
        """Format recommendations into structured, clinical-ready output."""
        if not recommendations:
            return "No specific recommendations generated. Please review patient data manually."
        
        formatted = []
        for i, rec in enumerate(recommendations, 1):
            # Clean up the recommendation
            clean_rec = rec.strip()
            if not clean_rec:
                continue
            
            # Ensure proper numbering
            if not clean_rec[0].isdigit():
                clean_rec = f"{i}. {clean_rec}"
            
            # Add clinical formatting
            if len(clean_rec) > 100:
                # Add line breaks for long recommendations
                clean_rec = clean_rec.replace('. ', '.\n   ')
            
            formatted.append(clean_rec)
        
        return "\n\n".join(formatted)
    
    def validate_recommendation(self, recommendation: str, patient_data: Dict) -> bool:
        """Enhanced validation for clinical appropriateness and safety."""
        try:
            # Basic validation
            if not recommendation or len(recommendation.strip()) < 15:
                logger.warning("Recommendation too short to be clinically useful")
                return False
            
            rec_lower = recommendation.lower()
            
            # Enhanced safety checks
            dangerous_patterns = [
                "discontinue all medications",
                "stop all treatment",
                "no treatment needed",
                "ignore symptoms",
                "definitely not serious",
                "100% certain",
                "never",
                "always safe"
            ]
            
            for pattern in dangerous_patterns:
                if pattern in rec_lower:
                    logger.warning(f"Potentially dangerous recommendation pattern detected: {pattern}")
                    return False
            
            # Patient-specific contraindication checks
            medications = patient_data.get("medications", [])
            allergies = patient_data.get("allergies", [])
            
            # Check against known allergies
            for allergy in allergies:
                allergen = allergy.get("allergen", "").lower()
                if allergen in rec_lower and len(allergen) > 3:
                    logger.warning(f"Recommendation may involve known allergen: {allergen}")
                    return False
            
            # Check for drug interactions (basic examples)
            drug_interactions = {
                "warfarin": ["aspirin", "ibuprofen", "naproxen"],
                "metformin": ["contrast dye"],
                "ace inhibitor": ["potassium supplement"],
            }
            
            current_drugs = [med.get("name", "").lower() for med in medications]
            
            for current_drug in current_drugs:
                for contraindicated_drug in drug_interactions.get(current_drug, []):
                    if contraindicated_drug in rec_lower:
                        logger.warning(f"Potential drug interaction: {current_drug} + {contraindicated_drug}")
                        # Don't automatically reject, but flag for review
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating recommendation: {e}")
            return True  # Default to allowing if validation fails
    
    def get_visit_type_prompt(self, visit_type: str, patient_data: Dict, symptoms: str) -> str:
        """Get appropriate prompt based on visit type with enhanced patient context."""
        visit_type_lower = visit_type.lower()
        
        if "emergency" in visit_type_lower or "urgent" in visit_type_lower or "er" in visit_type_lower:
            return self.create_emergency_prompt(patient_data, symptoms)
        elif "follow" in visit_type_lower or "followup" in visit_type_lower:
            return self.create_followup_prompt(patient_data, symptoms)
        elif "routine" in visit_type_lower or "annual" in visit_type_lower or "physical" in visit_type_lower:
            return self.create_routine_prompt(patient_data, symptoms)
        else:
            # Default to enhanced query for other visit types
            return self.create_enhanced_query(patient_data, visit_type, symptoms)
    
    def create_diagnostic_focused_prompt(self, patient_data: Dict, symptoms: str, suspected_conditions: List[str] = None) -> str:
        """Create a diagnostic-focused prompt for complex cases."""
        patient_context = self._format_comprehensive_patient_context(patient_data)
        
        diagnostic_template = f"""🔬 DIAGNOSTIC CONSULTATION - DIFFERENTIAL DIAGNOSIS SUPPORT

=== PATIENT CLINICAL PROFILE ===
{patient_context}

=== CLINICAL PRESENTATION ===
Presenting Symptoms: {symptoms}
"""
        
        if suspected_conditions:
            diagnostic_template += f"""
=== DIFFERENTIAL CONSIDERATIONS ===
Suspected Conditions: {', '.join(suspected_conditions)}
"""
        
        diagnostic_template += """
=== DIAGNOSTIC FRAMEWORK ===
Please provide systematic diagnostic recommendations:

1. DIFFERENTIAL DIAGNOSIS
   - Most likely diagnoses based on presentation and patient factors
   - Consider patient's medical history and risk factors
   - Rule-out conditions that require immediate attention

2. DIAGNOSTIC WORKUP STRATEGY
   - Laboratory tests indicated
   - Imaging studies needed
   - Specialized testing requirements
   - Priority and timing of tests

3. CLINICAL DECISION MAKING
   - Probability assessment of key diagnoses
   - Risk stratification for this patient
   - Factors that support or argue against each diagnosis

4. IMMEDIATE MANAGEMENT
   - Empiric treatments while awaiting results
   - Monitoring requirements
   - Safety considerations

Format recommendations with clear rationale and consideration of this patient's specific clinical context."""
        
        return diagnostic_template
    
    def create_medication_review_prompt(self, patient_data: Dict, concerns: str) -> str:
        """Create a specialized prompt for medication review and optimization."""
        patient_context = self._format_comprehensive_patient_context(patient_data)
        
        med_review_template = f"""💊 MEDICATION REVIEW & OPTIMIZATION

=== PATIENT CLINICAL PROFILE ===
{patient_context}

=== MEDICATION CONCERNS ===
Current Concerns: {concerns}

=== MEDICATION REVIEW FRAMEWORK ===
Comprehensive medication assessment:

1. MEDICATION APPROPRIATENESS
   - Review indication for each medication
   - Assess for therapeutic duplication
   - Evaluate dose appropriateness for patient factors

2. SAFETY ASSESSMENT
   - Drug-drug interactions
   - Drug-disease interactions
   - Contraindications based on patient history

3. ADHERENCE & OPTIMIZATION
   - Simplification opportunities
   - Cost-effectiveness considerations
   - Patient preference factors

4. MONITORING REQUIREMENTS
   - Laboratory monitoring needs
   - Clinical monitoring parameters
   - Timing of follow-up assessments

Provide specific medication recommendations that optimize efficacy while minimizing risks for this patient."""
        
        return med_review_template
    
    def get_context_summary_for_logging(self, patient_data: Dict) -> Dict:
        """Generate a summary of patient context for logging and debugging."""
        try:
            summary = {
                "patient_id": patient_data.get("patient_id", "Unknown"),
                "demographics_available": bool(patient_data.get("demographics")),
                "diagnoses_count": len(patient_data.get("diagnoses", [])),
                "medications_count": len(patient_data.get("medications", [])),
                "allergies_count": len(patient_data.get("allergies", [])),
                "recent_visits_count": len(patient_data.get("visit_history", [])),
                "lab_results_available": bool(patient_data.get("laboratory_results")),
                "social_history_available": bool(patient_data.get("social_history"))
            }
            
            # Extract key clinical elements
            if patient_data.get("diagnoses"):
                active_dx = [d["description"] for d in patient_data["diagnoses"] if d.get("status") == "active"]
                summary["active_diagnoses_count"] = len(active_dx)
                summary["sample_diagnoses"] = active_dx[:3]  # First 3 for logging
            
            if patient_data.get("medications"):
                active_meds = [m["name"] for m in patient_data["medications"] if m.get("status") == "active"]
                summary["active_medications_count"] = len(active_meds)
                summary["sample_medications"] = active_meds[:3]  # First 3 for logging
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating context summary: {e}")
            return {"error": str(e)}


def create_clinical_prompt_template() -> ClinicalPromptTemplate:
    """Factory function to create enhanced clinical prompt template."""
    return ClinicalPromptTemplate()


# Utility functions for prompt enhancement
def extract_clinical_keywords(text: str) -> List[str]:
    """Extract clinical keywords from text for better context matching."""
    clinical_terms = [
        # Symptoms
        "pain", "fever", "nausea", "vomiting", "shortness of breath", "chest pain",
        "headache", "dizziness", "fatigue", "weight loss", "weight gain",
        
        # Body systems
        "cardiac", "pulmonary", "respiratory", "gastrointestinal", "neurological",
        "musculoskeletal", "dermatological", "endocrine", "renal", "hepatic",
        
        # Common conditions
        "diabetes", "hypertension", "asthma", "copd", "depression", "anxiety",
        "arthritis", "cancer", "infection", "allergic reaction",
        
        # Urgency indicators
        "acute", "chronic", "emergency", "urgent", "routine", "follow-up"
    ]
    
    text_lower = text.lower()
    found_terms = []
    
    for term in clinical_terms:
        if term in text_lower:
            found_terms.append(term)
    
    return found_terms


def format_patient_summary_for_display(patient_data: Dict) -> str:
    """Format patient summary for display in UI or reports."""
    try:
        template = ClinicalPromptTemplate()
        context = template._format_comprehensive_patient_context(patient_data)
        
        # Convert to more readable format
        sections = context.split('\n')
        formatted_sections = []
        
        for section in sections:
            if section.startswith('DEMOGRAPHICS:'):
                formatted_sections.append(f"**Patient Information:** {section.replace('DEMOGRAPHICS: ', '')}")
            elif section.startswith('ACTIVE CONDITIONS:'):
                formatted_sections.append(f"**Medical Conditions:** {section.replace('ACTIVE CONDITIONS: ', '')}")
            elif section.startswith('CURRENT MEDICATIONS:'):
                formatted_sections.append(f"**Current Medications:** {section.replace('CURRENT MEDICATIONS: ', '')}")
            elif section.startswith('ALLERGIES:'):
                formatted_sections.append(f"**Known Allergies:** {section.replace('ALLERGIES: ', '')}")
            elif section.startswith('RISK FACTORS:'):
                formatted_sections.append(f"**Risk Factors:** {section.replace('RISK FACTORS: ', '')}")
        
        return '\n\n'.join(formatted_sections)
        
    except Exception as e:
        logger.error(f"Error formatting patient summary for display: {e}")
        return f"Patient ID: {patient_data.get('patient_id', 'Unknown')} - Error formatting summary"