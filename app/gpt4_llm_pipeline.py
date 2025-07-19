"""
Enhanced AI pipeline for clinical recommendations using GPT-4.1 via GitHub AI.
GPT-4.1 VERSION - Uses OpenAI's GPT-4.1 through GitHub AI inference endpoint.
Optimized for medical diagnostic reasoning and patient history analysis.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import logging
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field, PrivateAttr
import json
import hashlib
from datetime import datetime, timedelta

from .retriever import ClinicalRetriever
from .prompt import ClinicalPromptTemplate

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GPT4MedicalLLM(LLM):
    """
    Medical LLM using GPT-4.1 via GitHub AI inference endpoint.
    Optimized for clinical diagnostic reasoning and patient history analysis.
    """
    
    # Define fields using Pydantic Field for proper LangChain compatibility
    model_name: str = Field(default="gpt-4.1-preview")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=3000)  # GPT-4.1 can handle larger responses
    top_p: float = Field(default=0.95)
    api_key: Optional[str] = Field(default=None)
    base_url: str = Field(default="https://models.github.ai/inference")
    
    # Use PrivateAttr for internal attributes
    _client: Any = PrivateAttr(default=None)
    _client_initialized: bool = PrivateAttr(default=False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize private attributes
        self.__dict__['_client'] = None
        self.__dict__['_client_initialized'] = False
        self._setup_client()
    
    @property
    def _llm_type(self) -> str:
        """Return identifier of LLM type."""
        return "gpt4_medical_diagnostic_llm"
    
    def _setup_client(self):
        """Setup the GPT-4.1 client via GitHub AI."""
        try:
            # Get API key from instance or environment
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            
            if not api_key:
                logger.error("No GPT-4.1 API key found. Set OPENAI_API_KEY environment variable.")
                logger.info("Ensure your GitHub AI token is properly configured")
                self.__dict__['_client'] = None
                self.__dict__['_client_initialized'] = False
                return
            
            # Import and setup OpenAI client for GitHub AI
            try:
                from openai import OpenAI
                logger.info("OpenAI SDK available for GPT-4.1")
            except ImportError:
                logger.error("OpenAI SDK not installed. Install with: pip install openai")
                self.__dict__['_client'] = None
                self.__dict__['_client_initialized'] = False
                return
            
            # Create OpenAI client with GitHub AI endpoint
            client = OpenAI(
                base_url=self.base_url,
                api_key=api_key
            )
            
            self.__dict__['_client'] = client
            self.__dict__['_client_initialized'] = True
            logger.info(f"GPT-4.1 Medical Diagnostic LLM initialized successfully via GitHub AI")
            
            # Test the connection
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Error setting up GPT-4.1 client: {e}")
            self.__dict__['_client'] = None
            self.__dict__['_client_initialized'] = False
    
    def _test_connection(self):
        """Test the GPT-4.1 connection with a diagnostic query."""
        try:
            logger.info("Testing GPT-4.1 connection with diagnostic query...")
            
            # Test with a medical diagnostic question
            test_response = self.__dict__['_client'].chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a clinical diagnostic AI. Provide brief, accurate medical assessments."
                    },
                    {
                        "role": "user",
                        "content": "A 65-year-old patient presents with chest pain, shortness of breath, and has a history of diabetes. What are the primary diagnostic considerations?"
                    }
                ],
                max_tokens=300,
                temperature=0.1,
            )
            
            # Check response
            if test_response and test_response.choices and len(test_response.choices) > 0:
                response_text = test_response.choices[0].message.content
                logger.info(f"✅ GPT-4.1 diagnostic connection successful")
                logger.debug(f"Test response: {response_text[:150]}...")
                return True
            else:
                logger.error("Unexpected response format from GPT-4.1")
                self.__dict__['_client_initialized'] = False
                return False
                
        except Exception as e:
            logger.error(f"GPT-4.1 connection test failed: {e}")
            
            # Provide specific error guidance for GitHub AI
            error_str = str(e).lower()
            if "unauthorized" in error_str or "401" in error_str:
                logger.error("❌ Authentication failed. Please check your GitHub AI token")
                logger.info("💡 Ensure you have:")
                logger.info("   1. Valid GitHub AI access token")
                logger.info("   2. Proper permissions for GPT-4.1 model access")
                logger.info("   3. Token is correctly set as OPENAI_API_KEY environment variable")
            elif "rate limit" in error_str or "429" in error_str:
                logger.error("❌ Rate limit exceeded. Please try again later")
            elif "model" in error_str and "not found" in error_str:
                logger.error(f"❌ Model {self.model_name} not found")
                logger.info("💡 Trying alternative GPT models...")
                return self._try_alternative_gpt_models()
            else:
                logger.error(f"❌ Unexpected error: {e}")
            
            self.__dict__['_client_initialized'] = False
            return False
    
    def _try_alternative_gpt_models(self) -> bool:
        """Try alternative GPT models that might be available via GitHub AI."""
        alternative_models = [
            "openai/gpt-4.1",
            "openai/gpt-4o",
            "openai/gpt-4o-mini", 
            "openai/gpt-4-turbo",
            "openai/gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4"
        ]
        
        client = self.__dict__.get('_client', None)
        if not client:
            return False
        
        for model in alternative_models:
            try:
                logger.info(f"🧪 Trying GPT model: {model}")
                
                test_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Define diabetes in one sentence."
                        }
                    ],
                    max_tokens=50,
                    temperature=0.1,
                )
                
                if test_response and test_response.choices and len(test_response.choices) > 0:
                    response_text = test_response.choices[0].message.content
                    logger.info(f"✅ GPT model {model} successful")
                    self.model_name = model  # Update to working model
                    return True
                
            except Exception as e:
                logger.warning(f"❌ GPT model {model} failed: {e}")
                continue
        
        logger.info("🔄 All GPT models failed")
        return False
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Make a call to GPT-4.1 API for medical diagnostic analysis."""
        client_initialized = self.__dict__.get('_client_initialized', False)
        client = self.__dict__.get('_client', None)
        
        if not client_initialized or client is None:
            logger.error("GPT-4.1 client not available - cannot generate diagnostic recommendations")
            raise RuntimeError("GPT-4.1 client not initialized. Please check your GitHub AI token and configuration.")
        
        try:
            logger.info(f"Calling GPT-4.1 for diagnostic analysis with model: {self.model_name}")
            
            # Create diagnostic-focused system prompt optimized for GPT-4.1
            gpt4_diagnostic_system_prompt = """You are an expert clinical diagnostician with decades of experience in internal medicine and diagnostic reasoning. Provide comprehensive diagnostic assessments with clear, structured formatting.

DIAGNOSTIC EXCELLENCE FRAMEWORK:
1. SYSTEMATIC ANALYSIS: Review all patient data chronologically and by system
2. PATTERN RECOGNITION: Identify clinical patterns and disease progression
3. EVIDENCE-BASED REASONING: Support conclusions with specific patient data
4. DIFFERENTIAL DIAGNOSIS: Generate ranked alternatives with clear likelihood estimates
5. RISK STRATIFICATION: Assess immediate vs long-term patient risks
6. CLINICAL INTEGRATION: Consider all medications, comorbidities, and patient factors

CRITICAL FORMATTING REQUIREMENTS:
Structure your response EXACTLY as follows with clear section headers:

PRIMARY DIAGNOSIS
[State the single most likely diagnosis with confidence level (High/Medium/Low) and 3-5 key supporting evidence points from the patient data]

DIFFERENTIAL DIAGNOSES
[List 3-5 alternative diagnoses, each on a new line starting with a dash (-), including likelihood percentage and brief supporting evidence]

CLINICAL REASONING
[Provide step-by-step diagnostic thought process explaining how you integrated patient timeline, patterns, and risk factors]

RISK STRATIFICATION
[Categorize risks as Immediate (0-48 hours), Short-term (days-weeks), and Long-term, with specific patient risks]

RECOMMENDED DIAGNOSTIC WORKUP
[List specific tests needed, each on a new line starting with a dash (-), with timing and rationale]

IMMEDIATE MANAGEMENT PRIORITIES
[List urgent interventions needed, each on a new line starting with a dash (-), with specific actions and monitoring]

FOLLOW-UP CARE PLAN
[List follow-up recommendations, each on a new line starting with a dash (-), including timeline and specialist referrals]

FORMATTING RULES:
- Use the EXACT section headers shown above
- Start recommendation lists with dashes (-)
- Be specific and reference actual patient data
- Include confidence levels and percentages where appropriate
- Keep each recommendation actionable and specific
- Prioritize patient safety in all recommendations

BASE ALL CONCLUSIONS ON THE SPECIFIC PATIENT DATA PROVIDED."""

            # Make the API call to GPT-4.1
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": gpt4_diagnostic_system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            # Extract response text
            if response and response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                logger.info(f"✅ GPT-4.1 diagnostic analysis completed: {len(response_text)} characters")
                return response_text
            else:
                logger.error("Unexpected response format from GPT-4.1")
                raise RuntimeError("Invalid response format from GPT-4.1")
            
        except Exception as e:
            logger.error(f"Error calling GPT-4.1 for diagnostics: {e}")
            
            # Enhanced error handling for GitHub AI specific issues
            error_str = str(e).lower()
            if "unauthorized" in error_str or "401" in error_str:
                logger.error("❌ GPT-4.1 authentication failed")
                logger.info("💡 Check your GitHub AI token and model access permissions")
            elif "rate limit" in error_str or "429" in error_str:
                logger.error("❌ GPT-4.1 rate limit exceeded")
                logger.info("💡 Consider upgrading your GitHub AI plan or waiting before retry")
            elif "model" in error_str and "not found" in error_str:
                logger.error(f"❌ GPT model {self.model_name} not accessible via GitHub AI")
                logger.info("💡 Verify model availability in your GitHub AI account")
                if self._try_alternative_gpt_models():
                    # Retry with new model
                    return self._call(prompt, stop, run_manager, **kwargs)
            else:
                logger.error(f"❌ Unexpected GPT-4.1 error: {e}")
            
            # Re-raise the error instead of falling back to mock responses
            raise RuntimeError(f"GPT-4.1 diagnostic analysis failed: {str(e)}")

# Keep the same PatientDataProcessor from the original implementation
# (This class doesn't need changes as it's LLM-agnostic)

class PatientDataProcessor:
    """
    Handles efficient processing of large patient datasets.
    Implements smart chunking and summarization for datasets up to 10,000+ lines.
    This class is LLM-agnostic and works with both Mistral AI and GPT-4.1.
    """
    
    def __init__(self, max_chunk_size: int = 4000, overlap_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.cache = {}  # Simple in-memory cache for processed data
    
    def process_large_patient_data(self, patient_data: Dict) -> Dict:
        """
        Process large patient datasets efficiently.
        Returns optimized patient data structure for LLM analysis.
        """
        try:
            # Create a cache key for this patient's data
            cache_key = self._create_cache_key(patient_data)
            
            if cache_key in self.cache:
                logger.info("Using cached processed patient data")
                return self.cache[cache_key]
            
            logger.info("Processing large patient dataset for GPT-4.1...")
            
            # Optimize and structure the data
            processed_data = {
                "patient_summary": self._create_patient_summary(patient_data),
                "chronological_history": self._create_chronological_history(patient_data),
                "clinical_patterns": self._identify_clinical_patterns(patient_data),
                "risk_profile": self._create_risk_profile(patient_data),
                "recent_changes": self._identify_recent_changes(patient_data),
                "data_statistics": self._calculate_data_stats(patient_data)
            }
            
            # Cache the processed data
            self.cache[cache_key] = processed_data
            
            logger.info(f"Patient data processed for GPT-4.1: {processed_data['data_statistics']}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing patient data: {e}")
            return {"error": str(e), "raw_data": patient_data}
    
    def _create_cache_key(self, patient_data: Dict) -> str:
        """Create a cache key based on patient data content."""
        data_str = json.dumps(patient_data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _create_patient_summary(self, patient_data: Dict) -> Dict:
        """Create a concise summary of the patient's key information."""
        demographics = patient_data.get("demographics", {})
        diagnoses = patient_data.get("diagnoses", [])
        medications = patient_data.get("medications", [])
        
        # Get active conditions
        active_conditions = [d for d in diagnoses if d.get("status") == "active"]
        active_medications = [m for m in medications if m.get("status") == "active"]
        
        return {
            "patient_id": patient_data.get("patient_id"),
            "demographics": demographics,
            "active_conditions_count": len(active_conditions),
            "active_medications_count": len(active_medications),
            "primary_conditions": [d["description"] for d in active_conditions[:5]],
            "key_medications": [f"{m['name']} {m.get('dosage', '')}" for m in active_medications[:8]]
        }
    
    def _create_chronological_history(self, patient_data: Dict) -> List[Dict]:
        """Create a chronological timeline of patient's medical history."""
        timeline = []
        
        # Add diagnosis events
        for dx in patient_data.get("diagnoses", []):
            timeline.append({
                "date": dx.get("diagnosis_date", dx.get("date")),
                "type": "diagnosis",
                "description": dx["description"],
                "icd_code": dx.get("icd_code"),
                "status": dx.get("status", "active")
            })
        
        # Add medication events
        for med in patient_data.get("medications", []):
            timeline.append({
                "date": med.get("start_date"),
                "type": "medication_start",
                "description": f"Started {med['name']} {med.get('dosage', '')}",
                "status": med.get("status", "active")
            })
            
            if med.get("end_date"):
                timeline.append({
                    "date": med["end_date"],
                    "type": "medication_end",
                    "description": f"Stopped {med['name']}",
                    "status": "discontinued"
                })
        
        # Add visit events
        for visit in patient_data.get("visit_history", []):
            timeline.append({
                "date": visit.get("visit_date", visit.get("date")),
                "type": "visit",
                "description": f"{visit.get('visit_type', 'Visit')}: {visit.get('chief_complaint', 'No complaint')}",
                "provider": visit.get("provider"),
                "notes": visit.get("notes", "")[:200]  # Truncate long notes
            })
        
        # Sort by date and return recent events (last 50 to manage size)
        timeline.sort(key=lambda x: x.get("date", ""), reverse=True)
        return timeline[:50]
    
    def _identify_clinical_patterns(self, patient_data: Dict) -> Dict:
        """Identify patterns in the patient's clinical data."""
        patterns = {
            "recurring_symptoms": [],
            "medication_patterns": [],
            "visit_frequency": {},
            "condition_progression": []
        }
        
        try:
            # Analyze visit patterns
            visits = patient_data.get("visit_history", [])
            if visits:
                visit_types = {}
                recent_complaints = []
                
                for visit in visits[:20]:  # Last 20 visits
                    visit_type = visit.get("visit_type", "unknown")
                    visit_types[visit_type] = visit_types.get(visit_type, 0) + 1
                    
                    complaint = visit.get("chief_complaint", "")
                    if complaint:
                        recent_complaints.append(complaint.lower())
                
                patterns["visit_frequency"] = visit_types
                
                # Find recurring symptoms
                complaint_words = []
                for complaint in recent_complaints:
                    complaint_words.extend(complaint.split())
                
                word_counts = {}
                for word in complaint_words:
                    if len(word) > 3:  # Only significant words
                        word_counts[word] = word_counts.get(word, 0) + 1
                
                patterns["recurring_symptoms"] = [
                    word for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                ]
            
            # Analyze medication patterns
            medications = patient_data.get("medications", [])
            med_classes = {}
            
            for med in medications:
                if med.get("status") == "active":
                    med_name = med["name"].lower()
                    # Basic medication classification
                    if any(term in med_name for term in ["insulin", "metformin", "glyburide"]):
                        med_classes["diabetes"] = med_classes.get("diabetes", 0) + 1
                    elif any(term in med_name for term in ["lisinopril", "amlodipine", "metoprolol"]):
                        med_classes["cardiovascular"] = med_classes.get("cardiovascular", 0) + 1
                    elif any(term in med_name for term in ["atorvastatin", "simvastatin"]):
                        med_classes["lipid_management"] = med_classes.get("lipid_management", 0) + 1
            
            patterns["medication_patterns"] = med_classes
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
        
        return patterns
    
    def _create_risk_profile(self, patient_data: Dict) -> Dict:
        """Create a comprehensive risk profile for the patient."""
        risk_profile = {
            "age_risk": "low",
            "medication_risks": [],
            "condition_risks": [],
            "overall_complexity": "low"
        }
        
        try:
            # Age-based risk assessment
            demographics = patient_data.get("demographics", {})
            age = demographics.get("age", 0)
            
            if age > 80:
                risk_profile["age_risk"] = "very_high"
            elif age > 65:
                risk_profile["age_risk"] = "high"
            elif age > 50:
                risk_profile["age_risk"] = "moderate"
            
            # Condition-based risk assessment
            diagnoses = patient_data.get("diagnoses", [])
            high_risk_conditions = [
                "diabetes", "heart failure", "coronary", "stroke", "cancer", 
                "renal failure", "liver", "copd", "asthma"
            ]
            
            for dx in diagnoses:
                if dx.get("status") == "active":
                    dx_lower = dx["description"].lower()
                    for condition in high_risk_conditions:
                        if condition in dx_lower:
                            risk_profile["condition_risks"].append({
                                "condition": dx["description"],
                                "risk_level": "high" if condition in ["heart failure", "cancer", "stroke"] else "moderate"
                            })
            
            # Medication risk assessment
            medications = patient_data.get("medications", [])
            high_risk_meds = ["warfarin", "insulin", "chemotherapy", "immunosuppressive"]
            
            for med in medications:
                if med.get("status") == "active":
                    med_lower = med["name"].lower()
                    for risk_med in high_risk_meds:
                        if risk_med in med_lower:
                            risk_profile["medication_risks"].append({
                                "medication": med["name"],
                                "risk_type": "bleeding" if "warfarin" in med_lower else "metabolic"
                            })
            
            # Overall complexity assessment
            total_active_conditions = len([d for d in diagnoses if d.get("status") == "active"])
            total_active_meds = len([m for m in medications if m.get("status") == "active"])
            
            if total_active_conditions > 5 or total_active_meds > 8:
                risk_profile["overall_complexity"] = "high"
            elif total_active_conditions > 3 or total_active_meds > 5:
                risk_profile["overall_complexity"] = "moderate"
            
        except Exception as e:
            logger.error(f"Error creating risk profile: {e}")
        
        return risk_profile
    
    def _identify_recent_changes(self, patient_data: Dict) -> Dict:
        """Identify recent changes in patient's condition or treatment."""
        recent_changes = {
            "new_medications": [],
            "stopped_medications": [],
            "new_diagnoses": [],
            "recent_visits": []
        }
        
        try:
            # Define "recent" as last 90 days
            recent_date = datetime.now() - timedelta(days=90)
            
            # Check for recent medication changes
            medications = patient_data.get("medications", [])
            for med in medications:
                start_date = med.get("start_date")
                end_date = med.get("end_date")
                
                if start_date:
                    try:
                        med_start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        if med_start > recent_date:
                            recent_changes["new_medications"].append({
                                "name": med["name"],
                                "dosage": med.get("dosage"),
                                "start_date": start_date
                            })
                    except:
                        pass
                
                if end_date:
                    try:
                        med_end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        if med_end > recent_date:
                            recent_changes["stopped_medications"].append({
                                "name": med["name"],
                                "end_date": end_date
                            })
                    except:
                        pass
            
            # Check for recent diagnoses
            diagnoses = patient_data.get("diagnoses", [])
            for dx in diagnoses:
                dx_date = dx.get("diagnosis_date", dx.get("date"))
                if dx_date:
                    try:
                        diagnosis_date = datetime.fromisoformat(dx_date.replace('Z', '+00:00'))
                        if diagnosis_date > recent_date:
                            recent_changes["new_diagnoses"].append({
                                "description": dx["description"],
                                "date": dx_date,
                                "icd_code": dx.get("icd_code")
                            })
                    except:
                        pass
            
            # Recent visits
            visits = patient_data.get("visit_history", [])
            for visit in visits[:5]:  # Last 5 visits
                visit_date = visit.get("visit_date", visit.get("date"))
                if visit_date:
                    try:
                        v_date = datetime.fromisoformat(visit_date.replace('Z', '+00:00'))
                        if v_date > recent_date:
                            recent_changes["recent_visits"].append({
                                "date": visit_date,
                                "type": visit.get("visit_type"),
                                "complaint": visit.get("chief_complaint"),
                                "provider": visit.get("provider")
                            })
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"Error identifying recent changes: {e}")
        
        return recent_changes
    
    def _calculate_data_stats(self, patient_data: Dict) -> Dict:
        """Calculate statistics about the patient data."""
        return {
            "total_diagnoses": len(patient_data.get("diagnoses", [])),
            "active_diagnoses": len([d for d in patient_data.get("diagnoses", []) if d.get("status") == "active"]),
            "total_medications": len(patient_data.get("medications", [])),
            "active_medications": len([m for m in patient_data.get("medications", []) if m.get("status") == "active"]),
            "total_visits": len(patient_data.get("visit_history", [])),
            "data_completeness": self._assess_data_completeness(patient_data)
        }
    
    def _assess_data_completeness(self, patient_data: Dict) -> float:
        """Assess the completeness of patient data (0.0 to 1.0)."""
        completeness_score = 0.0
        total_fields = 6
        
        if patient_data.get("demographics"):
            completeness_score += 1
        if patient_data.get("diagnoses"):
            completeness_score += 1
        if patient_data.get("medications"):
            completeness_score += 1
        if patient_data.get("visit_history"):
            completeness_score += 1
        if patient_data.get("laboratory_results"):
            completeness_score += 1
        if patient_data.get("allergies"):
            completeness_score += 1
        
        return completeness_score / total_fields

class GPT4ClinicalRecommendationPipeline:
    """
    Enhanced pipeline for generating clinical recommendations using GPT-4.1 via GitHub AI.
    Focuses on diagnostic reasoning from comprehensive patient history analysis.
    """

    def __init__(self):
        self.retriever = None
        self.llm = None
        self.prompt_template = None
        self.data_processor = None
        self.initialized = False

    async def initialize(self):
        """Initialize the clinical recommendation pipeline with GPT-4.1."""
        try:
            logger.info("Initializing GPT-4.1 clinical diagnostic pipeline via GitHub AI...")
            
            # Initialize retriever for medical literature
            self.retriever = ClinicalRetriever()
            await self.retriever.initialize()
            
            # Initialize GPT-4.1 LLM
            await self._initialize_gpt4_llm()
            
            # Initialize prompt template
            self.prompt_template = ClinicalPromptTemplate()
            
            # Initialize data processor
            self.data_processor = PatientDataProcessor()
            
            self.initialized = True
            logger.info("GPT-4.1 clinical diagnostic pipeline initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPT-4.1 clinical pipeline: {e}")
            raise

    async def _initialize_gpt4_llm(self):
        """Initialize the GPT-4.1 LLM with diagnostic configuration."""
        try:
            logger.info("Initializing GPT-4.1 Diagnostic LLM via GitHub AI...")
            
            # Get token from environment
            token = os.environ.get("OPENAI_API_KEY")
            
            if not token:
                raise RuntimeError("OPENAI_API_KEY not found. Please set your GitHub AI token.")
            
            # Test if we can import required modules
            try:
                from openai import OpenAI
                logger.info("OpenAI SDK available for GPT-4.1")
            except ImportError:
                raise RuntimeError("OpenAI SDK not installed. Install with: pip install openai")
            
            # Create GPT-4.1 Medical LLM with diagnostic configuration
            self.llm = GPT4MedicalLLM(
                model_name="openai/gpt-4.1",  # Correct GitHub AI model name
                temperature=0.1,  # Low temperature for consistent medical advice
                max_tokens=3000,  # Increased for detailed diagnostic reports
                top_p=0.95,
                api_key=token,
                base_url="https://models.github.ai/inference"
            )
            
            # Test the LLM with a diagnostic scenario
            test_prompt = """
            Analyze this patient case:
            
            Patient: 72-year-old female
            History: Type 2 diabetes, hypertension, chronic kidney disease
            Recent symptoms: Progressive shortness of breath, ankle swelling, fatigue
            Current medications: Metformin, Lisinopril, Furosemide
            
            Provide diagnostic assessment.
            """
            
            try:
                test_response = self.llm._call(test_prompt)
                logger.info("✅ GPT-4.1 Diagnostic LLM initialized and tested successfully")
                logger.info(f"✅ Using model: {self.llm.model_name}")
            except Exception as e:
                logger.error(f"GPT-4.1 LLM test call failed: {e}")
                raise RuntimeError(f"GPT-4.1 diagnostic test failed: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing GPT-4.1 LLM: {e}")
            raise

    async def analyze_patient_history(
        self,
        patient_data: Dict
    ) -> Dict:
        """
        Main method: Analyze comprehensive patient history and generate diagnostic assessment using GPT-4.1.
        """
        try:
            if not self.initialized:
                raise RuntimeError("GPT-4.1 diagnostic pipeline not initialized")

            logger.info(f"Starting comprehensive patient history analysis with GPT-4.1 for patient {patient_data.get('patient_id')}")
            
            # Step 1: Process large patient dataset efficiently
            processed_data = self.data_processor.process_large_patient_data(patient_data)
            
            # Step 2: Retrieve relevant medical literature
            literature_context = await self._retrieve_relevant_literature(processed_data)
            
            # Step 3: Create comprehensive diagnostic prompt
            diagnostic_prompt = self._create_diagnostic_prompt(processed_data, literature_context)
            
            # Step 4: Get diagnostic analysis from GPT-4.1
            diagnostic_analysis = await self._execute_gpt4_diagnostic_analysis(diagnostic_prompt)
            
            # Step 5: Structure and validate the diagnostic response
            structured_diagnosis = self._structure_diagnostic_response(diagnostic_analysis, processed_data)

            return {
                "diagnostic_assessment": structured_diagnosis,
                "patient_summary": processed_data.get("patient_summary"),
                "clinical_patterns": processed_data.get("clinical_patterns"),
                "risk_profile": processed_data.get("risk_profile"),
                "recent_changes": processed_data.get("recent_changes"),
                "literature_sources": literature_context.get("sources", []),
                "confidence_score": self._calculate_diagnostic_confidence(diagnostic_analysis, processed_data),
                "processing_stats": processed_data.get("data_statistics"),
                "model_used": f"GPT-4.1 via GitHub AI - {self.llm.model_name}",
                "analysis_timestamp": datetime.now().isoformat(),
                "api_provider": "GitHub AI (OpenAI GPT-4.1)"
            }
            
        except Exception as e:
            logger.error(f"Error in GPT-4.1 patient history analysis: {e}")
            raise RuntimeError(f"GPT-4.1 patient history analysis failed: {str(e)}")

    async def _retrieve_relevant_literature(self, processed_data: Dict) -> Dict:
        """Retrieve relevant medical literature based on patient data."""
        try:
            literature_context = {"sources": [], "content": ""}
            
            # Extract key medical terms for literature search
            search_terms = []
            
            # Add primary conditions
            primary_conditions = processed_data.get("patient_summary", {}).get("primary_conditions", [])
            search_terms.extend(primary_conditions[:3])  # Top 3 conditions
            
            # Add recurring symptoms
            recurring_symptoms = processed_data.get("clinical_patterns", {}).get("recurring_symptoms", [])
            search_terms.extend(recurring_symptoms[:3])  # Top 3 symptoms
            
            # Search for relevant literature
            if search_terms:
                search_query = " ".join(search_terms)
                retrieved_docs = await self.retriever.search(search_query, k=5)
                
                literature_parts = []
                sources = []
                
                for doc in retrieved_docs:
                    literature_parts.append(doc.page_content[:500])  # Limit to 500 chars per doc
                    if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                        sources.append(doc.metadata['source'])
                
                literature_context = {
                    "content": "\n\n".join(literature_parts),
                    "sources": sources
                }
            
            logger.info(f"Retrieved {len(literature_context['sources'])} relevant literature sources for GPT-4.1")
            return literature_context
            
        except Exception as e:
            logger.error(f"Error retrieving literature: {e}")
            return {"sources": [], "content": ""}

    def _create_diagnostic_prompt(self, processed_data: Dict, literature_context: Dict) -> str:
        """Create a comprehensive diagnostic prompt for GPT-4.1 using the updated prompt system."""
        try:
            # Use the updated prompt template system optimized for GPT-4.1
            diagnostic_prompt = self.prompt_template.create_comprehensive_diagnostic_prompt(
                processed_data, literature_context
            )
            
            logger.info(f"Diagnostic prompt created for GPT-4.1: {len(diagnostic_prompt)} characters")
            return diagnostic_prompt
            
        except Exception as e:
            logger.error(f"Error creating diagnostic prompt for GPT-4.1: {e}")
            # Fallback to basic prompt creation
            return self._create_fallback_diagnostic_prompt(processed_data, literature_context)
    
    def _create_fallback_diagnostic_prompt(self, processed_data: Dict, literature_context: Dict) -> str:
        """Fallback diagnostic prompt creation if main system fails."""
        patient_summary = processed_data.get("patient_summary", {})
        
        fallback_prompt = f"""
=== GPT-4.1 PATIENT DIAGNOSTIC ANALYSIS ===

Patient: {patient_summary.get('patient_id', 'Unknown')}
Demographics: {patient_summary.get('demographics', 'Not available')}

Active Conditions: {', '.join(patient_summary.get('primary_conditions', []))}
Current Medications: {', '.join(patient_summary.get('key_medications', []))}

Please provide a comprehensive diagnostic assessment including:
1. Primary diagnosis with confidence level
2. Differential diagnoses ranked by likelihood
3. Clinical reasoning and evidence
4. Risk stratification
5. Recommended diagnostic workup
6. Immediate management priorities
7. Follow-up care plan

Base your analysis on the patient data provided and use your advanced reasoning capabilities.
"""
        
        return fallback_prompt

    async def _execute_gpt4_diagnostic_analysis(self, diagnostic_prompt: str) -> str:
        """Execute diagnostic analysis using GPT-4.1."""
        try:
            logger.info("Executing comprehensive diagnostic analysis with GPT-4.1...")
            
            # Call GPT-4.1 for diagnostic analysis
            analysis_result = self.llm._call(diagnostic_prompt)
            
            logger.info(f"GPT-4.1 diagnostic analysis completed: {len(analysis_result)} characters")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error executing GPT-4.1 diagnostic analysis: {e}")
            raise RuntimeError(f"GPT-4.1 diagnostic analysis failed: {str(e)}")

    def _structure_diagnostic_response(self, diagnostic_analysis: str, processed_data: Dict) -> Dict:
        """Structure the GPT-4.1 diagnostic response with improved parsing for meaningful output."""
        try:
            # Initialize structured response
            structured_response = {
                "primary_diagnosis": "",
                "differential_diagnoses": [],
                "clinical_reasoning": "",
                "risk_stratification": "",
                "recommended_workup": [],
                "immediate_management": [],
                "follow_up_recommendations": [],
                "full_analysis": diagnostic_analysis
            }
            
            # Clean and normalize the text for better parsing
            cleaned_text = self._clean_gpt4_response(diagnostic_analysis)
            
            # Split into logical sections
            sections = self._split_into_sections(cleaned_text)
            
            # Extract each section with improved logic
            structured_response["primary_diagnosis"] = self._extract_primary_diagnosis_improved(sections.get("primary_diagnosis", []))
            structured_response["differential_diagnoses"] = self._extract_differential_diagnoses_improved(sections.get("differential_diagnoses", []))
            structured_response["clinical_reasoning"] = self._extract_clinical_reasoning_improved(sections.get("clinical_reasoning", []))
            structured_response["risk_stratification"] = self._extract_risk_stratification_improved(sections.get("risk_stratification", []))
            structured_response["recommended_workup"] = self._extract_workup_improved(sections.get("recommended_workup", []))
            structured_response["immediate_management"] = self._extract_management_improved(sections.get("immediate_management", []))
            structured_response["follow_up_recommendations"] = self._extract_followup_improved(sections.get("follow_up_recommendations", []))
            
            logger.info("GPT-4.1 diagnostic response parsed with improved accuracy")
            return structured_response
            
        except Exception as e:
            logger.error(f"Error in improved GPT-4.1 response parsing: {e}")
            # Fallback with better error handling
            return self._fallback_response_extraction(diagnostic_analysis, processed_data)

    def _clean_gpt4_response(self, text: str) -> str:
        """Clean GPT-4.1 response text for better parsing."""
        # Remove markdown formatting
        cleaned = text.replace('**', '').replace('*', '')
        # Normalize line breaks
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        # Remove extra whitespace
        lines = [line.strip() for line in cleaned.split('\n')]
        return '\n'.join(line for line in lines if line)

    def _split_into_sections(self, text: str) -> Dict[str, List[str]]:
        """Split GPT-4.1 response into logical sections with improved recognition."""
        sections = {
            "primary_diagnosis": [],
            "differential_diagnoses": [],
            "clinical_reasoning": [],
            "risk_stratification": [],
            "recommended_workup": [],
            "immediate_management": [],
            "follow_up_recommendations": []
        }
        
        current_section = None
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_upper = line.upper()
            
            # Improved section detection with multiple patterns
            section_detected = None
            
            if any(pattern in line_upper for pattern in ["PRIMARY DIAGNOSIS", "MOST LIKELY DIAGNOSIS", "MAIN DIAGNOSIS"]):
                section_detected = "primary_diagnosis"
            elif any(pattern in line_upper for pattern in ["DIFFERENTIAL DIAGNOS", "ALTERNATIVE DIAGNOS", "OTHER DIAGNOS"]):
                section_detected = "differential_diagnoses"
            elif any(pattern in line_upper for pattern in ["CLINICAL REASONING", "DIAGNOSTIC REASONING", "ANALYSIS", "THOUGHT PROCESS"]):
                section_detected = "clinical_reasoning"
            elif any(pattern in line_upper for pattern in ["RISK STRATIFICATION", "RISK ASSESSMENT", "PATIENT RISKS"]):
                section_detected = "risk_stratification"
            elif any(pattern in line_upper for pattern in ["DIAGNOSTIC WORKUP", "RECOMMENDED WORKUP", "TESTS NEEDED", "LABORATORY", "IMAGING"]):
                section_detected = "recommended_workup"
            elif any(pattern in line_upper for pattern in ["IMMEDIATE MANAGEMENT", "URGENT CARE", "IMMEDIATE CARE", "THERAPEUTIC"]):
                section_detected = "immediate_management"
            elif any(pattern in line_upper for pattern in ["FOLLOW-UP", "FOLLOW UP", "FOLLOWUP", "NEXT STEPS", "MONITORING"]):
                section_detected = "follow_up_recommendations"
            
            if section_detected:
                # Save previous section
                if current_section and current_content:
                    sections[current_section].extend(current_content)
                
                # Start new section
                current_section = section_detected
                current_content = []
                
                # Don't include the header line in content unless it has useful info
                if len(line) > 30 and not line.endswith(':'):
                    current_content.append(line)
            else:
                # Add content to current section
                if current_section:
                    current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            sections[current_section].extend(current_content)
        
        return sections

    def _extract_primary_diagnosis_improved(self, content: List[str]) -> str:
        """Extract primary diagnosis with better accuracy."""
        if not content:
            return "Primary diagnosis not identified in response"
        
        # Look for actual diagnosis content
        diagnosis_parts = []
        
        for line in content:
            # Skip empty lines and pure formatting
            if not line or line in [':', '-', '•', '*']:
                continue
            
            # Skip lines that are just section headers
            if any(header in line.upper() for header in ["PRIMARY DIAGNOSIS", "CONFIDENCE", "SUPPORTING"]):
                continue
            
            # Look for meaningful diagnostic content
            if len(line) > 15:  # Meaningful length
                diagnosis_parts.append(line)
        
        if diagnosis_parts:
            # Take the first substantial line as the primary diagnosis
            primary = diagnosis_parts[0]
            
            # Clean up common artifacts
            primary = primary.lstrip('- • * 1. 2. 3.')
            primary = primary.replace('Diagnosis:', '').replace('Primary:', '').strip()
            
            return primary
        
        return "Primary diagnosis not clearly identified"

    def _extract_differential_diagnoses_improved(self, content: List[str]) -> List[Dict]:
        """Extract differential diagnoses with improved structure."""
        differentials = []
        current_diagnosis = None
        
        for line in content:
            if not line:
                continue
            
            # Check if this line starts a new diagnosis
            if (line.startswith(('-', '•', '*')) or 
                any(char.isdigit() and char in line[:5] for char in "12345") or
                (len(line) > 20 and line[0].isupper() and ':' not in line[:10])):
                
                # Save previous diagnosis
                if current_diagnosis and current_diagnosis.get("diagnosis"):
                    differentials.append(current_diagnosis)
                
                # Start new diagnosis
                diagnosis_text = line.lstrip('- • * 1234567890. ')
                
                # Extract likelihood percentage if present
                likelihood = "unknown"
                if '%' in diagnosis_text:
                    import re
                    percentage_match = re.search(r'(\d+%)', diagnosis_text)
                    if percentage_match:
                        likelihood = percentage_match.group(1)
                        diagnosis_text = diagnosis_text.replace(likelihood, '').strip()
                
                current_diagnosis = {
                    "diagnosis": diagnosis_text.strip(),
                    "evidence": "",
                    "likelihood": likelihood
                }
            
            elif current_diagnosis:
                # Add supporting information to current diagnosis
                if any(keyword in line.lower() for keyword in ["supporting", "evidence", "factor", "suggests", "indicates"]):
                    if current_diagnosis["evidence"]:
                        current_diagnosis["evidence"] += " " + line
                    else:
                        current_diagnosis["evidence"] = line
                elif "likelihood" in line.lower() and '%' in line:
                    import re
                    percentage_match = re.search(r'(\d+%)', line)
                    if percentage_match:
                        current_diagnosis["likelihood"] = percentage_match.group(1)
        
        # Add the last diagnosis
        if current_diagnosis and current_diagnosis.get("diagnosis"):
            differentials.append(current_diagnosis)
        
        return differentials[:5]  # Limit to 5 most relevant

    def _extract_clinical_reasoning_improved(self, content: List[str]) -> str:
        """Extract clinical reasoning with better formatting."""
        if not content:
            return "Clinical reasoning not provided"
        
        # Clean and format the reasoning
        reasoning_parts = []
        
        for line in content:
            if line and len(line) > 10:  # Meaningful content
                reasoning_parts.append(line)
        
        if reasoning_parts:
            # Join with proper formatting
            reasoning = '\n'.join(reasoning_parts)
            return reasoning
        
        return "Clinical reasoning not clearly provided"

    def _extract_risk_stratification_improved(self, content: List[str]) -> str:
        """Extract risk stratification with better structure."""
        if not content:
            return "Risk stratification not provided"
        
        risk_parts = []
        
        for line in content:
            if line and len(line) > 10:
                risk_parts.append(line)
        
        if risk_parts:
            return '\n'.join(risk_parts)
        
        return "Risk assessment not clearly provided"

    def _extract_workup_improved(self, content: List[str]) -> List[str]:
        """Extract recommended workup with better parsing."""
        workup_items = []
        current_item = ""
        
        for line in content:
            if not line:
                continue
            
            # Check if this starts a new workup item
            if (line.startswith(('-', '•', '*')) or 
                any(char.isdigit() and char in line[:5] for char in "12345") or
                line.endswith(':') or
                any(keyword in line.lower() for keyword in ["laboratory", "imaging", "test", "study", "exam"])):
                
                # Save previous item
                if current_item and len(current_item) > 15:
                    workup_items.append(current_item.strip())
                
                # Start new item
                current_item = line.lstrip('- • * 1234567890. ')
            else:
                # Continue current item
                if current_item:
                    current_item += " " + line
                else:
                    current_item = line
        
        # Add the last item
        if current_item and len(current_item) > 15:
            workup_items.append(current_item.strip())
        
        return workup_items[:8]  # Limit to 8 items

    def _extract_management_improved(self, content: List[str]) -> List[str]:
        """Extract immediate management with better parsing."""
        management_items = []
        current_item = ""
        
        for line in content:
            if not line:
                continue
            
            # Check if this starts a new management item
            if (line.startswith(('-', '•', '*')) or 
                any(char.isdigit() and char in line[:5] for char in "12345") or
                line.endswith(':') or
                any(keyword in line.lower() for keyword in ["monitor", "administer", "discontinue", "start", "stop", "assess"])):
                
                # Save previous item
                if current_item and len(current_item) > 15:
                    management_items.append(current_item.strip())
                
                # Start new item
                current_item = line.lstrip('- • * 1234567890. ')
            else:
                # Continue current item
                if current_item:
                    current_item += " " + line
                else:
                    current_item = line
        
        # Add the last item
        if current_item and len(current_item) > 15:
            management_items.append(current_item.strip())
        
        return management_items[:8]  # Limit to 8 items

    def _extract_followup_improved(self, content: List[str]) -> List[str]:
        """Extract follow-up recommendations with better parsing."""
        followup_items = []
        current_item = ""
        
        for line in content:
            if not line:
                continue
            
            # Check if this starts a new follow-up item
            if (line.startswith(('-', '•', '*')) or 
                any(char.isdigit() and char in line[:5] for char in "12345") or
                line.endswith(':') or
                any(keyword in line.lower() for keyword in ["referral", "appointment", "follow", "return", "schedule", "monitor"])):
                
                # Save previous item
                if current_item and len(current_item) > 15:
                    followup_items.append(current_item.strip())
                
                # Start new item
                current_item = line.lstrip('- • * 1234567890. ')
            else:
                # Continue current item
                if current_item:
                    current_item += " " + line
                else:
                    current_item = line
        
        # Add the last item
        if current_item and len(current_item) > 15:
            followup_items.append(current_item.strip())
        
        return followup_items[:8]  # Limit to 8 items

    def _fallback_response_extraction(self, diagnostic_analysis: str, processed_data: Dict) -> Dict:
        """Fallback extraction method when primary parsing fails."""
        return {
            "primary_diagnosis": "Unable to parse primary diagnosis - see full analysis",
            "differential_diagnoses": [
                {
                    "diagnosis": "Parsing error - refer to full analysis",
                    "evidence": "Unable to extract structured differential diagnoses",
                    "likelihood": "unknown"
                }
            ],
            "clinical_reasoning": diagnostic_analysis[:1000] + "..." if len(diagnostic_analysis) > 1000 else diagnostic_analysis,
            "risk_stratification": "Unable to parse risk stratification - see full analysis",
            "recommended_workup": ["Unable to parse workup recommendations - see full analysis"],
            "immediate_management": ["Unable to parse management recommendations - see full analysis"],
            "follow_up_recommendations": ["Unable to parse follow-up recommendations - see full analysis"],
            "full_analysis": diagnostic_analysis,
            "parsing_note": "Automated parsing failed - full analysis available in 'full_analysis' field"
        }

    def _extract_primary_diagnosis(self, content_lines: List[str]) -> str:
        """Extract primary diagnosis from content lines."""
        # Look for the actual diagnosis content
        diagnosis_parts = []
        for line in content_lines:
            if line and not line.startswith('-') and not line.startswith('•'):
                diagnosis_parts.append(line)
        
        return " ".join(diagnosis_parts) if diagnosis_parts else "Primary diagnosis not clearly identified"

    def _parse_gpt4_differential_diagnoses(self, content_lines: List[str]) -> List[Dict]:
        """Parse differential diagnoses from GPT-4.1 content lines with better formatting."""
        differentials = []
        current_dx = None
        
        for line in content_lines:
            line = line.strip()
            
            # Look for numbered items or diagnosis names
            if (line.startswith('-') or line.startswith('•') or 
                any(char.isdigit() and char in line[:3] for char in line[:3]) or
                (len(line) > 10 and line[0].isupper())):
                
                # Save previous diagnosis
                if current_dx and current_dx["diagnosis"]:
                    differentials.append(current_dx)
                
                # Start new diagnosis
                diagnosis_text = line.lstrip('-•0123456789. ')
                
                # Extract likelihood if present
                likelihood = "unknown"
                if "%" in diagnosis_text:
                    parts = diagnosis_text.split()
                    for part in parts:
                        if "%" in part:
                            likelihood = part
                            diagnosis_text = diagnosis_text.replace(part, "").strip()
                            break
                
                current_dx = {
                    "diagnosis": diagnosis_text,
                    "evidence": "",
                    "likelihood": likelihood
                }
            elif current_dx and line:
                # Add to current diagnosis evidence
                if "supporting" in line.lower() or "evidence" in line.lower():
                    current_dx["evidence"] += " " + line
                elif "likelihood" in line.lower() and "%" in line:
                    # Extract likelihood
                    parts = line.split()
                    for part in parts:
                        if "%" in part:
                            current_dx["likelihood"] = part
                            break
        
        # Add the last diagnosis
        if current_dx and current_dx["diagnosis"]:
            differentials.append(current_dx)
        
        return differentials[:5]  # Limit to 5 differential diagnoses

    def _parse_gpt4_recommendations(self, content_lines: List[str]) -> List[str]:
        """Parse recommendations from GPT-4.1 content lines with better handling."""
        recommendations = []
        current_rec = ""
        
        for line in content_lines:
            line = line.strip()
            
            if not line:
                continue
                
            # Check if this is a new recommendation item
            if (line.startswith('-') or line.startswith('•') or 
                any(char.isdigit() and char in line[:3] for char in line[:3]) or
                line.endswith(':') and len(line) < 50):
                
                # Save previous recommendation
                if current_rec:
                    clean_rec = current_rec.strip().lstrip('-•0123456789. ')
                    if len(clean_rec) > 10:
                        recommendations.append(clean_rec)
                
                # Start new recommendation
                current_rec = line
            else:
                # Continue current recommendation
                if current_rec:
                    current_rec += " " + line
                else:
                    current_rec = line
        
        # Add the last recommendation
        if current_rec:
            clean_rec = current_rec.strip().lstrip('-•0123456789. ')
            if len(clean_rec) > 10:
                recommendations.append(clean_rec)
        
        return recommendations[:8]  # Limit to 8 recommendations

    def _parse_differential_diagnoses(self, content_lines: List[str]) -> List[Dict]:
        """Parse differential diagnoses from GPT-4.1 content lines."""
        differentials = []
        current_dx = None
        
        for line in content_lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*') or any(char.isdigit() and char in line[:3] for char in line[:3]):
                # New differential diagnosis
                if current_dx:
                    differentials.append(current_dx)
                
                current_dx = {
                    "diagnosis": line.lstrip('-•*0123456789. '),
                    "evidence": "",
                    "likelihood": "unknown"
                }
            elif current_dx and line:
                # Add to current diagnosis description
                current_dx["evidence"] += " " + line
        
        if current_dx:
            differentials.append(current_dx)
        
        return differentials[:5]  # Limit to 5 differential diagnoses

    def _parse_recommendations(self, content_lines: List[str]) -> List[str]:
        """Parse recommendations from GPT-4.1 content lines."""
        recommendations = []
        
        for line in content_lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or any(char.isdigit() and char in line[:3] for char in line[:3])):
                clean_rec = line.lstrip('-•*0123456789. ')
                if len(clean_rec) > 10:  # Only meaningful recommendations
                    recommendations.append(clean_rec)
        
        return recommendations[:8]  # Limit to 8 recommendations

    def _calculate_diagnostic_confidence(self, diagnostic_analysis: str, processed_data: Dict) -> float:
        """Calculate confidence score for the GPT-4.1 diagnostic analysis."""
        try:
            confidence_factors = []
            
            # Factor 1: Data completeness
            data_completeness = processed_data.get("data_statistics", {}).get("data_completeness", 0.5)
            confidence_factors.append(data_completeness * 0.3)
            
            # Factor 2: Analysis detail and structure (GPT-4.1 typically provides detailed responses)
            analysis_length = len(diagnostic_analysis)
            length_factor = min(analysis_length / 2500, 1.0)  # Normalize to 2500 chars for GPT-4.1
            confidence_factors.append(length_factor * 0.2)
            
            # Factor 3: Number of data points available
            total_diagnoses = processed_data.get("data_statistics", {}).get("total_diagnoses", 0)
            total_medications = processed_data.get("data_statistics", {}).get("total_medications", 0)
            total_visits = processed_data.get("data_statistics", {}).get("total_visits", 0)
            
            data_richness = min((total_diagnoses + total_medications + total_visits) / 20, 1.0)
            confidence_factors.append(data_richness * 0.2)
            
            # Factor 4: Clinical patterns identified
            patterns = processed_data.get("clinical_patterns", {})
            pattern_score = 0
            if patterns.get("recurring_symptoms"):
                pattern_score += 0.1
            if patterns.get("visit_frequency"):
                pattern_score += 0.1
            if patterns.get("medication_patterns"):
                pattern_score += 0.1
            confidence_factors.append(pattern_score)
            
            # Factor 5: GPT-4.1 quality bonus (high-quality model)
            gpt4_bonus = 0.15  # Higher than Mistral due to GPT-4.1's advanced capabilities
            confidence_factors.append(gpt4_bonus)
            
            # Calculate final confidence
            final_confidence = sum(confidence_factors)
            return min(max(final_confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95
            
        except Exception as e:
            logger.error(f"Error calculating GPT-4.1 diagnostic confidence: {e}")
            return 0.7  # Default higher confidence for GPT-4.1

    # Legacy method for backward compatibility
    async def generate_recommendations(
        self,
        patient_data: Dict,
        visit_type: str = "comprehensive_analysis",
        symptoms: str = "comprehensive_history_review",
        additional_context: Optional[str] = None
    ) -> Dict:
        """
        Legacy method for backward compatibility.
        Redirects to the new GPT-4.1 analyze_patient_history method.
        """
        logger.info("Legacy generate_recommendations called - redirecting to GPT-4.1 analyze_patient_history")
        
        try:
            # Call the new comprehensive analysis method
            analysis_result = await self.analyze_patient_history(patient_data)
            
            # Convert to legacy format for compatibility
            diagnostic_assessment = analysis_result.get("diagnostic_assessment", {})
            
            # Extract recommendations from diagnostic assessment
            recommendations = []
            
            if diagnostic_assessment.get("primary_diagnosis"):
                recommendations.append(f"PRIMARY DIAGNOSIS: {diagnostic_assessment['primary_diagnosis']}")
            
            for i, diff_dx in enumerate(diagnostic_assessment.get("differential_diagnoses", [])[:3], 1):
                recommendations.append(f"DIFFERENTIAL {i}: {diff_dx.get('diagnosis', 'Unknown')}")
            
            for rec in diagnostic_assessment.get("immediate_management", [])[:3]:
                recommendations.append(f"MANAGEMENT: {rec}")
            
            for rec in diagnostic_assessment.get("recommended_workup", [])[:3]:
                recommendations.append(f"WORKUP: {rec}")
            
            # Return in legacy format
            return {
                "suggestions": recommendations,
                "confidence": analysis_result.get("confidence_score", 0.7),
                "sources": analysis_result.get("literature_sources", []),
                "patient_context": analysis_result.get("patient_summary", {}),
                "model_used": analysis_result.get("model_used", "GPT-4.1 via GitHub AI"),
                "api_provider": "GitHub AI (OpenAI GPT-4.1)",
                "diagnostic_analysis": analysis_result.get("diagnostic_assessment"),
                "comprehensive_analysis": analysis_result
            }
            
        except Exception as e:
            logger.error(f"Error in legacy GPT-4.1 generate_recommendations: {e}")
            return {
                "suggestions": [f"Error in GPT-4.1 diagnostic analysis: {str(e)}"],
                "confidence": 0.0,
                "sources": [],
                "patient_context": {},
                "error": str(e),
                "api_provider": "GitHub AI (GPT-4.1 Error)"
            }