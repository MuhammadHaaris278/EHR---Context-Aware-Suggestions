"""
Enhanced AI pipeline for clinical recommendations using Mistral AI API.
FIXED VERSION - Uses Mistral AI's official API for medical applications.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
import logging
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field, PrivateAttr
import json

from .retriever import ClinicalRetriever
from .prompt import ClinicalPromptTemplate

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class MistralMedicalLLM(LLM):
    """
    Medical LLM using Mistral AI's official API.
    Optimized for clinical decision support and medical reasoning.
    """
    
    # Define fields using Pydantic Field for proper LangChain compatibility
    model_name: str = Field(default="mistral-large-latest")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1024)
    top_p: float = Field(default=0.95)
    api_key: Optional[str] = Field(default=None)
    
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
        return "mistral_medical_llm"
    
    def _setup_client(self):
        """Setup the Mistral AI client."""
        try:
            # Get API key from instance or environment
            api_key = self.api_key or os.environ.get("MISTRAL_API_KEY")
            
            if not api_key:
                logger.error("No Mistral AI API key found. Set MISTRAL_API_KEY environment variable.")
                logger.info("Get your API key from: https://console.mistral.ai/")
                logger.info("1. Create account at console.mistral.ai")
                logger.info("2. Add billing information") 
                logger.info("3. Generate API key in 'API Keys' section")
                self.__dict__['_client'] = None
                self.__dict__['_client_initialized'] = False
                return
            
            # Import and setup Mistral AI client
            try:
                from mistralai import Mistral
                logger.info("Mistral AI SDK available")
            except ImportError:
                logger.error("Mistral AI SDK not installed. Install with: pip install mistralai")
                self.__dict__['_client'] = None
                self.__dict__['_client_initialized'] = False
                return
            
            # Create Mistral client
            client = Mistral(api_key=api_key)
            
            self.__dict__['_client'] = client
            self.__dict__['_client_initialized'] = True
            logger.info(f"Mistral AI Medical LLM client initialized successfully")
            
            # Test the connection
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Error setting up Mistral AI client: {e}")
            self.__dict__['_client'] = None
            self.__dict__['_client_initialized'] = False
    
    def _test_connection(self):
        """Test the Mistral AI connection with a medical query."""
        try:
            logger.info("Testing Mistral AI connection with medical query...")
            
            # Test with a simple medical question
            test_response = self.__dict__['_client'].chat.complete(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant. Provide brief, accurate medical information."
                    },
                    {
                        "role": "user",
                        "content": "What is hypertension?"
                    }
                ],
                max_tokens=100,
                temperature=0.1,
            )
            
            # Check response
            if hasattr(test_response, 'choices') and len(test_response.choices) > 0:
                response_text = test_response.choices[0].message.content
                logger.info(f"✅ Mistral AI connection successful: {response_text[:100]}...")
                return True
            else:
                logger.error("Unexpected response format from Mistral AI")
                self.__dict__['_client_initialized'] = False
                return False
                
        except Exception as e:
            logger.error(f"Mistral AI connection test failed: {e}")
            
            # Provide specific error guidance
            error_str = str(e).lower()
            if "unauthorized" in error_str or "401" in error_str:
                logger.error("❌ Authentication failed. Please check your MISTRAL_API_KEY")
                logger.info("💡 Ensure you have:")
                logger.info("   1. Valid API key from console.mistral.ai")
                logger.info("   2. Billing information added to your account")
                logger.info("   3. API key is correctly set in environment")
            elif "payment" in error_str or "billing" in error_str:
                logger.error("❌ Billing issue. Please add payment method to your Mistral account")
                logger.info("💡 Go to https://console.mistral.ai/billing/ to add billing info")
            elif "rate limit" in error_str or "429" in error_str:
                logger.error("❌ Rate limit exceeded. Please try again later")
            elif "model not found" in error_str or "404" in error_str:
                logger.error(f"❌ Model {self.model_name} not found")
                logger.info("💡 Trying alternative Mistral models...")
                return self._try_alternative_mistral_models()
            else:
                logger.error(f"❌ Unexpected error: {e}")
            
            self.__dict__['_client_initialized'] = False
            return False
    
    def _try_alternative_mistral_models(self) -> bool:
        """Try alternative Mistral models that might be available."""
        alternative_models = [
            "mistral-large-latest",
            "mistral-medium-latest", 
            "mistral-small-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b"
        ]
        
        client = self.__dict__.get('_client', None)
        if not client:
            return False
        
        for model in alternative_models:
            try:
                logger.info(f"🧪 Trying Mistral model: {model}")
                
                test_response = client.chat.complete(
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
                
                if hasattr(test_response, 'choices') and len(test_response.choices) > 0:
                    response_text = test_response.choices[0].message.content
                    logger.info(f"✅ Mistral model {model} successful")
                    self.model_name = model  # Update to working model
                    return True
                
            except Exception as e:
                logger.warning(f"❌ Mistral model {model} failed: {e}")
                continue
        
        logger.info("🔄 All Mistral models failed")
        return False
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Make a call to Mistral AI API for medical consultation."""
        client_initialized = self.__dict__.get('_client_initialized', False)
        client = self.__dict__.get('_client', None)
        
        if not client_initialized or client is None:
            logger.warning("Mistral AI client not available, using fallback medical response")
            return self._medical_mock_response(prompt)
        
        try:
            logger.info(f"Calling Mistral AI with model: {self.model_name}")
            
            # Create medical-focused system prompt
            medical_system_prompt = """You are an advanced clinical decision support AI assistant specialized in evidence-based medicine. Your responses should be:

1. EVIDENCE-BASED: Grounded in current medical guidelines and best practices
2. PATIENT-SAFE: Always prioritize patient safety and consider contraindications
3. SPECIFIC: Provide actionable, numbered clinical recommendations (1-6 items)
4. COMPREHENSIVE: Address diagnosis, treatment, monitoring, and follow-up
5. CONTEXTUAL: Consider the patient's complete medical profile

Format your response as numbered recommendations that are immediately actionable by healthcare providers. Always consider drug interactions, allergies, and comorbidities."""

            # Make the API call
            response = client.chat.complete(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": medical_system_prompt
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
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                logger.info(f"✅ Mistral AI medical response generated: {len(response_text)} characters")
                return response_text
            else:
                logger.error("Unexpected response format from Mistral AI")
                return self._medical_mock_response(prompt)
            
        except Exception as e:
            logger.error(f"Error calling Mistral AI: {e}")
            
            # Enhanced error handling
            error_str = str(e).lower()
            if "unauthorized" in error_str or "401" in error_str:
                logger.error("❌ Mistral AI authentication failed")
                logger.info("💡 Check your MISTRAL_API_KEY and billing status")
            elif "rate limit" in error_str or "429" in error_str:
                logger.error("❌ Mistral AI rate limit exceeded")
                logger.info("💡 Consider upgrading your Mistral AI plan")
            elif "payment" in error_str or "billing" in error_str:
                logger.error("❌ Mistral AI billing issue")
                logger.info("💡 Add payment method at console.mistral.ai/billing/")
            elif "model" in error_str and "not found" in error_str:
                logger.error(f"❌ Mistral model {self.model_name} not accessible")
                logger.info("💡 Trying alternative model...")
                if self._try_alternative_mistral_models():
                    # Retry with new model
                    return self._call(prompt, stop, run_manager, **kwargs)
            else:
                logger.error(f"❌ Unexpected Mistral AI error: {e}")
            
            logger.info("🔄 Falling back to medical mock response")
            return self._medical_mock_response(prompt)
    
    def _medical_mock_response(self, prompt: str) -> str:
        """Generate enhanced medical mock responses for fallback scenarios."""
        prompt_lower = prompt.lower()
        
        # Clear indicator this is a mock response
        mock_header = "🤖 [MEDICAL MOCK RESPONSE - Mistral AI unavailable] "
        
        # Enhanced medical responses based on clinical keywords
        if any(term in prompt_lower for term in ["chest pain", "cardiac", "heart", "coronary"]):
            return mock_header + """Based on the cardiac presentation, here are the clinical recommendations:

1. Obtain immediate 12-lead ECG within 10 minutes to evaluate for STEMI or other acute changes
2. Establish continuous cardiac monitoring and assess vital signs including oxygen saturation
3. Draw cardiac biomarkers (high-sensitivity troponin) and complete metabolic panel
4. Administer aspirin 325mg unless contraindicated, and consider nitroglycerin for chest pain
5. Perform chest X-ray to evaluate for cardiomegaly, pulmonary edema, or pneumothorax  
6. Arrange urgent cardiology consultation for risk stratification and further management"""
        
        elif any(term in prompt_lower for term in ["diabetes", "hyperglycemia", "insulin", "glucose"]):
            return mock_header + """Based on the diabetes-related presentation, here are the clinical recommendations:

1. Check current blood glucose, HbA1c, and review recent glucose monitoring patterns
2. Assess for diabetic ketoacidosis (DKA) if glucose >250 mg/dL with ketones and acidosis
3. Evaluate for diabetic complications: retinopathy, nephropathy, neuropathy, cardiovascular disease
4. Review diabetes medications for adherence, effectiveness, and potential dose adjustments
5. Provide comprehensive diabetes education on glucose monitoring, carb counting, and sick day management
6. Schedule appropriate diabetes care: ophthalmology (annual), podiatry, and endocrinology as needed"""
        
        elif any(term in prompt_lower for term in ["hypertension", "blood pressure", "bp", "antihypertensive"]):
            return mock_header + """Based on the hypertension presentation, here are the clinical recommendations:

1. Confirm elevated BP with multiple readings using proper technique and appropriate cuff size
2. Evaluate for hypertensive emergency (BP >180/120 with end-organ damage) requiring immediate treatment
3. Assess for secondary causes: sleep apnea, renal disease, endocrine disorders, medication-induced
4. Review current antihypertensive regimen for effectiveness, adherence, and drug interactions
5. Implement lifestyle modifications: sodium restriction (<2g/day), weight management, regular exercise
6. Monitor for target organ damage with ECG, urinalysis, and consider echocardiogram if indicated"""
        
        elif any(term in prompt_lower for term in ["infection", "sepsis", "fever", "antibiotic"]):
            return mock_header + """Based on the infectious presentation, here are the clinical recommendations:

1. Assess for sepsis using qSOFA criteria and obtain blood cultures before antibiotics
2. Identify likely source of infection with targeted history, physical exam, and imaging
3. Initiate empiric broad-spectrum antibiotics within 1 hour if sepsis suspected
4. Obtain appropriate cultures (blood, urine, sputum, wound) before antibiotic administration
5. Provide supportive care: IV fluids, fever control, and monitor for organ dysfunction
6. De-escalate antibiotics based on culture results and consider infectious disease consultation"""
        
        elif any(term in prompt_lower for term in ["shortness of breath", "dyspnea", "respiratory", "copd", "asthma"]):
            return mock_header + """Based on the respiratory presentation, here are the clinical recommendations:

1. Assess airway, breathing, circulation (ABCs) and provide supplemental oxygen if SpO2 <90%
2. Obtain chest X-ray, ABG or VBG, and peak flow measurement if asthma suspected
3. Consider pulmonary embolism if sudden onset with risk factors - obtain D-dimer or CTPA
4. Administer bronchodilators (albuterol/ipratropium) if obstructive disease suspected
5. Evaluate for heart failure with BNP/pro-BNP, ECG, and echocardiogram if indicated
6. Consider corticosteroids for COPD exacerbation or severe asthma, monitor response to treatment"""
        
        else:
            return mock_header + """Based on the clinical presentation, here are the general recommendations:

1. Perform systematic assessment using ABCDE approach (Airway, Breathing, Circulation, Disability, Exposure)
2. Obtain focused history including chief complaint, HPI, past medical history, medications, and allergies
3. Conduct targeted physical examination based on chief complaint and differential diagnosis
4. Order appropriate diagnostic studies guided by clinical suspicion and evidence-based guidelines
5. Implement patient-centered treatment plan considering comorbidities, contraindications, and preferences
6. Establish monitoring plan, provide patient education, and arrange appropriate follow-up care"""

class ClinicalRecommendationPipeline:
    """
    Enhanced pipeline for generating clinical recommendations using Mistral AI.
    Optimized for medical decision support with proper error handling.
    """

    def __init__(self):
        self.retriever = None
        self.llm = None
        self.prompt_template = None
        self.qa_chain = None
        self.initialized = False

    async def initialize(self):
        """Initialize the clinical recommendation pipeline with Mistral AI."""
        try:
            logger.info("Initializing Mistral AI clinical recommendation pipeline...")
            
            # Initialize retriever
            self.retriever = ClinicalRetriever()
            await self.retriever.initialize()
            
            # Initialize Mistral AI LLM
            await self._initialize_mistral_llm()
            
            # Initialize prompt template
            self.prompt_template = ClinicalPromptTemplate()
            
            # Create QA chain
            self._create_qa_chain()
            
            self.initialized = True
            logger.info("Mistral AI clinical pipeline initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Mistral AI clinical pipeline: {e}")
            raise

    async def _initialize_mistral_llm(self):
        """Initialize the Mistral AI LLM with proper configuration."""
        try:
            logger.info("Initializing Mistral AI Medical LLM...")
            
            # Get token from environment
            token = os.environ.get("MISTRAL_API_KEY")
            
            if not token:
                logger.warning("MISTRAL_API_KEY not found in environment variables")
                logger.warning("Please set your Mistral AI API key to use advanced medical reasoning")
                logger.warning("Get your API key from: https://console.mistral.ai/")
                logger.warning("Using fallback mock LLM for testing")
                self.llm = MistralMedicalLLM()  # Will use mock responses
                return
            
            # Test if we can import required modules
            try:
                from mistralai import Mistral
                logger.info("Mistral AI SDK available")
            except ImportError:
                logger.error("Mistral AI SDK not installed. Install with: pip install mistralai")
                self.llm = MistralMedicalLLM()  # Will use mock responses
                return
            
            # Create Mistral Medical LLM with optimal configuration for healthcare
            self.llm = MistralMedicalLLM(
                model_name="mistral-large-latest",  # Best model for complex medical reasoning
                temperature=0.1,  # Low temperature for consistent medical advice
                max_tokens=1024,
                top_p=0.95,
                api_key=token,
            )
            
            # Test the LLM with a medical question
            test_prompt = """Patient presents with chest pain and shortness of breath. 
            Patient has history of hypertension and diabetes. What are the key clinical considerations?"""
            
            try:
                test_response = self.llm._call(test_prompt)
                if "mock" not in test_response.lower():
                    logger.info("✅ Mistral AI Medical LLM initialized and tested successfully")
                    logger.info(f"✅ Using model: {self.llm.model_name}")
                else:
                    logger.warning("⚠️ Mistral AI LLM fell back to mock response - API may be unavailable")
            except Exception as e:
                logger.error(f"Mistral AI LLM test call failed: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing Mistral AI LLM: {e}")
            # Fallback to mock LLM
            self.llm = MistralMedicalLLM()

    def _create_qa_chain(self):
        """Create the QA chain with Mistral AI LLM."""
        try:
            if not self.llm:
                raise ValueError("Mistral AI LLM not initialized")
            
            if not self.retriever:
                raise ValueError("Retriever not initialized")
            
            if not self.prompt_template:
                raise ValueError("Prompt template not initialized")
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever.get_retriever(),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_template.get_prompt()}
            )
            logger.info("Mistral AI medical QA chain created successfully")
            
        except Exception as e:
            logger.error(f"Error creating Mistral AI QA chain: {e}")
            raise

    async def generate_recommendations(
        self,
        patient_data: Dict,
        visit_type: str,
        symptoms: str,
        additional_context: Optional[str] = None
    ) -> Dict:
        """Generate clinical recommendations using Mistral AI with enhanced patient context."""
        try:
            if not self.initialized:
                raise RuntimeError("Mistral AI pipeline not initialized")

            # Create enhanced medical query with patient context
            enhanced_query = self._create_enhanced_medical_query(
                patient_data, visit_type, symptoms, additional_context
            )
            
            # Log the enhanced query for debugging
            logger.info(f"Enhanced Mistral AI query created: {len(enhanced_query)} characters")
            logger.debug(f"Mistral query preview: {enhanced_query[:500]}...")
            
            # Execute chain with Mistral AI
            result = await self._execute_mistral_chain(enhanced_query, patient_data)
            
            # Process recommendations with medical focus
            recommendations = self._process_medical_recommendations(result, patient_data)

            return {
                "suggestions": recommendations,
                "confidence": self._calculate_medical_confidence(result, patient_data),
                "sources": self._extract_sources(result),
                "patient_context": self._get_patient_summary(patient_data),
                "query_used": enhanced_query[:200] + "..." if len(enhanced_query) > 200 else enhanced_query,
                "model_used": f"Mistral AI - {self.llm.model_name}" if self.llm else "Unknown",
                "api_provider": "Mistral AI"
            }
            
        except Exception as e:
            logger.error(f"Error generating Mistral AI recommendations: {e}")
            return {
                "suggestions": [
                    "Error generating Mistral AI recommendations. Please verify patient data and API configuration.",
                    "Ensure Mistral AI API key is properly configured and billing is active.",
                    "Consider manual clinical review while troubleshooting the AI system."
                ],
                "confidence": 0.0,
                "sources": [],
                "patient_context": {},
                "error": str(e),
                "api_provider": "Mistral AI (Error)"
            }

    def _create_enhanced_medical_query(
        self, 
        patient_data: Dict, 
        visit_type: str, 
        symptoms: str, 
        additional_context: Optional[str] = None
    ) -> str:
        """Create enhanced medical query optimized for Mistral AI's reasoning capabilities."""
        try:
            # Get comprehensive patient summary
            patient_summary = self._get_patient_summary(patient_data)
            
            # Build medical reasoning query for Mistral AI
            query_parts = [
                "=== CLINICAL CONSULTATION REQUEST ===",
                "You are providing clinical decision support for a real patient case.",
                "Please analyze the complete clinical picture and provide evidence-based recommendations.",
                "",
                "=== PATIENT CLINICAL PROFILE ===",
                f"Patient Demographics: {patient_summary.get('demographics', 'Not specified')}",
                "",
                f"Active Medical Conditions: {self._format_medical_list(patient_summary.get('active_diagnoses', []))}",
                "",
                f"Current Medications & Dosages: {self._format_medical_list(patient_summary.get('current_medications', []))}",
                "",
                f"Relevant Medical History: {self._format_medical_list(patient_summary.get('recent_visits', []))}",
                "",
                f"Risk Factors & Comorbidities: {self._format_medical_list(patient_summary.get('risk_factors', []))}",
                "",
                "=== CURRENT CLINICAL ENCOUNTER ===",
                f"Type of Visit: {visit_type}",
                f"Chief Complaint: {symptoms}",
            ]
            
            if additional_context:
                query_parts.extend([
                    f"Additional Clinical Information: {additional_context}",
                ])
            
            query_parts.extend([
                "",
                "=== CLINICAL DECISION SUPPORT REQUEST ===",
                "Please provide 4-6 specific, evidence-based clinical recommendations that address:",
                "",
                "🔍 DIAGNOSTIC ASSESSMENT:",
                "- What immediate diagnostic workup is indicated?",
                "- Are there any urgent conditions to rule out?",
                "",
                "💊 THERAPEUTIC MANAGEMENT:", 
                "- What treatment approaches are most appropriate?",
                "- How do comorbidities affect treatment selection?",
                "",
                "⚠️ SAFETY & CONTRAINDICATIONS:",
                "- What drug interactions or contraindications must be considered?",
                "- What monitoring is essential for patient safety?",
                "",
                "🔄 FOLLOW-UP & COORDINATION:",
                "- What specialist referrals are indicated?",
                "- What is the appropriate follow-up timeline?",
                "",
                "📚 PATIENT EDUCATION:",
                "- What key information should be provided to the patient?",
                "- What warning signs should they watch for?",
                "",
                "RESPONSE FORMAT:",
                "Please provide exactly 4-6 numbered recommendations (1., 2., 3., etc.) that are:",
                "✓ Specific and actionable for immediate clinical implementation",
                "✓ Evidence-based and guideline-concordant", 
                "✓ Safe and appropriate for this specific patient profile",
                "✓ Considerate of all medications, allergies, and comorbidities",
                "",
                "Begin each recommendation with a clear action verb and be specific about:",
                "- Exact diagnostic tests or procedures",
                "- Specific medications with dosages when appropriate",
                "- Timeline for follow-up or monitoring",
                "- Specific parameters to monitor"
            ])
            
            enhanced_query = "\n".join(query_parts)
            logger.info(f"Enhanced Mistral AI medical query created: {len(enhanced_query)} characters")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error creating enhanced Mistral AI query: {e}")
            return f"Medical consultation for patient with {symptoms} during {visit_type} visit. Patient ID: {patient_data.get('patient_id', 'Unknown')}"

    def _format_medical_list(self, items: List[str]) -> str:
        """Format medical information for optimal Mistral AI processing."""
        if not items:
            return "None documented"
        if len(items) == 1:
            return items[0]
        # Use bullet points for better structure
        if len(items) <= 3:
            return " | ".join(items)
        else:
            return " | ".join(items[:6]) + f" (and {len(items)-6} more)" if len(items) > 6 else " | ".join(items)

    async def _execute_mistral_chain(self, query: str, patient_data: Dict) -> Dict:
        """Execute the QA chain with Mistral AI."""
        try:
            logger.info(f"Executing Mistral AI medical QA chain...")
            
            # Execute the chain
            result = self.qa_chain.invoke({"query": query})
            
            # Add metadata
            result["patient_context"] = self._get_patient_summary(patient_data)
            result["model_info"] = {
                "provider": "Mistral AI",
                "model": self.llm.model_name if self.llm else "Unknown",
                "temperature": self.llm.temperature if self.llm else 0.1
            }
            
            logger.info("Mistral AI medical QA chain execution completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error executing Mistral AI chain: {e}")
            return await self._mistral_fallback_call(query, patient_data)

    async def _mistral_fallback_call(self, query: str, patient_data: Dict) -> Dict:
        """Enhanced fallback method using direct Mistral AI call."""
        try:
            logger.info("Using Mistral AI fallback direct call")
            
            # Get retrieved medical documents
            retrieved_docs = await self.retriever.search(query, k=5)
            
            # Format context for Mistral AI
            context_parts = []
            
            # Add medical literature context
            if retrieved_docs:
                context_parts.append("=== RELEVANT MEDICAL LITERATURE ===")
                for i, doc in enumerate(retrieved_docs, 1):
                    context_parts.append(f"Source {i}: {doc.page_content[:400]}...")
                context_parts.append("")
            
            # Combine context and query
            context = "\n".join(context_parts)
            full_prompt = f"{context}\n\n{query}"
            
            # Call Mistral AI directly
            response = self.llm._call(full_prompt)
            
            # Return in expected format
            return {
                "result": response,
                "source_documents": retrieved_docs,
                "patient_context": self._get_patient_summary(patient_data),
                "model_info": {"provider": "Mistral AI", "model": self.llm.model_name}
            }
            
        except Exception as e:
            logger.error(f"Error in Mistral AI fallback call: {e}")
            return {
                "result": "Error generating Mistral AI recommendations. Please review patient data manually.",
                "source_documents": [],
                "patient_context": {},
                "model_info": {"provider": "Mistral AI", "error": str(e)}
            }

    def _process_medical_recommendations(self, result: Dict, patient_data: Dict) -> List[str]:
        """Process Mistral AI recommendations with medical validation."""
        try:
            output = result.get("result", "")
            if not output or output.strip() == "":
                return ["No recommendations generated by Mistral AI"]
            
            # Extract numbered recommendations
            lines = output.split('\n')
            recommendations = []
            
            for line in lines:
                line = line.strip()
                # Look for numbered items (1., 2., etc.)
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Clean up the recommendation
                    clean_line = line.lstrip('0123456789.-•').strip()
                    if clean_line and len(clean_line) > 15:  # Minimum meaningful length
                        recommendations.append(clean_line)
            
            # If no structured recommendations found, try other extraction methods
            if not recommendations:
                # Try splitting by sentences and filtering for medical content
                sentences = [s.strip() for s in output.split('.') if s.strip()]
                medical_keywords = ['patient', 'monitor', 'assess', 'consider', 'obtain', 'administer', 'evaluate', 'review', 'provide', 'schedule']
                
                for sentence in sentences:
                    if len(sentence) > 30 and any(keyword in sentence.lower() for keyword in medical_keywords):
                        recommendations.append(sentence)
                        if len(recommendations) >= 6:  # Limit to 6 recommendations
                            break
            
            # If still no recommendations, return the full output
            if not recommendations:
                recommendations = [output.strip()]
            
            # Validate recommendations for medical appropriateness
            validated_recommendations = []
            for rec in recommendations:
                if self._validate_medical_recommendation(rec, patient_data):
                    validated_recommendations.append(rec)
            
            return validated_recommendations[:6]  # Limit to 6 recommendations
            
        except Exception as e:
            logger.error(f"Error processing Mistral AI recommendations: {e}")
            return ["Error processing Mistral AI recommendations. Please review manually."]

    def _validate_medical_recommendation(self, recommendation: str, patient_data: Dict) -> bool:
        """Validate recommendation for medical safety and appropriateness."""
        try:
            rec_lower = recommendation.lower()
            
            # Basic safety checks
            if len(recommendation.strip()) < 15:
                return False
            
            # Check for dangerous absolute statements
            dangerous_phrases = [
                "discontinue all medications",
                "stop all treatment", 
                "no treatment needed",
                "ignore symptoms",
                "definitely not serious",
                "never follow up",
                "avoid all medical care"
            ]
            
            for phrase in dangerous_phrases:
                if phrase in rec_lower:
                    logger.warning(f"Dangerous recommendation filtered: {recommendation}")
                    return False
            
            # Check for basic medical appropriateness
            if any(term in rec_lower for term in ["recommend", "consider", "assess", "monitor", "evaluate", "obtain", "review", "provide", "schedule", "administer"]):
                return True
            
            # Check minimum medical relevance
            medical_terms = ["patient", "treatment", "medication", "test", "follow-up", "consultation", "therapy", "diagnosis"]
            if any(term in rec_lower for term in medical_terms):
                return True
            
            return True  # Default to allowing if unclear
            
        except Exception as e:
            logger.error(f"Error validating medical recommendation: {e}")
            return True  # Default to allowing if validation fails

    def _calculate_medical_confidence(self, result: Dict, patient_data: Dict) -> float:
        """Calculate confidence score for Mistral AI medical recommendations."""
        try:
            source_docs = result.get("source_documents", [])
            response_length = len(result.get("result", ""))
            
            if response_length == 0:
                return 0.0
            
            # Base confidence factors
            source_confidence = min(len(source_docs) / 3.0, 1.0)  # Based on retrieved sources
            length_confidence = 0.7 if response_length < 100 else 0.9 if response_length > 500 else 0.8
            
            # Patient data completeness factor
            patient_completeness = 0.5  # Base score
            
            if patient_data.get("demographics"):
                patient_completeness += 0.1
            if patient_data.get("diagnoses"):
                patient_completeness += 0.2
            if patient_data.get("medications"):
                patient_completeness += 0.2
            
            patient_completeness = min(patient_completeness, 1.0)
            
            # Mistral AI quality bonus (high-quality model)
            mistral_bonus = 0.1
            
            # Combined confidence
            final_confidence = (source_confidence * 0.3 + 
                              length_confidence * 0.3 + 
                              patient_completeness * 0.3 +
                              mistral_bonus * 0.1)
            
            return min(final_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating medical confidence: {e}")
            return 0.6  # Default moderate confidence

    def _extract_sources(self, result: Dict) -> List[str]:
        """Extract source information from retrieved documents."""
        try:
            source_docs = result.get("source_documents", [])
            sources = []
            
            for doc in source_docs:
                if hasattr(doc, 'metadata') and doc.metadata:
                    source = doc.metadata.get('source', 'Unknown')
                    if source not in sources and source != 'Unknown':
                        sources.append(source)
                        
            return sources[:5]  # Limit to 5 sources
            
        except Exception as e:
            logger.error(f"Error extracting sources: {e}")
            return []

    def _get_patient_summary(self, patient_data: Dict) -> Dict:
        """Extract and format patient summary for medical context."""
        try:
            summary = {}
            
            # Demographics
            demographics = patient_data.get("demographics", {})
            if demographics:
                age = demographics.get('age', 'Unknown age')
                gender = demographics.get('gender', 'Unknown gender')
                summary['demographics'] = f"{age} year old {gender}"
            else:
                summary['demographics'] = "Demographics not available"
            
            # Active diagnoses
            diagnoses = patient_data.get("diagnoses", [])
            active_diagnoses = []
            for dx in diagnoses:
                if dx.get("status") == "active":
                    dx_info = dx["description"]
                    if dx.get("date"):
                        dx_info += f" (since {dx['date']})"
                    active_diagnoses.append(dx_info)
            summary['active_diagnoses'] = active_diagnoses[:8]  # Top 8
            
            # Current medications
            medications = patient_data.get("medications", [])
            current_medications = []
            for med in medications:
                if med.get("status") == "active":
                    med_info = med["name"]
                    if med.get("dosage"):
                        med_info += f" {med['dosage']}"
                    if med.get("frequency"):
                        med_info += f" {med['frequency']}"
                    current_medications.append(med_info)
            summary['current_medications'] = current_medications[:10]  # Top 10
            
            # Recent visits
            visits = patient_data.get("visit_history", [])
            recent_visits = []
            for visit in visits[:5]:  # Last 5 visits
                visit_info = f"{visit.get('type', 'Visit')}: {visit.get('chief_complaint', 'No complaint')}"
                if visit.get('date'):
                    visit_info += f" ({visit['date']})"
                recent_visits.append(visit_info)
            summary['recent_visits'] = recent_visits
            
            # Risk factors
            risk_factors = []
            
            # Age-based risk
            age = demographics.get('age', 0)
            if isinstance(age, (int, float)):
                if age > 75:
                    risk_factors.append("Advanced age (>75)")
                elif age > 65:
                    risk_factors.append("Elderly (65-75)")
            
            # Condition-based risks
            for dx in diagnoses:
                dx_lower = dx.get("description", "").lower()
                if "diabetes" in dx_lower:
                    risk_factors.append("Diabetes mellitus")
                elif "hypertension" in dx_lower:
                    risk_factors.append("Hypertension")
                elif any(term in dx_lower for term in ["cardiac", "heart", "coronary"]):
                    risk_factors.append("Cardiovascular disease")
                elif any(term in dx_lower for term in ["copd", "asthma"]):
                    risk_factors.append("Respiratory disease")
                elif any(term in dx_lower for term in ["renal", "kidney"]):
                    risk_factors.append("Renal disease")
            
            summary['risk_factors'] = list(set(risk_factors))  # Remove duplicates
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting patient summary: {e}")
            return {
                "demographics": f"Error retrieving demographics: {str(e)}",
                "active_diagnoses": [],
                "current_medications": [],
                "recent_visits": [],
                "risk_factors": []
            }