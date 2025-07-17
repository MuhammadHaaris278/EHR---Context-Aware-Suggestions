"""
Main AI pipeline for clinical recommendations.
Orchestrates retrieval, prompt construction, and LLM inference.
"""

import os
import asyncio
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint, BaseLLM
from langchain.chains import RetrievalQA
from langchain.schema import Document

from .retriever import ClinicalRetriever
from .prompt import ClinicalPromptTemplate

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class MockLLM(BaseLLM):
    """Custom mock LLM for testing, compatible with LangChain."""
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> str:
        """Generate mock responses based on input prompts."""
        prompt = prompts[0].lower() if prompts else ""
        if "chest pain" in prompt or "shortness of breath" in prompt:
            return """1. Order ECG and chest X-ray to evaluate cardiac and pulmonary causes
2. Review patient history for asthma or cardiovascular risk factors
3. Monitor SpO2 levels and initiate oxygen if below 92%
4. Consider troponin levels if cardiac cause suspected"""
        elif "abdominal pain" in prompt:
            return """1. Order abdominal ultrasound and CBC to assess for appendicitis or infection
2. Check medication history for NSAID use
3. Assess for signs of peritonitis and consider surgical consultation
4. Monitor vital signs and pain levels"""
        else:
            return """1. Complete comprehensive physical examination
2. Review patient's medical history and current medications
3. Order appropriate diagnostic tests based on symptoms
4. Consider differential diagnoses and follow clinical guidelines"""

    def _llm_type(self) -> str:
        """Return the LLM type."""
        return "mock_llm"

    def __call__(self, prompt: str, **kwargs) -> str:
        """Make the mock LLM callable."""
        return self._generate([prompt], stop=kwargs.get("stop"))

class ClinicalRecommendationPipeline:
    """
    Main pipeline for generating clinical recommendations.
    Combines retrieval, prompt engineering, and LLM inference.
    """

    def __init__(self):
        self.retriever = None
        self.llm = None
        self.prompt_template = None
        self.qa_chain = None
        self.initialized = False

    async def initialize(self):
        try:
            logger.info("Initializing clinical recommendation pipeline...")
            self.retriever = ClinicalRetriever()
            await self.retriever.initialize()
            await self._initialize_llm()
            self.prompt_template = ClinicalPromptTemplate()
            self._create_qa_chain()
            self.initialized = True
            logger.info("Pipeline initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    async def _initialize_llm(self):
        try:
            logger.info("Using Hugging Face Inference API for BioMistral-7B")
            token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            if not token:
                raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
            self.llm = HuggingFaceEndpoint(
                repo_id="BioMistral/BioMistral-7B",
                temperature=0.1,
                max_new_tokens=1024,
                top_p=0.95,
                huggingfacehub_api_token=token,
            )
            logger.info("Hugging Face Endpoint loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing Hugging Face Endpoint: {e}")
            await self._initialize_fallback_llm()

    async def _initialize_fallback_llm(self):
        logger.warning("Using fallback MockLLM for testing")
        self.llm = MockLLM()
        logger.info("Fallback MockLLM initialized successfully")

    def _create_qa_chain(self):
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever.get_retriever(),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_template.get_prompt()}
            )
            logger.info("QA chain created successfully")
        except Exception as e:
            logger.error(f"Error creating QA chain: {e}")
            raise

    async def generate_recommendations(
        self,
        patient_data: Dict,
        visit_type: str,
        symptoms: str,
        additional_context: Optional[str] = None
    ) -> Dict:
        try:
            if not self.initialized:
                raise RuntimeError("Pipeline not initialized")

            query = self._construct_query(patient_data, visit_type, symptoms, additional_context)
            result = await self._execute_chain(query, patient_data)
            recommendations = self._process_recommendations(result)

            return {
                "suggestions": recommendations,
                "confidence": self._calculate_confidence(result),
                "sources": self._extract_sources(result)
            }
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise

    def _construct_query(self, patient_data: Dict, visit_type: str, symptoms: str, additional_context: Optional[str] = None) -> str:
        query_parts = [
            f"Patient symptoms: {symptoms}",
            f"Visit type: {visit_type}"
        ]
        if patient_data.get("diagnoses"):
            diagnoses = [d["description"] for d in patient_data["diagnoses"][:3]]
            query_parts.append(f"Medical history: {', '.join(diagnoses)}")
        if patient_data.get("medications"):
            medications = [m["name"] for m in patient_data["medications"][:5]]
            query_parts.append(f"Current medications: {', '.join(medications)}")
        if additional_context:
            query_parts.append(f"Additional context: {additional_context}")
        return " | ".join(query_parts)

    async def _execute_chain(self, query: str, patient_data: Dict) -> Dict:
        try:
            # The QA chain will automatically retrieve documents and format the prompt
            # We just need to pass the query and let the chain handle the rest
            logger.info(f"Executing QA chain with query: {query[:100]}...")
            
            # For async execution
            if hasattr(self.qa_chain, 'ainvoke'):
                result = await self.qa_chain.ainvoke({"query": query})
            else:
                # For sync execution
                result = self.qa_chain({"query": query})
            
            logger.info("QA chain execution completed")
            return result
            
        except Exception as e:
            logger.error(f"Error executing chain: {e}")
            # Fallback to direct LLM call if QA chain fails
            return await self._fallback_direct_llm_call(query, patient_data)

    async def _fallback_direct_llm_call(self, query: str, patient_data: Dict) -> Dict:
        """Fallback method to call LLM directly if QA chain fails."""
        try:
            logger.info("Using fallback direct LLM call")
            
            # Get retrieved documents manually
            retrieved_docs = await self.retriever.search(query, k=5)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            
            # Format patient context
            patient_context = self.prompt_template._format_patient_context(patient_data)
            system_prompt = self.prompt_template._create_system_prompt()
            
            # Create the input dictionary with all required variables
            input_dict = {
                "system_prompt": system_prompt,
                "patient_context": patient_context,
                "query": query,
                "context": context
            }
            
            # Format the prompt
            formatted_prompt = self.prompt_template.get_prompt().format(**input_dict)
            
            # Call LLM directly
            if hasattr(self.llm, '__call__'):
                response = self.llm(formatted_prompt)
            else:
                response = self.llm._generate([formatted_prompt])
            
            # Return in the expected format
            return {
                "result": response,
                "source_documents": retrieved_docs
            }
            
        except Exception as e:
            logger.error(f"Error in fallback LLM call: {e}")
            # Return a basic error response
            return {
                "result": "Error generating recommendations. Please try again.",
                "source_documents": []
            }

    def _process_recommendations(self, result: Dict) -> List[str]:
        try:
            output = result.get("result", "")
            lines = output.split('\n')
            recommendations = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    recommendations.append(line.lstrip('0123456789.-•').strip())
            if not recommendations:
                recommendations = [s.strip() for s in output.split('.') if s.strip()]
            return recommendations[:6]
        except Exception as e:
            logger.error(f"Error processing recommendations: {e}")
            return ["Error processing recommendations"]

    def _calculate_confidence(self, result: Dict) -> float:
        try:
            source_docs = result.get("source_documents", [])
            response_length = len(result.get("result", ""))
            source_confidence = min(len(source_docs) / 3.0, 1.0)
            length_confidence = 0.6 if response_length < 50 else 0.8 if response_length > 1000 else 1.0
            return min(source_confidence * length_confidence, 1.0)
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _extract_sources(self, result: Dict) -> List[str]:
        try:
            source_docs = result.get("source_documents", [])
            sources = []
            for doc in source_docs:
                if hasattr(doc, 'metadata') and doc.metadata:
                    source = doc.metadata.get('source', 'Unknown')
                    if source not in sources:
                        sources.append(source)
            return sources[:5]
        except Exception as e:
            logger.error(f"Error extracting sources: {e}")
            return []