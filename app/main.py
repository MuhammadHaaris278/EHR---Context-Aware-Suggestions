"""
FastAPI application for clinical recommendation system.
UPDATED VERSION - Supports both GPT-4.1 (GitHub AI) and Mistral AI.
Main entry point for the EHR AI module with multi-model support.
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from contextlib import asynccontextmanager
import os

from .db import get_db, get_patient_data
from .llm_pipeline import ClinicalRecommendationPipeline  # Mistral AI pipeline
from .gpt4_llm_pipeline import GPT4ClinicalRecommendationPipeline  # GPT-4.1 pipeline
from .retriever import setup_retriever
from .prompt import create_clinical_prompt_template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instances
mistral_pipeline = None
gpt4_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize both AI pipelines on startup."""
    global mistral_pipeline, gpt4_pipeline
    
    try:
        logger.info("Initializing multi-model clinical diagnostic pipelines...")
        
        # Initialize Mistral AI pipeline
        try:
            mistral_pipeline = ClinicalRecommendationPipeline()
            await mistral_pipeline.initialize()
            logger.info("✅ Mistral AI pipeline initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Mistral AI pipeline failed to initialize: {e}")
            mistral_pipeline = None
        
        # Initialize GPT-4.1 pipeline
        try:
            gpt4_pipeline = GPT4ClinicalRecommendationPipeline()
            await gpt4_pipeline.initialize()
            logger.info("✅ GPT-4.1 pipeline initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ GPT-4.1 pipeline failed to initialize: {e}")
            gpt4_pipeline = None
        
        if not mistral_pipeline and not gpt4_pipeline:
            raise RuntimeError("No AI pipelines could be initialized")
        
        logger.info("Multi-model diagnostic system ready")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize diagnostic pipelines: {e}")
        raise
    finally:
        logger.info("Shutting down diagnostic pipelines...")

app = FastAPI(
    title="Multi-Model Clinical Diagnostic AI",
    description="AI-powered clinical diagnostic system supporting GPT-4.1 and Mistral AI",
    version="3.0.0",
    lifespan=lifespan
)

class DiagnosticAnalysisRequest(BaseModel):
    """Request model for comprehensive diagnostic analysis."""
    patient_id: str
    model_preference: Optional[str] = "auto"  # "gpt4", "mistral", or "auto"

class DiagnosticAnalysisResponse(BaseModel):
    """Response model for comprehensive diagnostic analysis."""
    patient_id: str
    primary_diagnosis: str
    differential_diagnoses: List[Dict[str, Any]]
    clinical_reasoning: str
    risk_stratification: str
    recommended_workup: List[str]
    immediate_management: List[str]
    follow_up_recommendations: List[str]
    confidence_score: float
    literature_sources: List[str]
    patient_summary: Dict[str, Any]
    clinical_patterns: Dict[str, Any]
    risk_profile: Dict[str, Any]
    recent_changes: Dict[str, Any]
    processing_stats: Dict[str, Any]
    analysis_timestamp: str
    model_used: str
    api_provider: str

class LegacyRecommendationRequest(BaseModel):
    """Legacy request model for backward compatibility."""
    patient_id: str
    visit_type: str = "comprehensive_analysis"
    symptoms: str = "comprehensive_history_review"
    additional_context: Optional[str] = None
    model_preference: Optional[str] = "auto"

class LegacyRecommendationResponse(BaseModel):
    """Legacy response model for backward compatibility."""
    recommendations: List[str]
    patient_id: str
    confidence_score: Optional[float] = None
    retrieved_sources: Optional[List[str]] = None
    model_used: Optional[str] = None

def get_optimal_pipeline(model_preference: str = "auto"):
    """Get the optimal pipeline based on preference and availability."""
    global mistral_pipeline, gpt4_pipeline
    
    if model_preference == "gpt4":
        if gpt4_pipeline:
            return gpt4_pipeline, "GPT-4.1"
        elif mistral_pipeline:
            logger.warning("GPT-4.1 requested but not available, falling back to Mistral AI")
            return mistral_pipeline, "Mistral AI (Fallback)"
        else:
            raise HTTPException(status_code=503, detail="No AI models available")
    
    elif model_preference == "mistral":
        if mistral_pipeline:
            return mistral_pipeline, "Mistral AI"
        elif gpt4_pipeline:
            logger.warning("Mistral AI requested but not available, falling back to GPT-4.1")
            return gpt4_pipeline, "GPT-4.1 (Fallback)"
        else:
            raise HTTPException(status_code=503, detail="No AI models available")
    
    else:  # "auto" - choose best available
        # Prefer GPT-4.1 for its advanced reasoning capabilities
        if gpt4_pipeline:
            return gpt4_pipeline, "GPT-4.1 (Auto-selected)"
        elif mistral_pipeline:
            return mistral_pipeline, "Mistral AI (Auto-selected)"
        else:
            raise HTTPException(status_code=503, detail="No AI models available")

@app.get("/health")
async def health_check():
    """Health check endpoint with multi-model status."""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "models_available": {
            "mistral_ai": mistral_pipeline is not None,
            "gpt4_1": gpt4_pipeline is not None
        },
        "features": [
            "multi_model_support",
            "diagnostic_analysis", 
            "patient_history_processing",
            "large_dataset_support"
        ],
        "preferred_model": "GPT-4.1" if gpt4_pipeline else "Mistral AI" if mistral_pipeline else "None"
    }

@app.post("/analyze-patient", response_model=DiagnosticAnalysisResponse)
async def analyze_patient_history(
    request: DiagnosticAnalysisRequest,
    db=Depends(get_db)
):
    """
    PRIMARY ENDPOINT: Analyze comprehensive patient history with model selection.
    
    Supports both GPT-4.1 (via GitHub AI) and Mistral AI for diagnostic analysis.
    Automatically selects the best available model unless specified.
    
    Args:
        request: Patient ID and optional model preference
        db: Database session
        
    Returns:
        Comprehensive diagnostic assessment with model information
    """
    try:
        # Get optimal pipeline based on preference
        pipeline, model_info = get_optimal_pipeline(request.model_preference)
        
        logger.info(f"Starting diagnostic analysis for patient {request.patient_id} using {model_info}")
        
        # Fetch complete patient data from database
        patient_data = get_patient_data(db, request.patient_id)
        if not patient_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Patient {request.patient_id} not found"
            )
        
        # Perform comprehensive patient history analysis
        analysis_result = await pipeline.analyze_patient_history(patient_data)
        
        # Extract diagnostic assessment
        diagnostic_assessment = analysis_result.get("diagnostic_assessment", {})
        
        return DiagnosticAnalysisResponse(
            patient_id=request.patient_id,
            primary_diagnosis=diagnostic_assessment.get("primary_diagnosis", "Unable to determine primary diagnosis"),
            differential_diagnoses=diagnostic_assessment.get("differential_diagnoses", []),
            clinical_reasoning=diagnostic_assessment.get("clinical_reasoning", "Clinical reasoning not available"),
            risk_stratification=diagnostic_assessment.get("risk_stratification", "Risk assessment not available"),
            recommended_workup=diagnostic_assessment.get("recommended_workup", []),
            immediate_management=diagnostic_assessment.get("immediate_management", []),
            follow_up_recommendations=diagnostic_assessment.get("follow_up_recommendations", []),
            confidence_score=analysis_result.get("confidence_score", 0.0),
            literature_sources=analysis_result.get("literature_sources", []),
            patient_summary=analysis_result.get("patient_summary", {}),
            clinical_patterns=analysis_result.get("clinical_patterns", {}),
            risk_profile=analysis_result.get("risk_profile", {}),
            recent_changes=analysis_result.get("recent_changes", {}),
            processing_stats=analysis_result.get("processing_stats", {}),
            analysis_timestamp=analysis_result.get("analysis_timestamp", ""),
            model_used=analysis_result.get("model_used", model_info),
            api_provider=analysis_result.get("api_provider", "Multi-Model System")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in diagnostic analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during diagnostic analysis: {str(e)}"
        )

@app.post("/analyze-patient-gpt4", response_model=DiagnosticAnalysisResponse)
async def analyze_patient_with_gpt4(
    request: DiagnosticAnalysisRequest,
    db=Depends(get_db)
):
    """
    DEDICATED GPT-4.1 ENDPOINT: Force analysis with GPT-4.1 via GitHub AI.
    
    Args:
        request: Patient ID for analysis
        db: Database session
        
    Returns:
        Comprehensive diagnostic assessment using GPT-4.1
    """
    if not gpt4_pipeline:
        raise HTTPException(
            status_code=503,
            detail="GPT-4.1 model not available. Check GitHub AI configuration."
        )
    
    request.model_preference = "gpt4"
    return await analyze_patient_history(request, db)

@app.post("/analyze-patient-mistral", response_model=DiagnosticAnalysisResponse) 
async def analyze_patient_with_mistral(
    request: DiagnosticAnalysisRequest,
    db=Depends(get_db)
):
    """
    DEDICATED MISTRAL ENDPOINT: Force analysis with Mistral AI.
    
    Args:
        request: Patient ID for analysis
        db: Database session
        
    Returns:
        Comprehensive diagnostic assessment using Mistral AI
    """
    if not mistral_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Mistral AI model not available. Check API configuration."
        )
    
    request.model_preference = "mistral"
    return await analyze_patient_history(request, db)

@app.post("/generate-suggestions", response_model=LegacyRecommendationResponse)
async def generate_clinical_suggestions(
    request: LegacyRecommendationRequest,
    db=Depends(get_db)
):
    """
    LEGACY ENDPOINT: Generate clinical recommendations (backward compatibility).
    
    Maintains backward compatibility while supporting multi-model selection.
    
    Args:
        request: Patient information with optional model preference
        db: Database session
        
    Returns:
        Clinical recommendations in legacy format
    """
    try:
        # Get optimal pipeline based on preference
        pipeline, model_info = get_optimal_pipeline(request.model_preference)
        
        logger.info(f"Legacy endpoint called for patient {request.patient_id} using {model_info}")
        
        # Fetch patient data from database
        patient_data = get_patient_data(db, request.patient_id)
        if not patient_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Patient {request.patient_id} not found"
            )
        
        # Use legacy method for backward compatibility
        recommendations = await pipeline.generate_recommendations(
            patient_data=patient_data,
            visit_type=request.visit_type,
            symptoms=request.symptoms,
            additional_context=request.additional_context
        )
        
        return LegacyRecommendationResponse(
            recommendations=recommendations["suggestions"],
            patient_id=request.patient_id,
            confidence_score=recommendations.get("confidence"),
            retrieved_sources=recommendations.get("sources"),
            model_used=model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in legacy recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error generating recommendations"
        )

@app.get("/models/available")
async def get_available_models():
    """Get information about available AI models and their status."""
    return {
        "available_models": {
            "gpt4_1": {
                "available": gpt4_pipeline is not None,
                "provider": "GitHub AI (OpenAI)",
                "model_name": gpt4_pipeline.llm.model_name if gpt4_pipeline else None,
                "capabilities": ["advanced_reasoning", "comprehensive_analysis", "structured_output"],
                "status": "ready" if gpt4_pipeline else "unavailable"
            },
            "mistral_ai": {
                "available": mistral_pipeline is not None,
                "provider": "Mistral AI",
                "model_name": mistral_pipeline.llm.model_name if mistral_pipeline else None,
                "capabilities": ["medical_specialization", "efficient_processing", "clinical_focus"],
                "status": "ready" if mistral_pipeline else "unavailable"
            }
        },
        "default_selection": "gpt4_1" if gpt4_pipeline else "mistral_ai" if mistral_pipeline else None,
        "selection_strategy": "Prefer GPT-4.1 for advanced reasoning, fallback to Mistral AI"
    }

@app.get("/models/compare")
async def compare_models():
    """Compare the capabilities and status of available models."""
    return {
        "model_comparison": {
            "gpt4_1": {
                "strengths": [
                    "Advanced reasoning capabilities",
                    "Better context understanding",
                    "More detailed diagnostic analysis",
                    "Superior differential diagnosis generation"
                ],
                "use_cases": [
                    "Complex multi-system cases",
                    "Rare disease diagnosis", 
                    "Detailed clinical reasoning",
                    "Teaching and education"
                ],
                "response_time": "3-8 seconds",
                "cost": "Higher",
                "availability": gpt4_pipeline is not None
            },
            "mistral_ai": {
                "strengths": [
                    "Medical domain specialization",
                    "Faster response times",
                    "Cost-effective",
                    "Reliable clinical recommendations"
                ],
                "use_cases": [
                    "Routine clinical decisions",
                    "High-volume processing",
                    "Standard diagnostic workflows",
                    "Production environments"
                ],
                "response_time": "2-5 seconds",
                "cost": "Lower", 
                "availability": mistral_pipeline is not None
            }
        },
        "recommendation": "Use GPT-4.1 for complex cases requiring detailed reasoning, Mistral AI for routine clinical decisions"
    }

@app.get("/patients/{patient_id}/summary")
async def get_patient_summary(patient_id: str, db=Depends(get_db)):
    """Get patient summary for debugging/verification."""
    try:
        patient_data = get_patient_data(db, patient_id)
        if not patient_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Patient {patient_id} not found"
            )
        
        return {
            "patient_id": patient_id,
            "demographics": patient_data.get("demographics", {}),
            "diagnoses": patient_data.get("diagnoses", []),
            "medications": patient_data.get("medications", []),
            "visit_history": patient_data.get("visit_history", []),
            "data_statistics": {
                "total_diagnoses": len(patient_data.get("diagnoses", [])),
                "active_diagnoses": len([d for d in patient_data.get("diagnoses", []) if d.get("status") == "active"]),
                "total_medications": len(patient_data.get("medications", [])),
                "active_medications": len([m for m in patient_data.get("medications", []) if m.get("status") == "active"]),
                "total_visits": len(patient_data.get("visit_history", []))
            },
            "recommended_model": "gpt4_1" if len(patient_data.get("diagnoses", [])) > 5 else "auto"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching patient summary: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error fetching patient data"
        )

@app.get("/patients/{patient_id}/analysis-preview")
async def get_analysis_preview(
    patient_id: str, 
    model: str = Query("auto", description="Model to use for preview: gpt4, mistral, or auto"),
    db=Depends(get_db)
):
    """
    Get a preview of what data will be analyzed for diagnostic assessment.
    Supports model selection for preview generation.
    """
    try:
        # Get optimal pipeline
        pipeline, model_info = get_optimal_pipeline(model)
        
        # Fetch patient data
        patient_data = get_patient_data(db, patient_id)
        if not patient_data:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found"
            )
        
        # Process the data to show what the AI will analyze
        processed_data = pipeline.data_processor.process_large_patient_data(patient_data)
        
        return {
            "patient_id": patient_id,
            "selected_model": model_info,
            "processing_preview": {
                "patient_summary": processed_data.get("patient_summary", {}),
                "chronological_events_count": len(processed_data.get("chronological_history", [])),
                "clinical_patterns": processed_data.get("clinical_patterns", {}),
                "risk_profile": processed_data.get("risk_profile", {}),
                "recent_changes_summary": {
                    "new_medications": len(processed_data.get("recent_changes", {}).get("new_medications", [])),
                    "new_diagnoses": len(processed_data.get("recent_changes", {}).get("new_diagnoses", [])),
                    "recent_visits": len(processed_data.get("recent_changes", {}).get("recent_visits", []))
                },
                "data_statistics": processed_data.get("data_statistics", {})
            },
            "sample_chronological_events": processed_data.get("chronological_history", [])[:10],
            "estimated_prompt_size": f"Optimized for {model_info} processing",
            "model_recommendation": "gpt4_1" if processed_data.get("data_statistics", {}).get("total_diagnoses", 0) > 5 else "auto"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating analysis preview: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error generating analysis preview"
        )

@app.get("/patients/{patient_id}/diagnostic-preview")
async def get_diagnostic_preview(
    patient_id: str,
    model: str = Query("auto", description="Model to use for preview: gpt4, mistral, or auto"),
    db=Depends(get_db)
):
    """
    Preview the diagnostic prompt that would be sent to the selected AI model.
    """
    try:
        # Get optimal pipeline
        pipeline, model_info = get_optimal_pipeline(model)
        
        # Fetch and process patient data
        patient_data = get_patient_data(db, patient_id)
        if not patient_data:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found"
            )
        
        # Process the data
        processed_data = pipeline.data_processor.process_large_patient_data(patient_data)
        
        # Get literature context
        literature_context = await pipeline._retrieve_relevant_literature(processed_data)
        
        # Create the diagnostic prompt
        diagnostic_prompt = pipeline._create_diagnostic_prompt(processed_data, literature_context)
        
        return {
            "patient_id": patient_id,
            "selected_model": model_info,
            "prompt_preview": {
                "total_length": len(diagnostic_prompt),
                "sections_included": [
                    "Patient Summary",
                    "Chronological History",
                    "Clinical Patterns", 
                    "Risk Assessment",
                    "Recent Changes",
                    "Medical Literature" if literature_context.get('content') else None,
                    "Diagnostic Analysis Request"
                ],
                "literature_sources_count": len(literature_context.get('sources', [])),
                "chronological_events_count": len(processed_data.get("chronological_history", [])),
                "estimated_tokens": len(diagnostic_prompt) // 4,
                "model_optimization": f"Optimized for {model_info} context window"
            },
            "prompt_sample": diagnostic_prompt[:1200] + "..." if len(diagnostic_prompt) > 1200 else diagnostic_prompt,
            "literature_sources": literature_context.get('sources', []),
            "model_specific_features": {
                "gpt4_1": ["Advanced reasoning", "Structured output", "Complex case handling"],
                "mistral_ai": ["Medical specialization", "Efficient processing", "Clinical focus"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating diagnostic preview: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error generating diagnostic preview"
        )

@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get detailed status of both diagnostic pipelines."""
    try:
        status = {
            "system_status": "operational" if (mistral_pipeline or gpt4_pipeline) else "degraded",
            "pipeline_version": "3.0.0",
            "multi_model_support": True,
            "models": {}
        }
        
        # Mistral AI pipeline status
        if mistral_pipeline:
            retriever_stats = await mistral_pipeline.retriever.get_stats() if mistral_pipeline.retriever else {}
            status["models"]["mistral_ai"] = {
                "status": "operational",
                "model": mistral_pipeline.llm.model_name if mistral_pipeline.llm else "Unknown",
                "provider": "Mistral AI",
                "max_tokens": mistral_pipeline.llm.max_tokens if mistral_pipeline.llm else 0,
                "temperature": mistral_pipeline.llm.temperature if mistral_pipeline.llm else 0.0,
                "retriever_status": retriever_stats,
                "cache_size": len(mistral_pipeline.data_processor.cache) if mistral_pipeline.data_processor else 0
            }
        else:
            status["models"]["mistral_ai"] = {
                "status": "unavailable",
                "error": "Pipeline not initialized"
            }
        
        # GPT-4.1 pipeline status
        if gpt4_pipeline:
            retriever_stats = await gpt4_pipeline.retriever.get_stats() if gpt4_pipeline.retriever else {}
            status["models"]["gpt4_1"] = {
                "status": "operational",
                "model": gpt4_pipeline.llm.model_name if gpt4_pipeline.llm else "Unknown",
                "provider": "GitHub AI (OpenAI)",
                "max_tokens": gpt4_pipeline.llm.max_tokens if gpt4_pipeline.llm else 0,
                "temperature": gpt4_pipeline.llm.temperature if gpt4_pipeline.llm else 0.0,
                "retriever_status": retriever_stats,
                "cache_size": len(gpt4_pipeline.data_processor.cache) if gpt4_pipeline.data_processor else 0
            }
        else:
            status["models"]["gpt4_1"] = {
                "status": "unavailable", 
                "error": "Pipeline not initialized"
            }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        return {
            "system_status": "error",
            "error": str(e)
        }

@app.post("/pipeline/clear-cache")
async def clear_processing_cache(
    model: str = Query("all", description="Model cache to clear: gpt4, mistral, or all")
):
    """Clear the data processing cache for specified models."""
    try:
        results = {}
        
        if model in ["mistral", "all"] and mistral_pipeline:
            cache_size_before = len(mistral_pipeline.data_processor.cache)
            mistral_pipeline.data_processor.cache.clear()
            results["mistral_ai"] = {
                "cache_cleared": cache_size_before,
                "status": "success"
            }
        
        if model in ["gpt4", "all"] and gpt4_pipeline:
            cache_size_before = len(gpt4_pipeline.data_processor.cache)
            gpt4_pipeline.data_processor.cache.clear()
            results["gpt4_1"] = {
                "cache_cleared": cache_size_before,
                "status": "success"
            }
        
        if not results:
            raise HTTPException(
                status_code=400,
                detail="No valid models specified or available for cache clearing"
            )
        
        return {
            "status": "success",
            "message": f"Cache cleared for {model} model(s)",
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error clearing processing cache"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)