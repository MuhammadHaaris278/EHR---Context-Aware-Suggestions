"""
FastAPI application for clinical recommendation system.
Main entry point for the EHR AI module.
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import logging
from contextlib import asynccontextmanager

from .db import get_db, get_patient_data
from .llm_pipeline import ClinicalRecommendationPipeline
from .retriever import setup_retriever
from .prompt import create_clinical_prompt_template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the AI pipeline on startup."""
    global pipeline
    try:
        logger.info("Initializing clinical recommendation pipeline...")
        pipeline = ClinicalRecommendationPipeline()
        await pipeline.initialize()
        logger.info("Pipeline initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    finally:
        logger.info("Shutting down pipeline...")

app = FastAPI(
    title="Clinical Recommendation AI",
    description="AI-powered clinical decision support system",
    version="1.0.0",
    lifespan=lifespan
)

class RecommendationRequest(BaseModel):
    """Request model for clinical recommendations."""
    patient_id: str
    visit_type: str
    symptoms: str
    additional_context: Optional[str] = None

class RecommendationResponse(BaseModel):
    """Response model for clinical recommendations."""
    recommendations: List[str]
    patient_id: str
    confidence_score: Optional[float] = None
    retrieved_sources: Optional[List[str]] = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "pipeline_ready": pipeline is not None}

@app.post("/generate-suggestions", response_model=RecommendationResponse)
async def generate_clinical_suggestions(
    request: RecommendationRequest,
    db=Depends(get_db)
):
    """
    Generate clinical recommendations based on patient data and symptoms.
    
    Args:
        request: Patient information and symptoms
        db: Database session
        
    Returns:
        Clinical recommendations with metadata
    """
    try:
        if not pipeline:
            raise HTTPException(
                status_code=503, 
                detail="AI pipeline not initialized"
            )
        
        # Fetch patient data from database
        patient_data = get_patient_data(db, request.patient_id)
        if not patient_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Patient {request.patient_id} not found"
            )
        
        # Generate recommendations
        recommendations = await pipeline.generate_recommendations(
            patient_data=patient_data,
            visit_type=request.visit_type,
            symptoms=request.symptoms,
            additional_context=request.additional_context
        )
        
        return RecommendationResponse(
            recommendations=recommendations["suggestions"],
            patient_id=request.patient_id,
            confidence_score=recommendations.get("confidence"),
            retrieved_sources=recommendations.get("sources")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error generating recommendations"
        )

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
            "diagnoses": patient_data.get("diagnoses", []),
            "medications": patient_data.get("medications", []),
            "visit_history": patient_data.get("visit_history", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching patient summary: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error fetching patient data"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)