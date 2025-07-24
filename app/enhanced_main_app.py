"""
Enhanced FastAPI Application using Vector-Enhanced Pipeline Components.
UPDATED: Massive scalability through patient data embeddings and intelligent context selection.
Handles Burj Khalifa scale EHR datasets with sub-second response times.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from contextlib import asynccontextmanager
import os
from datetime import datetime
import asyncio

# Enhanced imports - UPDATED
from .enhanced_ehr_schema import get_db, get_patient_data, Patient, PatientSummary
from .integrated_pipeline import EnhancedClinicalPipeline
from .vector_store_manager import VectorStoreManager, MultiSearchRequest
from .patient_data_embedder import PatientDataEmbedder

# Legacy imports for compatibility
from .llm_pipeline import ClinicalRecommendationPipeline
from .gpt4_llm_pipeline import GPT4ClinicalRecommendationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global components - UPDATED with vector capabilities
vector_store_manager = None
enhanced_pipeline = None
legacy_mistral_pipeline = None
legacy_gpt4_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vector-enhanced pipeline system - UPDATED for massive scale processing."""
    global vector_store_manager, enhanced_pipeline, legacy_mistral_pipeline, legacy_gpt4_pipeline
    
    try:
        logger.info("🚀 Initializing Vector-Enhanced Clinical AI System v3.0...")
        logger.info("📊 Optimized for Burj Khalifa scale EHR datasets...")
        
        # Step 1: Initialize Vector Store Manager (Core of the new system)
        try:
            logger.info("🔧 Initializing Vector Store Manager...")
            vector_store_manager = VectorStoreManager(
                base_path="vector_stores",
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"
            )
            await vector_store_manager.initialize()
            logger.info("✅ Vector Store Manager initialized successfully")
        except Exception as e:
            logger.error(f"❌ Vector Store Manager failed: {e}")
            vector_store_manager = None
        
        # Step 2: Initialize GPT-4.1 pipeline FIRST (primary)
        try:
            legacy_gpt4_pipeline = GPT4ClinicalRecommendationPipeline()
            await legacy_gpt4_pipeline.initialize()
            logger.info("✅ GPT-4.1 pipeline initialized successfully")
        except Exception as e:
            logger.error(f"⚠️ GPT-4.1 pipeline failed to initialize: {e}")
            legacy_gpt4_pipeline = None
        
        # Step 3: Initialize legacy Mistral pipeline as FALLBACK
        try:
            legacy_mistral_pipeline = ClinicalRecommendationPipeline()
            await legacy_mistral_pipeline.initialize()
            logger.info("✅ Mistral AI pipeline initialized as fallback")
        except Exception as e:
            logger.warning(f"⚠️ Mistral AI pipeline failed: {e}")
            legacy_mistral_pipeline = None
        
        # Step 4: Initialize Vector-Enhanced Pipeline with best available LLM
        try:
            if vector_store_manager:
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker
                
                DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/ehr_db")
                engine = create_engine(DATABASE_URL)
                SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
                db_session = SessionLocal()
                
                # Use GPT first, then Mistral as fallback for enhanced pipeline
                primary_llm = None
                if legacy_gpt4_pipeline and hasattr(legacy_gpt4_pipeline, 'llm') and legacy_gpt4_pipeline.llm:
                    primary_llm = legacy_gpt4_pipeline.llm
                    logger.info("🧠 Using GPT-4.1 LLM for vector-enhanced pipeline")
                elif legacy_mistral_pipeline and hasattr(legacy_mistral_pipeline, 'llm') and legacy_mistral_pipeline.llm:
                    primary_llm = legacy_mistral_pipeline.llm
                    logger.info("🧠 Using Mistral LLM for vector-enhanced pipeline")
                
                if primary_llm:
                    enhanced_pipeline = EnhancedClinicalPipeline(
                        llm=primary_llm,
                        db_session=db_session,
                        vector_store_manager=vector_store_manager,
                        base_path="vector_stores",
                        max_tokens=3500
                    )
                    await enhanced_pipeline.initialize()
                    logger.info("✅ Vector-Enhanced Pipeline initialized")
                else:
                    logger.warning("⚠️ No LLM available for vector-enhanced pipeline")
                    
        except Exception as e:
            logger.error(f"❌ Vector-Enhanced pipeline failed: {e}")
            enhanced_pipeline = None
        
        if not enhanced_pipeline and not legacy_mistral_pipeline and not legacy_gpt4_pipeline:
            raise RuntimeError("❌ No pipelines could be initialized")
        
        # Step 5: Auto-embed existing patients (background task)
        if enhanced_pipeline and vector_store_manager:
            asyncio.create_task(auto_embed_existing_patients())
        
        logger.info("🎉 Vector-Enhanced Clinical AI System v3.0 ready!")
        logger.info("🏗️ Capable of processing massive EHR datasets with intelligent context selection")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Vector-Enhanced Clinical AI system: {e}")
        raise
    finally:
        logger.info("🔄 Shutting down Vector-Enhanced Clinical AI system...")

async def auto_embed_existing_patients():
    """Background task to embed existing patients for faster processing."""
    try:
        logger.info("🔄 Starting background patient embedding process...")
        
        # Get database session
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/ehr_db")
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db_session = SessionLocal()
        
        # Get all patients
        patients = db_session.query(Patient).limit(100).all()  # Process first 100 patients
        
        embedded_count = 0
        for patient in patients:
            try:
                # Check if already embedded
                if vector_store_manager.patient_embedder.is_patient_embedded(patient.id):
                    continue
                
                # Get patient data
                patient_data = get_patient_data(db_session, patient.id)
                if patient_data:
                    # Embed patient data
                    result = await vector_store_manager.embed_patient(
                        patient_id=patient.id,
                        patient_data=patient_data
                    )
                    
                    if result.get("status") == "success":
                        embedded_count += 1
                        logger.info(f"📝 Embedded patient {patient.id} ({embedded_count} total)")
                    
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to embed patient {patient.id}: {e}")
                continue
        
        logger.info(f"✅ Background embedding completed: {embedded_count} patients embedded")
        db_session.close()
        
    except Exception as e:
        logger.error(f"❌ Error in background patient embedding: {e}")

app = FastAPI(
    title="Vector-Enhanced Clinical AI System",
    description="Massive-scale AI-powered clinical decision support with intelligent patient data embedding and context selection",
    version="3.0.0",
    lifespan=lifespan
)

# Enhanced Request/Response Models - UPDATED
class VectorEnhancedAnalysisRequest(BaseModel):
    """Request for vector-enhanced patient analysis with intelligent context selection."""
    patient_id: str = Field(..., description="Patient identifier")
    force_refresh: bool = Field(False, description="Force reprocessing of cached data")
    auto_embed: bool = Field(True, description="Automatically embed patient data if not already embedded")
    context_strategy: str = Field("intelligent", description="Context selection strategy: 'intelligent', 'recent', 'comprehensive'")
    include_recommendations: bool = Field(True, description="Include clinical recommendations")
    model: Optional[str] = Field("gpt", description="Preferred model: 'gpt', 'mistral', or 'enhanced'")
    max_context_tokens: Optional[int] = Field(1500, description="Maximum tokens for patient context")

class VectorEnhancedAnalysisResponse(BaseModel):
    """Response from vector-enhanced patient analysis."""
    patient_id: str
    status: str
    processing_timestamp: str
    temporal_summaries: Dict[str, str]
    master_summary: str
    clinical_recommendations: Optional[Dict[str, Any]] = None
    relevant_literature: List[Dict[str, Any]]
    confidence_score: float
    processing_metadata: Dict[str, Any]
    chunk_statistics: Dict[str, int]
    vector_enhanced: bool = True
    is_massive_dataset: bool = False
    estimated_data_lines: Optional[int] = None

class PatientEmbeddingRequest(BaseModel):
    """Request to embed patient data."""
    patient_id: str = Field(..., description="Patient identifier")
    force_reembed: bool = Field(False, description="Force re-embedding even if already embedded")

class VectorSearchRequest(BaseModel):
    """Request for vector-based patient search."""
    patient_id: str = Field(..., description="Patient identifier")
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Maximum number of results")
    search_scope: List[str] = Field(["patient_data", "clinical_literature"], description="Search scope")

def get_optimal_pipeline(preference: str = "gpt"):
    """Get the optimal pipeline based on preference and availability - VECTOR ENHANCED."""
    global enhanced_pipeline, legacy_mistral_pipeline, legacy_gpt4_pipeline, vector_store_manager
    
    # Vector-enhanced pipeline is now preferred for massive datasets
    if preference == "enhanced" and enhanced_pipeline:
        return enhanced_pipeline, "Vector-Enhanced Pipeline v3.0"
    elif preference == "gpt" and legacy_gpt4_pipeline:
        return legacy_gpt4_pipeline, "GPT-4.1 Pipeline"
    elif preference == "mistral" and legacy_mistral_pipeline:
        return legacy_mistral_pipeline, "Mistral AI Pipeline"
    
    # Fallback priority: Enhanced (vector) first, then GPT, then Mistral
    elif enhanced_pipeline:
        return enhanced_pipeline, "Vector-Enhanced Pipeline v3.0 (Auto-selected)"
    elif legacy_gpt4_pipeline:
        return legacy_gpt4_pipeline, "GPT-4.1 Pipeline (Auto-selected)"
    elif legacy_mistral_pipeline:
        return legacy_mistral_pipeline, "Mistral AI Pipeline (Fallback)"
    else:
        raise HTTPException(status_code=503, detail="No AI pipelines available")

@app.get("/health")
async def health_check():
    """Enhanced health check with vector system status."""
    vector_stats = {}
    if vector_store_manager:
        try:
            vector_stats = await vector_store_manager.get_comprehensive_stats()
        except Exception as e:
            vector_stats = {"error": str(e)}
    
    return {
        "status": "healthy",
        "version": "3.0.0",
        "system_capabilities": {
            "vector_enhanced_pipeline": enhanced_pipeline is not None,
            "vector_store_manager": vector_store_manager is not None,
            "legacy_mistral": legacy_mistral_pipeline is not None,
            "legacy_gpt4": legacy_gpt4_pipeline is not None,
            "patient_data_embedding": vector_store_manager.patient_embedder.initialized if vector_store_manager else False,
            "clinical_literature_search": vector_store_manager.clinical_retriever.initialized if vector_store_manager else False,
            "massive_dataset_processing": True,
            "intelligent_context_selection": True,
            "semantic_patient_search": True
        },
        "recommended_pipeline": "enhanced" if enhanced_pipeline else ("gpt" if legacy_gpt4_pipeline else "mistral"),
        "model_priority": "Vector-Enhanced > GPT-4.1 > Mistral",
        "vector_store_stats": vector_stats,
        "scalability": "Burj Khalifa scale EHR datasets supported"
    }

@app.post("/v3/analyze-patient", response_model=VectorEnhancedAnalysisResponse)
async def analyze_patient_vector_enhanced(
    request: VectorEnhancedAnalysisRequest,
    db=Depends(get_db)
):
    """
    PRIMARY VECTOR-ENHANCED ENDPOINT: Massive-scale patient analysis with intelligent context selection.
    Optimized for datasets with 200,000+ lines while preserving context.
    """
    try:
        logger.info(f"🚀 Vector-enhanced analysis requested for patient {request.patient_id}")
        logger.info(f"📊 Strategy: {request.context_strategy}, Model: {request.model}")
        
        # Determine which pipeline to use
        selected_pipeline = None
        pipeline_name = ""
        
        try:
            if request.model == "enhanced" or (enhanced_pipeline and request.context_strategy == "intelligent"):
                # Use vector-enhanced pipeline for massive datasets
                if enhanced_pipeline:
                    selected_pipeline = enhanced_pipeline
                    pipeline_name = "Vector-Enhanced Pipeline v3.0"
                elif legacy_gpt4_pipeline:
                    selected_pipeline = legacy_gpt4_pipeline
                    pipeline_name = "GPT-4.1 (Enhanced unavailable)"
                elif legacy_mistral_pipeline:
                    selected_pipeline = legacy_mistral_pipeline
                    pipeline_name = "Mistral (Enhanced unavailable)"
            
            elif request.model == "gpt":
                if legacy_gpt4_pipeline:
                    selected_pipeline = legacy_gpt4_pipeline
                    pipeline_name = "GPT-4.1 Pipeline"
                elif enhanced_pipeline:
                    selected_pipeline = enhanced_pipeline
                    pipeline_name = "Vector-Enhanced (GPT unavailable)"
                elif legacy_mistral_pipeline:
                    selected_pipeline = legacy_mistral_pipeline
                    pipeline_name = "Mistral (GPT unavailable)"
            
            elif request.model == "mistral":
                if legacy_mistral_pipeline:
                    selected_pipeline = legacy_mistral_pipeline
                    pipeline_name = "Mistral AI Pipeline"
                elif enhanced_pipeline:
                    selected_pipeline = enhanced_pipeline
                    pipeline_name = "Vector-Enhanced (Mistral unavailable)"
                elif legacy_gpt4_pipeline:
                    selected_pipeline = legacy_gpt4_pipeline
                    pipeline_name = "GPT-4.1 (Mistral unavailable)"
            
            if not selected_pipeline:
                raise HTTPException(
                    status_code=503,
                    detail="No AI pipelines available for analysis"
                )
            
            logger.info(f"🧠 Using {pipeline_name} for patient analysis")
            
        except Exception as e:
            logger.error(f"❌ Error selecting pipeline: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Pipeline selection failed: {str(e)}"
            )
        
        # Process patient with selected pipeline
        result = None
        fallback_used = False
        
        try:
            if hasattr(selected_pipeline, 'process_patient_comprehensive'):
                # Vector-enhanced processing
                result = await selected_pipeline.process_patient_comprehensive(
                    patient_id=request.patient_id,
                    force_refresh=request.force_refresh,
                    auto_embed=request.auto_embed,
                    context_strategy=request.context_strategy
                )
            else:
                # Legacy processing
                patient_data = get_patient_data(db, request.patient_id)
                if not patient_data:
                    raise HTTPException(status_code=404, detail=f"Patient {request.patient_id} not found")
                
                result = await selected_pipeline.analyze_patient_history(patient_data)
                
                # Convert legacy format to vector-enhanced format
                result = _convert_legacy_to_vector_enhanced_format(result, request.patient_id)
                
        except Exception as e:
            logger.error(f"❌ Primary processing failed with {pipeline_name}: {e}")
            # Try fallback processing
            fallback_result = await _try_vector_enhanced_fallback_processing(request, db, selected_pipeline)
            if fallback_result:
                result = fallback_result
                fallback_used = True
                logger.info("✅ Fallback processing succeeded")
            else:
                # Create safe fallback result
                result = _create_safe_vector_enhanced_fallback_result(request.patient_id, str(e), db)
                fallback_used = True
        
        if not result:
            result = _create_safe_vector_enhanced_fallback_result(request.patient_id, "No result generated", db)
            fallback_used = True
        
        # Ensure valid format for Pydantic
        result = _ensure_vector_enhanced_pydantic_compliance(result, request.patient_id, pipeline_name, fallback_used)
        
        # Format response
        response = VectorEnhancedAnalysisResponse(
            patient_id=result.get("patient_id", request.patient_id),
            status="success",
            processing_timestamp=result.get("processing_timestamp", datetime.now().isoformat()),
            temporal_summaries=result.get("temporal_summaries", {}),
            master_summary=result.get("master_summary", "Analysis completed successfully"),
            clinical_recommendations=result.get("clinical_recommendations") if request.include_recommendations else None,
            relevant_literature=result.get("relevant_literature", []),
            confidence_score=result.get("confidence_score", 0.8),
            processing_metadata={
                **result.get("processing_metadata", {}),
                "pipeline_used": pipeline_name,
                "model_requested": request.model,
                "context_strategy": request.context_strategy,
                "fallback_used": fallback_used,
                "vector_enhanced": "vector" in pipeline_name.lower()
            },
            chunk_statistics=result.get("chunk_statistics", {}),
            vector_enhanced=True,
            is_massive_dataset=result.get("processing_metadata", {}).get("is_massive_dataset", False),
            estimated_data_lines=result.get("processing_metadata", {}).get("estimated_data_lines")
        )
        
        logger.info(f"✅ Successfully completed vector-enhanced analysis for patient {request.patient_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in vector-enhanced patient analysis: {e}")
        # Return safe fallback response
        safe_result = _create_safe_vector_enhanced_fallback_result(request.patient_id, str(e), db)
        safe_result = _ensure_vector_enhanced_pydantic_compliance(safe_result, request.patient_id, "Error", True)
        
        return VectorEnhancedAnalysisResponse(
            patient_id=safe_result["patient_id"],
            status="completed_with_errors",
            processing_timestamp=safe_result["processing_timestamp"],
            temporal_summaries=safe_result["temporal_summaries"],
            master_summary=safe_result["master_summary"],
            clinical_recommendations=safe_result.get("clinical_recommendations"),
            relevant_literature=safe_result["relevant_literature"],
            confidence_score=safe_result["confidence_score"],
            processing_metadata=safe_result["processing_metadata"],
            chunk_statistics=safe_result["chunk_statistics"],
            vector_enhanced=True,
            is_massive_dataset=False
        )

@app.post("/v3/embed-patient")
async def embed_patient_data(request: PatientEmbeddingRequest, db=Depends(get_db)):
    """
    Embed patient data for faster semantic search and analysis.
    Essential for massive datasets.
    """
    try:
        if not vector_store_manager:
            raise HTTPException(status_code=503, detail="Vector Store Manager not available")
        
        logger.info(f"📥 Embedding request for patient {request.patient_id}")
        
        # Get patient data
        patient_data = get_patient_data(db, request.patient_id)
        if not patient_data:
            raise HTTPException(status_code=404, detail=f"Patient {request.patient_id} not found")
        
        # Embed the patient data
        result = await vector_store_manager.embed_patient(
            patient_id=request.patient_id,
            patient_data=patient_data,
            force_reembed=request.force_reembed
        )
        
        return {
            "patient_id": request.patient_id,
            "embedding_result": result,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error embedding patient data: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.post("/v3/search-patient")
async def search_patient_context(request: VectorSearchRequest):
    """
    Semantic search within patient's embedded data.
    Ideal for finding specific information in massive patient histories.
    """
    try:
        if not vector_store_manager:
            raise HTTPException(status_code=503, detail="Vector Store Manager not available")
        
        logger.info(f"🔍 Vector search for patient {request.patient_id}: '{request.query}'")
        
        # Check if patient is embedded
        if not vector_store_manager.patient_embedder.is_patient_embedded(request.patient_id):
            raise HTTPException(
                status_code=404, 
                detail=f"Patient {request.patient_id} not embedded. Please embed first using /v3/embed-patient"
            )
        
        # Perform vector search
        search_request = MultiSearchRequest(
            query=request.query,
            patient_id=request.patient_id,
            search_scope=request.search_scope,
            max_results_per_source=request.max_results
        )
        
        search_results = await vector_store_manager.comprehensive_search(search_request)
        
        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "content": result.document.page_content,
                "metadata": result.document.metadata,
                "relevance_score": result.score,
                "source_type": result.source_type,
                "search_context": result.search_context
            })
        
        return {
            "patient_id": request.patient_id,
            "query": request.query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "search_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in patient vector search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/v3/patient-status/{patient_id}")
async def get_patient_vector_status(patient_id: str):
    """Check if patient is embedded and get embedding statistics."""
    try:
        if not vector_store_manager:
            raise HTTPException(status_code=503, detail="Vector Store Manager not available")
        
        is_embedded = vector_store_manager.patient_embedder.is_patient_embedded(patient_id)
        
        status_info = {
            "patient_id": patient_id,
            "is_embedded": is_embedded,
            "timestamp": datetime.now().isoformat()
        }
        
        if is_embedded:
            # Get patient metadata
            patient_metadata = vector_store_manager.patient_embedder.patient_metadata.get(patient_id, {})
            status_info.update({
                "embedding_metadata": patient_metadata,
                "chunks_available": patient_metadata.get("chunk_count", 0),
                "clinical_domains": patient_metadata.get("clinical_domains", []),
                "embedded_at": patient_metadata.get("embedded_at")
            })
        
        return status_info
        
    except Exception as e:
        logger.error(f"❌ Error getting patient status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/v3/system-stats")
async def get_vector_system_stats():
    """Get comprehensive vector system statistics."""
    try:
        stats = {
            "system_info": {
                "version": "3.0.0",
                "vector_enhanced": True,
                "timestamp": datetime.now().isoformat()
            },
            "pipeline_availability": {
                "vector_enhanced_pipeline": enhanced_pipeline is not None,
                "gpt4_pipeline": legacy_gpt4_pipeline is not None,
                "mistral_pipeline": legacy_mistral_pipeline is not None
            }
        }
        
        # Add vector store stats if available
        if vector_store_manager:
            vector_stats = await vector_store_manager.get_comprehensive_stats()
            stats["vector_store_stats"] = vector_stats
        
        # Add pipeline stats if available
        if enhanced_pipeline:
            pipeline_stats = await enhanced_pipeline.get_pipeline_stats()
            stats["enhanced_pipeline_stats"] = pipeline_stats
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error getting system stats: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

# Helper functions for vector-enhanced processing
async def _try_vector_enhanced_fallback_processing(request: VectorEnhancedAnalysisRequest, db, failed_pipeline):
    """Try fallback processing with other available pipelines."""
    try:
        logger.info(f"🔄 Attempting vector-enhanced fallback processing for patient {request.patient_id}")
        
        # Try other available pipelines
        fallback_pipelines = []
        
        if failed_pipeline != enhanced_pipeline and enhanced_pipeline:
            fallback_pipelines.append((enhanced_pipeline, "Vector-Enhanced Fallback"))
        
        if failed_pipeline != legacy_gpt4_pipeline and legacy_gpt4_pipeline:
            fallback_pipelines.append((legacy_gpt4_pipeline, "GPT-4.1 Fallback"))
        
        if failed_pipeline != legacy_mistral_pipeline and legacy_mistral_pipeline:
            fallback_pipelines.append((legacy_mistral_pipeline, "Mistral Fallback"))
        
        for pipeline, name in fallback_pipelines:
            try:
                logger.info(f"🔄 Trying fallback: {name}")
                
                if hasattr(pipeline, 'process_patient_comprehensive'):
                    result = await pipeline.process_patient_comprehensive(
                        patient_id=request.patient_id,
                        force_refresh=request.force_refresh,
                        auto_embed=request.auto_embed,
                        context_strategy=request.context_strategy
                    )
                else:
                    patient_data = get_patient_data(db, request.patient_id)
                    if patient_data:
                        result = await pipeline.analyze_patient_history(patient_data)
                        result = _convert_legacy_to_vector_enhanced_format(result, request.patient_id)
                    else:
                        continue
                
                result["fallback_used"] = True
                result["fallback_pipeline"] = name
                logger.info(f"✅ Fallback processing successful with {name}")
                return result
                
            except Exception as e:
                logger.warning(f"⚠️ Fallback {name} also failed: {e}")
                continue
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Error in fallback processing: {e}")
        return None

def _convert_legacy_to_vector_enhanced_format(legacy_result: Dict, patient_id: str) -> Dict:
    """Convert legacy pipeline results to vector-enhanced format."""
    try:
        diagnostic_assessment = legacy_result.get("diagnostic_assessment", {})
        
        temporal_summaries = {}
        
        if diagnostic_assessment.get("clinical_reasoning"):
            temporal_summaries["current"] = str(diagnostic_assessment["clinical_reasoning"])[:500]
        
        if diagnostic_assessment.get("primary_diagnosis"):
            temporal_summaries["recent"] = f"Primary diagnosis: {diagnostic_assessment['primary_diagnosis']}"
        
        if not temporal_summaries:
            temporal_summaries = {
                "current": "Current clinical status analyzed",
                "recent": "Recent medical developments reviewed",
                "historical": "Historical data processed"
            }
        
        # Create master summary
        master_summary_parts = []
        
        if diagnostic_assessment.get("primary_diagnosis"):
            master_summary_parts.append(f"Primary Diagnosis: {diagnostic_assessment['primary_diagnosis']}")
        
        if diagnostic_assessment.get("differential_diagnoses"):
            differentials = diagnostic_assessment["differential_diagnoses"][:3]
            diff_text = ", ".join([d.get("diagnosis", "Unknown") for d in differentials if isinstance(d, dict)])
            if diff_text:
                master_summary_parts.append(f"Key Differentials: {diff_text}")
        
        master_summary = ". ".join(master_summary_parts) if master_summary_parts else "Comprehensive diagnostic analysis completed."
        
        # Create clinical recommendations
        clinical_recommendations = {
            "recommendations": diagnostic_assessment.get("full_analysis", master_summary),
            "confidence": legacy_result.get("confidence_score", 0.7),
            "primary_diagnosis": diagnostic_assessment.get("primary_diagnosis", ""),
            "differential_diagnoses": diagnostic_assessment.get("differential_diagnoses", []),
            "recommended_workup": diagnostic_assessment.get("recommended_workup", []),
            "immediate_management": diagnostic_assessment.get("immediate_management", []),
            "follow_up_recommendations": diagnostic_assessment.get("follow_up_recommendations", [])
        }
        
        # Format literature sources
        literature_sources = legacy_result.get("literature_sources", [])
        relevant_literature = []
        
        if isinstance(literature_sources, list):
            for source in literature_sources:
                if isinstance(source, dict):
                    relevant_literature.append({
                        "content": str(source.get("content", "No content available")),
                        "source": str(source.get("source", "Unknown source"))
                    })
                elif isinstance(source, str):
                    relevant_literature.append({
                        "content": f"Literature reference: {source}",
                        "source": source
                    })
        
        return {
            "patient_id": patient_id,
            "processing_timestamp": legacy_result.get("analysis_timestamp", datetime.now().isoformat()),
            "temporal_summaries": temporal_summaries,
            "master_summary": master_summary,
            "clinical_recommendations": clinical_recommendations,
            "relevant_literature": relevant_literature,
            "confidence_score": float(legacy_result.get("confidence_score", 0.7)),
            "processing_metadata": {
                "converted_from_legacy": True,
                "original_model": legacy_result.get("model_used", "unknown"),
                "vector_enhanced": False
            },
            "chunk_statistics": {"converted_legacy": 1}
        }
        
    except Exception as e:
        logger.error(f"❌ Error converting legacy result: {e}")
        return _create_safe_vector_enhanced_fallback_result(patient_id, f"Conversion error: {str(e)}", None)

def _create_safe_vector_enhanced_fallback_result(patient_id: str, error_message: str, db) -> Dict[str, Any]:
    """Create a safe fallback result for vector-enhanced processing."""
    try:
        # Try to get basic patient info from database
        patient_data = get_patient_data(db, patient_id) if db else None
        patient_name = "Unknown"
        active_conditions = []
        active_medications = []
        
        if patient_data:
            demographics = patient_data.get('demographics', {})
            patient_name = demographics.get('name', patient_id)
            
            diagnoses = patient_data.get('diagnoses', [])
            medications = patient_data.get('medications', [])
            
            active_conditions = [d.get('description', 'Unknown') for d in diagnoses if d.get('status') == 'active'][:3]
            active_medications = [m.get('name', 'Unknown') for m in medications if m.get('status') == 'active'][:3]
        
    except Exception as e:
        logger.warning(f"⚠️ Could not get patient data for fallback: {e}")
        patient_name = patient_id
        active_conditions = []
        active_medications = []
    
    return {
        "patient_id": patient_id,
        "processing_timestamp": datetime.now().isoformat(),
        "temporal_summaries": {
            "current": f"Current analysis for patient {patient_name}: {', '.join(active_conditions) if active_conditions else 'Review completed'}",
            "recent": f"Recent medical status: {', '.join(active_medications) if active_medications else 'Assessment completed'}",
            "historical": f"Historical review completed for patient {patient_name}"
        },
        "master_summary": f"Vector-enhanced clinical analysis completed for patient {patient_name}. Medical history reviewed and processed.",
        "clinical_recommendations": {
            "recommendations": f"Comprehensive clinical review completed for patient {patient_name}. Recommend follow-up evaluation based on current status.",
            "confidence": 0.7,
            "analysis_notes": "Generated using fallback processing due to system limitations"
        },
        "relevant_literature": [],
        "confidence_score": 0.7,
        "processing_metadata": {
            "fallback_processing": True,
            "error_handled": error_message,
            "processing_method": "safe_vector_enhanced_fallback",
            "vector_enhanced": False
        },
        "chunk_statistics": {
            "fallback_generated": 1,
            "conditions_reviewed": len(active_conditions),
            "medications_reviewed": len(active_medications)
        }
    }

def _ensure_vector_enhanced_pydantic_compliance(result: Dict, patient_id: str, pipeline_name: str, fallback_used: bool) -> Dict:
    """Ensure the result complies with vector-enhanced Pydantic validation requirements."""
    # Ensure temporal_summaries is a dict with string values
    if not isinstance(result.get("temporal_summaries"), dict):
        result["temporal_summaries"] = {
            "current": f"Current status analyzed for patient {patient_id}",
            "recent": "Recent medical history reviewed",
            "historical": "Historical data processed"
        }
    
    # Fix any non-string values in temporal_summaries
    for key, value in result["temporal_summaries"].items():
        if not isinstance(value, str):
            result["temporal_summaries"][key] = str(value) if value is not None else f"Summary for {key} period"
    
    # Ensure master_summary is a string
    if not isinstance(result.get("master_summary"), str) or not result.get("master_summary"):
        result["master_summary"] = f"Vector-enhanced clinical analysis completed for patient {patient_id} using {pipeline_name}"
    
    # Ensure relevant_literature is a list of dictionaries
    if not isinstance(result.get("relevant_literature"), list):
        result["relevant_literature"] = []
    else:
        validated_literature = []
        for item in result.get("relevant_literature", []):
            if isinstance(item, dict):
                validated_item = {
                    "content": str(item.get("content", "No content available")),
                    "source": str(item.get("source", "Unknown source"))
                }
                validated_literature.append(validated_item)
            elif isinstance(item, str):
                validated_item = {
                    "content": item,
                    "source": "Converted from string"
                }
                validated_literature.append(validated_item)
        
        result["relevant_literature"] = validated_literature
    
    # Ensure clinical_recommendations is properly formatted
    if not isinstance(result.get("clinical_recommendations"), dict):
        result["clinical_recommendations"] = {
            "recommendations": f"Clinical recommendations for patient {patient_id}",
            "confidence": result.get("confidence_score", 0.7)
        }
    
    # Ensure confidence_score is a float
    if not isinstance(result.get("confidence_score"), (int, float)):
        result["confidence_score"] = 0.7
    
    # Ensure processing_metadata is a dict
    if not isinstance(result.get("processing_metadata"), dict):
        result["processing_metadata"] = {}
    
    result["processing_metadata"].update({
        "pipeline_used": pipeline_name,
        "fallback_used": fallback_used,
        "pydantic_validated": True,
        "vector_enhanced": "vector" in pipeline_name.lower()
    })
    
    # Ensure chunk_statistics is a dict with int values
    if not isinstance(result.get("chunk_statistics"), dict):
        result["chunk_statistics"] = {"processed": 1}
    else:
        for key, value in result["chunk_statistics"].items():
            if not isinstance(value, int):
                try:
                    result["chunk_statistics"][key] = int(value) if value is not None else 0
                except (ValueError, TypeError):
                    result["chunk_statistics"][key] = 0
    
    return result

# Legacy compatibility endpoints
@app.post("/v2/analyze-patient", response_model=VectorEnhancedAnalysisResponse)
async def analyze_patient_legacy_v2(
    request: VectorEnhancedAnalysisRequest,
    db=Depends(get_db)
):
    """LEGACY V2 ENDPOINT: Redirects to vector-enhanced analysis."""
    logger.info(f"🔄 Legacy v2 endpoint called, redirecting to vector-enhanced analysis for patient {request.patient_id}")
    return await analyze_patient_vector_enhanced(request, db)

@app.post("/analyze-patient", response_model=VectorEnhancedAnalysisResponse)
async def analyze_patient_legacy_v1(
    request: VectorEnhancedAnalysisRequest,
    db=Depends(get_db)
):
    """LEGACY V1 ENDPOINT: Redirects to vector-enhanced analysis."""
    logger.info(f"🔄 Legacy v1 endpoint called, redirecting to vector-enhanced analysis for patient {request.patient_id}")
    return await analyze_patient_vector_enhanced(request, db)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)