Clinical AI Recommendation System
A production-grade AI module for Electronic Health Record (EHR) systems that provides context-aware clinical recommendations using Retrieval-Augmented Generation (RAG) with LangChain.
Features

Context-Aware Recommendations: Analyzes patient history, diagnoses, and visit type to provide relevant clinical suggestions
Hybrid Retrieval: Combines structured EHR data with semantic vector search of medical knowledge
Medical-Tuned LLM: Uses BioMistral-7B, a specialized medical language model
Production Ready: Built with FastAPI, SQLAlchemy, and comprehensive error handling
Scalable Vector Store: FAISS-based document retrieval with nomic-embed-text-v1 embeddings

Architecture
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangChain     │    │   FAISS Vector │
│   REST API      │───▶│   RAG Pipeline  │───▶│   Store         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   BioMistral-7B │    │   Clinical      │
│   EHR Database  │    │   LLM           │    │   Guidelines    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
Installation
Prerequisites

Python 3.8+
PostgreSQL (for EHR database)
CUDA-capable GPU (recommended for optimal performance)

Setup

Clone the repository:

bashgit clone <repository-url>
cd clinical-ai-system

Install dependencies:

bashpip install -r requirements.txt

Set up environment variables:

bashcp .env.example .env
# Edit .env with your configuration
Required environment variables:
envDATABASE_URL=postgresql://user:password@localhost/ehr_db
HUGGINGFACE_API_KEY=your_hf_key  # Optional for private models

Initialize the database:

bashpython -c "from app.db import create_tables; create_tables()"

Create the knowledge base:

bash# Create sample knowledge base
python -m app.embed_documents --sample

# Or embed your own documents
python -m app.embed_documents --directory /path/to/clinical/documents
Usage
Starting the Server
bash# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
API Endpoints
Generate Clinical Recommendations
bashcurl -X POST "http://localhost:8000/generate-suggestions" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "12345",
    "visit_type": "Follow-up",
    "symptoms": "Shortness of breath, chest tightness",
    "additional_context": "Patient has history of asthma"
  }'
Response:
json{
  "recommendations": [
    "Order ECG and chest X-ray to evaluate cardiac and pulmonary causes",
    "Review patient history for asthma or cardiovascular risk factors",
    "Monitor SpO2 levels and initiate oxygen if below 92%",
    "Consider pulmonary function tests to assess current asthma control"
  ],
  "patient_id": "12345",
  "confidence_score": 0.85,
  "retrieved_sources": ["chest_pain_guidelines.pdf", "asthma_management.pdf"]
}
Patient Summary
bashcurl -X GET "http://localhost:8000/patients/12345/summary"
Health Check
bashcurl -X GET "http://localhost:8000/health"
File Structure
app/
├── main.py              # FastAPI application and endpoints
├── db.py                # Database models and patient data access
├── llm_pipeline.py      # Main AI pipeline orchestration
├── retriever.py         # FAISS vector store and retrieval
├── prompt.py            # Clinical prompt templates
├── embed_documents.py   # Document embedding utilities
requirements.txt         # Python dependencies
README.md               # This file
Components
1. Database Layer (db.py)

SQLAlchemy models for patients, diagnoses, medications, visits
Patient data retrieval and formatting
Database connection management

2. AI Pipeline (llm_pipeline.py)

Orchestrates the entire RAG pipeline
Loads and manages BioMistral-7B LLM
Combines retrieval and generation
Handles different visit types (emergency, follow-up, routine)

3. Vector Retrieval (retriever.py)

FAISS-based document indexing and search
nomic-embed-text-v1 embeddings
Document chunking and metadata management
Similarity search with scoring

4. Prompt Engineering (prompt.py)

Context-aware prompt templates
Visit-type specific prompts
Patient data formatting
Recommendation validation

5. Document Embedding (embed_documents.py)

Utility for processing clinical documents
Supports PDF, Word, and text files
Batch processing and indexing
Sample knowledge base creation