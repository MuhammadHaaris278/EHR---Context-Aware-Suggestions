"""
Comprehensive Testing and Validation Framework
Tests for the enhanced clinical AI system including unit tests,
integration tests, and clinical validation scenarios.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

# Test fixtures and utilities
from .enhanced_ehr_schema import Patient, Condition, MedicationStatement, Encounter
from .integrated_pipeline import EnhancedClinicalPipeline
from .enhanced_summarizer import AdvancedPatientProcessor, TemporalChunker
from .enhanced_retriever import AdvancedClinicalRetriever
from .knowledge_base_setup import KnowledgeBaseManager

logger = logging.getLogger(__name__)

# Test Data Fixtures
class TestDataGenerator:
    """Generate realistic test data for clinical scenarios."""
    
    @staticmethod
    def create_test_patient_data(complexity_level: str = "moderate") -> Dict[str, Any]:
        """Create test patient data with varying complexity levels."""
        
        base_patient = {
            "patient_id": "test_patient_001",
            "demographics": {
                "age": 65,
                "gender": "Female",
                "name": "Jane Test Patient",
                "race": "Caucasian",
                "ethnicity": "Non-Hispanic"
            }
        }
        
        if complexity_level == "simple":
            return {
                **base_patient,
                "diagnoses": [
                    {
                        "code": "I10",
                        "description": "Essential hypertension",
                        "status": "active",
                        "diagnosis_date": "2023-01-15T00:00:00"
                    }
                ],
                "medications": [
                    {
                        "name": "Lisinopril",
                        "dosage": "10mg daily",
                        "status": "active",
                        "start_date": "2023-01-15T00:00:00"
                    }
                ],
                "visit_history": [
                    {
                        "visit_date": "2023-01-15T10:00:00",
                        "visit_type": "outpatient",
                        "chief_complaint": "Routine blood pressure check"
                    }
                ]
            }
        
        elif complexity_level == "complex":
            # 80-year-old with multiple chronic conditions
            return {
                **base_patient,
                "demographics": {**base_patient["demographics"], "age": 80},
                "diagnoses": [
                    {"code": "I10", "description": "Essential hypertension", "status": "active", "diagnosis_date": "2010-03-15T00:00:00"},
                    {"code": "E11.9", "description": "Type 2 diabetes mellitus", "status": "active", "diagnosis_date": "2012-06-22T00:00:00"},
                    {"code": "I25.10", "description": "Coronary artery disease", "status": "active", "diagnosis_date": "2018-09-10T00:00:00"},
                    {"code": "I50.9", "description": "Heart failure, unspecified", "status": "active", "diagnosis_date": "2020-11-05T00:00:00"},
                    {"code": "N18.3", "description": "Chronic kidney disease, stage 3", "status": "active", "diagnosis_date": "2021-04-12T00:00:00"},
                    {"code": "F32.9", "description": "Major depressive disorder", "status": "active", "diagnosis_date": "2022-02-18T00:00:00"},
                    {"code": "M79.3", "description": "Osteoarthritis", "status": "active", "diagnosis_date": "2019-07-30T00:00:00"},
                    {"code": "H25.9", "description": "Cataracts", "status": "resolved", "diagnosis_date": "2023-01-10T00:00:00"}
                ],
                "medications": [
                    {"name": "Lisinopril", "dosage": "20mg daily", "status": "active", "start_date": "2010-03-15T00:00:00"},
                    {"name": "Metformin", "dosage": "1000mg twice daily", "status": "active", "start_date": "2012-06-22T00:00:00"},
                    {"name": "Atorvastatin", "dosage": "40mg daily", "status": "active", "start_date": "2018-09-10T00:00:00"},
                    {"name": "Carvedilol", "dosage": "25mg twice daily", "status": "active", "start_date": "2020-11-05T00:00:00"},
                    {"name": "Furosemide", "dosage": "40mg daily", "status": "active", "start_date": "2020-11-05T00:00:00"},
                    {"name": "Sertraline", "dosage": "50mg daily", "status": "active", "start_date": "2022-02-18T00:00:00"},
                    {"name": "Insulin glargine", "dosage": "30 units daily", "status": "active", "start_date": "2023-03-15T00:00:00"},
                    {"name": "Warfarin", "dosage": "5mg daily", "status": "discontinued", "start_date": "2021-01-10T00:00:00", "end_date": "2023-06-01T00:00:00"}
                ],
                "visit_history": [
                    {"visit_date": "2024-01-15T10:00:00", "visit_type": "outpatient", "chief_complaint": "Routine follow-up"},
                    {"visit_date": "2023-12-20T14:30:00", "visit_type": "emergency", "chief_complaint": "Chest pain"},
                    {"visit_date": "2023-11-10T09:00:00", "visit_type": "outpatient", "chief_complaint": "Diabetes management"},
                    {"visit_date": "2023-10-05T11:15:00", "visit_type": "outpatient", "chief_complaint": "Heart failure follow-up"},
                    {"visit_date": "2023-09-12T08:30:00", "visit_type": "outpatient", "chief_complaint": "Medication review"}
                ] * 4  # Simulate multiple years of visits
            }
        
        else:  # moderate
            return {
                **base_patient,
                "diagnoses": [
                    {"code": "I10", "description": "Essential hypertension", "status": "active", "diagnosis_date": "2020-01-15T00:00:00"},
                    {"code": "E11.9", "description": "Type 2 diabetes mellitus", "status": "active", "diagnosis_date": "2021-06-22T00:00:00"},
                    {"code": "E78.5", "description": "Hyperlipidemia", "status": "active", "diagnosis_date": "2022-03-10T00:00:00"}
                ],
                "medications": [
                    {"name": "Lisinopril", "dosage": "10mg daily", "status": "active", "start_date": "2020-01-15T00:00:00"},
                    {"name": "Metformin", "dosage": "500mg twice daily", "status": "active", "start_date": "2021-06-22T00:00:00"},
                    {"name": "Atorvastatin", "dosage": "20mg daily", "status": "active", "start_date": "2022-03-10T00:00:00"}
                ],
                "visit_history": [
                    {"visit_date": "2024-01-15T10:00:00", "visit_type": "outpatient", "chief_complaint": "Routine diabetes check"},
                    {"visit_date": "2023-10-10T14:00:00", "visit_type": "outpatient", "chief_complaint": "Blood pressure management"},
                    {"visit_date": "2023-07-20T09:30:00", "visit_type": "outpatient", "chief_complaint": "Annual physical"}
                ]
            }

# Unit Tests
class TestTemporalChunker:
    """Test the temporal chunking functionality."""
    
    @pytest.fixture
    def temporal_chunker(self):
        return TemporalChunker()
    
    @pytest.fixture
    def sample_patient_data(self):
        return TestDataGenerator.create_test_patient_data("moderate")
    
    def test_temporal_chunking_basic(self, temporal_chunker, sample_patient_data):
        """Test basic temporal chunking functionality."""
        chunks = temporal_chunker.chunk_by_time(sample_patient_data)
        
        # Should have temporal periods
        assert isinstance(chunks, dict)
        assert len(chunks) > 0
        
        # Should contain expected periods
        expected_periods = ['acute', 'recent', 'current', 'chronic', 'historical']
        for period in chunks.keys():
            assert period in expected_periods
    
    def test_priority_assignment(self, temporal_chunker, sample_patient_data):
        """Test that clinical contexts are assigned appropriate priorities."""
        chunks = temporal_chunker.chunk_by_time(sample_patient_data)
        
        for period, contexts in chunks.items():
            for context in contexts:
                # Priority should be 1, 2, or 3
                assert context.priority in [1, 2, 3]
                
                # Active conditions should have higher priority
                if "active" in context.content.lower():
                    assert context.priority <= 2
    
    def test_domain_classification(self, temporal_chunker):
        """Test clinical domain classification."""
        # Test cardiovascular classification
        cv_diagnosis = {
            "description": "Essential hypertension",
            "status": "active",
            "diagnosis_date": "2023-01-01T00:00:00"
        }
        
        context = temporal_chunker._create_diagnosis_context(cv_diagnosis, datetime.now())
        assert context.domain == "cardiovascular"
        
        # Test endocrine classification
        endo_diagnosis = {
            "description": "Type 2 diabetes mellitus",
            "status": "active", 
            "diagnosis_date": "2023-01-01T00:00:00"
        }
        
        context = temporal_chunker._create_diagnosis_context(endo_diagnosis, datetime.now())
        assert context.domain == "endocrine"

class TestAdvancedSummarizer:
    """Test the advanced summarization functionality."""
    
    @pytest.fixture
    async def mock_llm(self):
        llm = AsyncMock()
        llm._call.return_value = "Test clinical summary with proper medical terminology and patient-specific recommendations."
        return llm
    
    @pytest.fixture
    async def summarizer(self, mock_llm):
        from .enhanced_summarizer import HierarchicalSummarizer
        return HierarchicalSummarizer(mock_llm)
    
    @pytest.mark.asyncio
    async def test_temporal_summarization(self, summarizer):
        """Test temporal chunk summarization."""
        # Create mock temporal chunks
        from .enhanced_summarizer import ClinicalContext
        
        test_chunks = {
            'recent': [
                ClinicalContext("Recent chest pain episode", 1, "cardiovascular", "recent", "encounter"),
                ClinicalContext("Started new medication Lisinopril", 2, "cardiovascular", "recent", "medication")
            ],
            'chronic': [
                ClinicalContext("Long-standing hypertension", 2, "cardiovascular", "chronic", "diagnosis"),
                ClinicalContext("Type 2 diabetes for 5 years", 2, "endocrine", "chronic", "diagnosis")
            ]
        }
        
        summaries = await summarizer.summarize_temporal_chunks(test_chunks)
        
        assert isinstance(summaries, dict)
        assert 'recent' in summaries
        assert 'chronic' in summaries
        assert len(summaries['recent']) > 50  # Should be substantial summary
        assert len(summaries['chronic']) > 50
    
    @pytest.mark.asyncio
    async def test_master_summary_creation(self, summarizer):
        """Test master summary creation from temporal summaries."""
        temporal_summaries = {
            'recent': "Patient recently presented with chest pain and started on Lisinopril for hypertension management.",
            'chronic': "Long-standing history of hypertension and type 2 diabetes mellitus with good control."
        }
        
        master_summary = await summarizer.create_master_summary(temporal_summaries)
        
        assert isinstance(master_summary, str)
        assert len(master_summary) > 100
        assert "patient" in master_summary.lower()
        # Should integrate information from different periods
        assert any(word in master_summary.lower() for word in ['recent', 'chronic', 'hypertension'])

class TestAdvancedRetriever:
    """Test the enhanced retrieval functionality."""
    
    @pytest.fixture
    async def mock_retriever(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever = AdvancedClinicalRetriever(index_path=temp_dir)
            
            # Mock the embedding and vectorstore components
            retriever.embeddings = Mock()
            retriever.clinical_embeddings = Mock()
            retriever.vectorstore = Mock()
            retriever.initialized = True
            
            # Mock search results
            mock_doc = Mock()
            mock_doc.page_content = "Clinical guideline for hypertension management including lifestyle modifications and pharmacotherapy."
            mock_doc.metadata = {
                "clinical_domain": "cardiovascular",
                "evidence_level": "A",
                "source": "test_guideline"
            }
            
            retriever.vectorstore.similarity_search.return_value = [mock_doc]
            
            return retriever
    
    @pytest.mark.asyncio
    async def test_clinical_context_search(self, mock_retriever):
        """Test clinical context-aware search."""
        clinical_context = {
            "primary_domain": "cardiovascular",
            "active_diagnoses": ["Hypertension", "Diabetes"],
            "current_medications": ["Lisinopril", "Metformin"]
        }
        
        results = await mock_retriever.search_clinical_context(
            query="hypertension management",
            clinical_context=clinical_context,
            k=5
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Should return documents with relevant metadata
        for doc in results:
            assert hasattr(doc, 'page_content')
            assert hasattr(doc, 'metadata')
            assert 'clinical_domain' in doc.metadata

# Integration Tests
class TestIntegratedPipeline:
    """Test the complete integrated pipeline."""
    
    @pytest.fixture
    async def mock_pipeline(self):
        """Create a mocked integrated pipeline for testing."""
        # Mock database session
        mock_db = Mock()
        
        # Mock LLM
        mock_llm = Mock()
        mock_llm._call.return_value = """
        PRIMARY DIAGNOSIS
        Essential hypertension with good control on current therapy.
        
        DIFFERENTIAL DIAGNOSES
        - Secondary hypertension (low likelihood - no supporting features)
        - White coat hypertension (ruled out by home readings)
        
        CLINICAL REASONING
        Patient has well-controlled essential hypertension on Lisinopril monotherapy.
        
        RECOMMENDED DIAGNOSTIC WORKUP
        - Continue current monitoring schedule
        - Annual laboratory assessment
        """
        
        # Create pipeline with mocks
        pipeline = EnhancedClinicalPipeline(
            llm=mock_llm,
            db_session=mock_db,
            max_tokens=3500
        )
        
        # Mock the initialization
        pipeline.initialized = True
        pipeline.retriever = Mock()
        pipeline.retriever.search_clinical_context = AsyncMock(return_value=[])
        
        # Mock patient data fetching
        async def mock_fetch_patient_data(patient_id):
            return TestDataGenerator.create_test_patient_data("moderate")
        
        pipeline._fetch_comprehensive_patient_data = mock_fetch_patient_data
        
        return pipeline
    
    @pytest.mark.asyncio
    async def test_comprehensive_patient_processing(self, mock_pipeline):
        """Test end-to-end patient processing."""
        result = await mock_pipeline.process_patient_comprehensive("test_patient_001")
        
        # Should return structured result
        assert isinstance(result, dict)
        assert "patient_id" in result
        assert "processing_timestamp" in result
        
        # Should have processing metadata
        assert "processing_metadata" in result
        metadata = result["processing_metadata"]
        assert "processing_duration_seconds" in metadata
        assert "pipeline_version" in metadata

# Performance Tests
class TestPerformance:
    """Performance and load testing scenarios."""
    
    @pytest.mark.asyncio
    async def test_large_patient_processing_performance(self):
        """Test performance with large patient datasets."""
        # Generate complex patient data
        large_patient_data = TestDataGenerator.create_test_patient_data("complex")
        
        # Mock processor with timing
        start_time = datetime.now()
        
        # Simulate processing (would use real processor in integration tests)
        chunker = TemporalChunker()
        chunks = chunker.chunk_by_time(large_patient_data)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        assert processing_time < 5.0  # Should process in under 5 seconds
        assert len(chunks) > 0
        
        # Should handle large datasets efficiently
        total_items = sum(len(contexts) for contexts in chunks.values())
        assert total_items > 10  # Complex patient should have many contexts
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent patient processing capability."""
        # Generate multiple patient datasets
        patient_data_list = [
            TestDataGenerator.create_test_patient_data("simple"),
            TestDataGenerator.create_test_patient_data("moderate"),
            TestDataGenerator.create_test_patient_data("complex")
        ]
        
        # Process concurrently
        start_time = datetime.now()
        
        async def process_patient(patient_data):
            chunker = TemporalChunker()
            return chunker.chunk_by_time(patient_data)
        
        tasks = [process_patient(data) for data in patient_data_list]
        results = await asyncio.gather(*tasks)
        
        concurrent_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete all processing
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
        
        # Concurrent processing should be efficient
        assert concurrent_time < 10.0  # Should complete in reasonable time

# Clinical Validation Tests
class TestClinicalValidation:
    """Clinical accuracy and validation tests."""
    
    def test_clinical_domain_accuracy(self):
        """Test accuracy of clinical domain classification."""
        chunker = TemporalChunker()
        
        test_cases = [
            ("Essential hypertension", "cardiovascular"),
            ("Type 2 diabetes mellitus", "endocrine"),
            ("Major depressive disorder", "psychiatric"),
            ("Chronic kidney disease", "renal"),
            ("Asthma", "respiratory")
        ]
        
        for description, expected_domain in test_cases:
            actual_domain = chunker._classify_medical_domain(description)
            assert actual_domain == expected_domain, f"Failed for {description}: got {actual_domain}, expected {expected_domain}"
    
    def test_medication_domain_classification(self):
        """Test medication domain classification accuracy."""
        chunker = TemporalChunker()
        
        test_cases = [
            ("Lisinopril", "cardiovascular"),
            ("Metformin", "endocrine"),
            ("Sertraline", "psychiatric"),
            ("Albuterol", "respiratory")
        ]
        
        for medication, expected_domain in test_cases:
            actual_domain = chunker._classify_medication_domain(medication)
            assert actual_domain == expected_domain, f"Failed for {medication}: got {actual_domain}, expected {expected_domain}"
    
    def test_temporal_prioritization_accuracy(self):
        """Test that temporal prioritization follows clinical logic."""
        chunker = TemporalChunker()
        now = datetime.now()
        
        # Recent active diagnosis should be high priority
        recent_active = {
            "description": "Acute myocardial infarction",
            "status": "active",
            "diagnosis_date": (now - timedelta(days=1)).isoformat()
        }
        
        context = chunker._create_diagnosis_context(recent_active, now)
        assert context.priority == 1  # Critical priority
        
        # Old resolved diagnosis should be lower priority
        old_resolved = {
            "description": "Resolved pneumonia",
            "status": "resolved",
            "diagnosis_date": (now - timedelta(days=365)).isoformat()
        }
        
        context = chunker._create_diagnosis_context(old_resolved, now)
        assert context.priority == 3  # Lower priority

# Test Utilities
class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def create_test_database():
        """Create test database with sample data."""
        # This would create a test database instance
        # with standardized test data for consistent testing
        pass
    
    @staticmethod
    def cleanup_test_data():
        """Clean up test data after tests complete."""
        pass
    
    @staticmethod
    def validate_clinical_response(response: str) -> bool:
        """Validate that a clinical response contains appropriate medical content."""
        required_elements = [
            "patient",
            "diagnosis", 
            "treatment",
            "recommendation"
        ]
        
        response_lower = response.lower()
        return all(element in response_lower for element in required_elements)

# Benchmarking Suite
class TestBenchmarks:
    """Performance benchmarking for the clinical system."""
    
    @pytest.mark.benchmark
    def test_temporal_chunking_benchmark(self, benchmark):
        """Benchmark temporal chunking performance."""
        patient_data = TestDataGenerator.create_test_patient_data("complex")
        chunker = TemporalChunker()
        
        def chunk_patient():
            return chunker.chunk_by_time(patient_data)
        
        result = benchmark(chunk_patient)
        
        # Verify result quality
        assert isinstance(result, dict)
        assert len(result) > 0
    
    @pytest.mark.benchmark
    async def test_summarization_benchmark(self, benchmark):
        """Benchmark summarization performance."""
        # This would benchmark the actual summarization process
        # with realistic clinical data
        pass

# Test Configuration
pytest_plugins = ["pytest_asyncio"]

# Test markers
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.timeout(30)  # 30 second timeout for all tests
]

# Test fixtures for database
@pytest.fixture(scope="session")
def test_database():
    """Create test database for integration tests."""
    # Would set up test database instance
    yield
    # Cleanup after tests

@pytest.fixture(scope="function") 
def clean_database(test_database):
    """Provide clean database for each test."""
    # Reset database state
    yield test_database
    # Cleanup test data

# Main test runner configuration
if __name__ == "__main__":
    # Run specific test suites
    pytest.main([
        "test_clinical_system.py",
        "-v",
        "--tb=short",
        "--timeout=60",
        "--asyncio-mode=auto"
    ])