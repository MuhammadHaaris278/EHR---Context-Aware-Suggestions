"""
Usage Examples and Implementation Guide
Demonstrates how to use the enhanced clinical AI system in various scenarios.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Example setup and configuration
async def setup_enhanced_clinical_system():
    """
    Complete setup example for the enhanced clinical system.
    Shows how to initialize all components and set up the knowledge base.
    """
    
    # 1. Database setup with enhanced schema
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from .enhanced_ehr_schema import Base, create_tables
    
    # Create database engine
    DATABASE_URL = "postgresql://username:password@localhost:5432/enhanced_ehr_db"
    engine = create_engine(DATABASE_URL)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    print("✅ Enhanced database schema created")
    
    # 2. Initialize LLM (using Mistral AI as example)
    from .llm_pipeline import MistralMedicalLLM
    
    llm = MistralMedicalLLM(
        model_name="mistral-large-latest",
        temperature=0.1,
        max_tokens=2048,
        api_key="your-mistral-api-key-here"
    )
    
    print("✅ LLM initialized")
    
    # 3. Initialize enhanced pipeline
    from .integrated_pipeline import EnhancedClinicalPipeline
    
    db_session = SessionLocal()
    
    pipeline = EnhancedClinicalPipeline(
        llm=llm,
        db_session=db_session,
        index_path="./enhanced_clinical_index",
        max_tokens=3500
    )
    
    await pipeline.initialize()
    print("✅ Enhanced clinical pipeline initialized")
    
    # 4. Setup initial knowledge base
    from .knowledge_base_setup import KnowledgeBaseManager
    
    kb_manager = KnowledgeBaseManager(pipeline.retriever)
    await kb_manager.setup_initial_knowledge_base()
    
    print("✅ Initial knowledge base created")
    
    return pipeline, kb_manager, db_session

async def example_large_patient_processing():
    """
    Example: Processing a patient with extensive medical history (80+ year old with 50+ conditions).
    Demonstrates how the system handles large patient histories efficiently.
    """
    
    print("\n=== LARGE PATIENT HISTORY PROCESSING EXAMPLE ===")
    
    # Setup
    pipeline, kb_manager, db_session = await setup_enhanced_clinical_system()
    
    # Simulate patient with extensive history
    patient_id = "patient_large_history_001"
    
    try:
        # Process comprehensive patient data
        start_time = datetime.now()
        
        result = await pipeline.process_patient_comprehensive(
            patient_id=patient_id,
            force_refresh=True  # Force fresh analysis
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Patient {patient_id} processed in {processing_time:.2f} seconds")
        print(f"Temporal periods identified: {list(result.get('temporal_summaries', {}).keys())}")
        print(f"Total contexts processed: {result.get('total_contexts_processed', 0)}")
        print(f"Confidence score: {result.get('confidence_score', 0.0):.2f}")
        
        # Display temporal summaries
        for period, summary in result.get('temporal_summaries', {}).items():
            print(f"\n{period.upper()} SUMMARY:")
            print(f"  {summary[:200]}...")
        
        # Display clinical recommendations if available
        clinical_recs = result.get('clinical_recommendations', {})
        if clinical_recs:
            print(f"\nCLINICAL RECOMMENDATIONS:")
            print(f"  Guidelines found: {len(clinical_recs.get('clinical_guidelines', []))}")
            print(f"  Recommendation confidence: {clinical_recs.get('recommendation_confidence', 0.0):.2f}")
        
        return result
        
    except Exception as e:
        print(f"Error processing large patient history: {e}")
        return None

async def example_bulk_processing():
    """
    Example: Bulk processing multiple patients for population health analysis.
    Shows how to efficiently process many patients concurrently.
    """
    
    print("\n=== BULK PATIENT PROCESSING EXAMPLE ===")
    
    pipeline, kb_manager, db_session = await setup_enhanced_clinical_system()
    
    # List of patient IDs to process (would come from database query)
    patient_ids = [
        "patient_001", "patient_002", "patient_003", "patient_004", "patient_005",
        "patient_006", "patient_007", "patient_008", "patient_009", "patient_010"
    ]
    
    print(f"Processing {len(patient_ids)} patients in bulk...")
    
    # Batch processing with concurrency control
    batch_size = 3  # Process 3 patients concurrently
    results = []
    
    for i in range(0, len(patient_ids), batch_size):
        batch = patient_ids[i:i + batch_size]
        
        print(f"Processing batch {i//batch_size + 1}: {batch}")
        
        # Process batch concurrently
        batch_tasks = [
            pipeline.process_patient_comprehensive(patient_id, force_refresh=False)
            for patient_id in batch
        ]
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle results and exceptions
        for patient_id, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                print(f"  ❌ {patient_id}: {result}")
            else:
                processing_time = result.get('processing_metadata', {}).get('processing_duration_seconds', 0)
                cached = result.get('cached', False)
                print(f"  ✅ {patient_id}: {processing_time:.2f}s {'(cached)' if cached else ''}")
                results.append(result)
        
        # Small delay between batches
        await asyncio.sleep(0.5)
    
    print(f"\nBulk processing completed: {len(results)}/{len(patient_ids)} successful")
    
    # Generate population insights
    generate_population_insights(results)
    
    return results

def generate_population_insights(patient_results: List[Dict]):
    """Generate insights from bulk patient processing results."""
    
    print("\n=== POPULATION HEALTH INSIGHTS ===")
    
    # Analyze processing performance
    processing_times = [
        r.get('processing_metadata', {}).get('processing_duration_seconds', 0) 
        for r in patient_results
    ]
    
    print(f"Processing Performance:")
    print(f"  Average time: {sum(processing_times) / len(processing_times):.2f} seconds")
    print(f"  Fastest: {min(processing_times):.2f} seconds")
    print(f"  Slowest: {max(processing_times):.2f} seconds")
    
    # Analyze confidence scores
    confidence_scores = [r.get('confidence_score', 0.0) for r in patient_results]
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    
    print(f"\nDiagnostic Confidence:")
    print(f"  Average confidence: {avg_confidence:.2f}")
    print(f"  High confidence (>0.8): {len([c for c in confidence_scores if c > 0.8])}/{len(confidence_scores)}")
    
    # Analyze temporal patterns
    temporal_distribution = {}
    for result in patient_results:
        for period in result.get('temporal_summaries', {}).keys():
            temporal_distribution[period] = temporal_distribution.get(period, 0) + 1
    
    print(f"\nTemporal Data Distribution:")
    for period, count in temporal_distribution.items():
        print(f"  {period}: {count} patients")

async def example_knowledge_base_expansion():
    """
    Example: Expanding the clinical knowledge base with new documents.
    Shows how to add clinical guidelines, protocols, and research papers.
    """
    
    print("\n=== KNOWLEDGE BASE EXPANSION EXAMPLE ===")
    
    pipeline, kb_manager, db_session = await setup_enhanced_clinical_system()
    
    # Get initial knowledge base stats
    initial_stats = await kb_manager.get_knowledge_base_report()
    print(f"Initial knowledge base: {initial_stats['knowledge_base_overview']['total_documents']} documents")
    
    # Example: Add clinical documents from different sources
    expansion_sources = {
        "directories": [
            "./clinical_guidelines/cardiology/",
            "./clinical_protocols/emergency_medicine/"
        ],
        "files": [
            "./documents/diabetes_management_2024.pdf",
            "./documents/hypertension_guidelines.pdf"
        ],
        "urls": [
            "https://www.uptodate.com/contents/management-of-hypertension",
            "https://care.diabetesjournals.org/content/diabetes-care-standards"
        ]
    }
    
    try:
        # Expand knowledge base
        expansion_result = await kb_manager.expand_knowledge_base(expansion_sources)
        
        print(f"Knowledge base expansion completed:")
        print(f"  Directories processed: {len(expansion_result['directories'])}")
        print(f"  Files processed: {len(expansion_result['files'])}")
        print(f"  URLs processed: {len(expansion_result['urls'])}")
        print(f"  Successful additions: {expansion_result['summary']['successful_additions']}")
        
        # Get updated stats
        updated_stats = await kb_manager.get_knowledge_base_report()
        new_total = updated_stats['knowledge_base_overview']['total_documents']
        initial_total = initial_stats['knowledge_base_overview']['total_documents']
        
        print(f"  Total documents: {initial_total} → {new_total} (+{new_total - initial_total})")
        
        # Show domain distribution
        print(f"\nDomain distribution after expansion:")
        for domain, count in updated_stats['content_analysis']['clinical_domains'].items():
            print(f"  {domain}: {count} documents")
        
        return expansion_result
        
    except Exception as e:
        print(f"Error expanding knowledge base: {e}")
        return None

async def example_temporal_analysis():
    """
    Example: Analyzing patient data with temporal awareness.
    Shows how the system organizes and prioritizes patient information by time periods.
    """
    
    print("\n=== TEMPORAL ANALYSIS EXAMPLE ===")
    
    pipeline, kb_manager, db_session = await setup_enhanced_clinical_system()
    
    patient_id = "patient_temporal_001"
    
    # Get comprehensive patient data
    patient_data = await pipeline._fetch_comprehensive_patient_data(patient_id)
    
    if not patient_data:
        print(f"Patient {patient_id} not found")
        return
    
    print(f"Analyzing temporal patterns for patient {patient_id}")
    print(f"Total diagnoses: {len(patient_data.get('diagnoses', []))}")
    print(f"Total medications: {len(patient_data.get('medications', []))}")
    print(f"Total encounters: {len(patient_data.get('encounters', []))}")
    
    # Create temporal chunks
    temporal_chunker = pipeline.patient_processor.temporal_chunker
    temporal_chunks = temporal_chunker.chunk_by_time(patient_data)
    
    print(f"\nTemporal organization:")
    for period, contexts in temporal_chunks.items():
        if contexts:
            print(f"\n{period.upper()} PERIOD ({len(contexts)} items):")
            
            # Group by priority
            by_priority = {}
            for context in contexts:
                priority = context.priority
                if priority not in by_priority:
                    by_priority[priority] = []
                by_priority[priority].append(context)
            
            for priority in sorted(by_priority.keys()):
                priority_name = {1: "Critical", 2: "Important", 3: "Relevant"}[priority]
                print(f"  {priority_name} ({len(by_priority[priority])} items):")
                
                for context in by_priority[priority][:3]:  # Show top 3
                    timestamp = context.timestamp.strftime("%Y-%m-%d") if context.timestamp else "Unknown"
                    print(f"    • {context.content[:80]}... [{timestamp}]")
    
    # Demonstrate hierarchical summarization
    print(f"\n=== HIERARCHICAL SUMMARIZATION ===")
    
    summarizer = pipeline.patient_processor.hierarchical_summarizer
    temporal_summaries = await summarizer.summarize_temporal_chunks(temporal_chunks)
    
    for period, summary in temporal_summaries.items():
        print(f"\n{period.upper()} SUMMARY:")
        print(f"  {summary[:150]}...")
    
    return temporal_chunks, temporal_summaries

async def example_clinical_context_retrieval():
    """
    Example: Using clinical context for intelligent document retrieval.
    Shows how the system finds relevant clinical guidelines based on patient context.
    """
    
    print("\n=== CLINICAL CONTEXT RETRIEVAL EXAMPLE ===")
    
    pipeline, kb_manager, db_session = await setup_enhanced_clinical_system()
    
    # Example clinical context from patient analysis
    clinical_context = {
        "primary_domain": "cardiovascular",
        "active_diagnoses": ["Hypertension", "Type 2 Diabetes", "Coronary Artery Disease"],
        "current_medications": ["Lisinopril", "Metformin", "Atorvastatin"],
        "recent_changes": {"new_medications": 1, "new_diagnoses": 0}
    }
    
    print("Patient Clinical Context:")
    print(f"  Primary domain: {clinical_context['primary_domain']}")
    print(f"  Active diagnoses: {', '.join(clinical_context['active_diagnoses'])}")
    print(f"  Current medications: {', '.join(clinical_context['current_medications'])}")
    
    # Search for relevant clinical guidelines
    search_queries = [
        "hypertension management guidelines",
        "diabetes cardiovascular risk management", 
        "statin therapy in coronary artery disease"
    ]
    
    print(f"\nSearching clinical knowledge base...")
    
    all_relevant_docs = []
    
    for query in search_queries:
        print(f"\nQuery: '{query}'")
        
        docs = await pipeline.retriever.search_clinical_context(
            query=query,
            clinical_context=clinical_context,
            k=3
        )
        
        print(f"  Found {len(docs)} relevant documents:")
        
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get('title', 'Untitled')
            domain = doc.metadata.get('clinical_domain', 'general')
            evidence = doc.metadata.get('evidence_level', 'D')
            
            print(f"    {i}. {title}")
            print(f"       Domain: {domain}, Evidence: Level {evidence}")
            print(f"       Preview: {doc.page_content[:100]}...")
        
        all_relevant_docs.extend(docs)
    
    # Analyze retrieval quality
    print(f"\n=== RETRIEVAL ANALYSIS ===")
    print(f"Total documents retrieved: {len(all_relevant_docs)}")
    
    # Domain distribution
    domains = [doc.metadata.get('clinical_domain', 'general') for doc in all_relevant_docs]
    domain_counts = {domain: domains.count(domain) for domain in set(domains)}
    print(f"Domain distribution: {domain_counts}")
    
    # Evidence level distribution
    evidence_levels = [doc.metadata.get('evidence_level', 'D') for doc in all_relevant_docs]
    evidence_counts = {level: evidence_levels.count(level) for level in set(evidence_levels)}
    print(f"Evidence levels: {evidence_counts}")
    
    return all_relevant_docs

async def example_performance_monitoring():
    """
    Example: Monitoring system performance and optimization.
    Shows how to track and optimize the enhanced clinical system.
    """
    
    print("\n=== PERFORMANCE MONITORING EXAMPLE ===")
    
    pipeline, kb_manager, db_session = await setup_enhanced_clinical_system()
    
    # Get comprehensive pipeline statistics
    stats = await pipeline.get_pipeline_stats()
    
    print("System Performance Overview:")
    print(f"  Pipeline status: {stats.get('pipeline_status', 'unknown')}")
    print(f"  Patients processed: {stats.get('processing_stats', {}).get('total_patients_processed', 0)}")
    print(f"  Average processing time: {stats.get('processing_stats', {}).get('average_processing_time', 0):.2f}s")
    
    # Calculate cache hit rate
    processing_stats = stats.get('processing_stats', {})
    cache_hits = processing_stats.get('cache_hits', 0)
    cache_misses = processing_stats.get('cache_misses', 0)
    total_requests = cache_hits + cache_misses
    cache_hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
    
    print(f"  Cache hit rate: {cache_hit_rate:.1f}% ({cache_hits}/{total_requests})")
    
    # Knowledge base statistics
    retriever_stats = stats.get('retriever_stats', {})
    print(f"\nKnowledge Base Statistics:")
    print(f"  Total documents: {retriever_stats.get('total_documents', 0)}")
    print(f"  Domain coverage: {len(retriever_stats.get('domain_distribution', {}))}")
    print(f"  Cache size: {retriever_stats.get('cache_size', 0)}")
    
    # Performance recommendations
    print(f"\nPerformance Recommendations:")
    
    if cache_hit_rate < 50:
        print("  • Consider increasing cache retention time")
    if stats.get('processing_stats', {}).get('average_processing_time', 0) > 10:
        print("  • Average processing time is high - consider optimizing chunking strategy")
    if retriever_stats.get('total_documents', 0) < 50:
        print("  • Knowledge base is small - add more clinical documents for better recommendations")
    
    # Memory usage simulation
    print(f"\nResource Usage (Estimated):")
    total_docs = retriever_stats.get('total_documents', 0)
    estimated_memory_mb = total_docs * 0.5  # Rough estimate
    print(f"  Estimated memory usage: {estimated_memory_mb:.1f} MB")
    print(f"  Recommended minimum RAM: {max(8, estimated_memory_mb * 2):.0f} GB")
    
    return stats

async def example_production_deployment():
    """
    Example: Production deployment configuration and best practices.
    Shows how to configure the system for production use.
    """
    
    print("\n=== PRODUCTION DEPLOYMENT EXAMPLE ===")
    
    # Production configuration example
    production_config = {
        "database": {
            "url": "postgresql://ehr_user:secure_password@prod-db:5432/ehr_production",
            "pool_size": 20,
            "max_overflow": 30,
            "pool_timeout": 30,
            "pool_recycle": 3600
        },
        "llm": {
            "provider": "mistral",
            "model": "mistral-large-latest",
            "temperature": 0.1,
            "max_tokens": 2048,
            "timeout": 30,
            "retry_attempts": 3
        },
        "pipeline": {
            "max_tokens": 3500,
            "cache_ttl_hours": 24,
            "batch_size": 10,
            "concurrent_patients": 5,
            "index_path": "/app/data/clinical_index"
        },
        "security": {
            "encrypt_patient_data": True,
            "audit_all_access": True,
            "require_auth": True,
            "session_timeout": 3600
        },
        "performance": {
            "enable_caching": True,
            "cache_size_mb": 1024,
            "enable_compression": True,
            "log_level": "INFO"
        }
    }
    
    print("Production Configuration:")
    print(json.dumps(production_config, indent=2))
    
    # Deployment checklist
    deployment_checklist = [
        "✅ Enhanced database schema deployed",
        "✅ LLM API keys configured and tested", 
        "✅ Knowledge base populated with clinical guidelines",
        "✅ Security configurations applied",
        "✅ Monitoring and logging configured",
        "✅ Backup and recovery procedures tested",
        "✅ Load testing completed",
        "✅ HIPAA compliance verified",
        "✅ Disaster recovery plan documented",
        "✅ Performance benchmarks established"
    ]
    
    print(f"\nDeployment Checklist:")
    for item in deployment_checklist:
        print(f"  {item}")
    
    # Production monitoring setup
    print(f"\nProduction Monitoring Setup:")
    print(f"  • Application metrics: Response times, error rates, throughput")
    print(f"  • System metrics: CPU, memory, disk usage, network I/O")
    print(f"  • Database metrics: Connection pool, query performance, deadlocks")
    print(f"  • LLM metrics: API latency, token usage, rate limits")
    print(f"  • Business metrics: Patients processed, diagnostic accuracy, user satisfaction")
    
    return production_config

# Main execution example
async def main():
    """Main example demonstrating the complete enhanced clinical system."""
    
    print("🚀 ENHANCED CLINICAL AI SYSTEM - COMPLETE DEMONSTRATION")
    print("=" * 70)
    
    try:
        # 1. System setup
        await setup_enhanced_clinical_system()
        
        # 2. Large patient processing
        await example_large_patient_processing()
        
        # 3. Bulk processing
        await example_bulk_processing()
        
        # 4. Knowledge base expansion
        await example_knowledge_base_expansion()
        
        # 5. Temporal analysis
        await example_temporal_analysis()
        
        # 6. Clinical context retrieval
        await example_clinical_context_retrieval()
        
        # 7. Performance monitoring
        await example_performance_monitoring()
        
        # 8. Production deployment guidance
        await example_production_deployment()
        
        print("\n✅ Complete demonstration finished successfully!")
        print("The enhanced clinical AI system is ready for production deployment.")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())