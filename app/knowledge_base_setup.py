"""
Clinical Knowledge Base Setup and Management
Advanced document processing for medical literature, guidelines, and protocols.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import re
from datetime import datetime
import aiofiles
import hashlib

from langchain.schema import Document
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    WebBaseLoader, JSONLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .enhanced_retriever import AdvancedClinicalRetriever

logger = logging.getLogger(__name__)

class ClinicalDocumentProcessor:
    """Advanced document processor for clinical knowledge base."""
    
    def __init__(self, retriever: AdvancedClinicalRetriever):
        self.retriever = retriever
        self.processed_documents = set()  # Track processed documents
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped_duplicates": 0
        }
    
    async def process_clinical_guidelines_directory(
        self,
        directory_path: str,
        file_patterns: List[str] = ["*.pdf", "*.txt", "*.docx", "*.json"]
    ) -> Dict[str, Any]:
        """
        Process all clinical documents in a directory.
        
        Args:
            directory_path: Path to directory containing clinical documents
            file_patterns: File patterns to match
            
        Returns:
            Processing statistics and results
        """
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            # Find all matching files
            all_files = []
            for pattern in file_patterns:
                all_files.extend(directory.glob(pattern))
            
            logger.info(f"Found {len(all_files)} clinical documents to process")
            
            # Process files in batches
            batch_size = 5
            results = []
            
            for i in range(0, len(all_files), batch_size):
                batch = all_files[i:i + batch_size]
                batch_results = await self._process_file_batch(batch)
                results.extend(batch_results)
                
                # Log progress
                processed = min(i + batch_size, len(all_files))
                logger.info(f"Processed {processed}/{len(all_files)} files")
            
            # Generate summary report
            return {
                "directory": str(directory),
                "total_files": len(all_files),
                "processing_results": results,
                "statistics": self.processing_stats,
                "successful_documents": [r for r in results if r["status"] == "success"],
                "failed_documents": [r for r in results if r["status"] == "error"]
            }
            
        except Exception as e:
            logger.error(f"Error processing clinical guidelines directory: {e}")
            raise
    
    async def _process_file_batch(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Process a batch of files concurrently."""
        tasks = [self._process_single_file(file_path) for file_path in files]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single clinical document file."""
        try:
            # Check for duplicates
            file_hash = await self._calculate_file_hash(file_path)
            if file_hash in self.processed_documents:
                self.processing_stats["skipped_duplicates"] += 1
                return {
                    "file": str(file_path),
                    "status": "skipped",
                    "reason": "duplicate",
                    "hash": file_hash
                }
            
            self.processed_documents.add(file_hash)
            
            # Load document based on file type
            documents = await self._load_document_by_type(file_path)
            
            if not documents:
                self.processing_stats["failed"] += 1
                return {
                    "file": str(file_path),
                    "status": "error",
                    "reason": "no_content_extracted"
                }
            
            # Enhance documents with clinical metadata
            enhanced_docs = []
            for doc in documents:
                enhanced_doc = await self._enhance_clinical_document(doc, file_path)
                enhanced_docs.append(enhanced_doc)
            
            # Determine document type and clinical domain
            doc_type, clinical_domain = self._classify_document(enhanced_docs[0])
            
            # Add to retriever
            stats = await self.retriever.add_clinical_documents(
                enhanced_docs, 
                document_type=doc_type
            )
            
            self.processing_stats["successful"] += 1
            self.processing_stats["total_processed"] += 1
            
            return {
                "file": str(file_path),
                "status": "success",
                "document_type": doc_type,
                "clinical_domain": clinical_domain,
                "chunks_created": stats.get("total_chunks", 0),
                "file_hash": file_hash
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.processing_stats["failed"] += 1
            self.processing_stats["total_processed"] += 1
            
            return {
                "file": str(file_path),
                "status": "error",
                "error": str(e)
            }
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return str(file_path)  # Fallback to file path
    
    async def _load_document_by_type(self, file_path: Path) -> List[Document]:
        """Load document based on file type."""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_extension == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif file_extension in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif file_extension == '.json':
                loader = JSONLoader(str(file_path), jq_schema='.', text_content=False)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            documents = loader.load()
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return []
    
    async def _enhance_clinical_document(
        self, 
        document: Document, 
        file_path: Path
    ) -> Document:
        """Enhance document with clinical metadata."""
        enhanced_metadata = document.metadata.copy()
        
        # Add file information
        enhanced_metadata.update({
            "source_file": str(file_path),
            "filename": file_path.name,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "processed_at": datetime.now().isoformat(),
            "processor_version": "clinical_v2.0"
        })
        
        # Extract clinical information from content
        clinical_info = self._extract_clinical_information(document.page_content)
        enhanced_metadata.update(clinical_info)
        
        # Clean and preprocess content
        cleaned_content = self._preprocess_clinical_content(document.page_content)
        
        return Document(
            page_content=cleaned_content,
            metadata=enhanced_metadata
        )
    
    def _extract_clinical_information(self, content: str) -> Dict[str, Any]:
        """Extract clinical information from document content."""
        clinical_info = {}
        
        # Extract medical codes
        icd_codes = re.findall(r'ICD-?10?[:\-\s]*([A-Z]\d{2}(?:\.\d+)?)', content, re.IGNORECASE)
        cpt_codes = re.findall(r'CPT[:\-\s]*(\d{5})', content, re.IGNORECASE)
        
        if icd_codes:
            clinical_info["icd_codes"] = list(set(icd_codes))
        if cpt_codes:
            clinical_info["cpt_codes"] = list(set(cpt_codes))
        
        # Extract drug names (simplified pattern)
        drug_pattern = r'\b(?:mg|mcg|units?)\b'
        potential_drugs = re.findall(r'\b[A-Z][a-z]+(?:in|ol|ide|ine|pam)\b(?:\s+\d+\s*(?:mg|mcg|units?))?', content)
        if potential_drugs:
            clinical_info["mentioned_medications"] = list(set(potential_drugs[:10]))  # Limit to 10
        
        # Extract dosages
        dosages = re.findall(r'\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)', content, re.IGNORECASE)
        if dosages:
            clinical_info["dosages_mentioned"] = list(set(dosages[:20]))  # Limit to 20
        
        # Identify document type indicators
        content_lower = content.lower()
        if any(term in content_lower for term in ['guideline', 'recommendation', 'consensus']):
            clinical_info["document_category"] = "guideline"
        elif any(term in content_lower for term in ['case report', 'case study']):
            clinical_info["document_category"] = "case_study"
        elif any(term in content_lower for term in ['protocol', 'procedure']):
            clinical_info["document_category"] = "protocol"
        else:
            clinical_info["document_category"] = "reference"
        
        return clinical_info
    
    def _preprocess_clinical_content(self, content: str) -> str:
        """Clean and preprocess clinical content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove page numbers and headers/footers (common patterns)
        content = re.sub(r'Page \d+ of \d+', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^\d+\s*$', '', content, flags=re.MULTILINE)
        
        # Normalize common medical abbreviations
        abbreviations = {
            r'\bPt\.?\b': 'Patient',
            r'\bDx\.?\b': 'Diagnosis',
            r'\bTx\.?\b': 'Treatment',
            r'\bHx\.?\b': 'History',
            r'\bF/U\b': 'Follow-up'
        }
        
        for abbrev, full_form in abbreviations.items():
            content = re.sub(abbrev, full_form, content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _classify_document(self, document: Document) -> tuple:
        """Classify document type and clinical domain."""
        content_lower = document.page_content.lower()
        filename_lower = document.metadata.get("filename", "").lower()
        
        # Document type classification
        doc_type = "clinical_reference"  # Default
        
        if any(term in content_lower for term in ['guideline', 'recommendation', 'consensus']):
            doc_type = "clinical_guideline"
        elif any(term in filename_lower for term in ['protocol', 'procedure']):
            doc_type = "clinical_protocol"
        elif any(term in content_lower for term in ['case report', 'case study']):
            doc_type = "case_study"
        elif any(term in content_lower for term in ['drug information', 'medication guide']):
            doc_type = "drug_information"
        
        # Clinical domain classification
        domain = "general"  # Default
        
        domain_keywords = {
            'cardiovascular': ['heart', 'cardiac', 'coronary', 'hypertension', 'cardiology'],
            'endocrine': ['diabetes', 'thyroid', 'hormone', 'endocrinology', 'metabolic'],
            'respiratory': ['lung', 'respiratory', 'pulmonary', 'asthma', 'copd'],
            'neurological': ['brain', 'neurological', 'neurology', 'stroke', 'seizure'],
            'psychiatric': ['mental health', 'psychiatric', 'psychology', 'depression', 'anxiety'],
            'oncology': ['cancer', 'oncology', 'tumor', 'malignant', 'chemotherapy'],
            'infectious': ['infection', 'infectious', 'antibiotic', 'bacterial', 'viral'],
            'emergency': ['emergency', 'trauma', 'critical care', 'intensive care']
        }
        
        domain_scores = {}
        for domain_name, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                domain_scores[domain_name] = score
        
        if domain_scores:
            domain = max(domain_scores, key=domain_scores.get)
        
        return doc_type, domain
    
    async def process_web_resources(self, urls: List[str]) -> Dict[str, Any]:
        """Process clinical documents from web URLs."""
        results = []
        
        for url in urls:
            try:
                logger.info(f"Processing web resource: {url}")
                
                # Load web content
                loader = WebBaseLoader(url)
                documents = loader.load()
                
                if not documents:
                    results.append({
                        "url": url,
                        "status": "error",
                        "reason": "no_content"
                    })
                    continue
                
                # Enhance with metadata
                for doc in documents:
                    doc.metadata.update({
                        "source_url": url,
                        "source_type": "web",
                        "processed_at": datetime.now().isoformat()
                    })
                
                # Classify and add to retriever
                doc_type, domain = self._classify_document(documents[0])
                stats = await self.retriever.add_clinical_documents(documents, doc_type)
                
                results.append({
                    "url": url,
                    "status": "success",
                    "document_type": doc_type,
                    "clinical_domain": domain,
                    "chunks_created": stats.get("total_chunks", 0)
                })
                
            except Exception as e:
                logger.error(f"Error processing web resource {url}: {e}")
                results.append({
                    "url": url,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "total_urls": len(urls),
            "results": results,
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"])
        }
    
    async def create_sample_knowledge_base(self):
        """Create a comprehensive sample knowledge base with realistic clinical content."""
        
        sample_documents = [
            # Cardiovascular Guidelines
            {
                "title": "Hypertension Management Guidelines",
                "content": """
                HYPERTENSION MANAGEMENT - CLINICAL PRACTICE GUIDELINES
                
                DEFINITION AND CLASSIFICATION
                Normal: SBP <120 AND DBP <80 mmHg
                Elevated: SBP 120-129 AND DBP <80 mmHg  
                Stage 1 HTN: SBP 130-139 OR DBP 80-89 mmHg
                Stage 2 HTN: SBP ≥140 OR DBP ≥90 mmHg
                
                INITIAL ASSESSMENT
                1. Medical History: Previous BP readings, family history, cardiovascular risk factors
                2. Physical Examination: Multiple BP measurements, fundoscopic exam, cardiac assessment
                3. Laboratory Tests: Basic metabolic panel, lipid profile, urinalysis, ECG
                4. Risk Stratification: Calculate 10-year ASCVD risk using pooled cohort equations
                
                TREATMENT APPROACH
                Non-Pharmacological (All Patients):
                - Dietary Approaches to Stop Hypertension (DASH) diet
                - Sodium restriction (<2.3g daily, ideal <1.5g)
                - Weight management (BMI 18.5-24.9 kg/m²)
                - Regular aerobic exercise (≥30 min, 3-4 days/week)
                - Alcohol moderation (≤2 drinks/day men, ≤1 drink/day women)
                - Smoking cessation
                
                Pharmacological Treatment:
                Stage 1 HTN with ASCVD risk ≥10%: Start antihypertensive
                Stage 2 HTN: Start combination therapy
                
                FIRST-LINE MEDICATIONS
                1. ACE Inhibitors: Lisinopril 10-40mg daily, Enalapril 5-20mg BID
                2. ARBs: Losartan 50-100mg daily, Valsartan 80-160mg daily  
                3. Calcium Channel Blockers: Amlodipine 2.5-10mg daily
                4. Thiazide Diuretics: Hydrochlorothiazide 12.5-25mg daily
                
                MONITORING
                - Follow-up in 1-2 weeks after initiation
                - Monthly visits until BP controlled
                - Every 3-6 months once stable
                - Annual laboratory monitoring
                
                COMPLICATIONS
                Target organ damage: LVH, chronic kidney disease, retinopathy
                Cardiovascular events: MI, stroke, heart failure
                """,
                "metadata": {
                    "clinical_domain": "cardiovascular",
                    "document_type": "clinical_guideline",
                    "evidence_level": "A",
                    "source": "American Heart Association"
                }
            },
            
            # Diabetes Management
            {
                "title": "Type 2 Diabetes Management Protocol",
                "content": """
                TYPE 2 DIABETES MELLITUS - COMPREHENSIVE MANAGEMENT
                
                DIAGNOSTIC CRITERIA
                - Fasting glucose ≥126 mg/dL (7.0 mmol/L)
                - Random glucose ≥200 mg/dL (11.1 mmol/L) with symptoms
                - HbA1c ≥6.5% (48 mmol/mol)
                - OGTT 2-hour glucose ≥200 mg/dL (11.1 mmol/L)
                
                INITIAL EVALUATION
                1. Complete Medical History: Duration of symptoms, family history, complications
                2. Physical Examination: BMI, blood pressure, foot examination, eye examination
                3. Laboratory Assessment:
                   - HbA1c, fasting glucose, lipid profile
                   - Comprehensive metabolic panel (kidney function)
                   - Urinalysis, urine microalbumin
                   - TSH (if indicated)
                
                TREATMENT GOALS
                - HbA1c <7.0% for most adults (individualize based on patient factors)
                - Preprandial glucose 80-130 mg/dL
                - Postprandial glucose <180 mg/dL
                - Blood pressure <130/80 mmHg
                - LDL cholesterol <100 mg/dL (or <70 mg/dL if ASCVD)
                
                PHARMACOLOGICAL MANAGEMENT
                First-line: Metformin 500-1000mg BID (start 500mg daily, titrate)
                
                Second-line options (add to metformin):
                - SGLT-2 inhibitors: Empagliflozin 10-25mg daily
                - GLP-1 agonists: Semaglutide 0.25-1mg weekly
                - DPP-4 inhibitors: Sitagliptin 100mg daily
                - Sulfonylureas: Glipizide 5-10mg daily
                - Insulin: If HbA1c >9% or glucose >300 mg/dL
                
                LIFESTYLE INTERVENTIONS
                - Medical nutrition therapy: Carbohydrate counting, portion control
                - Physical activity: ≥150 minutes moderate intensity per week
                - Weight management: 5-10% weight loss if overweight
                - Diabetes self-management education and support (DSMES)
                
                MONITORING SCHEDULE
                - HbA1c every 3-6 months
                - Annual comprehensive foot exam
                - Annual dilated eye exam
                - Annual nephropathy screening (eGFR, urine albumin)
                - Quarterly blood pressure monitoring
                
                COMPLICATIONS SCREENING
                Microvascular: Retinopathy, nephropathy, neuropathy
                Macrovascular: Coronary artery disease, stroke, peripheral arterial disease
                """,
                "metadata": {
                    "clinical_domain": "endocrine",
                    "document_type": "clinical_protocol",
                    "evidence_level": "A",
                    "source": "American Diabetes Association"
                }
            },
            
            # Mental Health Guidelines
            {
                "title": "Depression Screening and Management",
                "content": """
                MAJOR DEPRESSIVE DISORDER - SCREENING AND TREATMENT
                
                SCREENING TOOLS
                PHQ-9 (Patient Health Questionnaire-9):
                - Score 1-4: Minimal depression
                - Score 5-9: Mild depression
                - Score 10-14: Moderate depression
                - Score 15-19: Moderately severe depression
                - Score 20-27: Severe depression
                
                DSM-5 CRITERIA (≥5 symptoms for ≥2 weeks, including #1 or #2):
                1. Depressed mood most of the day, nearly every day
                2. Markedly diminished interest or pleasure in activities
                3. Significant weight loss/gain or appetite change
                4. Insomnia or hypersomnia nearly every day
                5. Psychomotor agitation or retardation
                6. Fatigue or loss of energy nearly every day
                7. Feelings of worthlessness or excessive guilt
                8. Diminished concentration or indecisiveness
                9. Recurrent thoughts of death or suicidal ideation
                
                TREATMENT APPROACH
                Mild Depression (PHQ-9 5-9):
                - Psychoeducation and supportive counseling
                - Problem-solving therapy or brief CBT
                - Lifestyle interventions (exercise, sleep hygiene)
                - Follow-up in 2-4 weeks
                
                Moderate to Severe Depression (PHQ-9 ≥10):
                - Antidepressant medication AND/OR psychotherapy
                - Consider combination therapy for severe cases
                
                FIRST-LINE ANTIDEPRESSANTS
                SSRIs:
                - Sertraline 25-200mg daily (start 25-50mg)
                - Escitalopram 5-20mg daily (start 5-10mg)
                - Fluoxetine 10-80mg daily (start 10-20mg)
                
                SNRIs:
                - Venlafaxine XR 37.5-225mg daily
                - Duloxetine 30-60mg daily
                
                Other:
                - Bupropion XL 150-450mg daily
                - Mirtazapine 15-45mg at bedtime
                
                MONITORING AND FOLLOW-UP
                - Initial follow-up: 1-2 weeks after starting medication
                - Reassess at 4-6 weeks for response
                - Full therapeutic trial: 8-12 weeks at adequate dose
                - Monitor for side effects and suicidal ideation
                - PHQ-9 at each visit to track progress
                
                SUICIDE RISK ASSESSMENT
                Risk factors: Previous attempts, family history, substance abuse, social isolation
                Protective factors: Social support, religious beliefs, children at home
                - Ask directly about suicidal thoughts
                - Assess plan, means, and intent
                - Create safety plan for high-risk patients
                - Consider hospitalization for imminent risk
                
                PSYCHOTHERAPY OPTIONS
                - Cognitive Behavioral Therapy (CBT)
                - Interpersonal Therapy (IPT)
                - Problem-Solving Therapy
                - Mindfulness-Based Cognitive Therapy
                """,
                "metadata": {
                    "clinical_domain": "psychiatric",
                    "document_type": "clinical_guideline", 
                    "evidence_level": "A",
                    "source": "American Psychiatric Association"
                }
            }
        ]
        
        # Process sample documents
        documents = []
        for doc_data in sample_documents:
            doc = Document(
                page_content=doc_data["content"],
                metadata={
                    **doc_data["metadata"],
                    "source": "sample_knowledge_base",
                    "title": doc_data["title"],
                    "created_at": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        # Add to retriever
        for doc in documents:
            doc_type = doc.metadata["document_type"]
            await self.retriever.add_clinical_documents([doc], doc_type)
        
        logger.info(f"Sample knowledge base created with {len(documents)} clinical documents")
        return {
            "documents_created": len(documents),
            "domains_covered": list(set(doc.metadata["clinical_domain"] for doc in documents)),
            "document_types": list(set(doc.metadata["document_type"] for doc in documents))
        }

class KnowledgeBaseManager:
    """High-level manager for clinical knowledge base operations."""
    
    def __init__(self, retriever: AdvancedClinicalRetriever):
        self.retriever = retriever
        self.document_processor = ClinicalDocumentProcessor(retriever)
        self.knowledge_base_stats = {}
    
    async def setup_initial_knowledge_base(self) -> Dict[str, Any]:
        """Set up initial knowledge base with sample clinical content."""
        try:
            logger.info("Setting up initial clinical knowledge base...")
            
            # Create sample knowledge base
            sample_result = await self.document_processor.create_sample_knowledge_base()
            
            # Get initial statistics
            stats = await self.retriever.get_retriever_stats()
            
            self.knowledge_base_stats = {
                "setup_timestamp": datetime.now().isoformat(),
                "initial_documents": sample_result["documents_created"],
                "covered_domains": sample_result["domains_covered"],
                "total_documents": stats.get("total_documents", 0),
                "status": "initialized"
            }
            
            logger.info("Initial knowledge base setup completed")
            return self.knowledge_base_stats
            
        except Exception as e:
            logger.error(f"Error setting up initial knowledge base: {e}")
            raise
    
    async def expand_knowledge_base(
        self,
        sources: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Expand knowledge base with additional sources.
        
        Args:
            sources: Dict with keys 'directories', 'files', 'urls' containing lists of sources
        """
        results = {
            "directories": [],
            "files": [],
            "urls": [],
            "summary": {}
        }
        
        try:
            # Process directories
            if sources.get("directories"):
                for directory in sources["directories"]:
                    dir_result = await self.document_processor.process_clinical_guidelines_directory(directory)
                    results["directories"].append(dir_result)
            
            # Process individual files
            if sources.get("files"):
                for file_path in sources["files"]:
                    file_result = await self.document_processor._process_single_file(Path(file_path))
                    results["files"].append(file_result)
            
            # Process web URLs
            if sources.get("urls"):
                url_result = await self.document_processor.process_web_resources(sources["urls"])
                results["urls"].append(url_result)
            
            # Generate summary
            total_successful = sum([
                len([r for r in results["directories"] if r.get("status") == "success"]),
                len([r for r in results["files"] if r.get("status") == "success"]),
                sum([r.get("successful", 0) for r in results["urls"]])
            ])
            
            results["summary"] = {
                "total_sources_processed": sum([
                    len(results["directories"]),
                    len(results["files"]),
                    sum([r.get("total_urls", 0) for r in results["urls"]])
                ]),
                "successful_additions": total_successful,
                "expansion_timestamp": datetime.now().isoformat()
            }
            
            # Update knowledge base stats
            updated_stats = await self.retriever.get_retriever_stats()
            self.knowledge_base_stats.update({
                "last_expansion": datetime.now().isoformat(),
                "total_documents": updated_stats.get("total_documents", 0),
                "domain_distribution": updated_stats.get("domain_distribution", {})
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error expanding knowledge base: {e}")
            raise
    
    async def get_knowledge_base_report(self) -> Dict[str, Any]:
        """Generate comprehensive knowledge base report."""
        try:
            stats = await self.retriever.get_retriever_stats()
            
            return {
                "knowledge_base_overview": {
                    "status": stats.get("status", "unknown"),
                    "total_documents": stats.get("total_documents", 0),
                    "last_updated": self.knowledge_base_stats.get("last_expansion", "Never"),
                    "setup_date": self.knowledge_base_stats.get("setup_timestamp", "Unknown")
                },
                "content_analysis": {
                    "clinical_domains": stats.get("domain_distribution", {}),
                    "evidence_levels": stats.get("evidence_level_distribution", {}),
                    "document_types": stats.get("document_type_distribution", {})
                },
                "quality_metrics": {
                    "high_evidence_percentage": self._calculate_high_evidence_percentage(stats),
                    "domain_coverage": len(stats.get("domain_distribution", {})),
                    "avg_documents_per_domain": self._calculate_avg_docs_per_domain(stats)
                },
                "recommendations": self._generate_knowledge_base_recommendations(stats)
            }
            
        except Exception as e:
            logger.error(f"Error generating knowledge base report: {e}")
            return {"error": str(e)}
    
    def _calculate_high_evidence_percentage(self, stats: Dict) -> float:
        """Calculate percentage of high-evidence documents."""
        evidence_dist = stats.get("evidence_level_distribution", {})
        high_evidence = evidence_dist.get("A", 0) + evidence_dist.get("B", 0)
        total = sum(evidence_dist.values())
        return (high_evidence / total * 100) if total > 0 else 0.0
    
    def _calculate_avg_docs_per_domain(self, stats: Dict) -> float:
        """Calculate average documents per clinical domain."""
        domain_dist = stats.get("domain_distribution", {})
        return sum(domain_dist.values()) / len(domain_dist) if domain_dist else 0.0
    
    def _generate_knowledge_base_recommendations(self, stats: Dict) -> List[str]:
        """Generate recommendations for improving knowledge base."""
        recommendations = []
        
        total_docs = stats.get("total_documents", 0)
        domain_dist = stats.get("domain_distribution", {})
        evidence_dist = stats.get("evidence_level_distribution", {})
        
        if total_docs < 50:
            recommendations.append("Consider adding more clinical documents to improve coverage")
        
        if len(domain_dist) < 5:
            recommendations.append("Add documents from additional clinical domains for better coverage")
        
        high_evidence_pct = self._calculate_high_evidence_percentage(stats)
        if high_evidence_pct < 60:
            recommendations.append("Focus on adding high-quality evidence-based guidelines (Level A/B)")
        
        if domain_dist:
            min_domain_docs = min(domain_dist.values())
            if min_domain_docs < 3:
                recommendations.append("Some clinical domains have limited documentation - consider balancing")
        
        return recommendations