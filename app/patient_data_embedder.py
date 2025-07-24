"""
Patient Data Embedder - Semantic Embedding System for EHR Data
Handles massive patient datasets (Burj Khalifa scale!) with intelligent chunking and vector storage.
Optimized for medical semantic search and context preservation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class PatientChunk:
    """Enhanced patient data chunk with medical context."""
    content: str
    chunk_type: str  # "diagnosis", "medication", "visit", "lab", "procedure"
    patient_id: str
    timestamp: Optional[datetime] = None
    clinical_domain: str = "general"
    priority: int = 2  # 1=critical, 2=important, 3=routine
    metadata: Dict[str, Any] = None
    chunk_id: str = ""

class MedicalPatientChunker:
    """Specialized chunker for patient medical data with context preservation."""
    
    def __init__(self):
        self.medical_domains = {
            'cardiovascular': ['heart', 'cardiac', 'coronary', 'hypertension', 'blood pressure', 'chest pain', 'angina'],
            'endocrine': ['diabetes', 'thyroid', 'hormone', 'insulin', 'glucose', 'metabolic', 'endocrine'],
            'respiratory': ['lung', 'respiratory', 'asthma', 'copd', 'breathing', 'pneumonia', 'dyspnea'],
            'neurological': ['brain', 'neuro', 'stroke', 'seizure', 'dementia', 'headache', 'migraine'],
            'psychiatric': ['depression', 'anxiety', 'mental', 'psychiatric', 'mood', 'bipolar', 'psychosis'],
            'renal': ['kidney', 'renal', 'urinary', 'nephro', 'dialysis', 'creatinine'],
            'gastrointestinal': ['stomach', 'intestinal', 'bowel', 'liver', 'hepatic', 'gastro'],
            'oncology': ['cancer', 'tumor', 'oncology', 'malignant', 'chemotherapy', 'radiation'],
            'hematology': ['blood', 'anemia', 'bleeding', 'coagulation', 'hemoglobin', 'platelet'],
            'infectious': ['infection', 'bacterial', 'viral', 'antibiotic', 'fever', 'sepsis']
        }
    
    def chunk_patient_data(self, patient_data: Dict) -> List[PatientChunk]:
        """Create semantically meaningful chunks from patient data."""
        chunks = []
        patient_id = patient_data.get('patient_id', 'unknown')
        
        # Process diagnoses with rich context
        chunks.extend(self._chunk_diagnoses(patient_data.get('diagnoses', []), patient_id))
        
        # Process medications with interaction context
        chunks.extend(self._chunk_medications(patient_data.get('medications', []), patient_id))
        
        # Process encounters/visits with clinical context
        chunks.extend(self._chunk_encounters(patient_data.get('encounters', []), patient_id))
        
        # Process laboratory results with reference ranges
        chunks.extend(self._chunk_laboratory_results(patient_data.get('laboratory_results', []), patient_id))
        
        # Process procedures with outcomes
        chunks.extend(self._chunk_procedures(patient_data.get('procedures', []), patient_id))
        
        # Process allergies with severity context
        chunks.extend(self._chunk_allergies(patient_data.get('allergies', []), patient_id))
        
        # Create comprehensive patient summary chunk
        summary_chunk = self._create_patient_summary_chunk(patient_data, patient_id)
        if summary_chunk:
            chunks.append(summary_chunk)
        
        logger.info(f"Created {len(chunks)} semantic chunks for patient {patient_id}")
        return chunks
    
    def _chunk_diagnoses(self, diagnoses: List[Dict], patient_id: str) -> List[PatientChunk]:
        """Create diagnosis chunks with clinical context."""
        chunks = []
        
        for i, diagnosis in enumerate(diagnoses):
            # Rich diagnosis context
            content_parts = [
                f"PATIENT DIAGNOSIS: {diagnosis.get('description', 'Unknown condition')}"
            ]
            
            if diagnosis.get('code'):
                content_parts.append(f"ICD Code: {diagnosis['code']}")
            
            if diagnosis.get('status'):
                content_parts.append(f"Status: {diagnosis['status'].upper()}")
            
            if diagnosis.get('diagnosis_date'):
                content_parts.append(f"Diagnosis Date: {diagnosis['diagnosis_date']}")
            
            if diagnosis.get('severity'):
                content_parts.append(f"Severity: {diagnosis['severity']}")
            
            if diagnosis.get('category'):
                content_parts.append(f"Category: {diagnosis['category']}")
            
            if diagnosis.get('notes'):
                content_parts.append(f"Clinical Notes: {diagnosis['notes']}")
            
            # Add clinical context
            clinical_domain = self._classify_clinical_domain(diagnosis.get('description', ''))
            content_parts.append(f"Clinical Domain: {clinical_domain}")
            
            # Determine priority
            priority = 1 if diagnosis.get('status', '').lower() in ['active', 'chronic'] else 2
            if any(urgent in diagnosis.get('description', '').lower() for urgent in ['acute', 'emergency', 'critical']):
                priority = 1
            
            chunk = PatientChunk(
                content="\n".join(content_parts),
                chunk_type="diagnosis",
                patient_id=patient_id,
                timestamp=self._parse_date(diagnosis.get('diagnosis_date')),
                clinical_domain=clinical_domain,
                priority=priority,
                metadata={
                    "icd_code": diagnosis.get('code'),
                    "diagnosis_status": diagnosis.get('status'),
                    "severity": diagnosis.get('severity'),
                    "original_index": i
                },
                chunk_id=f"{patient_id}_diagnosis_{i}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_medications(self, medications: List[Dict], patient_id: str) -> List[PatientChunk]:
        """Create medication chunks with interaction and therapeutic context."""
        chunks = []
        
        for i, medication in enumerate(medications):
            content_parts = [
                f"PATIENT MEDICATION: {medication.get('name', 'Unknown medication')}"
            ]
            
            if medication.get('generic_name'):
                content_parts.append(f"Generic Name: {medication['generic_name']}")
            
            if medication.get('dosage'):
                content_parts.append(f"Dosage: {medication['dosage']}")
            
            if medication.get('frequency'):
                content_parts.append(f"Frequency: {medication['frequency']}")
            
            if medication.get('route'):
                content_parts.append(f"Route: {medication['route']}")
            
            if medication.get('status'):
                content_parts.append(f"Status: {medication['status'].upper()}")
            
            if medication.get('indication'):
                content_parts.append(f"Indication: {medication['indication']}")
            
            if medication.get('start_date'):
                content_parts.append(f"Start Date: {medication['start_date']}")
            
            if medication.get('end_date'):
                content_parts.append(f"End Date: {medication['end_date']}")
            
            if medication.get('instructions'):
                content_parts.append(f"Instructions: {medication['instructions']}")
            
            # Add therapeutic class context
            therapeutic_class = self._classify_medication_therapeutic_class(medication.get('name', ''))
            content_parts.append(f"Therapeutic Class: {therapeutic_class}")
            
            # Determine clinical domain and priority
            clinical_domain = self._classify_medication_domain(medication.get('name', ''))
            priority = 1 if medication.get('status', '').lower() == 'active' else 2
            
            # High-risk medications get priority 1
            if any(risk_med in medication.get('name', '').lower() for risk_med in ['warfarin', 'insulin', 'chemotherapy', 'digoxin']):
                priority = 1
            
            chunk = PatientChunk(
                content="\n".join(content_parts),
                chunk_type="medication",
                patient_id=patient_id,
                timestamp=self._parse_date(medication.get('start_date')),
                clinical_domain=clinical_domain,
                priority=priority,
                metadata={
                    "medication_status": medication.get('status'),
                    "therapeutic_class": therapeutic_class,
                    "indication": medication.get('indication'),
                    "original_index": i
                },
                chunk_id=f"{patient_id}_medication_{i}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_encounters(self, encounters: List[Dict], patient_id: str) -> List[PatientChunk]:
        """Create encounter chunks with clinical visit context."""
        chunks = []
        
        for i, encounter in enumerate(encounters):
            content_parts = [
                f"PATIENT ENCOUNTER: {encounter.get('type', 'Unknown visit type')}"
            ]
            
            if encounter.get('start_time'):
                content_parts.append(f"Date: {encounter['start_time']}")
            
            if encounter.get('status'):
                content_parts.append(f"Status: {encounter['status']}")
            
            if encounter.get('chief_complaint'):
                content_parts.append(f"Chief Complaint: {encounter['chief_complaint']}")
            
            if encounter.get('provider'):
                content_parts.append(f"Provider: {encounter['provider']}")
            
            if encounter.get('location'):
                content_parts.append(f"Location: {encounter['location']}")
            
            if encounter.get('disposition'):
                content_parts.append(f"Disposition: {encounter['disposition']}")
            
            if encounter.get('duration_minutes'):
                content_parts.append(f"Duration: {encounter['duration_minutes']} minutes")
            
            # Determine priority based on encounter type
            encounter_type = encounter.get('type', '').lower()
            priority = 1 if encounter_type in ['emergency', 'inpatient', 'urgent'] else 2
            
            # Classify clinical domain based on chief complaint
            clinical_domain = self._classify_clinical_domain(encounter.get('chief_complaint', ''))
            
            chunk = PatientChunk(
                content="\n".join(content_parts),
                chunk_type="encounter",
                patient_id=patient_id,
                timestamp=self._parse_date(encounter.get('start_time')),
                clinical_domain=clinical_domain,
                priority=priority,
                metadata={
                    "encounter_type": encounter.get('type'),
                    "provider": encounter.get('provider'),
                    "chief_complaint": encounter.get('chief_complaint'),
                    "original_index": i
                },
                chunk_id=f"{patient_id}_encounter_{i}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_laboratory_results(self, lab_results: List[Dict], patient_id: str) -> List[PatientChunk]:
        """Create laboratory result chunks with reference ranges and interpretation."""
        chunks = []
        
        for i, lab in enumerate(lab_results):
            content_parts = [
                f"LABORATORY RESULT: {lab.get('name', 'Unknown test')}"
            ]
            
            if lab.get('code'):
                content_parts.append(f"LOINC Code: {lab['code']}")
            
            if lab.get('value_quantity') is not None:
                content_parts.append(f"Value: {lab['value_quantity']} {lab.get('unit', '')}")
            elif lab.get('value_string'):
                content_parts.append(f"Value: {lab['value_string']}")
            
            if lab.get('interpretation'):
                content_parts.append(f"Interpretation: {lab['interpretation']}")
            
            if lab.get('reference_range_low') is not None and lab.get('reference_range_high') is not None:
                content_parts.append(f"Reference Range: {lab['reference_range_low']}-{lab['reference_range_high']} {lab.get('unit', '')}")
            
            if lab.get('date'):
                content_parts.append(f"Test Date: {lab['date']}")
            
            # Determine priority based on abnormal results
            priority = 2
            if lab.get('interpretation', '').lower() in ['critical', 'high', 'low', 'abnormal']:
                priority = 1
            
            # Classify by lab category
            clinical_domain = self._classify_lab_domain(lab.get('name', ''))
            
            chunk = PatientChunk(
                content="\n".join(content_parts),
                chunk_type="laboratory",
                patient_id=patient_id,
                timestamp=self._parse_date(lab.get('date')),
                clinical_domain=clinical_domain,
                priority=priority,
                metadata={
                    "lab_code": lab.get('code'),
                    "interpretation": lab.get('interpretation'),
                    "value": lab.get('value_quantity'),
                    "original_index": i
                },
                chunk_id=f"{patient_id}_lab_{i}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_procedures(self, procedures: List[Dict], patient_id: str) -> List[PatientChunk]:
        """Create procedure chunks with outcomes and context."""
        chunks = []
        
        for i, procedure in enumerate(procedures):
            content_parts = [
                f"MEDICAL PROCEDURE: {procedure.get('name', 'Unknown procedure')}"
            ]
            
            if procedure.get('code'):
                content_parts.append(f"CPT Code: {procedure['code']}")
            
            if procedure.get('category'):
                content_parts.append(f"Category: {procedure['category']}")
            
            if procedure.get('performed_date'):
                content_parts.append(f"Performed Date: {procedure['performed_date']}")
            
            if procedure.get('status'):
                content_parts.append(f"Status: {procedure['status']}")
            
            if procedure.get('outcome'):
                content_parts.append(f"Outcome: {procedure['outcome']}")
            
            if procedure.get('complications'):
                content_parts.append(f"Complications: {procedure['complications']}")
            
            if procedure.get('indication'):
                content_parts.append(f"Indication: {procedure['indication']}")
            
            if procedure.get('body_site'):
                content_parts.append(f"Body Site: {procedure['body_site']}")
            
            # Determine priority based on procedure type
            priority = 2
            if any(urgent in procedure.get('name', '').lower() for urgent in ['emergency', 'urgent', 'stat', 'critical']):
                priority = 1
            
            clinical_domain = self._classify_procedure_domain(procedure.get('name', ''))
            
            chunk = PatientChunk(
                content="\n".join(content_parts),
                chunk_type="procedure",
                patient_id=patient_id,
                timestamp=self._parse_date(procedure.get('performed_date')),
                clinical_domain=clinical_domain,
                priority=priority,
                metadata={
                    "procedure_code": procedure.get('code'),
                    "category": procedure.get('category'),
                    "outcome": procedure.get('outcome'),
                    "original_index": i
                },
                chunk_id=f"{patient_id}_procedure_{i}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_allergies(self, allergies: List[Dict], patient_id: str) -> List[PatientChunk]:
        """Create allergy chunks with severity and reaction context."""
        chunks = []
        
        for i, allergy in enumerate(allergies):
            content_parts = [
                f"PATIENT ALLERGY: {allergy.get('allergen_name', 'Unknown allergen')}"
            ]
            
            if allergy.get('type'):
                content_parts.append(f"Allergy Type: {allergy['type']}")
            
            if allergy.get('status'):
                content_parts.append(f"Status: {allergy['status']}")
            
            if allergy.get('severity'):
                content_parts.append(f"Severity: {allergy['severity']}")
            
            if allergy.get('reaction'):
                content_parts.append(f"Reaction: {allergy['reaction']}")
            
            if allergy.get('onset_date'):
                content_parts.append(f"Onset Date: {allergy['onset_date']}")
            
            # High priority for severe allergies
            priority = 1 if allergy.get('severity', '').lower() in ['severe', 'life-threatening'] else 2
            
            # Classify allergy domain
            allergen_type = allergy.get('type', 'environmental')
            clinical_domain = "allergy_" + allergen_type
            
            chunk = PatientChunk(
                content="\n".join(content_parts),
                chunk_type="allergy",
                patient_id=patient_id,
                timestamp=self._parse_date(allergy.get('onset_date')),
                clinical_domain=clinical_domain,
                priority=priority,
                metadata={
                    "allergen_type": allergy.get('type'),
                    "severity": allergy.get('severity'),
                    "reaction": allergy.get('reaction'),
                    "original_index": i
                },
                chunk_id=f"{patient_id}_allergy_{i}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_patient_summary_chunk(self, patient_data: Dict, patient_id: str) -> Optional[PatientChunk]:
        """Create comprehensive patient summary chunk."""
        demographics = patient_data.get('demographics', {})
        
        if not demographics:
            return None
        
        content_parts = [
            f"PATIENT SUMMARY: {demographics.get('name', patient_id)}"
        ]
        
        if demographics.get('age'):
            content_parts.append(f"Age: {demographics['age']} years")
        
        if demographics.get('gender'):
            content_parts.append(f"Gender: {demographics['gender']}")
        
        if demographics.get('race'):
            content_parts.append(f"Race: {demographics['race']}")
        
        if demographics.get('ethnicity'):
            content_parts.append(f"Ethnicity: {demographics['ethnicity']}")
        
        # Add summary statistics
        diagnoses = patient_data.get('diagnoses', [])
        active_diagnoses = [d for d in diagnoses if d.get('status') == 'active']
        content_parts.append(f"Active Diagnoses: {len(active_diagnoses)}")
        
        medications = patient_data.get('medications', [])
        active_medications = [m for m in medications if m.get('status') == 'active']
        content_parts.append(f"Active Medications: {len(active_medications)}")
        
        # Top conditions and medications
        if active_diagnoses:
            top_conditions = [d.get('description', 'Unknown') for d in active_diagnoses[:5]]
            content_parts.append(f"Primary Conditions: {', '.join(top_conditions)}")
        
        if active_medications:
            top_medications = [m.get('name', 'Unknown') for m in active_medications[:5]]
            content_parts.append(f"Key Medications: {', '.join(top_medications)}")
        
        return PatientChunk(
            content="\n".join(content_parts),
            chunk_type="summary",
            patient_id=patient_id,
            timestamp=datetime.now(),
            clinical_domain="general",
            priority=1,  # Summary is always high priority
            metadata={
                "is_summary": True,
                "total_diagnoses": len(diagnoses),
                "total_medications": len(medications),
                "active_diagnoses": len(active_diagnoses),
                "active_medications": len(active_medications)
            },
            chunk_id=f"{patient_id}_summary"
        )
    
    def _classify_clinical_domain(self, text: str) -> str:
        """Classify text into clinical domain."""
        if not text:
            return "general"
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.medical_domains.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else "general"
    
    def _classify_medication_domain(self, medication_name: str) -> str:
        """Classify medication into clinical domain based on common drug classes."""
        if not medication_name:
            return "general"
        
        med_lower = medication_name.lower()
        
        # Cardiovascular medications
        if any(med in med_lower for med in ['lisinopril', 'amlodipine', 'metoprolol', 'atorvastatin', 'warfarin', 'aspirin']):
            return "cardiovascular"
        
        # Diabetes medications
        elif any(med in med_lower for med in ['insulin', 'metformin', 'glyburide', 'glipizide', 'pioglitazone']):
            return "endocrine"
        
        # Respiratory medications
        elif any(med in med_lower for med in ['albuterol', 'fluticasone', 'prednisone', 'montelukast']):
            return "respiratory"
        
        # Psychiatric medications
        elif any(med in med_lower for med in ['sertraline', 'fluoxetine', 'lorazepam', 'quetiapine', 'lithium']):
            return "psychiatric"
        
        # Antibiotics and infections
        elif any(med in med_lower for med in ['amoxicillin', 'azithromycin', 'cephalexin', 'ciprofloxacin']):
            return "infectious"
        
        # Pain management
        elif any(med in med_lower for med in ['morphine', 'oxycodone', 'tramadol', 'ibuprofen', 'acetaminophen']):
            return "pain_management"
        
        return "general"
    
    def _classify_medication_therapeutic_class(self, medication_name: str) -> str:
        """Classify medication into therapeutic class."""
        if not medication_name:
            return "unknown"
        
        med_lower = medication_name.lower()
        
        therapeutic_classes = {
            'ACE Inhibitor': ['lisinopril', 'enalapril', 'captopril'],
            'Beta Blocker': ['metoprolol', 'atenolol', 'propranolol'],
            'Calcium Channel Blocker': ['amlodipine', 'nifedipine', 'diltiazem'],
            'Statin': ['atorvastatin', 'simvastatin', 'rosuvastatin'],
            'Diuretic': ['furosemide', 'hydrochlorothiazide', 'spironolactone'],
            'Antidiabetic': ['metformin', 'insulin', 'glyburide', 'glipizide'],
            'SSRI': ['sertraline', 'fluoxetine', 'paroxetine', 'escitalopram'],
            'Benzodiazepine': ['lorazepam', 'diazepam', 'alprazolam'],
            'Antibiotic': ['amoxicillin', 'azithromycin', 'cephalexin'],
            'Bronchodilator': ['albuterol', 'salmeterol', 'ipratropium'],
            'Corticosteroid': ['prednisone', 'hydrocortisone', 'methylprednisolone']
        }
        
        for therapeutic_class, medications in therapeutic_classes.items():
            if any(med in med_lower for med in medications):
                return therapeutic_class
        
        return "other"
    
    def _classify_lab_domain(self, lab_name: str) -> str:
        """Classify laboratory test into clinical domain."""
        if not lab_name:
            return "general"
        
        lab_lower = lab_name.lower()
        
        if any(lab in lab_lower for lab in ['glucose', 'hba1c', 'insulin', 'c-peptide']):
            return "endocrine"
        elif any(lab in lab_lower for lab in ['creatinine', 'bun', 'gfr', 'protein', 'albumin']):
            return "renal"
        elif any(lab in lab_lower for lab in ['alt', 'ast', 'bilirubin', 'alkaline phosphatase']):
            return "hepatic"
        elif any(lab in lab_lower for lab in ['hemoglobin', 'hematocrit', 'platelet', 'wbc', 'rbc']):
            return "hematology"
        elif any(lab in lab_lower for lab in ['cholesterol', 'triglyceride', 'ldl', 'hdl']):
            return "cardiovascular"
        elif any(lab in lab_lower for lab in ['tsh', 'free t4', 'free t3']):
            return "endocrine"
        
        return "general"
    
    def _classify_procedure_domain(self, procedure_name: str) -> str:
        """Classify procedure into clinical domain."""
        if not procedure_name:
            return "general"
        
        proc_lower = procedure_name.lower()
        
        if any(proc in proc_lower for proc in ['cardiac', 'heart', 'angiogram', 'ecg', 'echo', 'stress test']):
            return "cardiovascular"
        elif any(proc in proc_lower for proc in ['colonoscopy', 'endoscopy', 'upper gi', 'lower gi']):
            return "gastrointestinal"
        elif any(proc in proc_lower for proc in ['bronchoscopy', 'pulmonary function', 'chest x-ray']):
            return "respiratory"
        elif any(proc in proc_lower for proc in ['mri', 'ct scan', 'ultrasound', 'x-ray']):
            return "radiology"
        elif any(proc in proc_lower for proc in ['surgery', 'surgical', 'operation', 'biopsy']):
            return "surgical"
        
        return "general"
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string with multiple format support."""
        if not date_str:
            return None
        
        try:
            # Try ISO format first
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            try:
                # Try common formats
                for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
            except:
                logger.warning(f"Could not parse date: {date_str}")
                return None

class PatientDataEmbedder:
    """
    Advanced patient data embedding system for massive EHR datasets.
    Handles Burj Khalifa scale data with semantic search capabilities.
    """
    
    def __init__(
        self, 
        index_path: str = "patient_embeddings_faiss",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto"
    ):
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.device = self._determine_device(device)
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.chunker = MedicalPatientChunker()
        
        # Patient management
        self.embedded_patients = set()
        self.patient_metadata = {}
        
        # Performance tracking
        self.embedding_stats = {
            "total_patients_embedded": 0,
            "total_chunks_created": 0,
            "total_embedding_time": 0,
            "average_chunks_per_patient": 0
        }
        
        self.initialized = False
    
    def _determine_device(self, device: str) -> str:
        """Determine optimal device for embeddings."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    async def initialize(self):
        """Initialize the patient data embedding system."""
        try:
            logger.info(f"Initializing Patient Data Embedder with model: {self.embedding_model}")
            
            # Setup embeddings model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={
                    'device': self.device,
                    'trust_remote_code': True
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32  # Optimize for large datasets
                }
            )
            
            # Setup or load vector store
            await self._setup_vectorstore()
            
            self.initialized = True
            logger.info(f"✅ Patient Data Embedder initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Patient Data Embedder: {e}")
            raise
    
    async def _setup_vectorstore(self):
        """Setup or load patient vector store."""
        try:
            if os.path.exists(self.index_path):
                logger.info(f"Loading existing patient vector store from {self.index_path}")
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Load patient metadata
                metadata_path = f"{self.index_path}_metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.embedded_patients = set(metadata.get('embedded_patients', []))
                        self.patient_metadata = metadata.get('patient_metadata', {})
                        self.embedding_stats = metadata.get('embedding_stats', self.embedding_stats)
                
                logger.info(f"Loaded vector store with {len(self.embedded_patients)} patients")
            else:
                logger.info("Creating new patient vector store")
                await self._create_empty_vectorstore()
                
        except Exception as e:
            logger.error(f"Error setting up patient vector store: {e}")
            raise
    
    async def _create_empty_vectorstore(self):
        """Create empty patient vector store."""
        try:
            # Create dummy document to initialize FAISS
            dummy_doc = Document(
                page_content="Patient data embedding system initialization.",
                metadata={
                    "patient_id": "system",
                    "chunk_type": "initialization",
                    "is_dummy": True
                }
            )
            
            self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            await self._save_vectorstore()
            
            logger.info("Empty patient vector store created")
            
        except Exception as e:
            logger.error(f"Error creating empty vector store: {e}")
            raise
    
    async def embed_patient_data(
        self, 
        patient_id: str, 
        patient_data: Dict,
        force_reembed: bool = False
    ) -> Dict[str, Any]:
        """
        Embed complete patient data into vector store.
        Handles massive patient datasets efficiently.
        """
        try:
            if not self.initialized:
                raise RuntimeError("Patient Data Embedder not initialized")
            
            start_time = datetime.now()
            
            # Check if patient already embedded
            if patient_id in self.embedded_patients and not force_reembed:
                logger.info(f"Patient {patient_id} already embedded, skipping...")
                return {
                    "patient_id": patient_id,
                    "status": "already_embedded",
                    "chunks_created": self.patient_metadata.get(patient_id, {}).get('chunk_count', 0)
                }
            
            logger.info(f"Embedding patient data for {patient_id}...")
            
            # Create semantic chunks
            chunks = self.chunker.chunk_patient_data(patient_data)
            
            if not chunks:
                logger.warning(f"No chunks created for patient {patient_id}")
                return {
                    "patient_id": patient_id,
                    "status": "no_data",
                    "chunks_created": 0
                }
            
            # Convert chunks to Documents
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.content,
                    metadata={
                        "patient_id": chunk.patient_id,
                        "chunk_type": chunk.chunk_type,
                        "chunk_id": chunk.chunk_id,
                        "clinical_domain": chunk.clinical_domain,
                        "priority": chunk.priority,
                        "timestamp": chunk.timestamp.isoformat() if chunk.timestamp else None,
                        "embedded_at": datetime.now().isoformat(),
                        **(chunk.metadata or {})
                    }
                )
                documents.append(doc)
            
            # Add to vector store in batches for efficiency
            batch_size = 100  # Process in batches for large datasets
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vectorstore.add_documents(batch)
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            # Update tracking
            self.embedded_patients.add(patient_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.patient_metadata[patient_id] = {
                "chunk_count": len(chunks),
                "embedded_at": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "clinical_domains": list(set(chunk.clinical_domain for chunk in chunks)),
                "chunk_types": list(set(chunk.chunk_type for chunk in chunks))
            }
            
            # Update stats
            self.embedding_stats["total_patients_embedded"] += 1
            self.embedding_stats["total_chunks_created"] += len(chunks)
            self.embedding_stats["total_embedding_time"] += processing_time
            self.embedding_stats["average_chunks_per_patient"] = (
                self.embedding_stats["total_chunks_created"] / 
                self.embedding_stats["total_patients_embedded"]
            )
            
            # Save vector store and metadata
            await self._save_vectorstore()
            
            logger.info(f"✅ Patient {patient_id} embedded: {len(chunks)} chunks in {processing_time:.2f}s")
            
            return {
                "patient_id": patient_id,
                "status": "success",
                "chunks_created": len(chunks),
                "processing_time_seconds": processing_time,
                "clinical_domains": self.patient_metadata[patient_id]["clinical_domains"],
                "chunk_types": self.patient_metadata[patient_id]["chunk_types"]
            }
            
        except Exception as e:
            logger.error(f"Error embedding patient {patient_id}: {e}")
            return {
                "patient_id": patient_id,
                "status": "error",
                "error": str(e),
                "chunks_created": 0
            }
    
    async def search_patient_context(
        self,
        patient_id: str,
        query: str,
        k: int = 5,
        filter_chunk_types: Optional[List[str]] = None,
        filter_clinical_domains: Optional[List[str]] = None,
        include_summary: bool = True
    ) -> List[Document]:
        """
        Semantic search within specific patient's embedded data.
        Optimized for finding relevant context quickly.
        """
        try:
            if not self.initialized:
                raise RuntimeError("Patient Data Embedder not initialized")
            
            if patient_id not in self.embedded_patients:
                logger.warning(f"Patient {patient_id} not found in embedded data")
                return []
            
            # Build filters
            search_filter = {"patient_id": patient_id}
            
            # Always include summary chunk if requested
            search_results = []
            
            if include_summary:
                summary_results = self.vectorstore.similarity_search(
                    query,
                    k=1,
                    filter={
                        "patient_id": patient_id,
                        "chunk_type": "summary"
                    }
                )
                search_results.extend(summary_results)
            
            # Search with filters
            remaining_k = k - len(search_results)
            if remaining_k > 0:
                # Create filter dict
                filter_dict = {"patient_id": patient_id}
                
                if filter_chunk_types:
                    # Search each chunk type separately and combine
                    for chunk_type in filter_chunk_types:
                        type_results = self.vectorstore.similarity_search(
                            query,
                            k=max(1, remaining_k // len(filter_chunk_types)),
                            filter={
                                "patient_id": patient_id,
                                "chunk_type": chunk_type
                            }
                        )
                        search_results.extend(type_results)
                else:
                    # General search within patient
                    general_results = self.vectorstore.similarity_search(
                        query,
                        k=remaining_k * 2,  # Get more then filter
                        filter=filter_dict
                    )
                    
                    # Filter by clinical domain if specified
                    if filter_clinical_domains:
                        filtered_results = [
                            doc for doc in general_results 
                            if doc.metadata.get('clinical_domain') in filter_clinical_domains
                        ]
                        search_results.extend(filtered_results[:remaining_k])
                    else:
                        search_results.extend(general_results[:remaining_k])
            
            # Remove duplicates while preserving order
            seen_chunk_ids = set()
            unique_results = []
            for doc in search_results:
                chunk_id = doc.metadata.get('chunk_id', '')
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    unique_results.append(doc)
            
            # Sort by priority and relevance
            unique_results.sort(key=lambda x: (
                x.metadata.get('priority', 3),  # Lower priority number = higher priority
                -len(x.page_content)  # Longer content slightly preferred
            ))
            
            logger.info(f"Found {len(unique_results)} relevant chunks for patient {patient_id}")
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"Error searching patient context for {patient_id}: {e}")
            return []
    
    async def get_patient_timeline_context(
        self,
        patient_id: str,
        days_back: int = 90,
        k: int = 10
    ) -> List[Document]:
        """Get recent patient context based on timeline."""
        try:
            if patient_id not in self.embedded_patients:
                return []
            
            # Search for recent chunks
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Get all patient chunks
            all_chunks = self.vectorstore.similarity_search(
                "",  # Empty query to get all
                k=1000,  # Get many chunks
                filter={"patient_id": patient_id}
            )
            
            # Filter by timestamp
            recent_chunks = []
            for chunk in all_chunks:
                timestamp_str = chunk.metadata.get('timestamp')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if timestamp >= cutoff_date:
                            recent_chunks.append(chunk)
                    except:
                        continue
            
            # Sort by timestamp (newest first) and priority
            recent_chunks.sort(key=lambda x: (
                x.metadata.get('priority', 3),
                -(datetime.fromisoformat(x.metadata.get('timestamp', '1900-01-01').replace('Z', '+00:00')).timestamp())
            ))
            
            return recent_chunks[:k]
            
        except Exception as e:
            logger.error(f"Error getting timeline context for {patient_id}: {e}")
            return []
    
    async def _save_vectorstore(self):
        """Save vector store and metadata."""
        try:
            # Save FAISS index
            self.vectorstore.save_local(self.index_path)
            
            # Save metadata
            metadata = {
                "embedded_patients": list(self.embedded_patients),
                "patient_metadata": self.patient_metadata,
                "embedding_stats": self.embedding_stats,
                "last_updated": datetime.now().isoformat(),
                "embedding_model": self.embedding_model,
                "device": self.device
            }
            
            metadata_path = f"{self.index_path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get comprehensive embedding statistics."""
        try:
            # Calculate additional stats
            total_chunks = self.embedding_stats.get("total_chunks_created", 0)
            total_patients = self.embedding_stats.get("total_patients_embedded", 0)
            
            # Domain distribution
            domain_counts = {}
            chunk_type_counts = {}
            
            for patient_id, metadata in self.patient_metadata.items():
                for domain in metadata.get("clinical_domains", []):
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
                for chunk_type in metadata.get("chunk_types", []):
                    chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
            
            return {
                "embedding_performance": self.embedding_stats,
                "patient_coverage": {
                    "total_patients_embedded": total_patients,
                    "embedded_patient_ids": list(self.embedded_patients),
                    "total_chunks": total_chunks
                },
                "content_distribution": {
                    "clinical_domains": domain_counts,
                    "chunk_types": chunk_type_counts
                },
                "system_info": {
                    "embedding_model": self.embedding_model,
                    "device": self.device,
                    "index_path": self.index_path,
                    "initialized": self.initialized
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            return {"error": str(e)}
    
    async def remove_patient_embeddings(self, patient_id: str) -> bool:
        """Remove all embeddings for a specific patient."""
        try:
            if patient_id not in self.embedded_patients:
                logger.warning(f"Patient {patient_id} not found in embeddings")
                return False
            
            # Note: FAISS doesn't support deletion by filter directly
            # For production, consider using a vector DB that supports deletion
            # For now, we'll mark as removed and rebuild periodically
            
            self.embedded_patients.discard(patient_id)
            if patient_id in self.patient_metadata:
                del self.patient_metadata[patient_id]
            
            await self._save_vectorstore()
            
            logger.info(f"Patient {patient_id} removed from embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error removing patient embeddings for {patient_id}: {e}")
            return False
    
    def is_patient_embedded(self, patient_id: str) -> bool:
        """Check if patient data is already embedded."""
        return patient_id in self.embedded_patients
    
    async def batch_embed_patients(
        self,
        patient_data_list: List[Tuple[str, Dict]],
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Embed multiple patients in batches for efficiency."""
        results = []
        
        for i in range(0, len(patient_data_list), batch_size):
            batch = patient_data_list[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(patient_data_list)-1)//batch_size + 1}")
            
            batch_tasks = [
                self.embed_patient_data(patient_id, patient_data)
                for patient_id, patient_data in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            
            # Brief pause between batches to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        return results