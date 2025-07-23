"""
Enhanced EHR Database Schema based on FHIR standards.
Designed for real-world clinical data complexity and AI processing.
FIXED VERSION - Added missing database setup functions.
"""

from sqlalchemy import (
    Column, String, Integer, DateTime, Text, ForeignKey, 
    Boolean, Numeric, JSON, Enum, Index, UniqueConstraint,
    create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum as PyEnum
import uuid
import os
import logging

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

# Enums for standardized values
class IdentifierTypeEnum(PyEnum):
    MRN = "medical_record_number"
    SSN = "social_security_number"
    DRIVER_LICENSE = "drivers_license"
    PASSPORT = "passport"
    INSURANCE = "insurance_id"

class EncounterStatusEnum(PyEnum):
    PLANNED = "planned"
    IN_PROGRESS = "in-progress" 
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ENTERED_IN_ERROR = "entered-in-error"

class EncounterTypeEnum(PyEnum):
    INPATIENT = "inpatient"
    OUTPATIENT = "outpatient"
    EMERGENCY = "emergency"
    VIRTUAL = "virtual"
    HOME_HEALTH = "home-health"

class DiagnosisStatusEnum(PyEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    RESOLVED = "resolved" 
    CHRONIC = "chronic"
    PROVISIONAL = "provisional"

class MedicationStatusEnum(PyEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISCONTINUED = "discontinued"
    COMPLETED = "completed"
    ON_HOLD = "on-hold"

class AllergyStatusEnum(PyEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    RESOLVED = "resolved"

class AllergySeverityEnum(PyEnum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life-threatening"

# Core Patient and Demographics
class Patient(Base):
    """Enhanced patient model with comprehensive demographics."""
    __tablename__ = "patients"
    
    # Primary identifiers
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Demographics
    first_name = Column(String(100), nullable=False)
    middle_name = Column(String(100))
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(DateTime, nullable=False)
    gender = Column(String(20))
    race = Column(String(50))
    ethnicity = Column(String(50))
    preferred_language = Column(String(50))
    
    # Contact information
    phone_primary = Column(String(20))
    phone_secondary = Column(String(20))
    email = Column(String(100))
    
    # Address information (JSON for flexibility)
    address = Column(JSON)  # {street, city, state, zip, country, type}
    
    # Status and metadata
    active = Column(Boolean, default=True)
    deceased = Column(Boolean, default=False)
    deceased_date = Column(DateTime)
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    identifiers = relationship("PatientIdentifier", back_populates="patient", cascade="all, delete-orphan")
    encounters = relationship("Encounter", back_populates="patient")
    conditions = relationship("Condition", back_populates="patient")
    medications = relationship("MedicationStatement", back_populates="patient")
    allergies = relationship("AllergyIntolerance", back_populates="patient")
    observations = relationship("Observation", back_populates="patient")
    procedures = relationship("Procedure", back_populates="patient")
    care_plans = relationship("CarePlan", back_populates="patient")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_patient_name', 'last_name', 'first_name'),
        Index('idx_patient_dob', 'date_of_birth'),
        Index('idx_patient_active', 'active'),
    )

class PatientIdentifier(Base):
    """Multiple identifier types per patient (MRN, SSN, etc.)."""
    __tablename__ = "patient_identifiers"
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    identifier_type = Column(Enum(IdentifierTypeEnum), nullable=False)
    value = Column(String(100), nullable=False)
    system = Column(String(200))  # Issuing organization/system
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    active = Column(Boolean, default=True)
    
    # Relationships
    patient = relationship("Patient", back_populates="identifiers")
    
    __table_args__ = (
        UniqueConstraint('identifier_type', 'value', name='uq_identifier_type_value'),
        Index('idx_identifier_value', 'identifier_type', 'value'),
    )

# Healthcare Encounters
class Encounter(Base):
    """Healthcare encounters (visits, admissions, etc.)."""
    __tablename__ = "encounters"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    
    # Encounter details
    status = Column(Enum(EncounterStatusEnum), nullable=False)
    encounter_type = Column(Enum(EncounterTypeEnum), nullable=False)
    priority = Column(String(20))  # routine, urgent, emergency
    
    # Timing
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration_minutes = Column(Integer)
    
    # Location and providers
    location = Column(String(200))
    department = Column(String(100))
    primary_provider_id = Column(String, ForeignKey("providers.id"))
    
    # Clinical data
    chief_complaint = Column(Text)
    reason_code = Column(String(50))  # ICD-10 or SNOMED
    reason_description = Column(Text)
    
    # Disposition
    disposition = Column(String(100))  # discharged home, admitted, transferred
    discharge_instructions = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="encounters")
    primary_provider = relationship("Provider")
    diagnoses = relationship("EncounterDiagnosis", back_populates="encounter")
    procedures = relationship("Procedure", back_populates="encounter")
    observations = relationship("Observation", back_populates="encounter")
    notes = relationship("ClinicalNote", back_populates="encounter")
    
    __table_args__ = (
        Index('idx_encounter_patient_date', 'patient_id', 'start_time'),
        Index('idx_encounter_type_date', 'encounter_type', 'start_time'),
    )

class Provider(Base):
    """Healthcare providers."""
    __tablename__ = "providers"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    npi = Column(String(20), unique=True)  # National Provider Identifier
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    specialty = Column(String(100))
    department = Column(String(100))
    title = Column(String(100))
    active = Column(Boolean, default=True)

# Medical Conditions and Diagnoses  
class Condition(Base):
    """Patient medical conditions (problems, diagnoses)."""
    __tablename__ = "conditions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    
    # Condition details
    code = Column(String(20))  # ICD-10, SNOMED-CT
    code_system = Column(String(50))  # Which coding system
    display_name = Column(String(500), nullable=False)
    category = Column(String(100))  # problem-list-item, encounter-diagnosis
    
    # Clinical status
    clinical_status = Column(Enum(DiagnosisStatusEnum), nullable=False)
    verification_status = Column(String(20))  # confirmed, provisional, differential
    severity = Column(String(20))  # mild, moderate, severe
    
    # Timing
    onset_date = Column(DateTime)
    onset_type = Column(String(50))  # sudden, gradual, chronic
    resolved_date = Column(DateTime)
    
    # Context
    recorded_date = Column(DateTime, default=datetime.utcnow)
    recorded_by_id = Column(String, ForeignKey("providers.id"))
    encounter_id = Column(String, ForeignKey("encounters.id"))
    
    # Additional data
    body_site = Column(String(100))  # anatomical location
    stage = Column(JSON)  # disease staging information
    evidence = Column(JSON)  # supporting evidence references
    notes = Column(Text)
    
    # Relationships
    patient = relationship("Patient", back_populates="conditions")
    recorded_by = relationship("Provider")
    encounter = relationship("Encounter")
    
    __table_args__ = (
        Index('idx_condition_patient_status', 'patient_id', 'clinical_status'),
        Index('idx_condition_code', 'code'),
        Index('idx_condition_date', 'onset_date'),
    )

class EncounterDiagnosis(Base):
    """Links diagnoses to specific encounters."""
    __tablename__ = "encounter_diagnoses"
    
    id = Column(Integer, primary_key=True)
    encounter_id = Column(String, ForeignKey("encounters.id"), nullable=False)
    condition_id = Column(String, ForeignKey("conditions.id"), nullable=False)
    rank = Column(Integer)  # primary, secondary, etc.
    diagnosis_type = Column(String(20))  # admitting, principal, discharge
    
    # Relationships
    encounter = relationship("Encounter", back_populates="diagnoses")
    condition = relationship("Condition")

# Medications
class MedicationStatement(Base):
    """Patient medication history and current medications."""
    __tablename__ = "medication_statements"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    
    # Medication identification
    medication_code = Column(String(50))  # RxNorm, NDC
    medication_name = Column(String(500), nullable=False)
    generic_name = Column(String(500))
    brand_name = Column(String(500))
    
    # Dosage and administration
    dosage_text = Column(String(500))  # "10mg twice daily"
    dose_quantity = Column(Numeric(10, 3))
    dose_unit = Column(String(50))  # mg, mcg, units
    frequency = Column(String(100))  # BID, TID, QD, PRN
    route = Column(String(50))  # oral, IV, IM, topical
    
    # Status and timing
    status = Column(Enum(MedicationStatusEnum), nullable=False)
    effective_start = Column(DateTime, nullable=False)
    effective_end = Column(DateTime)
    
    # Clinical context
    indication = Column(String(500))  # reason for prescription
    prescriber_id = Column(String, ForeignKey("providers.id"))
    encounter_id = Column(String, ForeignKey("encounters.id"))
    
    # Instructions and notes
    patient_instructions = Column(Text)
    pharmacy_instructions = Column(Text)
    notes = Column(Text)
    
    # Metadata
    recorded_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="medications")
    prescriber = relationship("Provider")
    encounter = relationship("Encounter")
    
    __table_args__ = (
        Index('idx_medication_patient_status', 'patient_id', 'status'),
        Index('idx_medication_name', 'medication_name'),
        Index('idx_medication_dates', 'effective_start', 'effective_end'),
    )

# Allergies and Intolerances
class AllergyIntolerance(Base):
    """Patient allergies and intolerances."""
    __tablename__ = "allergy_intolerances"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    
    # Allergen details
    allergen_code = Column(String(50))  # SNOMED, RxNorm
    allergen_name = Column(String(500), nullable=False)
    allergen_type = Column(String(50))  # medication, food, environment
    
    # Allergy characteristics
    status = Column(Enum(AllergyStatusEnum), nullable=False)
    type = Column(String(20))  # allergy, intolerance, contraindication
    category = Column(String(20))  # medication, food, environment
    criticality = Column(String(20))  # low, high, unable-to-assess
    severity = Column(Enum(AllergySeverityEnum))
    
    # Reaction details
    reaction_manifestation = Column(ARRAY(String))  # rash, nausea, anaphylaxis
    reaction_description = Column(Text)
    onset_date = Column(DateTime)
    
    # Context
    recorded_date = Column(DateTime, default=datetime.utcnow)
    recorded_by_id = Column(String, ForeignKey("providers.id"))
    
    # Relationships
    patient = relationship("Patient", back_populates="allergies")
    recorded_by = relationship("Provider")
    
    __table_args__ = (
        Index('idx_allergy_patient_status', 'patient_id', 'status'),
        Index('idx_allergy_type', 'allergen_type'),
    )

# Clinical Observations and Lab Results
class Observation(Base):
    """Clinical observations, vital signs, lab results."""
    __tablename__ = "observations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    encounter_id = Column(String, ForeignKey("encounters.id"))
    
    # Observation identification
    code = Column(String(50), nullable=False)  # LOINC, SNOMED
    display_name = Column(String(500), nullable=False)
    category = Column(String(100))  # vital-signs, laboratory, survey
    
    # Values
    value_quantity = Column(Numeric(12, 4))
    value_unit = Column(String(50))
    value_string = Column(String(500))
    value_boolean = Column(Boolean)
    value_datetime = Column(DateTime)
    
    # Reference ranges and interpretation
    reference_range_low = Column(Numeric(12, 4))
    reference_range_high = Column(Numeric(12, 4))
    interpretation = Column(String(20))  # normal, high, low, critical
    
    # Context
    status = Column(String(20), default='final')  # preliminary, final, amended
    effective_datetime = Column(DateTime, nullable=False)
    issued_datetime = Column(DateTime)
    
    # Performer and device
    performer_id = Column(String, ForeignKey("providers.id"))
    device_name = Column(String(200))
    method = Column(String(200))
    
    # Additional data
    body_site = Column(String(100))
    specimen_type = Column(String(100))
    notes = Column(Text)
    
    # Relationships
    patient = relationship("Patient", back_populates="observations")
    encounter = relationship("Encounter", back_populates="observations")
    performer = relationship("Provider")
    
    __table_args__ = (
        Index('idx_observation_patient_date', 'patient_id', 'effective_datetime'),
        Index('idx_observation_code', 'code'),
        Index('idx_observation_category', 'category'),
    )

# Procedures
class Procedure(Base):
    """Medical procedures performed."""
    __tablename__ = "procedures"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    encounter_id = Column(String, ForeignKey("encounters.id"))
    
    # Procedure identification
    code = Column(String(50))  # CPT, ICD-10-PCS, SNOMED
    display_name = Column(String(500), nullable=False)
    category = Column(String(100))  # surgical, diagnostic, therapeutic
    
    # Status and timing
    status = Column(String(20), nullable=False)  # completed, in-progress, stopped
    performed_datetime = Column(DateTime)
    performed_start = Column(DateTime)
    performed_end = Column(DateTime)
    
    # Clinical details
    body_site = Column(String(100))
    outcome = Column(String(200))
    complications = Column(Text)
    
    # Performers
    primary_performer_id = Column(String, ForeignKey("providers.id"))
    location = Column(String(200))
    
    # Additional data
    indication = Column(String(500))  # reason for procedure
    notes = Column(Text)
    follow_up = Column(Text)
    
    # Relationships
    patient = relationship("Patient", back_populates="procedures")
    encounter = relationship("Encounter", back_populates="procedures")
    primary_performer = relationship("Provider")
    
    __table_args__ = (
        Index('idx_procedure_patient_date', 'patient_id', 'performed_datetime'),
        Index('idx_procedure_code', 'code'),
    )

# Care Planning
class CarePlan(Base):
    """Patient care plans and treatment goals."""
    __tablename__ = "care_plans"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    
    # Plan details
    title = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(String(20), nullable=False)  # active, completed, cancelled
    intent = Column(String(20))  # proposal, plan, order
    
    # Timing
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Context
    created_by_id = Column(String, ForeignKey("providers.id"))
    addresses = Column(ARRAY(String))  # condition IDs this plan addresses
    
    # Goals and activities
    goals = Column(JSON)  # Array of goal objects
    activities = Column(JSON)  # Array of activity objects
    
    # Relationships
    patient = relationship("Patient", back_populates="care_plans")
    created_by = relationship("Provider")

# Clinical Documentation
class ClinicalNote(Base):
    """Clinical notes and documentation."""
    __tablename__ = "clinical_notes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    encounter_id = Column(String, ForeignKey("encounters.id"))
    
    # Note classification
    note_type = Column(String(100), nullable=False)  # progress, discharge, consultation
    specialty = Column(String(100))  # cardiology, psychiatry, etc.
    
    # Content
    title = Column(String(200))
    content = Column(Text, nullable=False)
    status = Column(String(20), default='final')  # draft, final, amended
    
    # Authorship
    author_id = Column(String, ForeignKey("providers.id"), nullable=False)
    dictated_datetime = Column(DateTime)
    transcribed_datetime = Column(DateTime)
    authenticated_datetime = Column(DateTime)
    
    # Metadata for AI processing
    processed_by_ai = Column(Boolean, default=False)
    extracted_entities = Column(JSON)  # NLP-extracted entities
    sentiment_score = Column(Numeric(3, 2))
    readability_score = Column(Numeric(5, 2))
    
    # Relationships
    patient = relationship("Patient")
    encounter = relationship("Encounter", back_populates="notes")
    author = relationship("Provider")
    
    __table_args__ = (
        Index('idx_note_patient_date', 'patient_id', 'dictated_datetime'),
        Index('idx_note_type', 'note_type'),
        Index('idx_note_ai_processed', 'processed_by_ai'),
    )

# Summary tables for AI optimization
class PatientSummary(Base):
    """AI-generated patient summaries for optimization."""
    __tablename__ = "patient_summaries"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False, unique=True)
    
    # Summaries by time period
    summary_current = Column(Text)  # Current active issues
    summary_recent = Column(Text)   # Last 6 months
    summary_historical = Column(Text)  # Historical overview
    summary_chronic = Column(Text)  # Chronic conditions
    
    # Key clinical facts (for quick reference)
    active_diagnoses = Column(ARRAY(String))
    current_medications = Column(ARRAY(String))
    allergies = Column(ARRAY(String))
    chronic_conditions = Column(ARRAY(String))
    
    # AI metadata
    last_updated = Column(DateTime, default=datetime.utcnow)
    generated_by_model = Column(String(100))
    confidence_score = Column(Numeric(3, 2))
    
    # Relationships
    patient = relationship("Patient")
    
    __table_args__ = (
        Index('idx_summary_updated', 'last_updated'),
    )

# Database Setup and Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./enhanced_ehr.db")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def initialize_database():
    """Initialize the database by creating all tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        return False

def get_db():
    """Dependency to get database session for FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_patient_data(db, patient_id: str) -> Optional[Dict]:
    """Get comprehensive patient data for a given patient ID."""
    try:
        # Query patient with eager loading of relationships
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        
        if not patient:
            logger.warning(f"Patient {patient_id} not found")
            return None
        
        # Build comprehensive patient data structure
        patient_data = {
            "patient_id": patient.id,
            "demographics": {
                "name": f"{patient.first_name} {patient.last_name}",
                "age": _calculate_age(patient.date_of_birth),
                "gender": patient.gender,
                "race": patient.race,
                "ethnicity": patient.ethnicity,
                "preferred_language": patient.preferred_language,
                "phone": patient.phone_primary,
                "email": patient.email
            },
            "identifiers": [
                {
                    "type": identifier.identifier_type.value,
                    "value": identifier.value,
                    "system": identifier.system
                }
                for identifier in patient.identifiers
            ],
            "diagnoses": [
                {
                    "code": condition.code,
                    "description": condition.display_name,
                    "status": condition.clinical_status.value if condition.clinical_status else "unknown",
                    "diagnosis_date": condition.onset_date.isoformat() if condition.onset_date else None,
                    "date": condition.onset_date.isoformat() if condition.onset_date else None,
                    "severity": condition.severity,
                    "icd_code": condition.code,
                    "category": condition.category,
                    "notes": condition.notes
                }
                for condition in patient.conditions
            ],
            "medications": [
                {
                    "name": med.medication_name,
                    "generic_name": med.generic_name,
                    "brand_name": med.brand_name,
                    "dosage": med.dosage_text,
                    "frequency": med.frequency,
                    "route": med.route,
                    "status": med.status.value if med.status else "unknown",
                    "start_date": med.effective_start.isoformat() if med.effective_start else None,
                    "end_date": med.effective_end.isoformat() if med.effective_end else None,
                    "indication": med.indication,
                    "instructions": med.patient_instructions
                }
                for med in patient.medications
            ],
            "allergies": [
                {
                    "allergen_name": allergy.allergen_name,
                    "type": allergy.type,
                    "status": allergy.status.value if allergy.status else "unknown",
                    "severity": allergy.severity.value if allergy.severity else "unknown",
                    "reaction": allergy.reaction_description,
                    "onset_date": allergy.onset_date.isoformat() if allergy.onset_date else None
                }
                for allergy in patient.allergies
            ],
            "visit_history": [
                {
                    "visit_date": encounter.start_time.isoformat() if encounter.start_time else None,
                    "date": encounter.start_time.isoformat() if encounter.start_time else None,
                    "visit_type": encounter.encounter_type.value if encounter.encounter_type else "unknown",
                    "chief_complaint": encounter.chief_complaint,
                    "provider": f"{encounter.primary_provider.first_name} {encounter.primary_provider.last_name}" if encounter.primary_provider else None,
                    "location": encounter.location,
                    "disposition": encounter.disposition,
                    "notes": encounter.discharge_instructions or ""
                }
                for encounter in patient.encounters
            ],
            "laboratory_results": [
                {
                    "code": obs.code,
                    "name": obs.display_name,
                    "value": float(obs.value_quantity) if obs.value_quantity else obs.value_string,
                    "unit": obs.value_unit,
                    "date": obs.effective_datetime.isoformat() if obs.effective_datetime else None,
                    "interpretation": obs.interpretation,
                    "reference_range_low": float(obs.reference_range_low) if obs.reference_range_low else None,
                    "reference_range_high": float(obs.reference_range_high) if obs.reference_range_high else None
                }
                for obs in patient.observations
            ]
        }
        
        logger.info(f"Retrieved comprehensive data for patient {patient_id}")
        return patient_data
        
    except Exception as e:
        logger.error(f"Error retrieving patient data for {patient_id}: {e}")
        return None

def _calculate_age(birth_date) -> int:
    """Calculate patient age from birth date."""
    if not birth_date:
        return 0
    
    today = datetime.now()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

# Test database connection
def test_database_connection():
    """Test database connection."""
    try:
        from sqlalchemy import text  # Add this import
        db = SessionLocal()
        # Try a simple query (fixed for newer SQLAlchemy versions)
        result = db.execute(text("SELECT 1")).scalar()  # Wrap in text()
        db.close()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False