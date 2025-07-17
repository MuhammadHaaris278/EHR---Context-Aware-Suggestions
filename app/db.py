"""
Database layer for patient data retrieval.
Handles connection to EHR database and patient information queries.
"""

import os
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Database URL from environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://user:password@localhost/ehr_db"
)

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Patient(Base):
    """Patient information model."""
    __tablename__ = "patients"
    
    id = Column(String, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    date_of_birth = Column(DateTime, nullable=False)
    gender = Column(String, nullable=True)
    
    # Relationships
    diagnoses = relationship("Diagnosis", back_populates="patient")
    medications = relationship("Medication", back_populates="patient")
    visits = relationship("Visit", back_populates="patient")

class Diagnosis(Base):
    """Patient diagnosis model."""
    __tablename__ = "diagnoses"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    icd_code = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    diagnosis_date = Column(DateTime, nullable=False)
    status = Column(String, default="active")  # active, resolved, chronic
    
    # Relationships
    patient = relationship("Patient", back_populates="diagnoses")

class Medication(Base):
    """Patient medication model."""
    __tablename__ = "medications"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    name = Column(String, nullable=False)
    dosage = Column(String, nullable=True)
    frequency = Column(String, nullable=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)
    status = Column(String, default="active")  # active, discontinued, completed
    
    # Relationships
    patient = relationship("Patient", back_populates="medications")

class Visit(Base):
    """Patient visit model."""
    __tablename__ = "visits"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    visit_date = Column(DateTime, nullable=False)
    visit_type = Column(String, nullable=False)  # routine, follow-up, emergency
    chief_complaint = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    provider = Column(String, nullable=True)
    
    # Relationships
    patient = relationship("Patient", back_populates="visits")

# Database dependency
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_patient_data(db: Session, patient_id: str) -> Optional[Dict]:
    """
    Retrieve comprehensive patient data for AI processing.
    
    Args:
        db: Database session
        patient_id: Patient identifier
        
    Returns:
        Dictionary containing patient information or None if not found
    """
    try:
        # Query patient with all related data
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        
        if not patient:
            logger.warning(f"Patient {patient_id} not found in database")
            return None
        
        # Get active diagnoses
        active_diagnoses = db.query(Diagnosis).filter(
            Diagnosis.patient_id == patient_id,
            Diagnosis.status == "active"
        ).all()
        
        # Get current medications
        current_medications = db.query(Medication).filter(
            Medication.patient_id == patient_id,
            Medication.status == "active"
        ).all()
        
        # Get recent visits (last 6 months)
        recent_visits = db.query(Visit).filter(
            Visit.patient_id == patient_id
        ).order_by(Visit.visit_date.desc()).limit(10).all()
        
        # Format patient data for AI processing
        patient_data = {
            "patient_id": patient.id,
            "demographics": {
                "age": calculate_age(patient.date_of_birth),
                "gender": patient.gender,
                "name": f"{patient.first_name} {patient.last_name}"
            },
            "diagnoses": [
                {
                    "icd_code": d.icd_code,
                    "description": d.description,
                    "date": d.diagnosis_date.isoformat(),
                    "status": d.status
                }
                for d in active_diagnoses
            ],
            "medications": [
                {
                    "name": m.name,
                    "dosage": m.dosage,
                    "frequency": m.frequency,
                    "start_date": m.start_date.isoformat(),
                    "status": m.status
                }
                for m in current_medications
            ],
            "visit_history": [
                {
                    "date": v.visit_date.isoformat(),
                    "type": v.visit_type,
                    "chief_complaint": v.chief_complaint,
                    "provider": v.provider,
                    "notes": v.notes[:500] if v.notes else None  # Truncate long notes
                }
                for v in recent_visits
            ]
        }
        
        logger.info(f"Retrieved data for patient {patient_id}")
        return patient_data
        
    except Exception as e:
        logger.error(f"Error retrieving patient data: {e}")
        return None

def calculate_age(birth_date) -> int:
    """Calculate patient age from birth date."""
    from datetime import datetime
    today = datetime.now()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

def create_tables():
    """Create database tables if they don't exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

# Initialize database on import
if __name__ == "__main__":
    create_tables()