"""
Enhanced EHR Clinical AI System
Advanced AI-powered clinical decision support with scalable patient history processing
"""

__version__ = "2.0.0"
__author__ = "EHR Clinical AI Team"
__description__ = "Enhanced AI-powered clinical decision support system"

# Package-level imports for convenience
from .enhanced_ehr_schema import (
    Patient, PatientIdentifier, Encounter, Provider, Condition,
    MedicationStatement, AllergyIntolerance, Observation, Procedure,
    ClinicalNote, CarePlan, PatientSummary,
    get_db, get_patient_data, initialize_database
)

__all__ = [
    "Patient", "PatientIdentifier", "Encounter", "Provider", "Condition",
    "MedicationStatement", "AllergyIntolerance", "Observation", "Procedure", 
    "ClinicalNote", "CarePlan", "PatientSummary",
    "get_db", "get_patient_data", "initialize_database"
]