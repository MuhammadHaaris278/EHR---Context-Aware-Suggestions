#!/usr/bin/env python3
"""
Comprehensive EHR Data Insertion Script
Inserts 60+ years of medical history for an 80-year-old patient
Uses the enhanced_ehr_schema.py models to populate PostgreSQL database
"""
import os
import sys
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Any

try:
    # Add parent directory to path to import enhanced_ehr_schema
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from .enhanced_ehr_schema import (
        Base, SessionLocal, initialize_database, engine,
        Patient, PatientIdentifier, Provider, Encounter, Condition, 
        MedicationStatement, AllergyIntolerance, Observation, Procedure,
        ClinicalNote, CarePlan, PatientSummary, EncounterDiagnosis,
        IdentifierTypeEnum, EncounterStatusEnum, EncounterTypeEnum,
        DiagnosisStatusEnum, MedicationStatusEnum, AllergyStatusEnum, 
        AllergySeverityEnum
    )
except ImportError as e:
    print(f"Error importing schema: {e}")
    print("Make sure enhanced_ehr_schema.py is in your Python path")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ehr_data_insertion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EHRDataInserter:
    """Comprehensive EHR data insertion class"""
    
    def __init__(self):
        self.session = SessionLocal()
        self.patient_id = "1"
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.session.rollback()
            logger.error(f"Error occurred: {exc_val}")
        else:
            self.session.commit()
        self.session.close()
        
    def insert_providers(self) -> None:
        """Insert healthcare providers"""
        logger.info("Inserting healthcare providers...")
        
        providers_data = [
            # Primary Care Physicians
            {"id": "prov_001", "npi": "1234567890", "first_name": "Robert", "last_name": "Thompson", "specialty": "Family Medicine", "department": "Primary Care", "title": "MD", "active": True},
            {"id": "prov_002", "npi": "1234567891", "first_name": "William", "last_name": "Anderson", "specialty": "Internal Medicine", "department": "Primary Care", "title": "MD", "active": False},
            {"id": "prov_003", "npi": "1234567892", "first_name": "Margaret", "last_name": "Foster", "specialty": "Geriatrics", "department": "Internal Medicine", "title": "MD", "active": True},
            
            # Specialists
            {"id": "prov_004", "npi": "1234567893", "first_name": "Sarah", "last_name": "Williams", "specialty": "Cardiology", "department": "Cardiology", "title": "MD", "active": True},
            {"id": "prov_005", "npi": "1234567894", "first_name": "Michael", "last_name": "Johnson", "specialty": "Endocrinology", "department": "Endocrinology", "title": "MD", "active": True},
            {"id": "prov_006", "npi": "1234567895", "first_name": "Lisa", "last_name": "Davis", "specialty": "Orthopedics", "department": "Orthopedics", "title": "MD", "active": True},
            {"id": "prov_007", "npi": "1234567896", "first_name": "James", "last_name": "Wilson", "specialty": "Gastroenterology", "department": "Gastroenterology", "title": "MD", "active": True},
            {"id": "prov_008", "npi": "1234567897", "first_name": "Patricia", "last_name": "Brown", "specialty": "Ophthalmology", "department": "Ophthalmology", "title": "MD", "active": True},
            {"id": "prov_009", "npi": "1234567898", "first_name": "David", "last_name": "Miller", "specialty": "Emergency Medicine", "department": "Emergency", "title": "MD", "active": True},
            {"id": "prov_010", "npi": "1234567899", "first_name": "Mary", "last_name": "Garcia", "specialty": "Nephrology", "department": "Nephrology", "title": "MD", "active": True},
            {"id": "prov_011", "npi": "1234568900", "first_name": "Richard", "last_name": "Martinez", "specialty": "Neurology", "department": "Neurology", "title": "MD", "active": True},
            {"id": "prov_012", "npi": "1234568901", "first_name": "Jennifer", "last_name": "Rodriguez", "specialty": "Oncology", "department": "Oncology", "title": "MD", "active": True},
            {"id": "prov_013", "npi": "1234568902", "first_name": "Charles", "last_name": "Lee", "specialty": "Pulmonology", "department": "Pulmonology", "title": "MD", "active": True},
            {"id": "prov_014", "npi": "1234568903", "first_name": "Susan", "last_name": "Taylor", "specialty": "Dermatology", "department": "Dermatology", "title": "MD", "active": True},
            {"id": "prov_015", "npi": "1234568904", "first_name": "Daniel", "last_name": "Clark", "specialty": "Urology", "department": "Urology", "title": "MD", "active": True},
            {"id": "prov_016", "npi": "1234568905", "first_name": "Linda", "last_name": "White", "specialty": "Rheumatology", "department": "Rheumatology", "title": "MD", "active": True},
            {"id": "prov_017", "npi": "1234568906", "first_name": "Paul", "last_name": "Harris", "specialty": "Psychiatry", "department": "Psychiatry", "title": "MD", "active": True},
            {"id": "prov_018", "npi": "1234568907", "first_name": "Nancy", "last_name": "Martin", "specialty": "Gynecology", "department": "Women's Health", "title": "MD", "active": False},
            {"id": "prov_019", "npi": "1234568908", "first_name": "Kevin", "last_name": "Thompson", "specialty": "Anesthesiology", "department": "Anesthesiology", "title": "MD", "active": True},
            {"id": "prov_020", "npi": "1234568909", "first_name": "Barbara", "last_name": "Jackson", "specialty": "Radiology", "department": "Radiology", "title": "MD", "active": True},
            {"id": "prov_021", "npi": "1234568910", "first_name": "Steven", "last_name": "Young", "specialty": "Cardiothoracic Surgery", "department": "Surgery", "title": "MD", "active": True},
            {"id": "prov_022", "npi": "1234568911", "first_name": "Helen", "last_name": "King", "specialty": "Infectious Disease", "department": "Internal Medicine", "title": "MD", "active": True},
            {"id": "prov_023", "npi": "1234568912", "first_name": "Robert", "last_name": "Wright", "specialty": "General Surgery", "department": "Surgery", "title": "MD", "active": True},
            {"id": "prov_024", "npi": "1234568913", "first_name": "Carol", "last_name": "Lopez", "specialty": "Endocrinology", "department": "Endocrinology", "title": "MD", "active": True},
            {"id": "prov_025", "npi": "1234568914", "first_name": "Mark", "last_name": "Hill", "specialty": "Pathology", "department": "Pathology", "title": "MD", "active": True},
        ]
        
        for provider_data in providers_data:
            provider = Provider(**provider_data)
            self.session.add(provider)
            
        logger.info(f"Inserted {len(providers_data)} providers")
        
    def insert_patient(self) -> None:
        """Insert main patient record"""
        logger.info("Inserting patient record...")
        
        address_json = {
            "street": "1234 Maple Street",
            "city": "Springfield", 
            "state": "IL",
            "zip": "62701",
            "country": "USA",
            "type": "home"
        }
        
        patient = Patient(
            id=self.patient_id,
            first_name="Eleanor",
            middle_name="Margaret", 
            last_name="Henderson",
            date_of_birth=datetime(1944, 3, 15),
            gender="female",
            race="White",
            ethnicity="Not Hispanic or Latino",
            preferred_language="English",
            phone_primary="555-123-4567",
            phone_secondary="555-987-6543", 
            email="eleanor.henderson@email.com",
            address=address_json,
            active=True,
            deceased=False,
            created_at=datetime(1962, 1, 1, 8, 0, 0),
            updated_at=datetime(2024, 12, 15, 10, 30, 0)
        )
        
        self.session.add(patient)
        logger.info("Inserted patient: Eleanor Margaret Henderson")
        
    def insert_patient_identifiers(self) -> None:
        """Insert patient identifiers"""
        logger.info("Inserting patient identifiers...")
        
        identifiers_data = [
            {"identifier_type": IdentifierTypeEnum.MRN, "value": "MRN123456789", "system": "Springfield General Hospital", "period_start": datetime(1962, 1, 1), "active": True},
            {"identifier_type": IdentifierTypeEnum.MRN, "value": "MRN987654321", "system": "University Medical Center", "period_start": datetime(1975, 6, 1), "active": True},
            {"identifier_type": IdentifierTypeEnum.SSN, "value": "123-45-6789", "system": "Social Security Administration", "period_start": datetime(1962, 1, 1), "active": True},
            {"identifier_type": IdentifierTypeEnum.INSURANCE, "value": "MEDICARE123456789A", "system": "Medicare", "period_start": datetime(2009, 3, 15), "active": True},
            {"identifier_type": IdentifierTypeEnum.INSURANCE, "value": "BCBS987654321", "system": "Blue Cross Blue Shield", "period_start": datetime(1965, 1, 1), "active": True},
            {"identifier_type": IdentifierTypeEnum.INSURANCE, "value": "AETNA456789123", "system": "Aetna", "period_start": datetime(1980, 1, 1), "active": False},
            {"identifier_type": IdentifierTypeEnum.DRIVER_LICENSE, "value": "IL123456789", "system": "Illinois DMV", "period_start": datetime(1962, 3, 15), "active": False},
        ]
        
        for identifier_data in identifiers_data:
            identifier = PatientIdentifier(
                patient_id=self.patient_id,
                **identifier_data
            )
            self.session.add(identifier)
            
        logger.info(f"Inserted {len(identifiers_data)} patient identifiers")
        
    def insert_conditions(self) -> None:
        """Insert medical conditions"""
        logger.info("Inserting medical conditions...")
        
        conditions_data = [
            # Active chronic conditions
            {
                "id": "cond_001", "code": "E11.9", "code_system": "ICD-10", 
                "display_name": "Type 2 diabetes mellitus without complications",
                "category": "problem-list-item", "clinical_status": DiagnosisStatusEnum.ACTIVE,
                "verification_status": "confirmed", "severity": "moderate",
                "onset_date": datetime(1985, 6, 15), "recorded_date": datetime(1985, 6, 15),
                "recorded_by_id": "prov_002", 
                "notes": "Diagnosed during routine physical. Initially managed with diet and exercise."
            },
            {
                "id": "cond_002", "code": "I10", "code_system": "ICD-10",
                "display_name": "Essential hypertension", 
                "category": "problem-list-item", "clinical_status": DiagnosisStatusEnum.ACTIVE,
                "verification_status": "confirmed", "severity": "moderate",
                "onset_date": datetime(1982, 11, 20), "recorded_date": datetime(1982, 11, 20),
                "recorded_by_id": "prov_002",
                "notes": "Blood pressure consistently elevated. Family history positive."
            },
            {
                "id": "cond_003", "code": "I25.10", "code_system": "ICD-10",
                "display_name": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
                "category": "problem-list-item", "clinical_status": DiagnosisStatusEnum.ACTIVE,
                "verification_status": "confirmed", "severity": "moderate", 
                "onset_date": datetime(1995, 3, 10), "recorded_date": datetime(1995, 3, 10),
                "recorded_by_id": "prov_004",
                "notes": "Discovered during cardiac catheterization. Two-vessel disease."
            },
            {
                "id": "cond_004", "code": "N18.3", "code_system": "ICD-10",
                "display_name": "Chronic kidney disease, stage 3 (moderate)",
                "category": "problem-list-item", "clinical_status": DiagnosisStatusEnum.ACTIVE,
                "verification_status": "confirmed", "severity": "moderate",
                "onset_date": datetime(2010, 2, 14), "recorded_date": datetime(2010, 2, 14),
                "recorded_by_id": "prov_010",
                "notes": "eGFR 45-59. Likely diabetic nephropathy."
            },
            {
                "id": "cond_005", "code": "I48.91", "code_system": "ICD-10",
                "display_name": "Unspecified atrial fibrillation",
                "category": "problem-list-item", "clinical_status": DiagnosisStatusEnum.ACTIVE,
                "verification_status": "confirmed", "severity": "moderate",
                "onset_date": datetime(2023, 2, 15), "recorded_date": datetime(2023, 2, 15),
                "recorded_by_id": "prov_004",
                "notes": "New onset atrial fibrillation. Started on anticoagulation."
            },
            {
                "id": "cond_006", "code": "M81.0", "code_system": "ICD-10",
                "display_name": "Age-related osteoporosis without current pathological fracture",
                "category": "problem-list-item", "clinical_status": DiagnosisStatusEnum.ACTIVE,
                "verification_status": "confirmed", "severity": "moderate",
                "onset_date": datetime(2019, 3, 10), "recorded_date": datetime(2019, 3, 10),
                "recorded_by_id": "prov_006",
                "notes": "DEXA scan shows T-score -2.8. Started on bisphosphonate therapy."
            },
            {
                "id": "cond_007", "code": "E03.9", "code_system": "ICD-10",
                "display_name": "Hypothyroidism, unspecified",
                "category": "problem-list-item", "clinical_status": DiagnosisStatusEnum.ACTIVE,
                "verification_status": "confirmed", "severity": "mild",
                "onset_date": datetime(2000, 12, 8), "recorded_date": datetime(2000, 12, 8),
                "recorded_by_id": "prov_005",
                "notes": "TSH elevated at 8.5 mIU/L. Started on levothyroxine."
            },
            {
                "id": "cond_008", "code": "K21.9", "code_system": "ICD-10",  
                "display_name": "Gastro-esophageal reflux disease without esophagitis",
                "category": "problem-list-item", "clinical_status": DiagnosisStatusEnum.ACTIVE,
                "verification_status": "confirmed", "severity": "mild",
                "onset_date": datetime(2008, 9, 5), "recorded_date": datetime(2008, 9, 5),
                "recorded_by_id": "prov_007",
                "notes": "Chronic GERD symptoms."
            },
            # Resolved conditions
            {
                "id": "cond_009", "code": "C50.911", "code_system": "ICD-10",
                "display_name": "Malignant neoplasm of unspecified site of right female breast",
                "category": "problem-list-item", "clinical_status": DiagnosisStatusEnum.RESOLVED,
                "verification_status": "confirmed", "severity": "severe",
                "onset_date": datetime(1998, 4, 12), "resolved_date": datetime(2003, 4, 12),
                "recorded_date": datetime(1998, 4, 12), "recorded_by_id": "prov_012",
                "body_site": "right breast",
                "notes": "Stage IIA invasive ductal carcinoma. Treated with lumpectomy, chemotherapy, and radiation."
            },
            {
                "id": "cond_010", "code": "S72.001A", "code_system": "ICD-10",
                "display_name": "Fracture of unspecified part of neck of right femur, initial encounter",
                "category": "encounter-diagnosis", "clinical_status": DiagnosisStatusEnum.RESOLVED,
                "verification_status": "confirmed", "severity": "severe",
                "onset_date": datetime(2018, 11, 15), "resolved_date": datetime(2019, 5, 15),
                "recorded_date": datetime(2018, 11, 15), "recorded_by_id": "prov_009",
                "body_site": "right hip",
                "notes": "Fall at home. Surgical repair with hip replacement."
            },
        ]
        
        for condition_data in conditions_data:
            condition = Condition(
                patient_id=self.patient_id,
                **condition_data
            )
            self.session.add(condition)
            
        logger.info(f"Inserted {len(conditions_data)} medical conditions")
        
    def insert_allergies(self) -> None:
        """Insert allergies and intolerances"""
        logger.info("Inserting allergies and intolerances...")
        
        allergies_data = [
            {
                "id": "allergy_001", "allergen_code": "387207008", "allergen_name": "Penicillin",
                "allergen_type": "medication", "status": AllergyStatusEnum.ACTIVE,
                "type": "allergy", "category": "medication", "criticality": "high",
                "severity": AllergySeverityEnum.MODERATE,
                "reaction_manifestation": ["rash", "hives", "itching"],
                "reaction_description": "Generalized urticaria and pruritus within 30 minutes of administration",
                "onset_date": datetime(1975, 5, 20), "recorded_date": datetime(1975, 5, 20),
                "recorded_by_id": "prov_002"
            },
            {
                "id": "allergy_002", "allergen_code": "256259004", "allergen_name": "Shellfish",
                "allergen_type": "food", "status": AllergyStatusEnum.ACTIVE,
                "type": "allergy", "category": "food", "criticality": "high", 
                "severity": AllergySeverityEnum.SEVERE,
                "reaction_manifestation": ["swelling", "difficulty breathing", "nausea"],
                "reaction_description": "Facial swelling and respiratory distress after eating shrimp",
                "onset_date": datetime(1988, 8, 12), "recorded_date": datetime(1988, 8, 12),
                "recorded_by_id": "prov_002"
            },
            {
                "id": "allergy_003", "allergen_code": "396064000", "allergen_name": "Sulfonamides",
                "allergen_type": "medication", "status": AllergyStatusEnum.ACTIVE,
                "type": "allergy", "category": "medication", "criticality": "low",
                "severity": AllergySeverityEnum.MILD,
                "reaction_manifestation": ["rash"],
                "reaction_description": "Mild skin rash with sulfa antibiotics",
                "onset_date": datetime(1992, 3, 8), "recorded_date": datetime(1992, 3, 8),
                "recorded_by_id": "prov_002"
            },
            {
                "id": "allergy_004", "allergen_code": "264287008", "allergen_name": "Tree pollen",
                "allergen_type": "environment", "status": AllergyStatusEnum.ACTIVE,
                "type": "allergy", "category": "environment", "criticality": "low",
                "severity": AllergySeverityEnum.MILD,
                "reaction_manifestation": ["sneezing", "watery eyes", "congestion"],
                "reaction_description": "Seasonal allergic rhinitis symptoms in spring",
                "onset_date": datetime(1980, 4, 15), "recorded_date": datetime(1980, 4, 15),
                "recorded_by_id": "prov_002"
            },
        ]
        
        for allergy_data in allergies_data:
            allergy = AllergyIntolerance(
                patient_id=self.patient_id,
                **allergy_data
            )
            self.session.add(allergy)
            
        logger.info(f"Inserted {len(allergies_data)} allergies and intolerances")
        
    def insert_medications(self) -> None:
        """Insert medication statements"""
        logger.info("Inserting medication statements...")
        
        medications_data = [
            # Current active medications
            {
                "id": "med_001", "medication_code": "310965", 
                "medication_name": "Metformin 1000mg tablets",
                "generic_name": "Metformin hydrochloride", "brand_name": "Glucophage",
                "dosage_text": "1000mg twice daily", "dose_quantity": Decimal("1000"),
                "dose_unit": "mg", "frequency": "BID", "route": "oral",
                "status": MedicationStatusEnum.ACTIVE,
                "effective_start": datetime(1985, 6, 15),
                "indication": "Type 2 diabetes mellitus", "prescriber_id": "prov_005",
                "patient_instructions": "Take with meals to reduce stomach upset",
                "notes": "Long-term diabetes management"
            },
            {
                "id": "med_002", "medication_code": "197361",
                "medication_name": "Lisinopril 10mg tablets", 
                "generic_name": "Lisinopril", "brand_name": "Prinivil",
                "dosage_text": "10mg once daily", "dose_quantity": Decimal("10"),
                "dose_unit": "mg", "frequency": "daily", "route": "oral",
                "status": MedicationStatusEnum.ACTIVE,
                "effective_start": datetime(1995, 3, 10),
                "indication": "Hypertension and cardioprotection", "prescriber_id": "prov_004",
                "patient_instructions": "Take at the same time each day",
                "notes": "ACE inhibitor for BP control"
            },
            {
                "id": "med_003", "medication_code": "617312",
                "medication_name": "Atorvastatin 40mg tablets",
                "generic_name": "Atorvastatin calcium", "brand_name": "Lipitor", 
                "dosage_text": "40mg once daily at bedtime", "dose_quantity": Decimal("40"),
                "dose_unit": "mg", "frequency": "daily", "route": "oral",
                "status": MedicationStatusEnum.ACTIVE,
                "effective_start": datetime(2000, 9, 20),
                "indication": "Hyperlipidemia", "prescriber_id": "prov_004",
                "patient_instructions": "Take in the evening",
                "notes": "Statin therapy for cholesterol"
            },
            {
                "id": "med_004", "medication_code": "855288",
                "medication_name": "Aspirin 81mg tablets",
                "generic_name": "Aspirin", "brand_name": "Bayer",
                "dosage_text": "81mg once daily", "dose_quantity": Decimal("81"),
                "dose_unit": "mg", "frequency": "daily", "route": "oral",
                "status": MedicationStatusEnum.ACTIVE,
                "effective_start": datetime(1995, 3, 10),
                "indication": "Cardiovascular protection", "prescriber_id": "prov_004",
                "patient_instructions": "Take with food",
                "notes": "Low-dose for cardioprotection"
            },
            {
                "id": "med_005", "medication_code": "966221",
                "medication_name": "Levothyroxine 75mcg tablets",
                "generic_name": "Levothyroxine sodium", "brand_name": "Synthroid",
                "dosage_text": "75mcg once daily on empty stomach", "dose_quantity": Decimal("75"),
                "dose_unit": "mcg", "frequency": "daily", "route": "oral",
                "status": MedicationStatusEnum.ACTIVE,
                "effective_start": datetime(2000, 12, 8),
                "indication": "Hypothyroidism", "prescriber_id": "prov_005",
                "patient_instructions": "Take 30-60 minutes before breakfast",
                "notes": "Thyroid hormone replacement"
            },
            {
                "id": "med_006", "medication_code": "1292884",
                "medication_name": "Warfarin 5mg tablets",
                "generic_name": "Warfarin sodium", "brand_name": "Coumadin",
                "dosage_text": "5mg daily, adjust per INR", "dose_quantity": Decimal("5"),
                "dose_unit": "mg", "frequency": "daily", "route": "oral",
                "status": MedicationStatusEnum.ACTIVE,
                "effective_start": datetime(2023, 2, 15),
                "indication": "Atrial fibrillation anticoagulation", "prescriber_id": "prov_004",
                "patient_instructions": "Regular INR monitoring required",
                "notes": "Anticoagulation for stroke prevention"
            },
            # Discontinued medications
            {
                "id": "med_007", "medication_code": "153666",
                "medication_name": "Glyburide 5mg tablets",
                "generic_name": "Glyburide", "brand_name": "DiaBeta",
                "dosage_text": "5mg twice daily", "dose_quantity": Decimal("5"),
                "dose_unit": "mg", "frequency": "BID", "route": "oral",
                "status": MedicationStatusEnum.DISCONTINUED,
                "effective_start": datetime(1990, 8, 15), "effective_end": datetime(2018, 3, 20),
                "indication": "Type 2 diabetes mellitus", "prescriber_id": "prov_005",
                "patient_instructions": "Take 30 minutes before meals",
                "notes": "Discontinued due to hypoglycemia risk"
            },
            # Historical cancer treatment
            {
                "id": "med_008", "medication_code": "212754",
                "medication_name": "Tamoxifen 20mg tablets",
                "generic_name": "Tamoxifen citrate", "brand_name": "Nolvadex",
                "dosage_text": "20mg once daily", "dose_quantity": Decimal("20"),
                "dose_unit": "mg", "frequency": "daily", "route": "oral",
                "status": MedicationStatusEnum.COMPLETED,
                "effective_start": datetime(1998, 12, 1), "effective_end": datetime(2003, 12, 1),
                "indication": "Breast cancer adjuvant therapy", "prescriber_id": "prov_012",
                "patient_instructions": "Take at same time daily",
                "notes": "5-year course completed"
            },
        ]
        
        for medication_data in medications_data:
            medication = MedicationStatement(
                patient_id=self.patient_id,
                recorded_date=medication_data.get("effective_start", datetime.now()),
                **medication_data
            )
            self.session.add(medication)
            
        logger.info(f"Inserted {len(medications_data)} medication statements")
        
    def insert_encounters(self) -> None:
        """Insert healthcare encounters"""
        logger.info("Inserting healthcare encounters...")
        
        encounters_data = [
            # Recent encounters
            {
                "id": "enc_001", "status": EncounterStatusEnum.COMPLETED,
                "encounter_type": EncounterTypeEnum.OUTPATIENT, "priority": "routine",
                "start_time": datetime(2024, 12, 1, 9, 0, 0),
                "end_time": datetime(2024, 12, 1, 10, 0, 0), "duration_minutes": 60,
                "location": "Springfield General Hospital", "department": "Geriatrics",
                "primary_provider_id": "prov_003", "chief_complaint": "Annual wellness visit",
                "reason_code": "Z00.00", "reason_description": "Comprehensive geriatric assessment",
                "disposition": "discharged home",
                "discharge_instructions": "Continue current medications. Follow up in 6 months."
            },
            {
                "id": "enc_002", "status": EncounterStatusEnum.COMPLETED,
                "encounter_type": EncounterTypeEnum.OUTPATIENT, "priority": "routine",
                "start_time": datetime(2024, 9, 15, 14, 30, 0),
                "end_time": datetime(2024, 9, 15, 15, 15, 0), "duration_minutes": 45,
                "location": "Springfield Endocrine Center", "department": "Endocrinology", 
                "primary_provider_id": "prov_005", "chief_complaint": "Diabetes follow-up",
                "reason_code": "E11.9", "reason_description": "Type 2 diabetes monitoring",
                "disposition": "discharged home",
                "discharge_instructions": "A1C improved. Continue metformin."
            },
            {
                "id": "enc_003", "status": EncounterStatusEnum.COMPLETED,
                "encounter_type": EncounterTypeEnum.EMERGENCY, "priority": "urgent",
                "start_time": datetime(2024, 6, 20, 14, 15, 0),
                "end_time": datetime(2024, 6, 20, 19, 30, 0), "duration_minutes": 315,
                "location": "Springfield General Hospital", "department": "Emergency",
                "primary_provider_id": "prov_009", "chief_complaint": "Chest pain and shortness of breath",
                "reason_code": "R06.02", "reason_description": "Dyspnea and chest discomfort",
                "disposition": "discharged home",
                "discharge_instructions": "EKG normal. Stress test ordered. Follow up cardiology."
            },
            # Historical major encounters
            {
                "id": "enc_004", "status": EncounterStatusEnum.COMPLETED,
                "encounter_type": EncounterTypeEnum.INPATIENT, "priority": "emergency",
                "start_time": datetime(2018, 11, 15, 20, 0, 0),
                "end_time": datetime(2018, 11, 25, 10, 0, 0), "duration_minutes": 14400,
                "location": "Springfield General Hospital", "department": "Orthopedics",
                "primary_provider_id": "prov_006", "chief_complaint": "Hip fracture surgery",
                "reason_code": "S72.001A", "reason_description": "Total hip arthroplasty",
                "disposition": "discharged home",
                "discharge_instructions": "Hip replacement surgery. Physical therapy. Weight bearing as tolerated."
            },
            {
                "id": "enc_005", "status": EncounterStatusEnum.COMPLETED,
                "encounter_type": EncounterTypeEnum.OUTPATIENT, "priority": "routine",
                "start_time": datetime(1998, 5, 15, 10, 0, 0),
                "end_time": datetime(1998, 5, 15, 14, 0, 0), "duration_minutes": 240,
                "location": "Springfield Cancer Center", "department": "Oncology",
                "primary_provider_id": "prov_012", "chief_complaint": "Chemotherapy cycle 1",
                "reason_code": "C50.911", "reason_description": "Breast cancer treatment", 
                "disposition": "discharged home",
                "discharge_instructions": "AC regimen cycle 1. Tolerated well. Next cycle in 3 weeks."
            },
        ]
        
        for encounter_data in encounters_data:
            encounter = Encounter(
                patient_id=self.patient_id,
                created_at=encounter_data["start_time"],
                updated_at=encounter_data["start_time"],
                **encounter_data
            )
            self.session.add(encounter)
            
        logger.info(f"Inserted {len(encounters_data)} healthcare encounters")
        
    def insert_observations(self) -> None:
        """Insert observations (lab results, vital signs)"""
        logger.info("Inserting observations...")
        
        observations_data = [
            # Recent lab results (2024)
            {
                "id": "obs_001", "encounter_id": "enc_001", "code": "33747-0",
                "display_name": "Hemoglobin A1c", "category": "laboratory",
                "value_quantity": Decimal("7.2"), "value_unit": "%",
                "reference_range_low": Decimal("4.0"), "reference_range_high": Decimal("6.0"),
                "interpretation": "high", "status": "final",
                "effective_datetime": datetime(2024, 12, 1, 9, 15, 0),
                "performer_id": "prov_003", "notes": "Diabetes control improved"
            },
            {
                "id": "obs_002", "encounter_id": "enc_001", "code": "2339-0",
                "display_name": "Glucose, fasting", "category": "laboratory",
                "value_quantity": Decimal("145"), "value_unit": "mg/dL", 
                "reference_range_low": Decimal("70"), "reference_range_high": Decimal("99"),
                "interpretation": "high", "status": "final",
                "effective_datetime": datetime(2024, 12, 1, 9, 15, 0),
                "performer_id": "prov_003", "notes": "Fasting glucose elevated"
            },
            {
                "id": "obs_003", "encounter_id": "enc_001", "code": "2160-0",
                "display_name": "Creatinine", "category": "laboratory",
                "value_quantity": Decimal("1.4"), "value_unit": "mg/dL",
                "reference_range_low": Decimal("0.6"), "reference_range_high": Decimal("1.2"),
                "interpretation": "high", "status": "final",
                "effective_datetime": datetime(2024, 12, 1, 9, 15, 0),
                "performer_id": "prov_003", "notes": "Stable CKD"
            },
            # Vital signs
            {
                "id": "obs_004", "encounter_id": "enc_001", "code": "8480-6",
                "display_name": "Systolic blood pressure", "category": "vital-signs",
                "value_quantity": Decimal("142"), "value_unit": "mmHg",
                "reference_range_low": Decimal("90"), "reference_range_high": Decimal("140"),
                "interpretation": "high", "status": "final",
                "effective_datetime": datetime(2024, 12, 1, 9, 0, 0),
                "performer_id": "prov_003", "notes": "Slightly elevated systolic BP"
            },
            {
                "id": "obs_005", "encounter_id": "enc_001", "code": "8462-4",
                "display_name": "Diastolic blood pressure", "category": "vital-signs",
                "value_quantity": Decimal("88"), "value_unit": "mmHg",
                "reference_range_low": Decimal("60"), "reference_range_high": Decimal("90"),
                "interpretation": "normal", "status": "final",
                "effective_datetime": datetime(2024, 12, 1, 9, 0, 0),
                "performer_id": "prov_003", "notes": "Within range"
            },
            {
                "id": "obs_006", "encounter_id": "enc_001", "code": "29463-7",
                "display_name": "Body weight", "category": "vital-signs",
                "value_quantity": Decimal("165"), "value_unit": "lb",
                "interpretation": "normal", "status": "final",
                "effective_datetime": datetime(2024, 12, 1, 9, 0, 0),
                "performer_id": "prov_003", "notes": "Stable weight"
            },
        ]
        
        for observation_data in observations_data:
            observation = Observation(
                patient_id=self.patient_id,
                **observation_data
            )
            self.session.add(observation)
            
        logger.info(f"Inserted {len(observations_data)} observations")
        
    def insert_procedures(self) -> None:
        """Insert medical procedures"""
        logger.info("Inserting procedures...")
        
        procedures_data = [
            {
                "id": "proc_001", "encounter_id": "enc_001", "code": "93000",
                "display_name": "Electrocardiogram", "category": "diagnostic",
                "status": "completed", "performed_datetime": datetime(2024, 12, 1, 9, 30, 0),
                "body_site": "chest", "outcome": "normal sinus rhythm",
                "primary_performer_id": "prov_003", "location": "Springfield General Hospital",
                "indication": "Routine cardiac screening",
                "notes": "Normal EKG with no acute changes"
            },
            {
                "id": "proc_002", "encounter_id": "enc_004", "code": "27130",
                "display_name": "Total hip arthroplasty", "category": "surgical",
                "status": "completed", "performed_datetime": datetime(2018, 11, 16, 8, 0, 0),
                "body_site": "right hip", "outcome": "successful hip replacement",
                "primary_performer_id": "prov_006", "location": "Springfield General Hospital",
                "indication": "Right femoral neck fracture",
                "notes": "Uncomplicated total hip replacement. Good postoperative recovery."
            },
            {
                "id": "proc_003", "encounter_id": "enc_005", "code": "96413",
                "display_name": "Chemotherapy administration", "category": "therapeutic",
                "status": "completed", "performed_datetime": datetime(1998, 5, 15, 10, 0, 0),
                "body_site": "intravenous", "outcome": "successful administration",
                "primary_performer_id": "prov_012", "location": "Springfield Cancer Center",
                "indication": "Breast cancer treatment",
                "notes": "AC regimen cycle 1 administered without complications"
            },
        ]
        
        for procedure_data in procedures_data:
            procedure = Procedure(
                patient_id=self.patient_id,
                **procedure_data
            )
            self.session.add(procedure)
            
        logger.info(f"Inserted {len(procedures_data)} procedures")
        
    def insert_clinical_notes(self) -> None:
        """Insert clinical notes"""
        logger.info("Inserting clinical notes...")
        
        comprehensive_note = """CHIEF COMPLAINT: Annual wellness visit and comprehensive geriatric assessment.

HISTORY OF PRESENT ILLNESS: 
Eleanor Henderson is an 80-year-old female with extensive medical history including type 2 diabetes mellitus (diagnosed 1985), essential hypertension (diagnosed 1982), coronary artery disease (diagnosed 1995), chronic kidney disease stage 3 (diagnosed 2010), atrial fibrillation (diagnosed 2023), and remote history of successfully treated breast cancer (1998-2003).

Patient reports feeling generally well with good functional status. She continues to live independently with family support. Diabetes management improved with recent A1C of 7.2%. Atrial fibrillation well controlled on warfarin with therapeutic INR.

ASSESSMENT AND PLAN:
1. Type 2 diabetes mellitus - WELL CONTROLLED (A1C 7.2%)
2. Essential hypertension - ADEQUATELY CONTROLLED  
3. Coronary artery disease - STABLE
4. Atrial fibrillation - WELL CONTROLLED on warfarin
5. Chronic kidney disease stage 3 - STABLE
6. Continue current medication regimen
7. Follow up in 6 months"""
        
        notes_data = [
            {
                "id": "note_001", "encounter_id": "enc_001", "note_type": "progress",
                "specialty": "Geriatrics", "title": "Annual Comprehensive Geriatric Assessment",
                "content": comprehensive_note, "status": "final", "author_id": "prov_003",
                "dictated_datetime": datetime(2024, 12, 1, 11, 0, 0),
                "authenticated_datetime": datetime(2024, 12, 1, 12, 0, 0),
                "processed_by_ai": True
            },
        ]
        
        for note_data in notes_data:
            note = ClinicalNote(
                patient_id=self.patient_id,
                **note_data
            )
            self.session.add(note)
            
        logger.info(f"Inserted {len(notes_data)} clinical notes")
        
    def insert_care_plans(self) -> None:
        """Insert care plans"""
        logger.info("Inserting care plans...")
        
        care_plans_data = [
            {
                "id": "care_001", "title": "Comprehensive Diabetes Management Plan",
                "description": "Evidence-based diabetes care plan targeting optimal glucose control",
                "status": "active", "intent": "plan",
                "period_start": datetime(2024, 1, 1), "created_date": datetime(2024, 1, 1),
                "created_by_id": "prov_005", "addresses": ["cond_001"],
                "goals": [
                    {"description": "Achieve HbA1c <7.5%", "target": "7.5", "measure": "HbA1c percentage"}
                ],
                "activities": [
                    {"activity": "Home glucose monitoring", "frequency": "twice daily"},
                    {"activity": "Take metformin as prescribed", "frequency": "1000mg BID"}
                ]
            },
        ]
        
        for care_plan_data in care_plans_data:
            care_plan = CarePlan(
                patient_id=self.patient_id,
                **care_plan_data
            )
            self.session.add(care_plan)
            
        logger.info(f"Inserted {len(care_plans_data)} care plans")
        
    def insert_patient_summary(self) -> None:
        """Insert AI-generated patient summary"""
        logger.info("Inserting patient summary...")
        
        summary = PatientSummary(
            id="summary_001",
            patient_id=self.patient_id,
            summary_current="Eleanor Henderson is an 80-year-old female with multiple well-managed chronic conditions including type 2 diabetes (A1C 7.2%), hypertension, coronary artery disease, stage 3 chronic kidney disease, and atrial fibrillation on warfarin anticoagulation.",
            summary_recent="Recent 6-month activity shows stable health with improved diabetes control. Atrial fibrillation remains well-controlled on warfarin.",
            summary_historical="Eleanor has extraordinary medical history spanning 60+ years including successful breast cancer treatment (1998-2003) and bilateral hip fractures with total hip replacement (2018).",
            summary_chronic="Chronic conditions requiring ongoing management: diabetes since 1985, hypertension since 1982, CAD since 1995, CKD since 2010, atrial fibrillation since 2023.",
            active_diagnoses=["Type 2 diabetes mellitus", "Essential hypertension", "Coronary artery disease", "Chronic kidney disease stage 3", "Atrial fibrillation"],
            current_medications=["Metformin 1000mg BID", "Lisinopril 10mg daily", "Atorvastatin 40mg daily", "Warfarin 5mg daily", "Levothyroxine 75mcg daily"],
            allergies=["Penicillin - urticaria", "Shellfish - anaphylaxis", "Sulfonamides - rash", "Tree pollen - rhinitis"],
            chronic_conditions=["Type 2 diabetes mellitus", "Essential hypertension", "Coronary artery disease", "Chronic kidney disease", "Atrial fibrillation"],
            last_updated=datetime(2024, 12, 15, 14, 0, 0),
            generated_by_model="Claude-4-Medical-AI-Enhanced",
            confidence_score=Decimal("0.98")
        )
        
        self.session.add(summary)
        logger.info("Inserted patient summary")
        
    def run_insertion(self) -> None:
        """Run the complete data insertion process"""
        try:
            logger.info("=== Starting Comprehensive EHR Data Insertion ===")
            logger.info("Patient: Eleanor Margaret Henderson (Age 80)")
            logger.info("Medical History: 1962-2024 (62 years)")
            
            # Insert data in proper order to maintain referential integrity
            self.insert_providers()
            self.session.flush()  # Flush to ensure providers are available
            
            self.insert_patient()
            self.session.flush()
            
            self.insert_patient_identifiers()
            self.insert_conditions()
            self.insert_allergies()
            self.insert_medications()
            self.insert_encounters()
            self.session.flush()  # Flush encounters before dependent records
            
            self.insert_observations()
            self.insert_procedures()
            self.insert_clinical_notes()
            self.insert_care_plans()
            self.insert_patient_summary()
            
            logger.info("=== Data Insertion Completed Successfully ===")
            
            # Print summary statistics
            self.print_summary_statistics()
            
        except Exception as e:
            logger.error(f"Error during data insertion: {e}")
            raise
            
    def print_summary_statistics(self) -> None:
        """Print summary of inserted data"""
        logger.info("\n=== INSERTION SUMMARY ===")
        
        # Count records in each table
        stats = {
            "Providers": self.session.query(Provider).count(),
            "Patients": self.session.query(Patient).count(),
            "Patient Identifiers": self.session.query(PatientIdentifier).filter_by(patient_id=self.patient_id).count(),
            "Conditions": self.session.query(Condition).filter_by(patient_id=self.patient_id).count(),
            "Medications": self.session.query(MedicationStatement).filter_by(patient_id=self.patient_id).count(),
            "Allergies": self.session.query(AllergyIntolerance).filter_by(patient_id=self.patient_id).count(),
            "Encounters": self.session.query(Encounter).filter_by(patient_id=self.patient_id).count(),
            "Observations": self.session.query(Observation).filter_by(patient_id=self.patient_id).count(),
            "Procedures": self.session.query(Procedure).filter_by(patient_id=self.patient_id).count(),
            "Clinical Notes": self.session.query(ClinicalNote).filter_by(patient_id=self.patient_id).count(),
            "Care Plans": self.session.query(CarePlan).filter_by(patient_id=self.patient_id).count(),
            "Patient Summaries": self.session.query(PatientSummary).filter_by(patient_id=self.patient_id).count(),
        }
        
        total_records = sum(stats.values())
        
        for category, count in stats.items():
            logger.info(f"{category}: {count}")
        
        logger.info(f"\nTOTAL RECORDS INSERTED: {total_records}")
        logger.info("✅ Comprehensive EHR database population completed successfully!")


def main():
    """Main execution function"""
    try:
        # Initialize database
        logger.info("Initializing database...")
        if not initialize_database():
            logger.error("Failed to initialize database")
            return False
            
        # Insert comprehensive EHR data
        with EHRDataInserter() as inserter:
            inserter.run_insertion()
            
        logger.info("🎉 EHR Data Insertion Process Completed Successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)