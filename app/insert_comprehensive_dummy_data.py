"""
Comprehensive Patient Data Insertion Script for Enhanced EHR Schema
Creates realistic patient records with proper relationships and foreign keys.
"""

import os
import sys
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
import logging
from sqlalchemy.orm import Session
from faker import Faker
import uuid

# Add the app directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.enhanced_ehr_schema import (
        initialize_database, SessionLocal, 
        Patient, PatientIdentifier, Provider, Encounter, Condition,
        MedicationStatement, AllergyIntolerance, Observation, Procedure,
        ClinicalNote, CarePlan, PatientSummary,
        IdentifierTypeEnum, EncounterStatusEnum, EncounterTypeEnum,
        DiagnosisStatusEnum, MedicationStatusEnum, AllergyStatusEnum, AllergySeverityEnum
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the correct directory and the enhanced_ehr_schema.py is fixed")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker for realistic data generation
fake = Faker()

class ComprehensivePatientDataInserter:
    """Insert comprehensive, realistic patient data into the enhanced EHR database."""
    
    def __init__(self):
        self.db: Session = None
        self.providers: List[Provider] = []
        
        # Medical data libraries for realistic entries
        self.conditions_library = [
            {"name": "Type 2 Diabetes Mellitus", "code": "E11.9", "system": "ICD-10"},
            {"name": "Essential Hypertension", "code": "I10", "system": "ICD-10"},
            {"name": "Hyperlipidemia", "code": "E78.5", "system": "ICD-10"},
            {"name": "Chronic Kidney Disease", "code": "N18.9", "system": "ICD-10"},
            {"name": "Coronary Artery Disease", "code": "I25.10", "system": "ICD-10"},
            {"name": "Heart Failure", "code": "I50.9", "system": "ICD-10"},
            {"name": "Asthma", "code": "J45.9", "system": "ICD-10"},
            {"name": "COPD", "code": "J44.1", "system": "ICD-10"},
            {"name": "Depression", "code": "F32.9", "system": "ICD-10"},
            {"name": "Anxiety Disorder", "code": "F41.9", "system": "ICD-10"},
            {"name": "Osteoarthritis", "code": "M19.90", "system": "ICD-10"},
            {"name": "Gastroesophageal Reflux Disease", "code": "K21.9", "system": "ICD-10"},
            {"name": "Sleep Apnea", "code": "G47.33", "system": "ICD-10"},
            {"name": "Hypothyroidism", "code": "E03.9", "system": "ICD-10"},
            {"name": "Atrial Fibrillation", "code": "I48.91", "system": "ICD-10"}
        ]
        
        self.medications_library = [
            {"name": "Metformin", "generic": "Metformin HCl", "dosage": "500mg", "frequency": "twice daily", "route": "oral", "indication": "Type 2 Diabetes"},
            {"name": "Lisinopril", "generic": "Lisinopril", "dosage": "10mg", "frequency": "once daily", "route": "oral", "indication": "Hypertension"},
            {"name": "Atorvastatin", "generic": "Atorvastatin", "dosage": "20mg", "frequency": "once daily", "route": "oral", "indication": "Hyperlipidemia"},
            {"name": "Amlodipine", "generic": "Amlodipine", "dosage": "5mg", "frequency": "once daily", "route": "oral", "indication": "Hypertension"},
            {"name": "Metoprolol", "generic": "Metoprolol Tartrate", "dosage": "25mg", "frequency": "twice daily", "route": "oral", "indication": "Hypertension"},
            {"name": "Furosemide", "generic": "Furosemide", "dosage": "40mg", "frequency": "once daily", "route": "oral", "indication": "Heart Failure"},
            {"name": "Albuterol", "generic": "Albuterol Sulfate", "dosage": "90mcg", "frequency": "as needed", "route": "inhaled", "indication": "Asthma"},
            {"name": "Omeprazole", "generic": "Omeprazole", "dosage": "20mg", "frequency": "once daily", "route": "oral", "indication": "GERD"},
            {"name": "Sertraline", "generic": "Sertraline HCl", "dosage": "50mg", "frequency": "once daily", "route": "oral", "indication": "Depression"},
            {"name": "Levothyroxine", "generic": "Levothyroxine Sodium", "dosage": "100mcg", "frequency": "once daily", "route": "oral", "indication": "Hypothyroidism"},
            {"name": "Warfarin", "generic": "Warfarin Sodium", "dosage": "5mg", "frequency": "once daily", "route": "oral", "indication": "Atrial Fibrillation"},
            {"name": "Insulin Glargine", "generic": "Insulin Glargine", "dosage": "20 units", "frequency": "once daily", "route": "subcutaneous", "indication": "Diabetes"}
        ]
        
        self.allergies_library = [
            {"name": "Penicillin", "type": "medication", "severity": AllergySeverityEnum.MODERATE, "reaction": "Skin rash and itching"},
            {"name": "Sulfa drugs", "type": "medication", "severity": AllergySeverityEnum.MILD, "reaction": "Nausea and dizziness"},
            {"name": "Shellfish", "type": "food", "severity": AllergySeverityEnum.SEVERE, "reaction": "Anaphylaxis, difficulty breathing"},
            {"name": "Peanuts", "type": "food", "severity": AllergySeverityEnum.SEVERE, "reaction": "Swelling, difficulty breathing"},
            {"name": "Latex", "type": "environment", "severity": AllergySeverityEnum.MODERATE, "reaction": "Contact dermatitis"},
            {"name": "Pollen", "type": "environment", "severity": AllergySeverityEnum.MILD, "reaction": "Seasonal allergic rhinitis"},
            {"name": "Codeine", "type": "medication", "severity": AllergySeverityEnum.MODERATE, "reaction": "Nausea and vomiting"},
            {"name": "Aspirin", "type": "medication", "severity": AllergySeverityEnum.MILD, "reaction": "Stomach irritation"}
        ]
        
        self.observations_library = [
            {"code": "33747-0", "name": "General blood chemistry panel", "category": "laboratory"},
            {"code": "2339-0", "name": "Glucose", "category": "laboratory", "unit": "mg/dL", "ref_low": 70, "ref_high": 100},
            {"code": "2093-3", "name": "Cholesterol total", "category": "laboratory", "unit": "mg/dL", "ref_low": 125, "ref_high": 200},
            {"code": "2571-8", "name": "Triglycerides", "category": "laboratory", "unit": "mg/dL", "ref_low": 35, "ref_high": 150},
            {"code": "33765-2", "name": "White blood cell count", "category": "laboratory", "unit": "K/uL", "ref_low": 4.5, "ref_high": 11.0},
            {"code": "8480-6", "name": "Systolic blood pressure", "category": "vital-signs", "unit": "mmHg", "ref_low": 90, "ref_high": 120},
            {"code": "8462-4", "name": "Diastolic blood pressure", "category": "vital-signs", "unit": "mmHg", "ref_low": 60, "ref_high": 80},
            {"code": "8310-5", "name": "Body temperature", "category": "vital-signs", "unit": "degF", "ref_low": 97.0, "ref_high": 99.5},
            {"code": "8867-4", "name": "Heart rate", "category": "vital-signs", "unit": "bpm", "ref_low": 60, "ref_high": 100},
            {"code": "29463-7", "name": "Body weight", "category": "vital-signs", "unit": "kg", "ref_low": 50, "ref_high": 120},
            {"code": "8302-2", "name": "Body height", "category": "vital-signs", "unit": "cm", "ref_low": 150, "ref_high": 200}
        ]
        
        self.procedures_library = [
            {"code": "45378", "name": "Colonoscopy", "category": "diagnostic"},
            {"code": "93000", "name": "Electrocardiogram", "category": "diagnostic"},
            {"code": "76700", "name": "Abdominal ultrasound", "category": "diagnostic"},
            {"code": "71020", "name": "Chest X-ray", "category": "diagnostic"},
            {"code": "36415", "name": "Blood draw", "category": "therapeutic"},
            {"code": "90658", "name": "Influenza vaccination", "category": "preventive"},
            {"code": "90715", "name": "Tetanus vaccination", "category": "preventive"},
            {"code": "99213", "name": "Office visit", "category": "evaluation"}
        ]
    
    def setup_database(self):
        """Initialize database and create session."""
        logger.info("🔧 Setting up database...")
        
        success = initialize_database()
        if not success:
            raise RuntimeError("❌ Failed to initialize database")
        
        self.db = SessionLocal()
        logger.info("✅ Database setup complete")
        
    def create_providers(self, count: int = 10):
        """Create healthcare providers."""
        logger.info(f"👩‍⚕️ Creating {count} healthcare providers...")
        
        specialties = [
            "Internal Medicine", "Family Medicine", "Cardiology", "Endocrinology",
            "Pulmonology", "Nephrology", "Gastroenterology", "Neurology",
            "Psychiatry", "Emergency Medicine", "General Surgery", "Orthopedics"
        ]
        
        for i in range(count):
            provider = Provider(
                id=str(uuid.uuid4()),
                npi=f"{1000000000 + i:010d}",  # Generate realistic NPI
                first_name=fake.first_name(),
                last_name=fake.last_name(),
                specialty=random.choice(specialties),
                department=random.choice(specialties),
                title=random.choice(["MD", "DO", "NP", "PA"]),
                active=True
            )
            self.db.add(provider)
            self.providers.append(provider)
        
        self.db.commit()
        logger.info(f"✅ Created {count} providers")
    
    def create_comprehensive_patient(self, patient_id: str = None) -> str:
        """Create a single patient with comprehensive medical history."""
        if not patient_id:
            patient_id = str(uuid.uuid4())
        
        logger.info(f"🏥 Creating comprehensive patient {patient_id}...")
        
        # Create patient demographics
        first_name = fake.first_name()
        last_name = fake.last_name()
        birth_date = fake.date_of_birth(minimum_age=25, maximum_age=85)
        
        patient = Patient(
            id=patient_id,
            first_name=first_name,
            middle_name=fake.first_name() if random.choice([True, False]) else None,
            last_name=last_name,
            date_of_birth=birth_date,
            gender=random.choice(["male", "female", "other"]),
            race=random.choice(["White", "Black or African American", "Asian", "Native American", "Other"]),
            ethnicity=random.choice(["Hispanic or Latino", "Not Hispanic or Latino"]),
            preferred_language=random.choice(["English", "Spanish", "Other"]),
            phone_primary=fake.phone_number(),
            phone_secondary=fake.phone_number() if random.choice([True, False]) else None,
            email=fake.email(),
            address={
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state(),
                "zip": fake.zipcode(),
                "country": "USA"
            },
            active=True,
            deceased=False
        )
        self.db.add(patient)
        self.db.flush()  # Get the patient ID
        
        # Create patient identifiers
        self._create_patient_identifiers(patient.id)
        
        # Create medical conditions (3-8 conditions per patient)
        conditions = self._create_patient_conditions(patient.id, random.randint(3, 8))
        
        # Create medications based on conditions (5-15 medications)
        self._create_patient_medications(patient.id, conditions, random.randint(5, 15))
        
        # Create allergies (0-4 allergies per patient)
        self._create_patient_allergies(patient.id, random.randint(0, 4))
        
        # Create encounters/visits (10-50 encounters over the past 5 years)
        encounters = self._create_patient_encounters(patient.id, random.randint(10, 50))
        
        # Create observations/lab results (50-200 observations)
        self._create_patient_observations(patient.id, encounters, random.randint(50, 200))
        
        # Create procedures (5-20 procedures)
        self._create_patient_procedures(patient.id, encounters, random.randint(5, 20))
        
        # Create clinical notes (10-30 notes)
        self._create_clinical_notes(patient.id, encounters, random.randint(10, 30))
        
        # Create care plan if patient has chronic conditions
        if len([c for c in conditions if c.clinical_status == DiagnosisStatusEnum.CHRONIC]) > 0:
            self._create_care_plan(patient.id)
        
        self.db.commit()
        logger.info(f"✅ Created comprehensive patient {patient_id} with full medical history")
        
        return patient_id
    
    def _create_patient_identifiers(self, patient_id: str):
        """Create patient identifiers (MRN, SSN, etc.)."""
        identifiers = [
            PatientIdentifier(
                patient_id=patient_id,
                identifier_type=IdentifierTypeEnum.MRN,
                value=f"MRN-{random.randint(100000, 999999)}",
                system="https://hospital.local",
                period_start=datetime.now() - timedelta(days=random.randint(365, 3650)),
                active=True
            ),
            PatientIdentifier(
                patient_id=patient_id,
                identifier_type=IdentifierTypeEnum.SSN,
                value=f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}",
                system="https://ssa.gov",
                period_start=datetime.now() - timedelta(days=random.randint(365, 3650)),
                active=True
            )
        ]
        
        for identifier in identifiers:
            self.db.add(identifier)
    
    def _create_patient_conditions(self, patient_id: str, count: int) -> List[Condition]:
        """Create patient medical conditions."""
        conditions = []
        selected_conditions = random.sample(self.conditions_library, min(count, len(self.conditions_library)))
        
        for i, condition_data in enumerate(selected_conditions):
            onset_date = fake.date_between(start_date='-5y', end_date='today')
            
            # Determine status based on condition type and onset
            if "diabetes" in condition_data["name"].lower() or "hypertension" in condition_data["name"].lower():
                status = DiagnosisStatusEnum.CHRONIC
            else:
                status = random.choice([DiagnosisStatusEnum.ACTIVE, DiagnosisStatusEnum.CHRONIC, DiagnosisStatusEnum.RESOLVED])
            
            condition = Condition(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                code=condition_data["code"],
                code_system=condition_data["system"],
                display_name=condition_data["name"],
                category="problem-list-item",
                clinical_status=status,
                verification_status="confirmed",
                severity=random.choice(["mild", "moderate", "severe"]),
                onset_date=onset_date,
                onset_type=random.choice(["sudden", "gradual", "chronic"]),
                recorded_date=onset_date + timedelta(days=random.randint(0, 30)),
                recorded_by_id=random.choice(self.providers).id if self.providers else None,
                notes=f"Patient diagnosed with {condition_data['name']} based on clinical presentation and diagnostic workup."
            )
            conditions.append(condition)
            self.db.add(condition)
        
        return conditions
    
    def _create_patient_medications(self, patient_id: str, conditions: List[Condition], count: int):
        """Create patient medications based on their conditions."""
        # Create medications that align with patient's conditions
        condition_names = [c.display_name.lower() for c in conditions]
        relevant_meds = []
        
        # Match medications to conditions
        for med in self.medications_library:
            indication_lower = med["indication"].lower()
            for condition_name in condition_names:
                if any(word in condition_name for word in indication_lower.split()):
                    relevant_meds.append(med)
                    break
        
        # Add some random medications if we don't have enough relevant ones
        if len(relevant_meds) < count:
            remaining_meds = [m for m in self.medications_library if m not in relevant_meds]
            relevant_meds.extend(random.sample(remaining_meds, min(count - len(relevant_meds), len(remaining_meds))))
        
        # Create medication records
        selected_meds = random.sample(relevant_meds, min(count, len(relevant_meds)))
        
        for med_data in selected_meds:
            start_date = fake.date_between(start_date='-3y', end_date='today')
            
            # Determine if medication is still active
            status = random.choices(
                [MedicationStatusEnum.ACTIVE, MedicationStatusEnum.DISCONTINUED, MedicationStatusEnum.COMPLETED],
                weights=[70, 20, 10]
            )[0]
            
            end_date = None
            if status != MedicationStatusEnum.ACTIVE:
                end_date = fake.date_between(start_date=start_date, end_date='today')
            
            medication = MedicationStatement(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                medication_code=f"RX{random.randint(10000, 99999)}",
                medication_name=med_data["name"],
                generic_name=med_data["generic"],
                dosage_text=f"{med_data['dosage']} {med_data['frequency']}",
                dose_quantity=float(med_data['dosage'].split('mg')[0]) if 'mg' in med_data['dosage'] else None,
                dose_unit="mg" if 'mg' in med_data['dosage'] else "mcg" if 'mcg' in med_data['dosage'] else "units",
                frequency=med_data["frequency"],
                route=med_data["route"],
                status=status,
                effective_start=start_date,
                effective_end=end_date,
                indication=med_data["indication"],
                prescriber_id=random.choice(self.providers).id if self.providers else None,
                patient_instructions=f"Take {med_data['dosage']} {med_data['frequency']} as directed.",
                recorded_date=start_date
            )
            self.db.add(medication)
    
    def _create_patient_allergies(self, patient_id: str, count: int):
        """Create patient allergies."""
        if count == 0:
            return
        
        selected_allergies = random.sample(self.allergies_library, min(count, len(self.allergies_library)))
        
        for allergy_data in selected_allergies:
            allergy = AllergyIntolerance(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                allergen_name=allergy_data["name"],
                allergen_type=allergy_data["type"],
                status=AllergyStatusEnum.ACTIVE,
                type="allergy",
                category=allergy_data["type"],
                criticality="high" if allergy_data["severity"] == AllergySeverityEnum.SEVERE else "low",
                severity=allergy_data["severity"],
                reaction_manifestation=[allergy_data["reaction"]],
                reaction_description=allergy_data["reaction"],
                onset_date=fake.date_between(start_date='-10y', end_date='-1y'),
                recorded_date=datetime.now() - timedelta(days=random.randint(30, 365)),
                recorded_by_id=random.choice(self.providers).id if self.providers else None
            )
            self.db.add(allergy)
    
    def _create_patient_encounters(self, patient_id: str, count: int) -> List[Encounter]:
        """Create patient encounters/visits."""
        encounters = []
        
        for i in range(count):
            start_time = fake.date_time_between(start_date='-5y', end_date='now')
            end_time = start_time + timedelta(minutes=random.randint(15, 180))
            
            encounter_type = random.choices(
                [EncounterTypeEnum.OUTPATIENT, EncounterTypeEnum.INPATIENT, EncounterTypeEnum.EMERGENCY, EncounterTypeEnum.VIRTUAL],
                weights=[70, 10, 15, 5]
            )[0]
            
            # Generate realistic chief complaints
            complaints = [
                "Routine follow-up", "Chest pain", "Shortness of breath", "Fatigue",
                "Medication refill", "Annual physical", "Back pain", "Headache",
                "Cough and cold symptoms", "Abdominal pain", "Joint pain", "Dizziness"
            ]
            
            encounter = Encounter(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                status=EncounterStatusEnum.COMPLETED,
                encounter_type=encounter_type,
                priority=random.choice(["routine", "urgent", "emergency"]),
                start_time=start_time,
                end_time=end_time,
                duration_minutes=int((end_time - start_time).total_seconds() / 60),
                location="Main Hospital" if encounter_type == EncounterTypeEnum.INPATIENT else "Outpatient Clinic",
                department=random.choice(["Internal Medicine", "Cardiology", "Emergency", "Family Medicine"]),
                primary_provider_id=random.choice(self.providers).id if self.providers else None,
                chief_complaint=random.choice(complaints),
                reason_code=f"Z{random.randint(10, 99)}.{random.randint(10, 99)}",
                disposition="Discharged home" if encounter_type != EncounterTypeEnum.INPATIENT else "Stable",
                discharge_instructions="Continue current medications. Follow up as needed." if random.choice([True, False]) else None
            )
            encounters.append(encounter)
            self.db.add(encounter)
        
        return encounters
    
    def _create_patient_observations(self, patient_id: str, encounters: List[Encounter], count: int):
        """Create patient observations/lab results."""
        for i in range(count):
            obs_data = random.choice(self.observations_library)
            encounter = random.choice(encounters) if encounters else None
            
            # Generate realistic values based on observation type
            value_quantity = None
            interpretation = "normal"
            
            if "ref_low" in obs_data and "ref_high" in obs_data:
                # Generate values that are sometimes abnormal
                if random.random() < 0.2:  # 20% chance of abnormal
                    if random.choice([True, False]):  # High
                        value_quantity = random.uniform(obs_data["ref_high"], obs_data["ref_high"] * 1.5)
                        interpretation = "high"
                    else:  # Low
                        value_quantity = random.uniform(obs_data["ref_low"] * 0.5, obs_data["ref_low"])
                        interpretation = "low"
                else:  # Normal range
                    value_quantity = random.uniform(obs_data["ref_low"], obs_data["ref_high"])
            else:
                value_quantity = random.uniform(10, 100)  # Default range
            
            observation = Observation(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                encounter_id=encounter.id if encounter else None,
                code=obs_data["code"],
                display_name=obs_data["name"],
                category=obs_data["category"],
                value_quantity=round(value_quantity, 2) if value_quantity else None,
                value_unit=obs_data.get("unit", ""),
                reference_range_low=obs_data.get("ref_low"),
                reference_range_high=obs_data.get("ref_high"),
                interpretation=interpretation,
                status="final",
                effective_datetime=encounter.start_time if encounter else fake.date_time_between(start_date='-2y', end_date='now'),
                issued_datetime=encounter.start_time + timedelta(hours=2) if encounter else fake.date_time_between(start_date='-2y', end_date='now'),
                performer_id=random.choice(self.providers).id if self.providers else None
            )
            self.db.add(observation)
    
    def _create_patient_procedures(self, patient_id: str, encounters: List[Encounter], count: int):
        """Create patient procedures."""
        for i in range(count):
            proc_data = random.choice(self.procedures_library)
            encounter = random.choice(encounters) if encounters else None
            
            performed_date = encounter.start_time if encounter else fake.date_time_between(start_date='-3y', end_date='now')
            
            procedure = Procedure(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                encounter_id=encounter.id if encounter else None,
                code=proc_data["code"],
                display_name=proc_data["name"],
                category=proc_data["category"],
                status="completed",
                performed_datetime=performed_date,
                performed_start=performed_date,
                performed_end=performed_date + timedelta(minutes=random.randint(10, 120)),
                outcome="Successful procedure without complications",
                primary_performer_id=random.choice(self.providers).id if self.providers else None,
                location="Main Hospital",
                indication=f"Clinical indication for {proc_data['name']}",
                notes=f"{proc_data['name']} performed successfully without complications."
            )
            self.db.add(procedure)
    
    def _create_clinical_notes(self, patient_id: str, encounters: List[Encounter], count: int):
        """Create clinical notes."""
        note_types = ["progress", "discharge", "consultation", "procedure", "admission"]
        
        for i in range(count):
            encounter = random.choice(encounters) if encounters else None
            note_type = random.choice(note_types)
            
            # Generate realistic clinical note content
            note_content = self._generate_clinical_note_content(note_type)
            
            note = ClinicalNote(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                encounter_id=encounter.id if encounter else None,
                note_type=note_type,
                specialty=random.choice(["Internal Medicine", "Cardiology", "Pulmonology", "Endocrinology"]),
                title=f"{note_type.title()} Note - {datetime.now().strftime('%Y-%m-%d')}",
                content=note_content,
                status="final",
                author_id=random.choice(self.providers).id if self.providers else None,
                dictated_datetime=encounter.start_time if encounter else fake.date_time_between(start_date='-2y', end_date='now'),
                transcribed_datetime=encounter.start_time + timedelta(hours=2) if encounter else fake.date_time_between(start_date='-2y', end_date='now'),
                authenticated_datetime=encounter.end_time if encounter else fake.date_time_between(start_date='-2y', end_date='now'),
                processed_by_ai=False
            )
            self.db.add(note)
    
    def _generate_clinical_note_content(self, note_type: str) -> str:
        """Generate realistic clinical note content."""
        if note_type == "progress":
            return """PROGRESS NOTE

CHIEF COMPLAINT: Follow-up for diabetes and hypertension

HISTORY OF PRESENT ILLNESS:
Patient returns for routine follow-up of type 2 diabetes mellitus and hypertension. Reports good adherence to medications. Blood sugar levels have been well controlled with current regimen. No episodes of hypoglycemia. Blood pressure readings at home have been within target range.

REVIEW OF SYSTEMS: 
Negative for chest pain, shortness of breath, palpitations, dizziness, visual changes, polyuria, or polydipsia.

PHYSICAL EXAMINATION:
Vital signs: BP 128/78, HR 72, RR 16, Temp 98.6°F
General: Well-appearing, no acute distress
HEENT: Normal
Cardiovascular: Regular rate and rhythm, no murmurs
Pulmonary: Clear bilaterally
Extremities: No edema

ASSESSMENT AND PLAN:
1. Type 2 Diabetes Mellitus - well controlled
   - Continue current metformin regimen
   - HbA1c due, will order today
   - Continue dietary counseling
   
2. Hypertension - well controlled
   - Continue lisinopril
   - Monitor blood pressure at home
   
3. Health maintenance
   - Annual diabetic eye exam due
   - Continue statin therapy

FOLLOW-UP: Return in 3 months or sooner if concerns."""
        
        elif note_type == "discharge":
            return """DISCHARGE SUMMARY

ADMISSION DATE: [Date]
DISCHARGE DATE: [Date]

PRINCIPAL DIAGNOSIS: Acute exacerbation of heart failure

HOSPITAL COURSE:
Patient admitted with acute shortness of breath and lower extremity edema. Initial chest X-ray showed pulmonary congestion. Patient was treated with IV diuretics with good response. Echocardiogram showed reduced ejection fraction at 35%.

DISCHARGE MEDICATIONS:
1. Furosemide 40mg daily
2. Lisinopril 10mg daily  
3. Metoprolol 25mg twice daily
4. Atorvastatin 20mg daily

DISCHARGE INSTRUCTIONS:
- Daily weights, call if weight gain >2 lbs in 24 hours
- Low sodium diet <2g daily
- Follow up with cardiology in 2 weeks
- Return to ED if worsening shortness of breath

FOLLOW-UP APPOINTMENTS:
Cardiology: Dr. Smith in 2 weeks
Primary Care: Dr. Jones in 1 week"""
        
        else:
            return f"""CLINICAL NOTE - {note_type.upper()}

Patient evaluated for routine {note_type}. Comprehensive assessment performed including history, physical examination, and review of systems. Clinical findings documented and appropriate management plan established. Patient counseled on treatment options and follow-up care. All questions answered satisfactorily.

Plan discussed with patient and family. Follow-up arranged as appropriate."""
    
    def _create_care_plan(self, patient_id: str):
        """Create a care plan for patients with chronic conditions."""
        care_plan = CarePlan(
            id=str(uuid.uuid4()),
            patient_id=patient_id,
            title="Comprehensive Chronic Disease Management Plan",
            description="Integrated care plan for management of chronic medical conditions including diabetes, hypertension, and associated comorbidities.",
            status="active",
            intent="plan",
            period_start=datetime.now(),
            period_end=datetime.now() + timedelta(days=365),
            created_date=datetime.now(),
            created_by_id=random.choice(self.providers).id if self.providers else None,
            goals=[
                {
                    "description": "Maintain HbA1c < 7.0%",
                    "target_date": (datetime.now() + timedelta(days=90)).isoformat(),
                    "status": "in-progress"
                },
                {
                    "description": "Blood pressure < 130/80 mmHg",
                    "target_date": (datetime.now() + timedelta(days=90)).isoformat(),
                    "status": "in-progress"
                }
            ],
            activities=[
                {
                    "description": "Quarterly HbA1c monitoring",
                    "frequency": "Every 3 months",
                    "status": "scheduled"
                },
                {
                    "description": "Monthly blood pressure monitoring",
                    "frequency": "Monthly",
                    "status": "active"
                }
            ]
        )
        self.db.add(care_plan)
    
    def create_multiple_patients(self, count: int = 5):
        """Create multiple comprehensive patients."""
        logger.info(f"🏥 Creating {count} comprehensive patients...")
        
        patient_ids = []
        for i in range(count):
            patient_id = self.create_comprehensive_patient()
            patient_ids.append(patient_id)
        
        logger.info(f"✅ Created {count} comprehensive patients: {patient_ids}")
        return patient_ids
    
    def cleanup(self):
        """Close database connection."""
        if self.db:
            self.db.close()
            logger.info("🔒 Database connection closed")

def main():
    """Main function to run the comprehensive patient data insertion."""
    try:
        # Initialize the inserter
        inserter = ComprehensivePatientDataInserter()
        
        # Setup database
        inserter.setup_database()
        
        # Create providers first (required for foreign key relationships)
        inserter.create_providers(15)
        
        # Create comprehensive patients
        patient_count = 5  # Change this number to create more or fewer patients
        patient_ids = inserter.create_multiple_patients(patient_count)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 PATIENT DATA INSERTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successfully created {patient_count} comprehensive patients")
        logger.info(f"✅ Created 15 healthcare providers")
        logger.info(f"✅ Patient IDs: {patient_ids}")
        logger.info("=" * 60)
        logger.info("🚀 Database is ready for the EHR application!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error during patient data insertion: {e}")
        raise
    finally:
        # Cleanup
        if 'inserter' in locals():
            inserter.cleanup()

if __name__ == "__main__":
    main()