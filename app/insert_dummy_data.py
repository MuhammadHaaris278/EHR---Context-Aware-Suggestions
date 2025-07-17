from datetime import datetime, timedelta
from app.db import SessionLocal, Base, Patient, Diagnosis, Medication, Visit
import random

def insert_dummy_data():
    # Create a database session
    db = SessionLocal()
    
    try:
        # List of first names, last names, genders, ICD codes, and medications
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa", "Thomas", "Anna"]
        last_names = ["Doe", "Smith", "Johnson", "Brown", "Taylor", "Wilson", "Anderson", "Thomas", "Jackson", "White"]
        genders = ["M", "F"]
        icd_codes = [
            "I10", "E11.9", "J45", "K80.0", "M17.9", "N39.0", "I25.10", "G43.909", "F32.9", "E78.5"
        ]  # Hypertension, Diabetes, Asthma, Cholelithiasis, Osteoarthritis, UTI, CAD, Migraine, Depression, Hyperlipidemia
        diagnoses_desc = [
            "Essential hypertension", "Type 2 diabetes mellitus", "Asthma", "Cholelithiasis", "Osteoarthritis",
            "Urinary tract infection", "Coronary artery disease", "Migraine", "Major depressive disorder", "Hyperlipidemia"
        ]
        medications = [
            {"name": "Lisinopril", "dosage": "10mg", "frequency": "Daily"},
            {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily"},
            {"name": "Albuterol", "dosage": "Inhaler", "frequency": "As needed"},
            {"name": "Ursodiol", "dosage": "300mg", "frequency": "Daily"},
            {"name": "Ibuprofen", "dosage": "200mg", "frequency": "As needed"},
            {"name": "Nitrofurantoin", "dosage": "100mg", "frequency": "Twice daily"},
            {"name": "Aspirin", "dosage": "81mg", "frequency": "Daily"},
            {"name": "Sumatriptan", "dosage": "50mg", "frequency": "As needed"},
            {"name": "Sertraline", "dosage": "50mg", "frequency": "Daily"},
            {"name": "Atorvastatin", "dosage": "20mg", "frequency": "Daily"}
        ]
        visit_types = ["follow-up", "emergency", "routine"]
        chief_complaints = [
            "Routine check-up", "Chest pain", "Shortness of breath", "Abdominal pain", "Joint pain",
            "Urinary symptoms", "Follow-up on heart condition", "Headache", "Mood changes", "Cholesterol review"
        ]
        providers = ["Dr. Smith", "Dr. Jones", "Dr. Lee", "Dr. Patel", "Dr. Kim"]

        # Insert 10 patients
        for i in range(10):
            patient_id = f"100{i:02d}"  # Generates IDs like 10000, 10001, ..., 10009
            birth_year = random.randint(1960, 2000)
            patient = Patient(
                id=patient_id,
                first_name=first_names[i],
                last_name=last_names[i],
                date_of_birth=datetime(birth_year, random.randint(1, 12), random.randint(1, 28)),
                gender=random.choice(genders)
            )
            db.add(patient)

            # Add 1-2 diagnoses per patient
            num_diagnoses = random.randint(1, 2)
            for _ in range(num_diagnoses):
                diag_index = random.randint(0, len(icd_codes) - 1)
                diagnosis = Diagnosis(
                    patient_id=patient_id,
                    icd_code=icd_codes[diag_index],
                    description=diagnoses_desc[diag_index],
                    diagnosis_date=datetime(2022, random.randint(1, 12), random.randint(1, 28)),
                    status=random.choice(["active", "resolved"])
                )
                db.add(diagnosis)

            # Add 1-2 medications per patient
            num_meds = random.randint(1, 2)
            for _ in range(num_meds):
                med_index = random.randint(0, len(medications) - 1)
                medication = Medication(
                    patient_id=patient_id,
                    name=medications[med_index]["name"],
                    dosage=medications[med_index]["dosage"],
                    frequency=medications[med_index]["frequency"],
                    start_date=datetime(2023, random.randint(1, 12), random.randint(1, 28)),
                    status=random.choice(["active", "discontinued"])
                )
                db.add(medication)

            # Add 1-3 visits per patient
            num_visits = random.randint(1, 3)
            for _ in range(num_visits):
                visit_date = datetime(2025, random.randint(1, 7), random.randint(1, 28))
                visit = Visit(
                    patient_id=patient_id,
                    visit_date=visit_date,
                    visit_type=random.choice(visit_types),
                    chief_complaint=random.choice(chief_complaints),
                    notes=f"Patient visited for {random.choice(chief_complaints).lower()}. Assessment pending.",
                    provider=random.choice(providers)
                )
                db.add(visit)

        # Commit the transaction
        db.commit()
        print("Dummy data for 10 patients inserted successfully!")

    except Exception as e:
        db.rollback()
        print(f"Error inserting dummy data: {e}")

    finally:
        db.close()

if __name__ == "__main__":
    insert_dummy_data()