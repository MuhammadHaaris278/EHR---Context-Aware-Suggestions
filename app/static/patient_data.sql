-- ===============================================================================
-- COMPREHENSIVE EHR DATA FOR 80-YEAR-OLD PATIENT
-- Patient: Eleanor Margaret Henderson (ID: 1)
-- Born: March 15, 1944 (Age: 80)
-- Medical History Span: 1962-2024 (62 years of comprehensive medical data)
-- ===============================================================================

-- ===========================================
-- HEALTHCARE PROVIDERS (Extensive provider network)
-- ===========================================

INSERT INTO providers (id, npi, first_name, last_name, specialty, department, title, active) VALUES
-- Primary Care Physicians (over the decades)
('prov_001', '1234567890', 'Robert', 'Thompson', 'Family Medicine', 'Primary Care', 'MD', true),
('prov_002', '1234567891', 'William', 'Anderson', 'Internal Medicine', 'Primary Care', 'MD', false), -- Retired
('prov_003', '1234567892', 'Margaret', 'Foster', 'Geriatrics', 'Internal Medicine', 'MD', true),

-- Specialists
('prov_004', '1234567893', 'Sarah', 'Williams', 'Cardiology', 'Cardiology', 'MD', true),
('prov_005', '1234567894', 'Michael', 'Johnson', 'Endocrinology', 'Endocrinology', 'MD', true),
('prov_006', '1234567895', 'Lisa', 'Davis', 'Orthopedics', 'Orthopedics', 'MD', true),
('prov_007', '1234567896', 'James', 'Wilson', 'Gastroenterology', 'Gastroenterology', 'MD', true),
('prov_008', '1234567897', 'Patricia', 'Brown', 'Ophthalmology', 'Ophthalmology', 'MD', true),
('prov_009', '1234567898', 'David', 'Miller', 'Emergency Medicine', 'Emergency', 'MD', true),
('prov_010', '1234567899', 'Mary', 'Garcia', 'Nephrology', 'Nephrology', 'MD', true),
('prov_011', '1234568900', 'Richard', 'Martinez', 'Neurology', 'Neurology', 'MD', true),
('prov_012', '1234568901', 'Jennifer', 'Rodriguez', 'Oncology', 'Oncology', 'MD', true),
('prov_013', '1234568902', 'Charles', 'Lee', 'Pulmonology', 'Pulmonology', 'MD', true),
('prov_014', '1234568903', 'Susan', 'Taylor', 'Dermatology', 'Dermatology', 'MD', true),
('prov_015', '1234568904', 'Daniel', 'Clark', 'Urology', 'Urology', 'MD', true),
('prov_016', '1234568905', 'Linda', 'White', 'Rheumatology', 'Rheumatology', 'MD', true),
('prov_017', '1234568906', 'Paul', 'Harris', 'Psychiatry', 'Psychiatry', 'MD', true),
('prov_018', '1234568907', 'Nancy', 'Martin', 'Gynecology', 'Women''s Health', 'MD', false), -- Retired
('prov_019', '1234568908', 'Kevin', 'Thompson', 'Anesthesiology', 'Anesthesiology', 'MD', true),
('prov_020', '1234568909', 'Barbara', 'Jackson', 'Radiology', 'Radiology', 'MD', true),

-- Additional specialists
('prov_021', '1234568910', 'Steven', 'Young', 'Cardiothoracic Surgery', 'Surgery', 'MD', true),
('prov_022', '1234568911', 'Helen', 'King', 'Infectious Disease', 'Internal Medicine', 'MD', true),
('prov_023', '1234568912', 'Robert', 'Wright', 'General Surgery', 'Surgery', 'MD', true),
('prov_024', '1234568913', 'Carol', 'Lopez', 'Endocrinology', 'Endocrinology', 'MD', true),
('prov_025', '1234568914', 'Mark', 'Hill', 'Pathology', 'Pathology', 'MD', true);

-- ===========================================
-- PATIENT RECORD (Comprehensive demographics)
-- ===========================================

INSERT INTO patients (
    id, first_name, middle_name, last_name, date_of_birth, gender, race, ethnicity, 
    preferred_language, phone_primary, phone_secondary, email, address, active, 
    deceased, created_at, updated_at
) VALUES (
    '1',
    'Eleanor',
    'Margaret',
    'Henderson',
    '1944-03-15',
    'female',
    'White',
    'Not Hispanic or Latino',
    'English',
    '555-123-4567',
    '555-987-6543',
    'eleanor.henderson@email.com',
    '{"street": "1234 Maple Street", "city": "Springfield", "state": "IL", "zip": "62701", "country": "USA", "type": "home"}',
    true,
    false,
    '1962-01-01 08:00:00',
    '2024-12-15 10:30:00'
);

-- ===========================================
-- PATIENT IDENTIFIERS (Multiple ID systems)
-- ===========================================

INSERT INTO patient_identifiers (patient_id, identifier_type, value, system, period_start, active) VALUES
('1', 'medical_record_number', 'MRN123456789', 'Springfield General Hospital', '1962-01-01', true),
('1', 'medical_record_number', 'MRN987654321', 'University Medical Center', '1975-06-01', true),
('1', 'social_security_number', '123-45-6789', 'Social Security Administration', '1962-01-01', true),
('1', 'insurance_id', 'MEDICARE123456789A', 'Medicare', '2009-03-15', true),
('1', 'insurance_id', 'BCBS987654321', 'Blue Cross Blue Shield', '1965-01-01', true),
('1', 'insurance_id', 'AETNA456789123', 'Aetna', '1980-01-01', false), -- Historical insurance
('1', 'drivers_license', 'IL123456789', 'Illinois DMV', '1962-03-15', false); -- Expired

-- ===========================================
-- MEDICAL CONDITIONS (Comprehensive 60+ year history)
-- ===========================================

INSERT INTO conditions (
    id, patient_id, code, code_system, display_name, category, clinical_status, 
    verification_status, severity, onset_date, resolved_date, recorded_date, 
    recorded_by_id, body_site, notes
) VALUES 
-- Early adulthood conditions (1960s-1970s)
('cond_001', '1', 'N30.9', 'ICD-10', 'Cystitis, unspecified', 'encounter-diagnosis', 'resolved', 'confirmed', 'mild', '1965-08-15', '1965-08-25', '1965-08-15', 'prov_002', 'bladder', 'First documented UTI. Treated with antibiotics.'),
('cond_002', '1', 'O80.1', 'ICD-10', 'Spontaneous vertex delivery', 'encounter-diagnosis', 'resolved', 'confirmed', NULL, '1967-11-20', '1967-11-20', '1967-11-20', 'prov_018', 'uterus', 'First pregnancy and delivery. Healthy baby girl.'),
('cond_003', '1', 'O80.1', 'ICD-10', 'Spontaneous vertex delivery', 'encounter-diagnosis', 'resolved', 'confirmed', NULL, '1970-04-10', '1970-04-10', '1970-04-10', 'prov_018', 'uterus', 'Second pregnancy and delivery. Healthy baby boy.'),
('cond_004', '1', 'S72.002A', 'ICD-10', 'Fracture of unspecified part of neck of left femur, initial encounter', 'encounter-diagnosis', 'resolved', 'confirmed', 'moderate', '1975-12-03', '1976-04-03', '1975-12-03', 'prov_006', 'left hip', 'Skiing accident. Treated with internal fixation.'),

-- Middle age conditions (1980s-1990s)
('cond_005', '1', 'I10', 'ICD-10', 'Essential hypertension', 'problem-list-item', 'active', 'confirmed', 'moderate', '1982-11-20', NULL, '1982-11-20', 'prov_002', NULL, 'Blood pressure consistently elevated >140/90. Family history positive.'),
('cond_006', '1', 'E11.9', 'ICD-10', 'Type 2 diabetes mellitus without complications', 'problem-list-item', 'active', 'confirmed', 'moderate', '1985-06-15', NULL, '1985-06-15', 'prov_002', NULL, 'Diagnosed during routine physical. Fasting glucose 165 mg/dL. Initially managed with diet.'),
('cond_007', '1', 'M15.9', 'ICD-10', 'Polyosteoarthritis, unspecified', 'problem-list-item', 'active', 'confirmed', 'moderate', '1988-09-12', NULL, '1988-09-12', 'prov_006', 'bilateral knees, hands', 'Progressive joint pain and morning stiffness. X-rays show joint space narrowing.'),
('cond_008', '1', 'E78.5', 'ICD-10', 'Hyperlipidemia, unspecified', 'problem-list-item', 'active', 'confirmed', 'mild', '1990-03-08', NULL, '1990-03-08', 'prov_002', NULL, 'Total cholesterol 285 mg/dL, LDL 190 mg/dL. Started dietary modifications.'),
('cond_009', '1', 'N95.1', 'ICD-10', 'Menopausal and female climacteric states', 'problem-list-item', 'resolved', 'confirmed', 'mild', '1992-01-01', '1997-01-01', '1992-01-01', 'prov_018', NULL, 'Natural menopause. Hot flashes and mood changes managed with lifestyle modifications.'),
('cond_010', '1', 'I25.10', 'ICD-10', 'Atherosclerotic heart disease of native coronary artery without angina pectoris', 'problem-list-item', 'active', 'confirmed', 'moderate', '1995-03-10', NULL, '1995-03-10', 'prov_004', NULL, 'Discovered during cardiac catheterization. Two-vessel disease (LAD and RCA).'),

-- Cancer history (Late 1990s)
('cond_011', '1', 'C50.911', 'ICD-10', 'Malignant neoplasm of unspecified site of right female breast', 'problem-list-item', 'resolved', 'confirmed', 'severe', '1998-04-12', '2003-04-12', '1998-04-12', 'prov_012', 'right breast', 'Stage IIA invasive ductal carcinoma. Successfully treated with surgery, chemotherapy, and radiation.'),
('cond_012', '1', 'Z85.3', 'ICD-10', 'Personal history of malignant neoplasm of breast', 'problem-list-item', 'active', 'confirmed', NULL, '2003-04-12', NULL, '2003-04-12', 'prov_012', NULL, 'History of breast cancer. Requires ongoing surveillance.'),

-- 2000s conditions
('cond_013', '1', 'E03.9', 'ICD-10', 'Hypothyroidism, unspecified', 'problem-list-item', 'active', 'confirmed', 'mild', '2000-12-08', NULL, '2000-12-08', 'prov_005', NULL, 'TSH elevated at 8.5 mIU/L. Started on levothyroxine.'),
('cond_014', '1', 'K80.20', 'ICD-10', 'Calculus of gallbladder without obstruction', 'encounter-diagnosis', 'resolved', 'confirmed', 'moderate', '2005-07-22', '2005-08-15', '2005-07-22', 'prov_007', 'gallbladder', 'Symptomatic gallstones. Treated with laparoscopic cholecystectomy.'),
('cond_015', '1', 'K21.9', 'ICD-10', 'Gastro-esophageal reflux disease without esophagitis', 'problem-list-item', 'active', 'confirmed', 'mild', '2008-09-05', NULL, '2008-09-05', 'prov_007', NULL, 'Chronic heartburn and regurgitation. Responsive to PPI therapy.'),

-- 2010s conditions
('cond_016', '1', 'N18.3', 'ICD-10', 'Chronic kidney disease, stage 3 (moderate)', 'problem-list-item', 'active', 'confirmed', 'moderate', '2010-02-14', NULL, '2010-02-14', 'prov_010', NULL, 'eGFR 45-59. Likely diabetic nephropathy. Started ACE inhibitor.'),
('cond_017', '1', 'J44.1', 'ICD-10', 'Chronic obstructive pulmonary disease with acute exacerbation', 'encounter-diagnosis', 'resolved', 'confirmed', 'moderate', '2012-01-20', '2012-03-20', '2012-01-20', 'prov_013', NULL, 'COPD exacerbation requiring hospitalization. 40 pack-year smoking history.'),
('cond_018', '1', 'Z87.891', 'ICD-10', 'Personal history of nicotine dependence', 'problem-list-item', 'active', 'confirmed', NULL, '2012-02-01', NULL, '2012-02-01', 'prov_013', NULL, 'Former smoker. Quit after COPD exacerbation. 40 pack-years total.'),
('cond_019', '1', 'H25.9', 'ICD-10', 'Unspecified age-related cataract', 'problem-list-item', 'active', 'confirmed', 'mild', '2015-09-20', NULL, '2015-09-20', 'prov_008', 'bilateral', 'Slowly progressive vision changes. Bilateral cataracts noted on exam.'),
('cond_020', '1', 'H91.90', 'ICD-10', 'Unspecified hearing loss, unspecified ear', 'problem-list-item', 'active', 'confirmed', 'mild', '2016-06-10', NULL, '2016-06-10', 'prov_003', 'bilateral', 'Age-related sensorineural hearing loss. Hearing aids recommended.'),
('cond_021', '1', 'S72.001A', 'ICD-10', 'Fracture of unspecified part of neck of right femur, initial encounter', 'encounter-diagnosis', 'resolved', 'confirmed', 'severe', '2018-11-15', '2019-05-15', '2018-11-15', 'prov_009', 'right hip', 'Fall at home. Surgical repair with total hip replacement.'),

-- Recent conditions (2020s)
('cond_022', '1', 'F32.0', 'ICD-10', 'Major depressive disorder, single episode, mild', 'problem-list-item', 'active', 'confirmed', 'mild', '2020-03-15', NULL, '2020-03-15', 'prov_017', NULL, 'Depression related to social isolation during COVID-19 pandemic.'),
('cond_023', '1', 'M79.3', 'ICD-10', 'Panniculitis, unspecified', 'encounter-diagnosis', 'resolved', 'confirmed', 'mild', '2021-08-10', '2021-10-10', '2021-08-10', 'prov_014', 'lower legs', 'Inflammatory condition of subcutaneous fat. Resolved with steroid treatment.'),
('cond_024', '1', 'G93.1', 'ICD-10', 'Anoxic brain damage, not elsewhere classified', 'encounter-diagnosis', 'resolved', 'confirmed', 'mild', '2022-04-20', '2022-06-20', '2022-04-20', 'prov_011', NULL, 'Mild cognitive changes post-anesthesia. Resolved over time.'),
('cond_025', '1', 'I48.91', 'ICD-10', 'Unspecified atrial fibrillation', 'problem-list-item', 'active', 'confirmed', 'moderate', '2023-02-15', NULL, '2023-02-15', 'prov_004', NULL, 'New onset atrial fibrillation. Started on anticoagulation.'),

-- Additional chronic conditions
('cond_026', '1', 'M81.0', 'ICD-10', 'Age-related osteoporosis without current pathological fracture', 'problem-list-item', 'active', 'confirmed', 'moderate', '2019-03-10', NULL, '2019-03-10', 'prov_006', NULL, 'DEXA scan shows T-score -2.8. Started on bisphosphonate therapy.'),
('cond_027', '1', 'H35.30', 'ICD-10', 'Unspecified macular degeneration', 'problem-list-item', 'active', 'confirmed', 'mild', '2020-11-05', NULL, '2020-11-05', 'prov_008', 'bilateral', 'Early age-related macular degeneration. Regular monitoring required.'),
('cond_028', '1', 'I83.90', 'ICD-10', 'Asymptomatic varicose veins of unspecified lower extremity', 'problem-list-item', 'active', 'confirmed', 'mild', '2017-07-15', NULL, '2017-07-15', 'prov_003', 'bilateral legs', 'Cosmetic and mild symptoms. Compression stockings recommended.'),
('cond_029', '1', 'K59.00', 'ICD-10', 'Constipation, unspecified', 'problem-list-item', 'active', 'confirmed', 'mild', '2019-09-20', NULL, '2019-09-20', 'prov_003', NULL, 'Chronic constipation. Managed with dietary fiber and laxatives.'),
('cond_030', '1', 'R06.02', 'ICD-10', 'Shortness of breath', 'encounter-diagnosis', 'resolved', 'confirmed', 'mild', '2024-06-20', '2024-06-25', '2024-06-20', 'prov_009', NULL, 'Episode of dyspnea. Cardiac workup negative.');

-- ===========================================
-- ALLERGIES AND INTOLERANCES (Comprehensive allergy history)
-- ===========================================

INSERT INTO allergy_intolerances (
    id, patient_id, allergen_code, allergen_name, allergen_type, status, type, 
    category, criticality, severity, reaction_manifestation, reaction_description, 
    onset_date, recorded_date, recorded_by_id
) VALUES
('allergy_001', '1', '387207008', 'Penicillin', 'medication', 'active', 'allergy', 'medication', 'high', 'moderate', '{"rash", "hives", "itching"}', 'Generalized urticaria and pruritus within 30 minutes of oral penicillin administration. No respiratory symptoms.', '1975-05-20', '1975-05-20', 'prov_002'),
('allergy_002', '1', '256259004', 'Shellfish', 'food', 'active', 'allergy', 'food', 'high', 'severe', '{"swelling", "difficulty breathing", "nausea", "vomiting"}', 'Facial and throat swelling with respiratory distress and GI symptoms after eating shrimp. Required epinephrine.', '1988-08-12', '1988-08-12', 'prov_002'),
('allergy_003', '1', '396064000', 'Sulfonamides', 'medication', 'active', 'allergy', 'medication', 'low', 'mild', '{"rash"}', 'Mild maculopapular rash on trunk and extremities with sulfa antibiotics. No systemic symptoms.', '1992-03-08', '1992-03-08', 'prov_002'),
('allergy_004', '1', '264287008', 'Tree pollen', 'environment', 'active', 'allergy', 'environment', 'low', 'mild', '{"sneezing", "watery eyes", "congestion", "runny nose"}', 'Seasonal allergic rhinitis symptoms in spring months. Responsive to antihistamines.', '1980-04-15', '1980-04-15', 'prov_002'),
('allergy_005', '1', '226842001', 'Latex', 'environment', 'active', 'allergy', 'environment', 'moderate', 'moderate', '{"skin irritation", "rash", "swelling"}', 'Contact dermatitis and localized swelling with latex gloves. Developed during medical procedures.', '1995-09-10', '1995-09-10', 'prov_012'),
('allergy_006', '1', '226955001', 'Dust mites', 'environment', 'active', 'allergy', 'environment', 'low', 'mild', '{"sneezing", "congestion", "itchy eyes"}', 'Year-round allergic rhinitis symptoms, worse in bedroom. Dust mite allergy confirmed by testing.', '1985-11-20', '1985-11-20', 'prov_002'),
('allergy_007', '1', '387458008', 'Aspirin', 'medication', 'inactive', 'intolerance', 'medication', 'low', 'mild', '{"stomach upset", "nausea"}', 'GI intolerance to full-dose aspirin. Tolerates low-dose 81mg for cardioprotection.', '2005-06-15', '2005-06-15', 'prov_004'),
('allergy_008', '1', '227493005', 'Cashews', 'food', 'active', 'allergy', 'food', 'moderate', 'moderate', '{"oral tingling", "lip swelling", "hives"}', 'Oral allergy syndrome with lip swelling and urticaria. Avoids all tree nuts as precaution.', '2010-03-22', '2010-03-22', 'prov_003');

-- ===========================================
-- MEDICATION STATEMENTS (Extensive medication history - 60+ years)
-- ===========================================

INSERT INTO medication_statements (
    id, patient_id, medication_code, medication_name, generic_name, brand_name,
    dosage_text, dose_quantity, dose_unit, frequency, route, status,
    effective_start, effective_end, indication, prescriber_id, patient_instructions, notes
) VALUES

-- === CURRENT ACTIVE MEDICATIONS ===
('med_001', '1', '310965', 'Metformin 1000mg tablets', 'Metformin hydrochloride', 'Glucophage', '1000mg twice daily', 1000, 'mg', 'BID', 'oral', 'active', '1985-06-15', NULL, 'Type 2 diabetes mellitus', 'prov_005', 'Take with meals to reduce stomach upset. Monitor for lactic acidosis symptoms.', 'Long-term diabetes management. Dose increased over time.'),
('med_002', '1', '197361', 'Lisinopril 10mg tablets', 'Lisinopril', 'Prinivil', '10mg once daily', 10, 'mg', 'daily', 'oral', 'active', '1995-03-10', NULL, 'Hypertension and cardioprotection', 'prov_004', 'Take at same time each day. Report persistent cough.', 'ACE inhibitor for BP control and renal protection'),
('med_003', '1', '617312', 'Atorvastatin 40mg tablets', 'Atorvastatin calcium', 'Lipitor', '40mg once daily at bedtime', 40, 'mg', 'daily', 'oral', 'active', '2000-09-20', NULL, 'Hyperlipidemia', 'prov_004', 'Take in evening. Report muscle pain or weakness.', 'Statin therapy for cholesterol and CV protection'),
('med_004', '1', '855288', 'Aspirin 81mg tablets', 'Aspirin', 'Bayer', '81mg once daily', 81, 'mg', 'daily', 'oral', 'active', '1995-03-10', NULL, 'Cardiovascular protection', 'prov_004', 'Take with food to reduce stomach irritation.', 'Low-dose for cardioprotection'),
('med_005', '1', '966221', 'Levothyroxine 75mcg tablets', 'Levothyroxine sodium', 'Synthroid', '75mcg once daily on empty stomach', 75, 'mcg', 'daily', 'oral', 'active', '2000-12-08', NULL, 'Hypothyroidism', 'prov_005', 'Take 30-60 minutes before breakfast. Separate from other medications.', 'Thyroid hormone replacement'),
('med_006', '1', '308136', 'Omeprazole 20mg capsules', 'Omeprazole', 'Prilosec', '20mg once daily', 20, 'mg', 'daily', 'oral', 'active', '2008-09-05', NULL, 'GERD', 'prov_007', 'Take before first meal of day. Long-term use monitored.', 'PPI for acid suppression'),
('med_007', '1', '206878', 'Calcium carbonate 500mg tablets', 'Calcium carbonate', 'Tums', '500mg twice daily with meals', 500, 'mg', 'BID', 'oral', 'active', '2010-02-14', NULL, 'Calcium supplementation and osteoporosis prevention', 'prov_010', 'Take with vitamin D for better absorption.', 'Bone health support'),
('med_008', '1', '317541', 'Vitamin D3 2000 IU tablets', 'Cholecalciferol', 'Nature Made', '2000 IU once daily', 2000, 'IU', 'daily', 'oral', 'active', '2010-02-14', NULL, 'Vitamin D deficiency and bone health', 'prov_010', 'Can take with or without food.', 'Increased dose for osteoporosis'),
('med_009', '1', '1292884', 'Warfarin 5mg tablets', 'Warfarin sodium', 'Coumadin', '5mg daily, adjust per INR', 5, 'mg', 'daily', 'oral', 'active', '2023-02-15', NULL, 'Atrial fibrillation anticoagulation', 'prov_004', 'Regular INR monitoring required. Consistent diet.', 'Anticoagulation for stroke prevention'),
('med_010', '1', '1233815', 'Alendronate 70mg tablets', 'Alendronate sodium', 'Fosamax', '70mg once weekly', 70, 'mg', 'weekly', 'oral', 'active', '2019-03-10', NULL, 'Osteoporosis', 'prov_006', 'Take on empty stomach with full glass water. Remain upright 30 min.', 'Bisphosphonate for bone density'),

-- === PAIN MANAGEMENT ===
('med_011', '1', '161', 'Acetaminophen 650mg tablets', 'Acetaminophen', 'Tylenol', '650mg every 6 hours as needed', 650, 'mg', 'PRN', 'oral', 'active', '1988-09-12', NULL, 'Arthritis pain and general analgesia', 'prov_006', 'Do not exceed 3000mg per day. Monitor liver function.', 'Primary pain medication for arthritis'),
('med_012', '1', '5640', 'Ibuprofen 400mg tablets', 'Ibuprofen', 'Advil', '400mg every 8 hours as needed', 400, 'mg', 'PRN', 'oral', 'on_hold', '1990-01-01', NULL, 'Arthritis pain', 'prov_006', 'Take with food. Monitor kidney function closely.', 'On hold due to CKD progression'),

-- === RECENTLY DISCONTINUED MEDICATIONS ===
('med_013', '1', '153666', 'Glyburide 5mg tablets', 'Glyburide', 'DiaBeta', '5mg twice daily', 5, 'mg', 'BID', 'oral', 'discontinued', '1990-08-15', '2018-03-20', 'Type 2 diabetes mellitus', 'prov_005', 'Take 30 minutes before meals.', 'Discontinued due to hypoglycemia risk in elderly'),
('med_014', '1', '104375', 'Furosemide 40mg tablets', 'Furosemide', 'Lasix', '40mg once daily', 40, 'mg', 'daily', 'oral', 'discontinued', '2012-06-10', '2020-11-15', 'Heart failure and edema', 'prov_004', 'Take in morning. Monitor electrolytes.', 'Discontinued when heart failure resolved'),
('med_015', '1', '1190521', 'Sertraline 50mg tablets', 'Sertraline hydrochloride', 'Zoloft', '50mg once daily', 50, 'mg', 'daily', 'oral', 'discontinued', '2020-03-15', '2023-03-15', 'Major depressive disorder', 'prov_017', 'Take with food. Gradual discontinuation.', 'Completed 3-year course for depression'),

-- === COMPLETED MEDICATION COURSES ===
('med_016', '1', '213269', 'Amoxicillin 500mg capsules', 'Amoxicillin', 'Amoxil', '500mg three times daily', 500, 'mg', 'TID', 'oral', 'completed', '2024-03-15', '2024-03-25', 'Bacterial pneumonia', 'prov_003', 'Complete full course even if feeling better.', '10-day course for community-acquired pneumonia'),
('med_017', '1', '996740', 'Prednisone 20mg tablets', 'Prednisone', 'Deltasone', '20mg daily with taper', 20, 'mg', 'daily', 'oral', 'completed', '2023-09-10', '2023-09-24', 'COPD exacerbation', 'prov_013', 'Take with food. Taper as directed.', '14-day steroid course with taper'),
('med_018', '1', '617640', 'Ciprofloxacin 500mg tablets', 'Ciprofloxacin', 'Cipro', '500mg twice daily', 500, 'mg', 'BID', 'oral', 'completed', '2023-07-20', '2023-07-27', 'Urinary tract infection', 'prov_015', 'Drink plenty of water. Avoid dairy products.', '7-day course for complicated UTI'),
('med_019', '1', '1099858', 'Azithromycin 250mg tablets', 'Azithromycin', 'Z-Pack', '500mg day 1, then 250mg daily', 250, 'mg', 'daily', 'oral', 'completed', '2022-11-10', '2022-11-15', 'Acute bronchitis', 'prov_013', 'Take on empty stomach or with food.', '5-day Z-pack for respiratory infection'),

-- === HISTORICAL CANCER TREATMENT ===
('med_020', '1', '40048', 'Doxorubicin injection', 'Doxorubicin hydrochloride', 'Adriamycin', '60 mg/m2 IV every 21 days', 60, 'mg/m2', 'cycle', 'intravenous', 'completed', '1998-05-15', '1998-11-15', 'Breast cancer', 'prov_012', 'Pre-medication with antiemetics. Cardiac monitoring.', 'AC regimen cycle 1-4 for breast cancer'),
('med_021', '1', '56946', 'Cyclophosphamide injection', 'Cyclophosphamide', 'Cytoxan', '600 mg/m2 IV every 21 days', 600, 'mg/m2', 'cycle', 'intravenous', 'completed', '1998-05-15', '1998-11-15', 'Breast cancer', 'prov_012', 'Adequate hydration. Monitor CBC.', 'AC regimen cycle 1-4 for breast cancer'),
('med_022', '1', '212754', 'Tamoxifen 20mg tablets', 'Tamoxifen citrate', 'Nolvadex', '20mg once daily', 20, 'mg', 'daily', 'oral', 'completed', '1998-12-01', '2003-12-01', 'Breast cancer adjuvant therapy', 'prov_012', 'Take at same time daily. Monitor for blood clots.', '5-year adjuvant hormone therapy'),

-- === HISTORICAL HORMONE REPLACEMENT ===
('med_023', '1', '197447', 'Estradiol/Progesterone patch', 'Estradiol/Norethindrone', 'CombiPatch', 'Apply twice weekly', NULL, 'mcg', 'twice weekly', 'transdermal', 'completed', '1992-01-01', '1998-01-01', 'Menopausal symptoms', 'prov_018', 'Rotate application sites. Monitor for breast changes.', 'HRT discontinued before cancer diagnosis'),

-- === HISTORICAL ANTIBIOTICS (Selected examples) ===
('med_024', '1', '723', 'Amoxicillin/Clavulanate 875mg', 'Amoxicillin/Clavulanic acid', 'Augmentin', '875mg twice daily', 875, 'mg', 'BID', 'oral', 'completed', '2019-12-05', '2019-12-15', 'Sinusitis', 'prov_003', 'Take with food to reduce GI upset.', '10-day course for acute bacterial sinusitis'),
('med_025', '1', '105078', 'Cephalexin 500mg capsules', 'Cephalexin', 'Keflex', '500mg four times daily', 500, 'mg', 'QID', 'oral', 'completed', '2021-06-10', '2021-06-17', 'Cellulitis', 'prov_014', 'Complete full course. Monitor for rash.', '7-day course for lower extremity cellulitis'),

-- === HISTORICAL PAIN MANAGEMENT ===
('med_026', '1', '1049502', 'Tramadol 50mg tablets', 'Tramadol hydrochloride', 'Ultram', '50mg every 6 hours as needed', 50, 'mg', 'PRN', 'oral', 'discontinued', '2018-11-15', '2019-02-15', 'Post-surgical pain', 'prov_006', 'May cause drowsiness. Risk of dependence.', 'Post hip replacement pain management'),
('med_027', '1', '1049221', 'Oxycodone 5mg tablets', 'Oxycodone hydrochloride', 'OxyContin', '5mg every 4-6 hours as needed', 5, 'mg', 'PRN', 'oral', 'completed', '2018-11-15', '2018-12-15', 'Severe post-operative pain', 'prov_006', 'Risk of dependence. Gradual taper.', 'Short-term use post hip replacement'),

-- === SUPPLEMENTS AND VITAMINS ===
('med_028', '1', '316069', 'Multivitamin tablets', 'Multivitamin/multimineral', 'Centrum Silver', 'One tablet daily', 1, 'tablet', 'daily', 'oral', 'active', '2010-01-01', NULL, 'Nutritional supplementation', 'prov_003', 'Take with food for better absorption.', 'Senior-specific vitamin formulation'),
('med_029', '1', '236', 'Vitamin B12 1000mcg tablets', 'Cyanocobalamin', 'Nature Made', '1000mcg once daily', 1000, 'mcg', 'daily', 'oral', 'active', '2015-05-20', NULL, 'B12 deficiency', 'prov_003', 'Sublingual absorption preferred.', 'Age-related B12 malabsorption'),
('med_030', '1', '8591', 'Omega-3 fatty acids 1000mg', 'Fish oil', 'Nature Made', '1000mg twice daily', 1000, 'mg', 'BID', 'oral', 'active', '2005-03-10', NULL, 'Cardiovascular health', 'prov_004', 'Take with meals to reduce fishy aftertaste.', 'Cardioprotective supplementation');

-- Continue with more encounters...
-- ===========================================
-- ENCOUNTERS (Comprehensive visit history - 60+ years)
-- ===========================================

INSERT INTO encounters (
    id, patient_id, status, encounter_type, priority, start_time, end_time,
    duration_minutes, location, department, primary_provider_id, chief_complaint,
    reason_code, reason_description, disposition, discharge_instructions, created_at
) VALUES

-- === 2024 ENCOUNTERS ===
('enc_001', '1', 'completed', 'outpatient', 'routine', '2024-12-01 09:00:00', '2024-12-01 10:00:00', 60, 'Springfield General Hospital', 'Geriatrics', 'prov_003', 'Annual wellness visit', 'Z00.00', 'Comprehensive geriatric assessment', 'discharged home', 'Continue current medications. Mammogram scheduled. Fall risk assessment completed.', '2024-12-01'),
('enc_002', '1', 'completed', 'outpatient', 'routine', '2024-09-15 14:30:00', '2024-09-15 15:15:00', 45, 'Springfield Endocrine Center', 'Endocrinology', 'prov_005', 'Diabetes follow-up', 'E11.9', 'Type 2 diabetes monitoring', 'discharged home', 'A1C improved to 7.2%. Continue metformin. Diabetic foot exam normal.', '2024-09-15'),
('enc_003', '1', 'completed', 'emergency', 'urgent', '2024-06-20 14:15:00', '2024-06-20 19:30:00', 315, 'Springfield General Hospital', 'Emergency', 'prov_009', 'Chest pain and shortness of breath', 'R06.02', 'Dyspnea and chest discomfort', 'discharged home', 'EKG normal. Chest X-ray clear. Stress test ordered. Follow up cardiology.', '2024-06-20'),
('enc_004', '1', 'completed', 'outpatient', 'routine', '2024-03-10 08:00:00', '2024-03-10 09:00:00', 60, 'Springfield Cardiology Associates', 'Cardiology', 'prov_004', 'Cardiac follow-up', 'I25.10', 'CAD monitoring and atrial fibrillation', 'discharged home', 'Stable CAD. Afib rate controlled. INR therapeutic. Continue warfarin.', '2024-03-10'),
('enc_005', '1', 'completed', 'outpatient', 'routine', '2024-01-25 10:30:00', '2024-01-25 11:00:00', 30, 'Springfield Laboratory', 'Laboratory', 'prov_003', 'Routine lab work', 'Z00.00', 'Annual laboratory screening', 'discharged home', 'Lab results will be reviewed at next appointment.', '2024-01-25'),

-- === 2023 ENCOUNTERS ===
('enc_006', '1', 'completed', 'outpatient', 'routine', '2023-11-28 11:00:00', '2023-11-28 12:00:00', 60, 'Springfield General Hospital', 'Primary Care', 'prov_003', 'Annual physical exam', 'Z00.00', 'Comprehensive physical examination', 'discharged home', 'Overall stable health. Mammogram and colonoscopy up to date. Hearing aid check recommended.', '2023-11-28'),
('enc_007', '1', 'completed', 'outpatient', 'urgent', '2023-08-10 13:45:00', '2023-08-10 14:45:00', 60, 'Springfield General Hospital', 'Primary Care', 'prov_003', 'Bilateral leg swelling', 'M79.3', 'Lower extremity edema', 'discharged home', 'Panniculitis diagnosed. Prednisone prescribed. Compression stockings. Follow up 2 weeks.', '2023-08-10'),
('enc_008', '1', 'completed', 'outpatient', 'routine', '2023-05-15 09:15:00', '2023-05-15 10:15:00', 60, 'Springfield Nephrology Center', 'Nephrology', 'prov_010', 'CKD monitoring', 'N18.3', 'Chronic kidney disease follow-up', 'discharged home', 'Stable kidney function. eGFR 52. Continue ACE inhibitor. Avoid nephrotoxic medications.', '2023-05-15'),
('enc_009', '1', 'completed', 'outpatient', 'urgent', '2023-03-15 14:00:00', '2023-03-15 15:30:00', 90, 'Springfield General Hospital', 'Primary Care', 'prov_003', 'Cough and fever', 'J18.9', 'Community-acquired pneumonia', 'discharged home', 'Pneumonia diagnosed. Antibiotics prescribed. Rest and increased fluids. Follow up if worsening.', '2023-03-15'),
('enc_010', '1', 'completed', 'outpatient', 'routine', '2023-02-15 08:30:00', '2023-02-15 09:30:00', 60, 'Springfield Cardiology Associates', 'Cardiology', 'prov_004', 'Irregular heartbeat', 'I48.91', 'New onset atrial fibrillation', 'discharged home', 'Atrial fibrillation confirmed. Anticoagulation started. Rate control initiated. Follow up 1 week.', '2023-02-15'),

-- === 2022 ENCOUNTERS ===
('enc_011', '1', 'completed', 'outpatient', 'routine', '2022-12-05 10:00:00', '2022-12-05 11:00:00', 60, 'Springfield General Hospital', 'Primary Care', 'prov_003', 'Annual wellness visit', 'Z00.00', 'Medicare annual wellness visit', 'discharged home', 'Cognitive assessment normal. Depression screening positive. Psychiatry referral made.', '2022-12-05'),
('enc_012', '1', 'completed', 'outpatient', 'urgent', '2022-11-10 15:30:00', '2022-11-10 16:15:00', 45, 'Springfield Pulmonary Associates', 'Pulmonology', 'prov_013', 'Persistent cough', 'J40', 'Acute bronchitis', 'discharged home', 'Acute bronchitis. Antibiotic prescribed. Inhaler for symptom relief. No smoking.', '2022-11-10'),
('enc_013', '1', 'completed', 'outpatient', 'routine', '2022-06-20 14:00:00', '2022-06-20 15:00:00', 60, 'Springfield Eye Center', 'Ophthalmology', 'prov_008', 'Vision changes', 'H35.30', 'Macular degeneration evaluation', 'discharged home', 'Early macular degeneration noted. AREDS vitamins recommended. Follow up 6 months.', '2022-06-20'),
('enc_014', '1', 'completed', 'outpatient', 'routine', '2022-04-20 09:00:00', '2022-04-20 10:30:00', 90, 'Springfield Neurology Center', 'Neurology', 'prov_011', 'Memory concerns', 'G93.1', 'Post-operative cognitive changes', 'discharged home', 'Mild cognitive changes post-anesthesia. Neuropsychological testing ordered. Reassess in 3 months.', '2022-04-20'),

-- === MAJOR HISTORICAL ENCOUNTERS ===

-- 2021
('enc_015', '1', 'completed', 'outpatient', 'urgent', '2021-08-10 11:00:00', '2021-08-10 12:00:00', 60, 'Springfield Dermatology', 'Dermatology', 'prov_014', 'Skin lesions on legs', 'M79.3', 'Panniculitis', 'discharged home', 'Panniculitis diagnosed. Topical steroids prescribed. Follow up 2 weeks if no improvement.', '2021-08-10'),
('enc_016', '1', 'completed', 'outpatient', 'routine', '2021-06-10 13:00:00', '2021-06-10 14:00:00', 60, 'Springfield Infectious Disease', 'Infectious Disease', 'prov_022', 'Cellulitis follow-up', 'L03.90', 'Cellulitis treatment monitoring', 'discharged home', 'Cellulitis responding to antibiotics. Complete 7-day course. Wound care instructions.', '2021-06-10'),

-- 2020 - COVID year
('enc_017', '1', 'completed', 'virtual', 'routine', '2020-11-05 10:00:00', '2020-11-05 10:30:00', 30, 'Springfield General Hospital', 'Ophthalmology', 'prov_008', 'Vision screening', 'H35.30', 'Macular degeneration monitoring', 'discharged home', 'Macular degeneration stable. Continue AREDS vitamins. Next visit 6 months.', '2020-11-05'),
('enc_018', '1', 'completed', 'virtual', 'urgent', '2020-03-25 14:00:00', '2020-03-25 14:45:00', 45, 'Springfield Mental Health', 'Psychiatry', 'prov_017', 'Depression and anxiety', 'F32.0', 'COVID-related depression', 'discharged home', 'Depression related to isolation. Sertraline started. Telehealth follow-up in 2 weeks.', '2020-03-25'),

-- 2019 - Osteoporosis diagnosis
('enc_019', '1', 'completed', 'outpatient', 'routine', '2019-03-10 08:00:00', '2019-03-10 09:00:00', 60, 'Springfield Bone Density Center', 'Radiology', 'prov_020', 'DEXA scan', 'M81.0', 'Osteoporosis screening', 'discharged home', 'DEXA shows osteoporosis T-score -2.8. Bisphosphonate therapy recommended. Calcium/Vitamin D.', '2019-03-10'),
('enc_020', '1', 'completed', 'outpatient', 'routine', '2019-09-20 15:00:00', '2019-09-20 15:30:00', 30, 'Springfield General Hospital', 'Primary Care', 'prov_003', 'Constipation', 'K59.00', 'Chronic constipation', 'discharged home', 'Chronic constipation. Increase fiber, fluids. Docusate prescribed. Follow up if worsening.', '2019-09-20'),

-- 2018 - Major hip fracture and surgery
('enc_021', '1', 'completed', 'emergency', 'emergency', '2018-11-15 16:30:00', '2018-11-15 20:00:00', 210, 'Springfield General Hospital', 'Emergency', 'prov_009', 'Fall with hip pain', 'S72.001A', 'Right femoral neck fracture', 'admitted', 'Emergency surgery required. Hip fracture confirmed on X-ray.', '2018-11-15'),
('enc_022', '1', 'completed', 'inpatient', 'emergency', '2018-11-15 20:00:00', '2018-11-25 10:00:00', 14400, 'Springfield General Hospital', 'Orthopedics', 'prov_006', 'Hip fracture surgery', 'S72.001A', 'Total hip arthroplasty', 'discharged home', 'Successful total hip replacement. Physical therapy. Weight bearing as tolerated.', '2018-11-15'),
('enc_023', '1', 'completed', 'outpatient', 'routine', '2018-12-10 14:00:00', '2018-12-10 15:00:00', 60, 'Springfield General Hospital', 'Orthopedics', 'prov_006', 'Post-operative follow-up', 'Z47.1', 'Post-surgical follow-up', 'discharged home', 'Excellent healing. Continue physical therapy. Next follow-up 6 weeks.', '2018-12-10'),

-- 2017
('enc_024', '1', 'completed', 'outpatient', 'routine', '2017-07-15 11:00:00', '2017-07-15 11:45:00', 45, 'Springfield Vascular Center', 'Vascular Surgery', 'prov_023', 'Varicose veins', 'I83.90', 'Varicose vein evaluation', 'discharged home', 'Mild varicose veins. Compression stockings recommended. Cosmetic treatment optional.', '2017-07-15'),

-- 2016 - Hearing loss
('enc_025', '1', 'completed', 'outpatient', 'routine', '2016-06-10 09:30:00', '2016-06-10 10:30:00', 60, 'Springfield Audiology Center', 'ENT', 'prov_011', 'Hearing difficulties', 'H91.90', 'Hearing assessment', 'discharged home', 'Moderate sensorineural hearing loss. Hearing aids recommended. Follow up 3 months.', '2016-06-10'),

-- 2015 - Cataract surgery
('enc_026', '1', 'completed', 'outpatient', 'routine', '2015-09-20 10:00:00', '2015-09-20 11:30:00', 90, 'Springfield Eye Center', 'Ophthalmology', 'prov_008', 'Vision changes', 'H25.9', 'Cataract evaluation', 'discharged home', 'Bilateral cataracts affecting vision. Surgery recommended for right eye first.', '2015-09-20'),
('enc_027', '1', 'completed', 'outpatient', 'routine', '2015-11-05 08:00:00', '2015-11-05 09:30:00', 90, 'Springfield Eye Surgery Center', 'Ophthalmology', 'prov_008', 'Cataract surgery', 'H25.9', 'Right cataract extraction', 'discharged home', 'Successful phacoemulsification with IOL. Eye shield overnight. Follow up tomorrow.', '2015-11-05'),
('enc_028', '1', 'completed', 'outpatient', 'routine', '2016-02-15 08:00:00', '2016-02-15 09:30:00', 90, 'Springfield Eye Surgery Center', 'Ophthalmology', 'prov_008', 'Cataract surgery', 'H25.9', 'Left cataract extraction', 'discharged home', 'Successful left cataract surgery. Bilateral vision now improved. Reading glasses needed.', '2016-02-15'),

-- 2012 - COPD exacerbation
('enc_029', '1', 'completed', 'emergency', 'urgent', '2012-01-20 20:15:00', '2012-01-20 23:30:00', 195, 'Springfield General Hospital', 'Emergency', 'prov_009', 'Shortness of breath', 'J44.1', 'COPD exacerbation', 'admitted', 'Severe dyspnea and cough. Chest X-ray shows hyperinflation. Admit for treatment.', '2012-01-20'),
('enc_030', '1', 'completed', 'inpatient', 'urgent', '2012-01-20 23:30:00', '2012-01-27 14:00:00', 9870, 'Springfield General Hospital', 'Pulmonology', 'prov_013', 'COPD exacerbation', 'J44.1', 'Chronic obstructive pulmonary disease', 'discharged home', 'COPD exacerbation treated with steroids and bronchodilators. Smoking cessation counseling. Oxygen at home.', '2012-01-20'),

-- 2010 - CKD diagnosis
('enc_031', '1', 'completed', 'outpatient', 'routine', '2010-02-14 13:00:00', '2010-02-14 14:00:00', 60, 'Springfield Nephrology Center', 'Nephrology', 'prov_010', 'Elevated creatinine', 'N18.3', 'Chronic kidney disease evaluation', 'discharged home', 'CKD stage 3 diagnosed. Likely diabetic nephropathy. ACE inhibitor started. Monitor closely.', '2010-02-14'),

-- 2008 - GERD diagnosis
('enc_032', '1', 'completed', 'outpatient', 'routine', '2008-09-05 11:00:00', '2008-09-05 12:00:00', 60, 'Springfield Gastroenterology', 'Gastroenterology', 'prov_007', 'Heartburn and acid reflux', 'K21.9', 'GERD evaluation', 'discharged home', 'GERD diagnosed. PPI therapy initiated. Dietary modifications. Upper endoscopy if no improvement.', '2008-09-05'),

-- 2005 - Gallbladder surgery
('enc_033', '1', 'completed', 'outpatient', 'urgent', '2005-07-22 14:30:00', '2005-07-22 16:00:00', 90, 'Springfield General Hospital', 'Emergency', 'prov_009', 'Severe abdominal pain', 'K80.20', 'Biliary colic', 'admitted', 'Right upper quadrant pain. Ultrasound shows gallstones. Surgery consultation.', '2005-07-22'),
('enc_034', '1', 'completed', 'inpatient', 'routine', '2005-08-15 07:00:00', '2005-08-17 11:00:00', 3060, 'Springfield General Hospital', 'Surgery', 'prov_023', 'Laparoscopic cholecystectomy', 'K80.20', 'Gallbladder removal', 'discharged home', 'Successful laparoscopic cholecystectomy. No complications. Low-fat diet. Follow up 2 weeks.', '2005-08-15'),

-- 2000 - Thyroid diagnosis
('enc_035', '1', 'completed', 'outpatient', 'routine', '2000-12-08 09:00:00', '2000-12-08 10:00:00', 60, 'Springfield Endocrine Center', 'Endocrinology', 'prov_005', 'Fatigue and weight gain', 'E03.9', 'Hypothyroidism evaluation', 'discharged home', 'TSH elevated at 8.5. Hypothyroidism diagnosed. Levothyroxine initiated. Recheck in 6 weeks.', '2000-12-08'),

-- === CANCER TREATMENT ENCOUNTERS (1998-2003) ===
('enc_036', '1', 'completed', 'outpatient', 'urgent', '1998-04-12 11:30:00', '1998-04-12 13:00:00', 90, 'Springfield General Hospital', 'Surgery', 'prov_023', 'Breast lump biopsy', 'C50.911', 'Breast mass evaluation', 'discharged home', 'Core biopsy shows invasive ductal carcinoma. Oncology referral urgent.', '1998-04-12'),
('enc_037', '1', 'completed', 'inpatient', 'routine', '1998-04-14 07:00:00', '1998-04-16 16:00:00', 4020, 'Springfield General Hospital', 'Surgery', 'prov_023', 'Lumpectomy', 'C50.911', 'Breast cancer surgery', 'discharged home', 'Successful lumpectomy with sentinel lymph node biopsy. Margins clear. Oncology follow-up.', '1998-04-14'),
('enc_038', '1', 'completed', 'outpatient', 'routine', '1998-05-15 10:00:00', '1998-05-15 14:00:00', 240, 'Springfield Cancer Center', 'Oncology', 'prov_012', 'Chemotherapy cycle 1', 'C50.911', 'Breast cancer treatment', 'discharged home', 'AC regimen cycle 1. Pre-medications given. Tolerated well. Next cycle in 3 weeks.', '1998-05-15'),
('enc_039', '1', 'completed', 'outpatient', 'routine', '1998-12-01 14:00:00', '1998-12-01 15:00:00', 60, 'Springfield Radiation Oncology', 'Radiation Oncology', 'prov_012', 'Radiation planning', 'C50.911', 'Radiation therapy planning', 'discharged home', 'Radiation simulation completed. Treatment to begin next week. 6 weeks total.', '1998-12-01'),

-- === EARLY MEDICAL HISTORY (1970s-1990s) ===

-- 1995 - CAD diagnosis
('enc_040', '1', 'completed', 'outpatient', 'urgent', '1995-03-10 10:00:00', '1995-03-10 12:00:00', 120, 'Springfield Cardiology Associates', 'Cardiology', 'prov_004', 'Chest pain', 'I25.10', 'Coronary artery disease evaluation', 'discharged home', 'Cardiac catheterization shows 2-vessel disease. Medical management recommended. Lifestyle changes.', '1995-03-10'),

-- 1992 - Menopause
('enc_041', '1', 'completed', 'outpatient', 'routine', '1992-01-15 14:00:00', '1992-01-15 15:00:00', 60, 'Springfield Women''s Health', 'Gynecology', 'prov_018', 'Irregular periods', 'N95.1', 'Menopausal transition', 'discharged home', 'Menopause confirmed. HRT discussed. Lifestyle modifications for symptoms.', '1992-01-15'),

-- 1988 - Arthritis diagnosis
('enc_042', '1', 'completed', 'outpatient', 'routine', '1988-09-12 11:00:00', '1988-09-12 12:00:00', 60, 'Springfield Orthopedics', 'Orthopedics', 'prov_006', 'Joint pain and stiffness', 'M15.9', 'Arthritis evaluation', 'discharged home', 'Osteoarthritis diagnosed. NSAIDs prescribed. Physical therapy recommended. Weight management.', '1988-09-12'),

-- 1985 - Diabetes diagnosis
('enc_043', '1', 'completed', 'outpatient', 'routine', '1985-06-15 14:00:00', '1985-06-15 15:00:00', 60, 'Springfield Family Medicine', 'Primary Care', 'prov_002', 'Routine physical', 'E11.9', 'Annual exam with diabetes diagnosis', 'discharged home', 'Diabetes diagnosed. Fasting glucose 165. Diabetes education scheduled. Diet modifications.', '1985-06-15'),

-- 1982 - Hypertension diagnosis
('enc_044', '1', 'completed', 'outpatient', 'routine', '1982-11-20 09:30:00', '1982-11-20 10:15:00', 45, 'Springfield Family Medicine', 'Primary Care', 'prov_002', 'High blood pressure', 'I10', 'Hypertension diagnosis', 'discharged home', 'Blood pressure consistently elevated. DASH diet education. Recheck in 3 months.', '1982-11-20'),

-- 1975 - Hip fracture from skiing
('enc_045', '1', 'completed', 'emergency', 'emergency', '1975-12-03 18:45:00', '1975-12-03 22:00:00', 195, 'Mountain View Hospital', 'Emergency', 'prov_009', 'Hip injury from fall', 'S72.002A', 'Left femoral neck fracture', 'admitted', 'Skiing accident. Left hip fracture confirmed. Surgery required.', '1975-12-03'),
('enc_046', '1', 'completed', 'inpatient', 'routine', '1975-12-04 08:00:00', '1975-12-10 14:00:00', 8760, 'Mountain View Hospital', 'Orthopedics', 'prov_006', 'Hip fracture repair', 'S72.002A', 'Internal fixation of hip fracture', 'discharged home', 'Successful ORIF of left hip fracture. Physical therapy. Weight bearing restrictions.', '1975-12-04'),

-- 1970 - Second childbirth
('enc_047', '1', 'completed', 'inpatient', 'routine', '1970-04-10 02:30:00', '1970-04-12 10:00:00', 3330, 'Springfield General Hospital', 'Obstetrics', 'prov_018', 'Labor and delivery', 'O80.1', 'Normal delivery', 'discharged home', 'Uncomplicated vaginal delivery. Healthy baby boy, 7 lbs 8 oz. Breastfeeding established.', '1970-04-10'),

-- 1967 - First childbirth
('enc_048', '1', 'completed', 'inpatient', 'routine', '1967-11-20 14:15:00', '1967-11-22 11:00:00', 2925, 'Springfield General Hospital', 'Obstetrics', 'prov_018', 'Labor and delivery', 'O80.1', 'Normal delivery', 'discharged home', 'First pregnancy. Uncomplicated delivery. Healthy baby girl, 6 lbs 12 oz. No complications.', '1967-11-20'),

-- 1965 - First UTI
('enc_049', '1', 'completed', 'outpatient', 'urgent', '1965-08-15 16:00:00', '1965-08-15 16:45:00', 45, 'Springfield Family Medicine', 'Primary Care', 'prov_002', 'Burning urination', 'N30.9', 'Urinary tract infection', 'discharged home', 'First documented UTI. Urine culture positive. Antibiotics prescribed. Hydration counseling.', '1965-08-15'),

-- 1962 - Establishment of care
('enc_050', '1', 'completed', 'outpatient', 'routine', '1962-05-20 10:00:00', '1962-05-20 11:00:00', 60, 'Springfield Family Medicine', 'Primary Care', 'prov_002', 'New patient visit', 'Z00.00', 'Establish primary care', 'discharged home', 'Healthy 18-year-old female. Comprehensive exam normal. Counseling on reproductive health.', '1962-05-20');

-- Continue in next part due to length limits...

-- ===========================================
-- ENCOUNTER DIAGNOSES (Link diagnoses to encounters)
-- ===========================================

INSERT INTO encounter_diagnoses (encounter_id, condition_id, rank, diagnosis_type) VALUES
-- Recent encounters
('enc_001', 'cond_005', 1, 'principal'), -- HTN
('enc_001', 'cond_006', 2, 'secondary'), -- DM
('enc_001', 'cond_010', 3, 'secondary'), -- CAD
('enc_002', 'cond_006', 1, 'principal'), -- DM follow-up
('enc_003', 'cond_030', 1, 'principal'), -- SOB
('enc_004', 'cond_010', 1, 'principal'), -- CAD
('enc_004', 'cond_025', 2, 'secondary'), -- Afib
('enc_006', 'cond_005', 1, 'principal'), -- Annual exam
('enc_007', 'cond_023', 1, 'principal'), -- Panniculitis
('enc_008', 'cond_016', 1, 'principal'), -- CKD
('enc_009', 'cond_017', 1, 'principal'), -- Pneumonia
('enc_010', 'cond_025', 1, 'principal'), -- New Afib

-- Historical major encounters
('enc_021', 'cond_021', 1, 'principal'), -- Hip fracture
('enc_033', 'cond_014', 1, 'principal'), -- Gallbladder
('enc_036', 'cond_011', 1, 'principal'), -- Breast cancer
('enc_040', 'cond_010', 1, 'principal'), -- CAD diagnosis
('enc_043', 'cond_006', 1, 'principal'), -- DM diagnosis
('enc_044', 'cond_005', 1, 'principal'), -- HTN diagnosis
('enc_045', 'cond_004', 1, 'principal'); -- First hip fracture

-- Continue with extensive observations in next section...

-- ===========================================
-- OBSERVATIONS (Comprehensive lab and vital signs - 60+ years)
-- ===========================================

INSERT INTO observations (
    id, patient_id, encounter_id, code, display_name, category, value_quantity,
    value_unit, reference_range_low, reference_range_high, interpretation,
    status, effective_datetime, performer_id, notes
) VALUES

-- === 2024 RECENT LABS ===
('obs_001', '1', 'enc_001', '33747-0', 'Hemoglobin A1c', 'laboratory', 7.2, '%', 4.0, 6.0, 'high', 'final', '2024-12-01 09:15:00', 'prov_003', 'Diabetes control improved from 7.8%'),
('obs_002', '1', 'enc_001', '2339-0', 'Glucose, fasting', 'laboratory', 145, 'mg/dL', 70, 99, 'high', 'final', '2024-12-01 09:15:00', 'prov_003', 'Fasting glucose elevated but improved'),
('obs_003', '1', 'enc_001', '2085-9', 'Cholesterol, HDL', 'laboratory', 48, 'mg/dL', 40, 100, 'normal', 'final', '2024-12-01 09:15:00', 'prov_003', 'HDL adequate for cardiac protection'),
('obs_004', '1', 'enc_001', '2089-1', 'Cholesterol, LDL', 'laboratory', 95, 'mg/dL', 0, 100, 'normal', 'final', '2024-12-01 09:15:00', 'prov_003', 'LDL at target with statin therapy'),
('obs_005', '1', 'enc_001', '2093-3', 'Cholesterol, total', 'laboratory', 175, 'mg/dL', 0, 200, 'normal', 'final', '2024-12-01 09:15:00', 'prov_003', 'Total cholesterol well controlled'),
('obs_006', '1', 'enc_001', '2571-8', 'Triglycerides', 'laboratory', 160, 'mg/dL', 0, 150, 'high', 'final', '2024-12-01 09:15:00', 'prov_003', 'Slightly elevated triglycerides'),

-- Kidney function tests
('obs_007', '1', 'enc_001', '2160-0', 'Creatinine', 'laboratory', 1.4, 'mg/dL', 0.6, 1.2, 'high', 'final', '2024-12-01 09:15:00', 'prov_003', 'Stable CKD stage 3'),
('obs_008', '1', 'enc_001', '33914-3', 'eGFR', 'laboratory', 52, 'mL/min/1.73m2', 60, 120, 'low', 'final', '2024-12-01 09:15:00', 'prov_003', 'Chronic kidney disease stage 3'),
('obs_009', '1', 'enc_001', '2888-6', 'Blood urea nitrogen', 'laboratory', 28, 'mg/dL', 8, 20, 'high', 'final', '2024-12-01 09:15:00', 'prov_003', 'Elevated BUN due to CKD'),
('obs_010', '1', 'enc_001', '2823-3', 'Potassium', 'laboratory', 4.2, 'mEq/L', 3.5, 5.0, 'normal', 'final', '2024-12-01 09:15:00', 'prov_003', 'Normal potassium with ACE inhibitor'),

-- Complete blood count
('obs_011', '1', 'enc_001', '718-7', 'Hemoglobin', 'laboratory', 12.8, 'g/dL', 12.0, 15.0, 'normal', 'final', '2024-12-01 09:15:00', 'prov_003', 'Normal hemoglobin'),
('obs_012', '1', 'enc_001', '4544-3', 'Hematocrit', 'laboratory', 38.5, '%', 36.0, 44.0, 'normal', 'final', '2024-12-01 09:15:00', 'prov_003', 'Normal hematocrit'),
('obs_013', '1', 'enc_001', '6690-2', 'White blood cell count', 'laboratory', 7.2, '10*3/uL', 4.0, 11.0, 'normal', 'final', '2024-12-01 09:15:00', 'prov_003', 'Normal WBC count'),
('obs_014', '1', 'enc_001', '777-3', 'Platelet count', 'laboratory', 285, '10*3/uL', 150, 400, 'normal', 'final', '2024-12-01 09:15:00', 'prov_003', 'Normal platelet count'),

-- Vital signs 2024
('obs_015', '1', 'enc_001', '8480-6', 'Systolic blood pressure', 'vital-signs', 142, 'mmHg', 90, 140, 'high', 'final', '2024-12-01 09:00:00', 'prov_003', 'Slightly elevated systolic BP'),
('obs_016', '1', 'enc_001', '8462-4', 'Diastolic blood pressure', 'vital-signs', 88, 'mmHg', 60, 90, 'normal', 'final', '2024-12-01 09:00:00', 'prov_003', 'Diastolic BP within range'),
('obs_017', '1', 'enc_001', '8867-4', 'Heart rate', 'vital-signs', 72, 'beats/min', 60, 100, 'normal', 'final', '2024-12-01 09:00:00', 'prov_003', 'Regular heart rate'),
('obs_018', '1', 'enc_001', '8310-5', 'Body temperature', 'vital-signs', 98.6, 'degF', 97.0, 99.5, 'normal', 'final', '2024-12-01 09:00:00', 'prov_003', 'Normal temperature'),
('obs_019', '1', 'enc_001', '29463-7', 'Body weight', 'vital-signs', 165, 'lb', NULL, NULL, 'normal', 'final', '2024-12-01 09:00:00', 'prov_003', 'Stable weight'),
('obs_020', '1', 'enc_001', '8302-2', 'Body height', 'vital-signs', 64, 'in', NULL, NULL, 'normal', 'final', '2024-12-01 09:00:00', 'prov_003', 'Height stable'),

-- Coagulation studies (for warfarin monitoring)
('obs_021', '1', 'enc_004', '6301-6', 'INR', 'laboratory', 2.3, 'ratio', 2.0, 3.0, 'normal', 'final', '2024-03-10 08:30:00', 'prov_004', 'Therapeutic INR for atrial fibrillation'),
('obs_022', '1', 'enc_004', '5902-2', 'Prothrombin time', 'laboratory', 25.8, 'seconds', 11.0, 13.0, 'high', 'final', '2024-03-10 08:30:00', 'prov_004', 'Prolonged PT on warfarin therapy'),

-- === 2023 OBSERVATIONS ===
('obs_023', '1', 'enc_002', '33747-0', 'Hemoglobin A1c', 'laboratory', 7.8, '%', 4.0, 6.0, 'high', 'final', '2024-09-15 10:45:00', 'prov_005', 'A1C elevated, medication adjustment needed'),
('obs_024', '1', 'enc_008', '2160-0', 'Creatinine', 'laboratory', 1.3, 'mg/dL', 0.6, 1.2, 'high', 'final', '2023-05-15 09:30:00', 'prov_010', 'CKD progression stable'),
('obs_025', '1', 'enc_008', '33914-3', 'eGFR', 'laboratory', 55, 'mL/min/1.73m2', 60, 120, 'low', 'final', '2023-05-15 09:30:00', 'prov_010', 'Stage 3 CKD unchanged'),

-- Thyroid function
('obs_026', '1', 'enc_006', '3016-3', 'TSH', 'laboratory', 2.8, 'mIU/L', 0.4, 4.0, 'normal', 'final', '2023-11-28 11:15:00', 'prov_003', 'Thyroid function stable on levothyroxine'),
('obs_027', '1', 'enc_006', '3024-7', 'Free T4', 'laboratory', 1.2, 'ng/dL', 0.8, 1.8, 'normal', 'final', '2023-11-28 11:15:00', 'prov_003', 'Normal free T4 level'),

-- Emergency visit cardiac markers
('obs_028', '1', 'enc_003', '10839-9', 'Troponin I', 'laboratory', 0.02, 'ng/mL', 0.0, 0.04, 'normal', 'final', '2024-06-20 15:30:00', 'prov_009', 'No evidence of myocardial infarction'),
('obs_029', '1', 'enc_003', '2154-3', 'Creatine kinase MB', 'laboratory', 3.2, 'ng/mL', 0.0, 6.0, 'normal', 'final', '2024-06-20 15:30:00', 'prov_009', 'Normal cardiac enzymes'),
('obs_030', '1', 'enc_003', '33762-9', 'NT-proBNP', 'laboratory', 285, 'pg/mL', 0, 125, 'high', 'final', '2024-06-20 15:30:00', 'prov_009', 'Mildly elevated BNP, age-related'),

-- Pneumonia workup
('obs_031', '1', 'enc_009', '1988-5', 'C-reactive protein', 'laboratory', 45, 'mg/L', 0.0, 3.0, 'high', 'final', '2023-03-15 14:30:00', 'prov_003', 'Elevated CRP indicating inflammation'),
('obs_032', '1', 'enc_009', '26464-8', 'White blood cell count', 'laboratory', 13.5, '10*3/uL', 4.0, 11.0, 'high', 'final', '2023-03-15 14:30:00', 'prov_003', 'Elevated WBC consistent with pneumonia'),
('obs_033', '1', 'enc_009', '5905-5', 'Procalcitonin', 'laboratory', 0.85, 'ng/mL', 0.0, 0.25, 'high', 'final', '2023-03-15 14:30:00', 'prov_003', 'Elevated procalcitonin suggests bacterial infection'),

-- === HISTORICAL TRENDING LABS ===

-- 2022 labs
('obs_034', '1', 'enc_011', '33747-0', 'Hemoglobin A1c', 'laboratory', 8.1, '%', 4.0, 6.0, 'high', 'final', '2022-12-05 10:30:00', 'prov_003', 'Diabetes control worsening'),
('obs_035', '1', 'enc_011', '2160-0', 'Creatinine', 'laboratory', 1.2, 'mg/dL', 0.6, 1.2, 'normal', 'final', '2022-12-05 10:30:00', 'prov_003', 'Borderline kidney function'),

-- 2021 labs
('obs_036', '1', NULL, '33747-0', 'Hemoglobin A1c', 'laboratory', 7.9, '%', 4.0, 6.0, 'high', 'final', '2021-06-15 09:00:00', 'prov_003', 'Diabetes suboptimal control'),
('obs_037', '1', NULL, '2160-0', 'Creatinine', 'laboratory', 1.1, 'mg/dL', 0.6, 1.2, 'normal', 'final', '2021-06-15 09:00:00', 'prov_003', 'Kidney function stable'),

-- 2020 labs
('obs_038', '1', NULL, '33747-0', 'Hemoglobin A1c', 'laboratory', 7.5, '%', 4.0, 6.0, 'high', 'final', '2020-09-20 10:00:00', 'prov_003', 'Diabetes moderately controlled'),
('obs_039', '1', NULL, '2093-3', 'Cholesterol, total', 'laboratory', 185, 'mg/dL', 0, 200, 'normal', 'final', '2020-09-20 10:00:00', 'prov_003', 'Cholesterol well controlled'),

-- 2019 - Osteoporosis workup
('obs_040', '1', 'enc_019', '25-006', 'DEXA T-score lumbar spine', 'diagnostic', -2.8, 'T-score', -1.0, 1.0, 'low', 'final', '2019-03-10 08:30:00', 'prov_020', 'Osteoporosis diagnosed'),
('obs_041', '1', 'enc_019', '25-007', 'DEXA T-score hip', 'diagnostic', -2.5, 'T-score', -1.0, 1.0, 'low', 'final', '2019-03-10 08:30:00', 'prov_020', 'Osteoporosis at hip'),
('obs_042', '1', 'enc_019', '1937-7', 'Vitamin D 25-hydroxy', 'laboratory', 18, 'ng/mL', 30, 100, 'low', 'final', '2019-03-10 09:00:00', 'prov_020', 'Vitamin D deficiency'),

-- 2018 - Post-surgical labs
('obs_043', '1', 'enc_022', '718-7', 'Hemoglobin', 'laboratory', 10.2, 'g/dL', 12.0, 15.0, 'low', 'final', '2018-11-16 06:00:00', 'prov_006', 'Post-operative anemia'),
('obs_044', '1', 'enc_022', '4544-3', 'Hematocrit', 'laboratory', 30.5, '%', 36.0, 44.0, 'low', 'final', '2018-11-16 06:00:00', 'prov_006', 'Low hematocrit post-surgery'),

-- 2015 - Multiple historical values
('obs_045', '1', NULL, '33747-0', 'Hemoglobin A1c', 'laboratory', 7.2, '%', 4.0, 6.0, 'high', 'final', '2015-04-20 10:00:00', 'prov_003', 'Diabetes well controlled'),
('obs_046', '1', NULL, '2160-0', 'Creatinine', 'laboratory', 1.0, 'mg/dL', 0.6, 1.2, 'normal', 'final', '2015-04-20 10:00:00', 'prov_003', 'Normal kidney function'),

-- 2010 - CKD diagnosis workup
('obs_047', '1', 'enc_031', '2160-0', 'Creatinine', 'laboratory', 1.3, 'mg/dL', 0.6, 1.2, 'high', 'final', '2010-02-14 13:30:00', 'prov_010', 'First elevated creatinine'),
('obs_048', '1', 'enc_031', '33914-3', 'eGFR', 'laboratory', 48, 'mL/min/1.73m2', 60, 120, 'low', 'final', '2010-02-14 13:30:00', 'prov_010', 'CKD stage 3a'),
('obs_049', '1', 'enc_031', '2889-4', 'Protein total urine', 'laboratory', 150, 'mg/24hr', 0, 150, 'normal', 'final', '2010-02-14 13:30:00', 'prov_010', 'Mild proteinuria'),

-- 2005 - Pre-operative labs
('obs_050', '1', 'enc_033', '718-7', 'Hemoglobin', 'laboratory', 13.5, 'g/dL', 12.0, 15.0, 'normal', 'final', '2005-07-22 15:00:00', 'prov_009', 'Normal pre-op hemoglobin'),
('obs_051', '1', 'enc_033', '6301-6', 'INR', 'laboratory', 1.0, 'ratio', 0.8, 1.2, 'normal', 'final', '2005-07-22 15:00:00', 'prov_009', 'Normal coagulation'),

-- 2000 - Thyroid diagnosis
('obs_052', '1', 'enc_035', '3016-3', 'TSH', 'laboratory', 8.5, 'mIU/L', 0.4, 4.0, 'high', 'final', '2000-12-08 09:30:00', 'prov_005', 'Elevated TSH, hypothyroidism'),
('obs_053', '1', 'enc_035', '3024-7', 'Free T4', 'laboratory', 0.6, 'ng/dL', 0.8, 1.8, 'low', 'final', '2000-12-08 09:30:00', 'prov_005', 'Low free T4'),

-- 1998 - Cancer treatment monitoring
('obs_054', '1', 'enc_038', '718-7', 'Hemoglobin', 'laboratory', 9.8, 'g/dL', 12.0, 15.0, 'low', 'final', '1998-06-15 10:00:00', 'prov_012', 'Chemotherapy-induced anemia'),
('obs_055', '1', 'enc_038', '6690-2', 'White blood cell count', 'laboratory', 3.2, '10*3/uL', 4.0, 11.0, 'low', 'final', '1998-06-15 10:00:00', 'prov_012', 'Chemotherapy-induced leukopenia'),
('obs_056', '1', 'enc_038', '777-3', 'Platelet count', 'laboratory', 125, '10*3/uL', 150, 400, 'low', 'final', '1998-06-15 10:00:00', 'prov_012', 'Mild thrombocytopenia'),

-- 1995 - CAD diagnosis
('obs_057', '1', 'enc_040', '2093-3', 'Cholesterol, total', 'laboratory', 285, 'mg/dL', 0, 200, 'high', 'final', '1995-03-10 11:00:00', 'prov_004', 'Hyperlipidemia diagnosed'),
('obs_058', '1', 'enc_040', '2089-1', 'Cholesterol, LDL', 'laboratory', 190, 'mg/dL', 0, 100, 'high', 'final', '1995-03-10 11:00:00', 'prov_004', 'Elevated LDL cholesterol'),

-- 1985 - Diabetes diagnosis
('obs_059', '1', 'enc_043', '2339-0', 'Glucose, fasting', 'laboratory', 165, 'mg/dL', 70, 99, 'high', 'final', '1985-06-15 14:30:00', 'prov_002', 'Diabetes diagnosed'),
('obs_060', '1', 'enc_043', '33747-0', 'Hemoglobin A1c', 'laboratory', 9.2, '%', 4.0, 6.0, 'high', 'final', '1985-06-15 14:30:00', 'prov_002', 'Initial A1C at diagnosis'),

-- Continue with extensive vital signs over decades...
-- Additional vital signs from various encounters
('obs_061', '1', 'enc_044', '8480-6', 'Systolic blood pressure', 'vital-signs', 158, 'mmHg', 90, 140, 'high', 'final', '1982-11-20 09:30:00', 'prov_002', 'Initial hypertension diagnosis'),
('obs_062', '1', 'enc_044', '8462-4', 'Diastolic blood pressure', 'vital-signs', 95, 'mmHg', 60, 90, 'high', 'final', '1982-11-20 09:30:00', 'prov_002', 'Elevated diastolic BP'),

-- Weight progression over decades
('obs_063', '1', 'enc_043', '29463-7', 'Body weight', 'vital-signs', 155, 'lb', NULL, NULL, 'normal', 'final', '1985-06-15 14:00:00', 'prov_002', 'Weight at diabetes diagnosis'),
('obs_064', '1', 'enc_040', '29463-7', 'Body weight', 'vital-signs', 165, 'lb', NULL, NULL, 'normal', 'final', '1995-03-10 10:00:00', 'prov_004', 'Weight gain over decade'),
('obs_065', '1', 'enc_035', '29463-7', 'Body weight', 'vital-signs', 170, 'lb', NULL, NULL, 'normal', 'final', '2000-12-08 09:00:00', 'prov_005', 'Peak weight'),
('obs_066', '1', 'enc_019', '29463-7', 'Body weight', 'vital-signs', 168, 'lb', NULL, NULL, 'normal', 'final', '2019-03-10 08:00:00', 'prov_020', 'Weight stable in recent years'),
('obs_067', '1', 'enc_022', '29463-7', 'Body weight', 'vital-signs', 162, 'lb', NULL, NULL, 'normal', 'final', '2018-11-25 10:00:00', 'prov_006', 'Weight loss post-surgery'),

-- Additional specialized lab tests
('obs_068', '1', NULL, '14957-5', 'Microalbumin/Creatinine ratio', 'laboratory', 45, 'mg/g', 0, 30, 'high', 'final', '2020-08-15 09:00:00', 'prov_010', 'Early diabetic nephropathy'),
('obs_069', '1', NULL, '13457-7', 'Cholesterol LDL/HDL ratio', 'laboratory', 2.0, 'ratio', 0, 3.5, 'normal', 'final', '2022-05-20 10:00:00', 'prov_004', 'Good cholesterol ratio'),
('obs_070', '1', NULL, '42719-5', 'Hemoglobin A1c estimated average glucose', 'laboratory', 158, 'mg/dL', 0, 126, 'high', 'final', '2024-12-01 09:15:00', 'prov_003', 'eAG corresponds to A1C of 7.2%');

-- ===========================================
-- PROCEDURES (Comprehensive procedure history)
-- ===========================================

INSERT INTO procedures (
    id, patient_id, encounter_id, code, display_name, category, status,
    performed_datetime, body_site, outcome, primary_performer_id, location,
    indication, notes
) VALUES

-- === 2024 PROCEDURES ===
('proc_001', '1', 'enc_001', '93000', 'Electrocardiogram', 'diagnostic', 'completed', '2024-12-01 09:30:00', 'chest', 'normal sinus rhythm with controlled afib', 'prov_003', 'Springfield General Hospital', 'Routine cardiac monitoring', 'EKG shows controlled atrial fibrillation on warfarin'),
('proc_002', '1', 'enc_003', '36415', 'Collection of venous blood by venipuncture', 'diagnostic', 'completed', '2024-06-20 15:00:00', 'antecubital vein', 'successful collection', 'prov_009', 'Springfield General Hospital', 'Cardiac enzyme testing', 'Blood drawn for troponin and cardiac markers'),
('proc_003', '1', 'enc_003', '71010', 'Chest X-ray, single view', 'diagnostic', 'completed', '2024-06-20 15:15:00', 'chest', 'no acute cardiopulmonary disease', 'prov_020', 'Springfield General Hospital', 'Chest pain evaluation', 'No infiltrates, effusions, or cardiomegaly'),
('proc_004', '1', 'enc_004', '93015', 'Cardiovascular stress test', 'diagnostic', 'completed', '2024-03-10 08:15:00', 'heart', 'negative for inducible ischemia', 'prov_004', 'Springfield Cardiology Associates', 'CAD monitoring', 'Exercise stress test negative, good functional capacity'),

-- === 2023 PROCEDURES ===
('proc_005', '1', 'enc_006', '77057', 'Screening mammography, bilateral', 'diagnostic', 'completed', '2023-11-28 12:00:00', 'bilateral breasts', 'no evidence of malignancy', 'prov_020', 'Springfield Imaging Center', 'Breast cancer surveillance', 'Annual surveillance mammogram normal, BI-RADS 1'),
('proc_006', '1', 'enc_006', '45378', 'Colonoscopy, flexible', 'diagnostic', 'completed', '2023-10-15 09:00:00', 'colon', 'normal colonoscopy', 'prov_007', 'Springfield Endoscopy Center', 'Colon cancer screening', 'No polyps or abnormalities. Next screening in 10 years'),
('proc_007', '1', 'enc_009', '71020', 'Chest X-ray, 2 views', 'diagnostic', 'completed', '2023-03-15 14:30:00', 'chest', 'right lower lobe pneumonia', 'prov_020', 'Springfield General Hospital', 'Pneumonia evaluation', 'Right lower lobe consolidation consistent with pneumonia'),

-- === 2022 PROCEDURES ===
('proc_008', '1', 'enc_013', '92250', 'Fundus photography', 'diagnostic', 'completed', '2022-06-20 14:30:00', 'bilateral retinas', 'early macular degeneration changes', 'prov_008', 'Springfield Eye Center', 'Macular degeneration monitoring', 'Drusen and pigmentary changes noted in both maculae'),
('proc_009', '1', 'enc_014', '96116', 'Neuropsychological testing', 'diagnostic', 'completed', '2022-05-10 10:00:00', 'brain', 'mild cognitive changes', 'prov_011', 'Springfield Neurology Center', 'Cognitive assessment', 'MMSE 26/30, mild short-term memory deficits'),

-- === MAJOR HISTORICAL PROCEDURES ===

-- 2021
('proc_010', '1', 'enc_015', '11042', 'Debridement of skin and subcutaneous tissue', 'therapeutic', 'completed', '2021-08-10 11:30:00', 'bilateral lower legs', 'successful debridement', 'prov_014', 'Springfield Dermatology', 'Panniculitis treatment', 'Inflammatory tissue debrided, wound care initiated'),

-- 2019 - Osteoporosis workup
('proc_011', '1', 'enc_019', '77080', 'Dual-energy X-ray absorptiometry (DEXA)', 'diagnostic', 'completed', '2019-03-10 08:30:00', 'lumbar spine and hip', 'osteoporosis confirmed', 'prov_020', 'Springfield Bone Density Center', 'Bone density assessment', 'T-scores: L1-L4 -2.8, Total hip -2.5, confirming osteoporosis'),

-- 2018 - Major hip surgery
('proc_012', '1', 'enc_022', '27130', 'Total hip arthroplasty', 'surgical', 'completed', '2018-11-16 08:00:00', 'right hip', 'successful hip replacement', 'prov_006', 'Springfield General Hospital', 'Right femoral neck fracture', 'Cementless total hip replacement, 32mm ceramic head on highly cross-linked polyethylene'),
('proc_013', '1', 'enc_022', '00630', 'Anesthesia for procedures on hip joint', 'supportive', 'completed', '2018-11-16 07:30:00', 'right hip', 'successful anesthesia', 'prov_019', 'Springfield General Hospital', 'Surgical anesthesia', 'General endotracheal anesthesia with arterial line monitoring'),

-- 2016 - Bilateral cataract surgery
('proc_014', '1', 'enc_027', '66984', 'Extracapsular cataract removal with insertion of intraocular lens prosthesis', 'surgical', 'completed', '2015-11-05 08:30:00', 'right eye', 'successful cataract extraction', 'prov_008', 'Springfield Eye Surgery Center', 'Right cataract', 'Phacoemulsification with monofocal IOL implantation'),
('proc_015', '1', 'enc_028', '66984', 'Extracapsular cataract removal with insertion of intraocular lens prosthesis', 'surgical', 'completed', '2016-02-15 08:30:00', 'left eye', 'successful cataract extraction', 'prov_008', 'Springfield Eye Surgery Center', 'Left cataract', 'Phacoemulsification with monofocal IOL implantation'),

-- 2012 - COPD exacerbation
('proc_016', '1', 'enc_029', '94060', 'Bronchodilator responsiveness', 'diagnostic', 'completed', '2012-01-21 09:00:00', 'lungs', 'positive response to bronchodilators', 'prov_013', 'Springfield General Hospital', 'COPD assessment', 'Spirometry showed obstructive pattern with bronchodilator response'),
('proc_017', '1', 'enc_030', '94002', 'Ventilation assist and management', 'therapeutic', 'completed', '2012-01-21 06:00:00', 'respiratory system', 'successful ventilatory support', 'prov_013', 'Springfield General Hospital', 'COPD exacerbation', 'BiPAP ventilation support during acute exacerbation'),

-- 2008 - GERD workup
('proc_018', '1', 'enc_032', '43235', 'Esophagogastroduodenoscopy', 'diagnostic', 'completed', '2008-12-10 10:00:00', 'upper GI tract', 'mild esophagitis', 'prov_007', 'Springfield Endoscopy Center', 'GERD evaluation', 'Grade A esophagitis, no Barrett''s esophagus, H. pylori negative'),

-- 2005 - Gallbladder surgery
('proc_019', '1', 'enc_034', '47562', 'Laparoscopic cholecystectomy', 'surgical', 'completed', '2005-08-15 09:30:00', 'gallbladder', 'successful removal', 'prov_023', 'Springfield General Hospital', 'Symptomatic cholelithiasis', 'Four-port laparoscopic technique, no conversion to open'),
('proc_020', '1', 'enc_034', '47563', 'Laparoscopic cholecystectomy with cholangiography', 'surgical', 'completed', '2005-08-15 10:00:00', 'bile ducts', 'normal cholangiogram', 'prov_023', 'Springfield General Hospital', 'Intraoperative bile duct assessment', 'Intraoperative cholangiogram normal, no retained stones'),

-- 2000s - Cancer screening
('proc_021', '1', NULL, '77057', 'Screening mammography, bilateral', 'diagnostic', 'completed', '2004-11-15 10:00:00', 'bilateral breasts', 'no abnormalities', 'prov_020', 'Springfield Imaging Center', 'Post-cancer surveillance', 'First surveillance mammogram post-treatment, normal'),
('proc_022', '1', NULL, '88305', 'Surgical pathology examination', 'diagnostic', 'completed', '2003-04-12 14:00:00', 'breast tissue', 'no residual malignancy', 'prov_025', 'Springfield Pathology', 'Post-treatment assessment', '5-year post-treatment biopsy negative for recurrence'),

-- === CANCER TREATMENT PROCEDURES (1998-1999) ===
('proc_023', '1', 'enc_037', '19120', 'Excision of lesion of breast', 'surgical', 'completed', '1998-04-14 10:00:00', 'right breast', 'complete excision with clear margins', 'prov_023', 'Springfield General Hospital', 'Breast cancer', 'Lumpectomy with sentinel lymph node biopsy, margins clear'),
('proc_024', '1', 'enc_037', '38500', 'Biopsy or excision of lymph node', 'surgical', 'completed', '1998-04-14 10:30:00', 'right axilla', 'sentinel node negative', 'prov_023', 'Springfield General Hospital', 'Breast cancer staging', 'Sentinel lymph node biopsy negative for metastases'),
('proc_025', '1', 'enc_038', '96413', 'Chemotherapy administration, intravenous infusion technique', 'therapeutic', 'completed', '1998-05-15 10:00:00', 'central venous access', 'successful administration', 'prov_012', 'Springfield Cancer Center', 'Breast cancer treatment', 'AC regimen cycle 1: Adriamycin 60mg/m2 + Cytoxan 600mg/m2'),
('proc_026', '1', NULL, '96413', 'Chemotherapy administration, intravenous infusion technique', 'therapeutic', 'completed', '1998-06-05 10:00:00', 'central venous access', 'successful administration', 'prov_012', 'Springfield Cancer Center', 'Breast cancer treatment', 'AC regimen cycle 2'),
('proc_027', '1', NULL, '96413', 'Chemotherapy administration, intravenous infusion technique', 'therapeutic', 'completed', '1998-06-26 10:00:00', 'central venous access', 'successful administration', 'prov_012', 'Springfield Cancer Center', 'Breast cancer treatment', 'AC regimen cycle 3'),
('proc_028', '1', NULL, '96413', 'Chemotherapy administration, intravenous infusion technique', 'therapeutic', 'completed', '1998-07-17 10:00:00', 'central venous access', 'successful administration', 'prov_012', 'Springfield Cancer Center', 'Breast cancer treatment', 'AC regimen cycle 4, final cycle'),
('proc_029', '1', 'enc_039', '77301', 'Intensity modulated radiation therapy planning', 'therapeutic', 'completed', '1998-12-01 14:00:00', 'right breast', 'treatment planning complete', 'prov_012', 'Springfield Radiation Oncology', 'Post-lumpectomy radiation', 'IMRT planning for whole breast radiation with boost to tumor bed'),
('proc_030', '1', NULL, '77418', 'Intensity modulated radiation therapy delivery', 'therapeutic', 'completed', '1998-12-15 09:00:00', 'right breast', 'successful radiation delivery', 'prov_012', 'Springfield Radiation Oncology', 'Breast cancer adjuvant therapy', 'Daily IMRT treatments, 50.4 Gy in 28 fractions plus boost'),

-- === OBSTETRIC PROCEDURES (1967, 1970) ===
('proc_031', '1', 'enc_048', '59400', 'Routine obstetric care including antepartum care, vaginal delivery and postpartum care', 'obstetric', 'completed', '1967-11-20 14:15:00', 'uterus', 'successful vaginal delivery', 'prov_018', 'Springfield General Hospital', 'First pregnancy', 'Uncomplicated spontaneous vaginal delivery, healthy female infant'),
('proc_032', '1', 'enc_047', '59400', 'Routine obstetric care including antepartum care, vaginal delivery and postpartum care', 'obstetric', 'completed', '1970-04-10 02:30:00', 'uterus', 'successful vaginal delivery', 'prov_018', 'Springfield General Hospital', 'Second pregnancy', 'Uncomplicated spontaneous vaginal delivery, healthy male infant'),

-- === ORTHOPEDIC PROCEDURES ===
-- 1975 - First hip fracture
('proc_033', '1', 'enc_046', '27244', 'Treatment of intertrochanteric, peritrochanteric, or subtrochanteric femoral fracture', 'surgical', 'completed', '1975-12-04 10:00:00', 'left hip', 'successful internal fixation', 'prov_006', 'Mountain View Hospital', 'Left femoral neck fracture', 'Open reduction internal fixation with dynamic hip screw'),

-- === DIAGNOSTIC PROCEDURES OVER DECADES ===
('proc_034', '1', NULL, '93307', 'Echocardiography, transthoracic', 'diagnostic', 'completed', '2023-02-20 14:00:00', 'heart', 'mild mitral regurgitation', 'prov_004', 'Springfield Cardiology Associates', 'Atrial fibrillation evaluation', 'Echo shows preserved EF 60%, mild MR, normal LA size'),
('proc_035', '1', NULL, '94010', 'Spirometry', 'diagnostic', 'completed', '2020-01-15 11:00:00', 'lungs', 'mild restrictive pattern', 'prov_013', 'Springfield Pulmonary Associates', 'Annual pulmonary function', 'FEV1/FVC 0.78, mild restriction, no significant change'),
('proc_036', '1', NULL, '78452', 'Myocardial perfusion imaging', 'diagnostic', 'completed', '2019-08-20 09:00:00', 'heart', 'normal perfusion', 'prov_004', 'Springfield Nuclear Medicine', 'CAD monitoring', 'Normal myocardial perfusion, no inducible ischemia'),
('proc_037', '1', NULL, '93017', 'Cardiovascular stress test with interpretation and report', 'diagnostic', 'completed', '2018-05-15 08:00:00', 'heart', 'negative stress test', 'prov_004', 'Springfield Cardiology Associates', 'Cardiac risk assessment', 'Negative exercise stress test, 85% maximum predicted heart rate achieved'),
('proc_038', '1', NULL, '71260', 'Computed tomography, chest', 'diagnostic', 'completed', '2016-09-10 15:00:00', 'chest', 'no pulmonary nodules', 'prov_013', 'Springfield Radiology', 'Lung cancer screening', 'Low-dose CT chest negative for pulmonary nodules'),
('proc_039', '1', NULL, '72148', 'Magnetic resonance imaging, lumbar spine', 'diagnostic', 'completed', '2014-07-20 16:00:00', 'lumbar spine', 'degenerative changes', 'prov_006', 'Springfield Radiology', 'Back pain evaluation', 'MRI shows multilevel degenerative disc disease, no nerve compression'),
('proc_040', '1', NULL, '76700', 'Abdominal ultrasound', 'diagnostic', 'completed', '2012-03-15 10:00:00', 'abdomen', 'normal abdominal organs', 'prov_003', 'Springfield Radiology', 'Abdominal pain workup', 'Normal liver, gallbladder (s/p cholecystectomy), kidneys, and pancreas');

-- ===========================================
-- CLINICAL NOTES (Comprehensive documentation)
-- ===========================================

INSERT INTO clinical_notes (
    id, patient_id, encounter_id, note_type, specialty, title, content, status,
    author_id, dictated_datetime, authenticated_datetime, processed_by_ai
) VALUES

-- === RECENT CLINICAL NOTES ===
('note_001', '1', 'enc_001', 'progress', 'Geriatrics', 'Annual Comprehensive Geriatric Assessment', 
'CHIEF COMPLAINT: Annual wellness visit and comprehensive geriatric assessment.

HISTORY OF PRESENT ILLNESS: 
Eleanor Henderson is an 80-year-old female with an extensive past medical history spanning over 60 years, including type 2 diabetes mellitus (diagnosed 1985), essential hypertension (diagnosed 1982), coronary artery disease with 2-vessel disease (diagnosed 1995), chronic kidney disease stage 3 (diagnosed 2010), atrial fibrillation (diagnosed 2023), osteoporosis (diagnosed 2019), and remote history of successfully treated breast cancer (1998-2003). She also has a history of bilateral hip fractures, the most recent in 2018 requiring total hip replacement.

The patient reports feeling generally well with good functional status. She continues to live independently in her own home with support from her daughter who lives nearby. She ambulates without assistive devices since her hip replacement recovery. Her diabetes management has improved with recent A1C of 7.2% down from 7.8% six months ago. She has been compliant with home glucose monitoring with recent readings averaging 140-160 mg/dL fasting and 180-200 mg/dL post-prandial.

Her atrial fibrillation was diagnosed 10 months ago and has been well controlled on warfarin with regular INR monitoring. She denies chest pain, palpitations, or dyspnea on exertion. Her last INR was therapeutic at 2.3. She has had no bleeding complications.

She continues her cardiac medications including lisinopril and atorvastatin with good tolerance. Her chronic kidney disease has remained stable with latest creatinine 1.4 mg/dL and eGFR 52 mL/min/1.73m2. She follows a renal diet and avoids nephrotoxic medications.

Her osteoporosis is managed with weekly alendronate, calcium, and vitamin D supplementation. She has had no fractures since her hip replacement and performs weight-bearing exercises regularly.

PAST MEDICAL HISTORY:
1. Type 2 diabetes mellitus (1985) - currently on metformin, A1C 7.2%
2. Essential hypertension (1982) - on lisinopril, well controlled
3. Coronary artery disease, 2-vessel (1995) - on optimal medical therapy
4. Chronic kidney disease, stage 3 (2010) - stable function
5. Atrial fibrillation (2023) - on warfarin anticoagulation
6. Osteoporosis (2019) - on bisphosphonate therapy
7. History of breast cancer, Stage IIA (1998-2003) - successfully treated with lumpectomy, AC chemotherapy, radiation, and 5 years of tamoxifen
8. Bilateral hip fractures: Left hip (1975, skiing accident) and right hip (2018, fall at home with total hip replacement)
9. History of cholecystectomy (2005)
10. History of COPD exacerbation (2012) - former smoker, 40 pack-years, quit 12 years ago
11. Hypothyroidism (2000) - stable on levothyroxine
12. GERD (2008) - controlled on omeprazole
13. Polyosteoarthritis (1988) - managed with acetaminophen
14. Bilateral cataracts (2015-2016) - surgically corrected
15. Age-related hearing loss (2016) - uses hearing aids
16. Age-related macular degeneration (2020) - stable, early changes

PAST SURGICAL HISTORY:
1. Right total hip arthroplasty (2018)
2. Bilateral cataract extraction with IOL implantation (2015, 2016)
3. Laparoscopic cholecystectomy (2005)
4. Right breast lumpectomy with sentinel lymph node biopsy (1998)
5. Left hip fracture repair with internal fixation (1975)
6. Two uncomplicated vaginal deliveries (1967, 1970)

MEDICATIONS:
1. Metformin 1000mg twice daily - diabetes
2. Lisinopril 10mg once daily - hypertension and cardioprotection
3. Atorvastatin 40mg once daily at bedtime - hyperlipidemia
4. Low-dose aspirin 81mg once daily - cardioprotection
5. Levothyroxine 75mcg once daily on empty stomach - hypothyroidism
6. Omeprazole 20mg once daily - GERD
7. Warfarin 5mg daily (adjusted per INR) - atrial fibrillation
8. Alendronate 70mg once weekly - osteoporosis
9. Calcium carbonate 500mg twice daily with meals - bone health
10. Vitamin D3 2000 IU once daily - vitamin D deficiency and bone health
11. Acetaminophen 650mg every 6 hours as needed - arthritis pain
12. Multivitamin daily - nutritional support

ALLERGIES: 
1. Penicillin - generalized urticaria and pruritus
2. Shellfish - anaphylaxis with facial swelling and respiratory distress
3. Sulfonamides - skin rash
4. Tree pollen - seasonal allergic rhinitis
5. Latex - contact dermatitis
6. Dust mites - year-round allergic rhinitis
7. Cashews/tree nuts - oral allergy syndrome with lip swelling

SOCIAL HISTORY: 
Former smoker with 40 pack-year history, quit in 2012 after COPD exacerbation. Drinks 1-2 glasses of wine per week socially. Lives independently in her own home. Daughter lives 10 minutes away and provides regular support with shopping and transportation. Widowed since 2015. Retired teacher. Drives during daytime only. Active in church and senior center activities. Performs activities of daily living independently. Uses hearing aids bilaterally.

FAMILY HISTORY:
Father: Died at age 78 from myocardial infarction, history of diabetes and hypertension
Mother: Died at age 85 from stroke, history of hypertension
Two siblings: Brother died at age 72 from lung cancer (smoker), sister alive at age 75 with diabetes
Two children: Daughter age 57, healthy; son age 54, history of hypertension
Strong family history of diabetes and cardiovascular disease

REVIEW OF SYSTEMS:
Constitutional: Denies fever, chills, night sweats, or unintentional weight loss
HEENT: Uses hearing aids bilaterally, vision corrected with reading glasses post-cataract surgery
Cardiovascular: Denies chest pain, palpitations, orthopnea, or PND
Respiratory: Denies dyspnea at rest, cough, or wheezing
Gastrointestinal: Mild heartburn controlled with omeprazole, regular bowel movements
Genitourinary: Denies dysuria, frequency, or incontinence
Musculoskeletal: Chronic joint pain managed with acetaminophen, good mobility post-hip replacement
Neurological: Denies dizziness, syncope, weakness, or memory problems
Psychiatric: Mood stable, previously treated for mild depression 2020-2023

PHYSICAL EXAMINATION:
Vital Signs: BP 142/88 mmHg, HR 72 bpm (irregularly irregular), RR 16, Temp 98.6°F, Weight 165 lbs, Height 64 inches, BMI 28.3, O2 Sat 97% on room air

General: Well-appearing, well-nourished elderly female in no acute distress. Appears stated age. Good historian and cooperative with examination.

HEENT: Normocephalic, atraumatic. PERRLA, EOMI. Fundoscopic exam shows mild macular changes consistent with early AMD. Hearing aids in place bilaterally. Oropharynx clear without lesions.

Neck: Supple, no lymphadenopathy, no thyromegaly, no carotid bruits, JVP normal

Cardiovascular: Irregularly irregular rhythm consistent with atrial fibrillation, normal S1 and S2, grade 2/6 systolic murmur at apex consistent with mitral regurgitation, no rubs or gallops, no peripheral edema

Pulmonary: Clear to auscultation bilaterally, no wheezes, rales, or rhonchi, good air movement

Abdomen: Soft, non-tender, non-distended, normal bowel sounds, no hepatosplenomegaly, no masses, well-healed laparoscopic cholecystectomy scars

Extremities: No edema, good peripheral pulses, well-healed right hip replacement scar, bilateral knee crepitus, no joint swelling or erythema

Skin: Warm and dry, no rashes or lesions, good turgor for age

Neurological: Alert and oriented x3, MMSE 28/30, gait steady with normal stance, no focal deficits, reflexes appropriate for age

ASSESSMENT AND PLAN:

1. Type 2 diabetes mellitus - WELL CONTROLLED
   - A1C improved to 7.2% from 7.8%, meeting target <7.5% for her age
   - Continue metformin 1000mg twice daily
   - Continue home glucose monitoring
   - Annual diabetic foot exam normal
   - Annual ophthalmologic exam due (scheduled)
   - Reinforce dietary compliance and regular exercise

2. Essential hypertension - ADEQUATELY CONTROLLED  
   - Blood pressure 142/88, acceptable for age and comorbidities
   - Continue lisinopril 10mg daily
   - Consider dose adjustment if persistently >150/90

3. Coronary artery disease - STABLE
   - No anginal symptoms, good functional capacity
   - Continue optimal medical therapy with aspirin and statin
   - Last stress test 2024 negative for ischemia
   - Continue lifestyle modifications

4. Atrial fibrillation - WELL CONTROLLED
   - Rate controlled, no symptoms
   - INR therapeutic at 2.3, continue warfarin
   - Regular INR monitoring every 4 weeks
   - CHA2DS2-VASc score 6, appropriate anticoagulation

5. Chronic kidney disease, stage 3 - STABLE
   - eGFR 52, creatinine 1.4, stable over past year
   - Continue ACE inhibitor for renoprotection
   - Monitor every 6 months
   - Avoid nephrotoxic medications

6. Osteoporosis - MANAGED
   - No fractures since hip replacement
   - Continue alendronate weekly, calcium, and vitamin D
   - DEXA scan due in 2025
   - Encourage weight-bearing exercise

7. History of breast cancer - SURVEILLANCE
   - 26 years post-treatment, excellent prognosis
   - Annual mammography (completed 2023, normal)
   - Continue surveillance per oncology

8. Hypothyroidism - STABLE
   - TSH last checked 2023, normal on current dose
   - Continue levothyroxine 75mcg daily
   - Recheck TSH annually

9. GERD - CONTROLLED
   - Symptoms well controlled on omeprazole
   - Continue current therapy
   - Monitor for long-term PPI effects

10. Health maintenance:
    - Mammography: Up to date (2023)
    - Colonoscopy: Up to date (2023, next due 2033)
    - Bone density: Due 2025
    - Ophthalmology: Annual due to AMD and diabetes
    - Hearing assessment: Due 2025
    - Vaccinations: Up to date including COVID-19, influenza, pneumococcal
    - Fall risk assessment: Low risk with good mobility
    - Cognitive assessment: Normal (MMSE 28/30)
    - Depression screening: Negative
    - Advanced directives: Completed and on file

PLAN:
- Continue current medication regimen
- INR check in 4 weeks
- Routine labs in 6 months (A1C, CMP, lipids)
- Ophthalmology referral for diabetic eye exam and AMD monitoring
- Follow up in 6 months or sooner if concerns arise
- Patient counseled on medication compliance, diet, exercise, and when to seek medical attention

The patient is doing remarkably well for her age with multiple chronic conditions well managed. She maintains excellent functional status and independence. Prognosis is good with continued medical management and family support.', 
'final', 'prov_003', '2024-12-01 11:00:00', '2024-12-01 12:00:00', true),

-- === HISTORICAL SIGNIFICANT NOTES ===
('note_002', '1', 'enc_022', 'operative', 'Orthopedics', 'Total Hip Replacement Surgery Report',
'PREOPERATIVE DIAGNOSIS: Right femoral neck fracture, displaced, secondary to fall

POSTOPERATIVE DIAGNOSIS: Right femoral neck fracture, displaced, secondary to fall

PROCEDURE PERFORMED: Right total hip arthroplasty (THA), cementless

SURGEON: Dr. Lisa Davis, MD
ASSISTANT: Dr. Kevin Thompson, MD  
ANESTHESIA: General endotracheal anesthesia
ESTIMATED BLOOD LOSS: 250 mL
COMPLICATIONS: None
IMPLANTS USED: 
- Acetabular cup: 52mm porous-coated titanium cup
- Femoral stem: Size 11 tapered titanium stem
- Femoral head: 32mm ceramic head
- Liner: Highly cross-linked polyethylene liner

INDICATIONS: 
This is a 74-year-old female who sustained a displaced right femoral neck fracture after a mechanical fall at home yesterday evening. Given her age, activity level, and fracture pattern, total hip replacement was recommended over hemiarthroplasty to provide the best long-term functional outcome.

DESCRIPTION OF PROCEDURE:
The patient was brought to the operating room and placed supine on the operating table. General anesthesia was administered by anesthesiology without complications. The patient was then positioned in the left lateral decubitus position on the fracture table with all bony prominences well padded. The right hip and lower extremity were prepped and draped in a sterile fashion.

A posterolateral approach was utilized. An incision was made from 5cm proximal to the greater trochanter extending distally 10cm along the posterior border of the greater trochanter. The fascia lata was incised in line with the skin incision. The gluteus maximus muscle was split bluntly in the direction of its fibers. The short external rotators (piriformis, superior and inferior gemelli, and obturator internus) were identified and divided close to their insertion on the greater trochanter, tagged for later repair.

The hip was then dislocated posteriorly. The fractured femoral head was removed and measured. The acetabulum was inspected and noted to be in good condition with intact cartilage medially. The acetabulum was reamed progressively starting with a 46mm reamer up to 52mm. A 52mm porous-coated titanium acetabular cup was inserted with excellent press fit and stability. A highly cross-linked polyethylene liner was inserted.

Attention was then turned to the femur. The femoral canal was opened with a box osteotome. Sequential reaming was performed, and the canal was prepared for a size 11 tapered titanium stem. Trial reduction was performed with excellent stability and appropriate leg lengths. The definitive size 11 stem was inserted with excellent press fit. A 32mm ceramic femoral head was placed on the stem.

The hip was reduced and found to be stable through full range of motion with no impingement. Leg lengths were equal. The short external rotators were repaired to the greater trochanter. The fascia lata and subcutaneous tissues were closed in layers with absorbable sutures. The skin was closed with skin staples.

The patient was awakened from anesthesia and transferred to the recovery room in stable condition. Post-operative X-rays showed excellent component positioning with no evidence of fracture or dislocation.

POSTOPERATIVE PLAN:
1. Weight bearing as tolerated with walker
2. Physical therapy starting post-operative day 1
3. DVT prophylaxis with enoxaparin
4. Pain management with multimodal approach
5. Dislocation precautions education
6. Follow up in clinic at 2 weeks, 6 weeks, 3 months, and 1 year
7. Home safety evaluation and equipment

PROGNOSIS: Excellent. The patient should expect significant improvement in pain and function with appropriate rehabilitation.', 
'final', 'prov_006', '2018-11-16 14:00:00', '2018-11-16 15:00:00', false),

('note_003', '1', 'enc_038', 'progress', 'Oncology', 'Chemotherapy Treatment Note - Cycle 1',
'PATIENT: Eleanor Henderson
DATE: May 15, 1998
CYCLE: 1 of 4 planned AC cycles

DIAGNOSIS: Invasive ductal carcinoma, right breast, Stage IIA (T2N0M0)
ER positive (95%), PR positive (85%), HER2 negative

TREATMENT PLAN: Adjuvant AC x 4 cycles followed by radiation therapy and 5 years of tamoxifen

CURRENT CYCLE: AC Cycle 1
- Adriamycin (doxorubicin) 60 mg/m2 IV push
- Cytoxan (cyclophosphamide) 600 mg/m2 IV over 30 minutes
- Pre-medications: Ondansetron 8mg IV, dexamethasone 12mg IV, diphenhydramine 25mg IV

INTERVAL HISTORY:
Patient returns for her first cycle of adjuvant chemotherapy. She has recovered well from her lumpectomy 4 weeks ago. Surgical site is well healed with no signs of infection. She reports good energy level and appetite. No fever, chills, or concerning symptoms. She has been taking the prescribed anti-nausea medications as directed.

PHYSICAL EXAMINATION:
VS: T 98.4°F, BP 138/82, HR 76, RR 16, Weight 158 lbs
General: Well-appearing woman in no distress
HEENT: PERRL, EOMI, oropharynx clear
Neck: No lymphadenopathy
Chest: Clear to auscultation bilaterally, surgical site well healed
Heart: RRR, no murmurs
Abdomen: Soft, non-tender
Extremities: No edema
Skin: No rash

LABORATORY RESULTS:
WBC: 6.8 (normal)
Hemoglobin: 13.2 g/dL (normal)
Platelets: 285,000 (normal)
Creatinine: 0.9 mg/dL (normal)
Bilirubin: 0.8 mg/dL (normal)
AST/ALT: Normal

ASSESSMENT:
54-year-old woman with newly diagnosed Stage IIA breast cancer, ER/PR positive, HER2 negative, status post lumpectomy with clear margins and negative sentinel lymph nodes. Ready to begin adjuvant chemotherapy with AC regimen.

PLAN:
1. Proceed with AC cycle 1 as planned
2. Pre-medications administered without reaction
3. Adriamycin 60 mg/m2 (105 mg) given IV push over 5 minutes through peripheral IV without extravasation
4. Cyclophosphamide 600 mg/m2 (1050 mg) given IV over 30 minutes
5. Post-chemotherapy hydration with normal saline
6. Patient tolerated treatment well without immediate reactions
7. Discharge home with anti-emetic regimen:
   - Compazine 10mg PO q6h PRN nausea
   - Ativan 0.5mg PO q6h PRN nausea/anxiety
8. Follow-up labs on day 10-14 to check counts
9. Return for cycle 2 in 21 days (June 5, 1998)
10. Call immediately for fever >100.4°F, persistent vomiting, or other concerning symptoms

PATIENT EDUCATION:
- Reviewed potential side effects including nausea, fatigue, hair loss, infection risk
- Importance of taking anti-nausea medications as prescribed
- When to call the oncology team
- Infection precautions
- Hair loss typically begins 2-3 weeks after first treatment
- Encourage adequate nutrition and hydration

Patient verbalized understanding of treatment plan and emergency precautions. She is motivated and has good family support. Prognosis is excellent with this treatment regimen.

Next appointment: June 5, 1998 for AC cycle 2', 
'final', 'prov_012', '1998-05-15 16:00:00', '1998-05-15 17:00:00', false),

('note_004', '1', 'enc_010', 'progress', 'Cardiology', 'New Onset Atrial Fibrillation Evaluation',
'CHIEF COMPLAINT: Irregular heartbeat discovered on routine monitoring

HISTORY OF PRESENT ILLNESS:
Eleanor Henderson is a 79-year-old female with a significant past medical history of type 2 diabetes, hypertension, coronary artery disease, and chronic kidney disease who presents for evaluation of newly discovered atrial fibrillation. The patient was noted to have an irregular pulse during a routine visit with her primary care physician last week. She denies palpitations, chest pain, shortness of breath, lightheadedness, or syncope. She has not noticed any change in her exercise tolerance or daily activities.

Her current medications include metformin for diabetes, lisinopril for hypertension, atorvastatin for hyperlipidemia, and low-dose aspirin for cardioprotection. She has been compliant with all medications.

PAST MEDICAL HISTORY:
- Type 2 diabetes mellitus (1985) - well controlled, recent A1C 7.8%
- Essential hypertension (1982) - controlled on lisinopril
- Coronary artery disease, 2-vessel (1995) - stable, on optimal medical therapy
- Chronic kidney disease, stage 3 (2010) - stable, eGFR ~55
- History of breast cancer (1998-2003) - successfully treated
- Status post right total hip replacement (2018)
- Hypothyroidism - stable on levothyroxine

PHYSICAL EXAMINATION:
Vital Signs: BP 148/86, HR 78 (irregularly irregular), RR 16, O2 sat 96%
General: Well-appearing elderly female in no acute distress
Cardiovascular: Irregularly irregular rhythm, no murmurs, rubs, or gallops
Lungs: Clear to auscultation bilaterally
Extremities: No edema

DIAGNOSTIC STUDIES:
EKG: Atrial fibrillation with controlled ventricular response, rate 70-85 bpm, no acute ST changes
Echocardiogram: Normal left ventricular systolic function (EF 60%), mild mitral regurgitation, left atrial dimension 4.2 cm (mildly enlarged), no valvular abnormalities

LABORATORY:
TSH: 2.1 (normal)
BNP: 145 (mildly elevated for age)
CMP: Normal except creatinine 1.3

ASSESSMENT AND PLAN:

ATRIAL FIBRILLATION, newly diagnosed
- CHA2DS2-VASc score: 6 points (age ≥75: 2 points, hypertension: 1 point, diabetes: 1 point, vascular disease: 1 point, female sex: 1 point)
- High stroke risk, anticoagulation indicated
- Rate appears controlled at rest

Plan:
1. Initiate anticoagulation with warfarin (INR goal 2.0-3.0)
   - Start warfarin 5mg daily
   - INR check in 3 days, then adjust dosing
   - Bridge with enoxaparin until INR therapeutic
   - Patient counseled on warfarin interactions and dietary considerations

2. Rate control appears adequate currently
   - Monitor heart rate and symptoms
   - May need rate control medication if symptoms develop

3. Rhythm control not pursued given asymptomatic presentation and age

4. Continue current cardiac medications (lisinopril, atorvastatin)

5. Discontinue aspirin once anticoagulated (bleeding risk)

6. Patient education provided regarding atrial fibrillation, stroke risk, and anticoagulation

7. Follow-up in 1 week to check INR and assess tolerance
   - Then monthly visits until INR stable
   - Home INR monitoring to be arranged

8. Annual echocardiogram to monitor LA size and LV function

PROGNOSIS: Good with appropriate anticoagulation and rate control. Patient counseled on the importance of compliance with warfarin therapy and regular monitoring.

The patient was receptive to the treatment plan and expressed understanding of the importance of anticoagulation for stroke prevention.', 
'final', 'prov_004', '2023-02-15 10:00:00', '2023-02-15 11:00:00', true);

-- ===========================================
-- CARE PLANS (Comprehensive care management)
-- ===========================================

INSERT INTO care_plans (
    id, patient_id, title, description, status, intent, period_start, period_end,
    created_date, created_by_id, addresses, goals, activities
) VALUES

('care_001', '1', 'Comprehensive Diabetes Management Plan', 'Evidence-based diabetes care plan targeting optimal glucose control while preventing complications in elderly patient with multiple comorbidities', 'active', 'plan', '2024-01-01', NULL, '2024-01-01', 'prov_005', 
'["cond_006"]', 
'[
  {"description": "Achieve HbA1c target of <7.5% for elderly patient", "target": "7.5", "measure": "HbA1c percentage", "timeframe": "6 months"},
  {"description": "Maintain fasting glucose 100-140 mg/dL", "target": "140", "measure": "fasting glucose mg/dL", "timeframe": "daily"},
  {"description": "Prevent diabetic complications (retinopathy, nephropathy, neuropathy)", "target": "no progression", "measure": "annual screening exams", "timeframe": "annually"},
  {"description": "Maintain stable weight", "target": "165", "measure": "body weight pounds", "timeframe": "quarterly"}
]',
'[
  {"activity": "Home blood glucose monitoring", "frequency": "twice daily", "instructions": "Check fasting and 2-hour post-meal glucose"},
  {"activity": "Take metformin as prescribed", "frequency": "1000mg twice daily with meals", "instructions": "Monitor for GI side effects and lactic acidosis symptoms"},
  {"activity": "Follow diabetic diet", "frequency": "ongoing", "instructions": "Carbohydrate counting, portion control, limit simple sugars"},
  {"activity": "Regular aerobic exercise", "frequency": "30 minutes daily", "instructions": "Walking, swimming, or other low-impact activities"},
  {"activity": "Annual diabetic eye exam", "frequency": "annually", "instructions": "Ophthalmology referral for dilated retinal exam"},
  {"activity": "Annual foot examination", "frequency": "annually", "instructions": "Screen for diabetic neuropathy and foot complications"},
  {"activity": "HbA1c monitoring", "frequency": "every 6 months", "instructions": "Assess long-term glucose control"},
  {"activity": "Diabetes education reinforcement", "frequency": "as needed", "instructions": "Review diabetes management, complications, and self-care"}
]'),

('care_002', '1', 'Cardiovascular Risk Management and Atrial Fibrillation Care Plan', 'Comprehensive plan to manage coronary artery disease, hypertension, hyperlipidemia, and atrial fibrillation with focus on stroke and cardiovascular event prevention', 'active', 'plan', '2023-02-15', NULL, '2023-02-15', 'prov_004', 
'["cond_005", "cond_010", "cond_025"]',
'[
  {"description": "Maintain blood pressure <150/90 mmHg for elderly patient", "target": "150/90", "measure": "blood pressure mmHg", "timeframe": "monthly"},
  {"description": "Achieve LDL cholesterol <100 mg/dL", "target": "100", "measure": "LDL cholesterol mg/dL", "timeframe": "quarterly"},
  {"description": "Maintain therapeutic INR 2.0-3.0 for stroke prevention", "target": "2.5", "measure": "INR", "timeframe": "monthly"},
  {"description": "Control atrial fibrillation heart rate 60-100 bpm", "target": "80", "measure": "heart rate bpm", "timeframe": "monthly"},
  {"description": "Prevent stroke and cardiovascular events", "target": "zero events", "measure": "clinical events", "timeframe": "ongoing"}
]',
'[
  {"activity": "Take antihypertensive medication (lisinopril)", "frequency": "10mg daily", "instructions": "Monitor for hypotension, hyperkalemia, and cough"},
  {"activity": "Take statin medication (atorvastatin)", "frequency": "40mg daily at bedtime", "instructions": "Monitor for muscle symptoms and liver enzymes"},
  {"activity": "Warfarin anticoagulation", "frequency": "5mg daily, adjust per INR", "instructions": "Regular INR monitoring, dietary vitamin K consistency"},
  {"activity": "Regular INR monitoring", "frequency": "every 4 weeks when stable", "instructions": "Home INR testing when available"},
  {"activity": "Heart-healthy diet (DASH diet)", "frequency": "ongoing", "instructions": "Low sodium <2g daily, increase fruits/vegetables, limit saturated fats"},
  {"activity": "Regular moderate exercise", "frequency": "daily as tolerated", "instructions": "Walking, avoid high-intensity activities due to anticoagulation"},
  {"activity": "Annual lipid panel", "frequency": "annually", "instructions": "Monitor response to statin therapy"},
  {"activity": "Periodic echocardiogram", "frequency": "annually", "instructions": "Monitor left atrial size and left ventricular function"},
  {"activity": "Blood pressure monitoring", "frequency": "weekly at home", "instructions": "Home BP monitoring, log results"}
]'),

('care_003', '1', 'Chronic Kidney Disease Stage 3 Management Plan', 'Comprehensive plan for CKD stage 3 focusing on slowing progression, managing complications, and optimizing overall health', 'active', 'plan', '2010-02-14', NULL, '2010-02-14', 'prov_010', 
'["cond_016"]',
'[
  {"description": "Slow progression of kidney disease", "target": "stable eGFR >45", "measure": "eGFR mL/min/1.73m2", "timeframe": "every 6 months"},
  {"description": "Maintain optimal blood pressure for renal protection", "target": "130/80", "measure": "blood pressure mmHg", "timeframe": "monthly"},
  {"description": "Prevent bone disease complications", "target": "normal calcium/phosphorus", "measure": "serum calcium/phosphorus", "timeframe": "every 6 months"},
  {"description": "Avoid nephrotoxic medications", "target": "zero nephrotoxic exposures", "measure": "medication review", "timeframe": "ongoing"}
]',
'[
  {"activity": "ACE inhibitor therapy (lisinopril)", "frequency": "10mg daily", "instructions": "Renoprotective, monitor potassium and creatinine"},
  {"activity": "Monitor kidney function", "frequency": "every 6 months", "instructions": "Serum creatinine, eGFR, urinalysis, microalbumin"},
  {"activity": "Calcium and vitamin D supplementation", "frequency": "calcium 500mg BID, vitamin D 2000 IU daily", "instructions": "Prevent bone disease, monitor serum levels"},
  {"activity": "Avoid nephrotoxic medications", "frequency": "ongoing", "instructions": "NSAIDs, contrast agents, certain antibiotics"},
  {"activity": "Protein intake modification", "frequency": "ongoing", "instructions": "Moderate protein intake 0.8-1.0 g/kg/day"},
  {"activity": "Blood pressure optimization", "frequency": "ongoing", "instructions": "Target <140/90, ideally <130/80 for CKD"},
  {"activity": "Annual nephrology follow-up", "frequency": "annually", "instructions": "Specialist management and preparation for advanced CKD if progression"},
  {"activity": "Vaccination updates", "frequency": "as recommended", "instructions": "Hepatitis B, influenza, pneumococcal per CKD guidelines"}
]'),

('care_004', '1', 'Osteoporosis Management and Fall Prevention Plan', 'Comprehensive bone health management focusing on fracture prevention, bone density optimization, and fall risk reduction', 'active', 'plan', '2019-03-10', NULL, '2019-03-10', 'prov_006', 
'["cond_026"]',
'[
  {"description": "Improve bone mineral density", "target": "T-score >-2.5", "measure": "DEXA scan T-score", "timeframe": "every 2 years"},
  {"description": "Prevent osteoporotic fractures", "target": "zero fractures", "measure": "clinical assessment", "timeframe": "ongoing"},
  {"description": "Minimize fall risk", "target": "zero falls", "measure": "fall risk assessment", "timeframe": "quarterly"},
  {"description": "Optimize calcium and vitamin D status", "target": "vitamin D >30 ng/mL", "measure": "serum 25-OH vitamin D", "timeframe": "annually"}
]',
'[
  {"activity": "Bisphosphonate therapy (alendronate)", "frequency": "70mg once weekly", "instructions": "Take on empty stomach, remain upright 30 minutes"},
  {"activity": "Calcium supplementation", "frequency": "500mg twice daily with meals", "instructions": "Improve calcium absorption, monitor for kidney stones"},
  {"activity": "Vitamin D supplementation", "frequency": "2000 IU daily", "instructions": "Optimize vitamin D status for bone health"},
  {"activity": "Weight-bearing exercise", "frequency": "30 minutes daily", "instructions": "Walking, resistance training as tolerated"},
  {"activity": "Fall risk assessment", "frequency": "quarterly", "instructions": "Home safety evaluation, medication review, balance assessment"},
  {"activity": "Home safety modifications", "frequency": "as needed", "instructions": "Remove trip hazards, improve lighting, install grab bars"},
  {"activity": "DEXA scan monitoring", "frequency": "every 2 years", "instructions": "Monitor response to therapy"},
  {"activity": "Avoid tobacco and limit alcohol", "frequency": "ongoing", "instructions": "Both negatively impact bone health"}
]'),

('care_005', '1', 'Comprehensive Geriatric Care Plan', 'Holistic care plan addressing multiple chronic conditions, functional status, cognitive health, and quality of life in an elderly patient', 'active', 'plan', '2020-01-01', NULL, '2020-01-01', 'prov_003', 
'["cond_005", "cond_006", "cond_010", "cond_016", "cond_025", "cond_026"]',
'[
  {"description": "Maintain functional independence", "target": "independent ADLs", "measure": "functional assessment", "timeframe": "quarterly"},
  {"description": "Optimize cognitive function", "target": "MMSE >24", "measure": "cognitive screening", "timeframe": "annually"},
  {"description": "Prevent hospitalizations", "target": "<1 hospitalization per year", "measure": "hospital admissions", "timeframe": "annually"},
  {"description": "Maintain quality of life", "target": "patient satisfaction >8/10", "measure": "quality of life assessment", "timeframe": "annually"}
]',
'[
  {"activity": "Comprehensive geriatric assessment", "frequency": "annually", "instructions": "Multidisciplinary evaluation of medical, functional, cognitive, and social domains"},
  {"activity": "Medication reconciliation and optimization", "frequency": "quarterly", "instructions": "Review for drug interactions, side effects, and appropriateness"},
  {"activity": "Cognitive screening", "frequency": "annually", "instructions": "MMSE or similar cognitive assessment tool"},
  {"activity": "Depression screening", "frequency": "annually", "instructions": "PHQ-9 or similar depression screening tool"},
  {"activity": "Functional status assessment", "frequency": "quarterly", "instructions": "Activities of daily living and instrumental ADLs"},
  {"activity": "Social support evaluation", "frequency": "annually", "instructions": "Assess family support, community resources, and social engagement"},
  {"activity": "Advance care planning", "frequency": "annually", "instructions": "Review goals of care, advance directives, and preferences"},
  {"activity": "Preventive care coordination", "frequency": "ongoing", "instructions": "Age-appropriate screenings, vaccinations, and health maintenance"}
]');

-- ===========================================
-- PATIENT SUMMARY (AI-generated comprehensive summary)
-- ===========================================

INSERT INTO patient_summaries (
    id, patient_id, summary_current, summary_recent, summary_historical, summary_chronic,
    active_diagnoses, current_medications, allergies, chronic_conditions,
    last_updated, generated_by_model, confidence_score
) VALUES (
    'summary_001',
    '1',
    'Eleanor Henderson is an 80-year-old female with multiple well-managed chronic conditions including type 2 diabetes (A1C 7.2%), hypertension, coronary artery disease, stage 3 chronic kidney disease, and newly diagnosed atrial fibrillation on warfarin anticoagulation. She maintains excellent functional independence living in her own home with family support. Recent concerns include optimal anticoagulation management for stroke prevention and ongoing surveillance for her multiple conditions. She has a remarkable history of breast cancer survivorship (26 years post-treatment) and successful recovery from bilateral hip fractures. Current status is stable with good quality of life and strong family support system.',
    
    'Recent 6-month activity: December 2024 annual wellness visit showed overall stable health with improved diabetes control (A1C down from 7.8% to 7.2%). Atrial fibrillation remains well-controlled on warfarin with therapeutic INR. June 2024 emergency visit for chest pain and dyspnea was negative for acute coronary syndrome with normal cardiac enzymes and imaging. March 2024 cardiology follow-up confirmed stable coronary artery disease and good atrial fibrillation management. No hospitalizations or acute exacerbations of chronic conditions. Completed routine health maintenance including mammography (2023) and colonoscopy (2023), both normal. Medication regimen remains stable with good tolerance.',
    
    'Eleanor has an extraordinary medical history spanning over 60 years of comprehensive healthcare documentation. Born in 1944, her medical journey began in 1962 with establishment of primary care. Major life events include two uncomplicated pregnancies and deliveries (1967, 1970), establishment of chronic diseases in middle age including hypertension (1982) and diabetes (1985), and a significant breast cancer diagnosis and successful treatment (1998-2003) with lumpectomy, AC chemotherapy, radiation, and 5 years of tamoxifen. She has experienced two hip fractures - first at age 31 from a skiing accident (1975) and second at age 74 from a home fall (2018) requiring total hip replacement. Other major surgical procedures include cholecystectomy (2005) and bilateral cataract surgeries (2015-2016). She was a 40 pack-year smoker who successfully quit in 2012 after a COPD exacerbation. Her medical complexity has increased with age, adding coronary artery disease (1995), chronic kidney disease (2010), osteoporosis (2019), and atrial fibrillation (2023).',
    
    'Chronic conditions requiring ongoing management span multiple organ systems: 1) Type 2 diabetes mellitus since 1985 - currently well-controlled on metformin with recent A1C 7.2%, requires ongoing monitoring for diabetic complications; 2) Essential hypertension since 1982 - managed with ACE inhibitor, provides renal and cardiac protection; 3) Coronary artery disease since 1995 - 2-vessel disease managed medically with aspirin, statin, and ACE inhibitor, remains stable without anginal symptoms; 4) Chronic kidney disease stage 3 since 2010 - likely diabetic nephropathy, stable function with eGFR ~52, managed with ACE inhibitor and monitoring; 5) Atrial fibrillation since 2023 - managed with anticoagulation (warfarin) for stroke prevention, rate-controlled; 6) Osteoporosis since 2019 - managed with bisphosphonate, calcium, and vitamin D supplementation; 7) Hypothyroidism since 2000 - stable on levothyroxine replacement; 8) GERD since 2008 - controlled with PPI therapy; 9) Polyosteoarthritis since 1988 - managed with acetaminophen and lifestyle modifications.',
    
    '["Type 2 diabetes mellitus", "Essential hypertension", "Coronary artery disease", "Chronic kidney disease stage 3", "Atrial fibrillation", "Osteoporosis", "Hypothyroidism", "GERD", "Polyosteoarthritis", "Age-related macular degeneration", "Age-related hearing loss", "History of breast cancer"]',
    
    '["Metformin 1000mg BID", "Lisinopril 10mg daily", "Atorvastatin 40mg daily", "Warfarin 5mg daily", "Levothyroxine 75mcg daily", "Omeprazole 20mg daily", "Alendronate 70mg weekly", "Calcium carbonate 500mg BID", "Vitamin D3 2000 IU daily", "Aspirin 81mg daily", "Acetaminophen 650mg PRN", "Multivitamin daily"]',
    
    '["Penicillin - urticaria and pruritus", "Shellfish - anaphylaxis with respiratory distress", "Sulfonamides - skin rash", "Tree pollen - seasonal allergic rhinitis", "Latex - contact dermatitis", "Dust mites - perennial allergic rhinitis", "Cashews/tree nuts - oral allergy syndrome"]',
    
    '["Type 2 diabetes mellitus", "Essential hypertension", "Coronary artery disease", "Chronic kidney disease", "Atrial fibrillation", "Osteoporosis", "Hypothyroidism", "GERD", "Polyosteoarthritis"]',
    
    '2024-12-15 14:00:00',
    'Claude-4-Medical-AI-Enhanced',
    0.98
);