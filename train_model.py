import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score

# ==============================================================================
# STEP 1: DATA ACQUISITION & BIOLOGICAL CLEANING
# ==============================================================================
print("Initializing Pipeline...")

input_file = 'healthcare_claims.csv' 

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Source file '{input_file}' not found. Please verify the file path.")

print(f"Loading raw claims data: {input_file}")
df = pd.read_csv(input_file)

# Drop biologically impossible database records (Strict physical rule cleaning)
pregnancy_related = ['Pregnancy', 'Cesarean Section']
biological_anomaly = (df['Gender'] == 'Male') & (df['Diagnosis'].isin(pregnancy_related))
cleaned_df = df[~biological_anomaly].copy()
print(f"Filtered out {len(df) - len(cleaned_df)} biological anomalies from the source dataset.")

# ==============================================================================
# STEP 2: FEATURE ENGINEERING
# ==============================================================================
print("\nEngineering features...")

# 1. Map individual diagnoses to broad clinical groups
diagnosis_to_group = {
    "Diabetes": "Chronic", "Hypertension": "Chronic", "Asthma": "Chronic",
    "HIV/AIDS": "Chronic", "Hepatitis B": "Chronic",
    "Kidney Disease": "Renal",
    "Cancer Treatment": "Major Surgical", "Cataract Surgery": "Major Surgical",
    "Appendectomy": "Major Surgical", "Advanced Spinal Surgery": "Major Surgical",
    "Complex Heart Surgery": "Major Surgical", "Neurosurgery": "Major Surgical",
    "Organ Transplant": "Major Surgical", "Cosmetic Surgery": "Major Surgical",
    "Epilepsy Surgery": "Major Surgical",
    "Gastroenteritis": "Infectious", "Pneumonia": "Infectious",
    "Tuberculosis": "Infectious", "Peptic Ulcer": "Infectious",
    "Stroke": "Cardiac",
    "Pregnancy": "Reproductive", "Cesarean Section": "Reproductive",
    "Infertility Treatment (IVF)": "Reproductive",
    "Migraine": "Routine", "Routine Check-up": "Routine"
}
cleaned_df['Diagnosis_Group'] = cleaned_df['Diagnosis'].map(diagnosis_to_group).fillna('Other')

# 2. Convert date sequences into hospitalized durations
cleaned_df['Date Admitted'] = pd.to_datetime(cleaned_df['Date Admitted'])
cleaned_df['Date Discharged'] = pd.to_datetime(cleaned_df['Date Discharged'])
cleaned_df['Days_Hospitalized'] = (cleaned_df['Date Discharged'] - cleaned_df['Date Admitted']).dt.days

def get_stay_category(days):
    if days == 0: return "Same-day"
    elif days <= 3: return "Short stay"
    elif days <= 7: return "Standard stay"
    elif days <= 14: return "Extended stay"
    else: return "Long stay"

cleaned_df['Stay_Category'] = cleaned_df['Days_Hospitalized'].apply(get_stay_category)

# 3. Categorize financial exposures into Bill Tiers
def get_bill_tier(amount):
    if amount < 25000: return "Very Low"
    elif amount < 50000: return "Low"
    elif amount < 75000: return "Low-Moderate"
    elif amount < 100000: return "Moderate"
    elif amount < 125000: return "Moderate-High"
    elif amount < 150000: return "High"
    elif amount < 200000: return "Very High"
    else: return "Extreme"

cleaned_df['Bill_Tier'] = cleaned_df['Amount Billed'].apply(get_bill_tier)

# 4. Convert target variable to binary format
cleaned_df['Fraud_Binary'] = cleaned_df['Fraud Type'].apply(lambda x: 0 if x == 'No Fraud' else 1)

# 5. Engine demographic-medical mismatches as a model feature
def biological_mismatch(row):
    age = row['Age']
    gender = row['Gender']
    diag = row['Diagnosis_Group']

    # Flags reproductive mappings with inconsistent age/demographic patterns
    if diag == "Reproductive":
        if gender != "Female":
            return 1
        if age < 12 or age > 55:
            return 1
            
    # Flags atypical pediatric cardiac diagnoses
    if diag == "Cardiac" and age < 10:
        return 1

    return 0

cleaned_df['Bio_Mismatch'] = cleaned_df.apply(biological_mismatch, axis=1)

# ==============================================================================
# STEP 3: PREPROCESSING PIPELINE
# ==============================================================================
print("\nConfiguring Preprocessor Pipeline...")

# Separate target and select training feature vectors
X = cleaned_df.drop(columns=['Fraud Type', 'Fraud_Binary'])
y = cleaned_df['Fraud_Binary']

categorical_cols = ['Gender', 'Diagnosis_Group', 'Stay_Category', 'Bill_Tier']
numerical_cols = ['Age', 'Days_Hospitalized', 'Bio_Mismatch']

# Sub-select features for direct pipeline feeding
X = X[categorical_cols + numerical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Fit-transform features
X_encoded = preprocessor.fit_transform(X)

# Stratified split to maintain uniform class proportions in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42, stratify=y
)

# ==============================================================================
# STEP 4: MODEL TRAINING AND SELECTION
# ==============================================================================
print("\nTraining and evaluating models...")

# Candidate A: Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

# Candidate B: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)

print("=" * 60)
print(f"Random Forest Classifier Accuracy:   {rf_acc:.4f}")
print(f"Logistic Regression Accuracy:        {lr_acc:.4f}")
print("=" * 60)

# Display detailed results for Random Forest Classifier
print("\n" + "="*20 + " RANDOM FOREST PERFORMANCE " + "="*20)
print(classification_report(y_test, rf_preds, target_names=['No Fraud', 'Fraud']))

# Display detailed results for Logistic Regression
print("\n" + "="*20 + " LOGISTIC REGRESSION PERFORMANCE " + "="*20)
print(classification_report(y_test, lr_preds, target_names=['No Fraud', 'Fraud']))
print("=" * 60)

# Dynamically export the stronger model
if rf_acc >= lr_acc:
    best_model = rf_model
    model_name = "Random Forest Classifier"
else:
    best_model = lr_model
    model_name = "Logistic Regression"

print(f"\nModel selected for final production serialization: {model_name}")

# ==============================================================================
# STEP 5: MODEL SERIALIZATION & DATA EXPORT
# ==============================================================================
print("\nWriting artifacts to disk...")
joblib.dump(best_model, 'fraud_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

# Export final enriched dataset for modeling archives
cleaned_df.to_csv('cleaned_healthcare_claims_final.csv', index=False)

print("-" * 55)
print("SUCCESS: Pipeline complete.")
print("Saved artifacts: 'fraud_model.pkl', 'preprocessor.pkl'")
print("Saved clean data: 'cleaned_healthcare_claims_final.csv'")
print("-" * 55)
