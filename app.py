import streamlit as st
import pandas as pd
import joblib

# Set page config for a wide dashboard layout
st.set_page_config(
    page_title="Health Insurance Claims Risk Scanner",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load AI model files
@st.cache_resource
def load_artifacts():
    model = joblib.load('fraud_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

try:
    model, preprocessor = load_artifacts()
except FileNotFoundError:
    st.error("Error: The core model files are missing from the server.")
    st.stop()

# --- Dropdown Categories ---
ILLNESS_MAPPING = {
    "Pregnancy / Childbirth delivery": "Reproductive",
    "Ovarian Cyst / Fertility check": "Reproductive",
    "Stroke Treatment / Assessment": "Routine",
    "Hypertension / High Blood Pressure": "Chronic",
    "Diabetes Management": "Chronic",
    "Chronic Kidney Disease": "Renal",
    "Appendectomy / Gallbladder Removal": "Major Surgical",
    "Pneumonia / Severe Flu": "Infectious",
    "General Checkup / Physical": "Routine"
}

def calculate_stay_category(days):
    if days == 0: return "Same-day"
    elif 1 <= days <= 3: return "Short stay"
    elif 4 <= days <= 7: return "Standard stay"
    elif 8 <= days <= 14: return "Extended stay"
    else: return "Long stay"

def calculate_bill_tier(amount):
    if amount < 500: return "Very Low"
    elif amount < 1500: return "Low"
    elif amount < 3000: return "Low-Moderate"
    elif amount < 6000: return "Moderate"
    elif amount < 12000: return "Moderate-High"
    elif amount < 25000: return "High"
    elif amount < 50000: return "Very High"
    else: return "Extreme"

# --- Mismatch Warning Checker ---
def check_biological_mismatch(age, gender, illness):
    if "Pregnancy" in illness or "Ovarian" in illness:
        if gender != "Female":
            return True, "A male person cannot have such illness."
        if age < 12 or age > 55:
            return True, f"The patient's age ({age}) can not be pregnant."
    if "Stroke" in illness and age < 10:
        return True, f"Rare Case Warning: Stroke in children under 10 is very rare."
    return False, ""

# --- Visual Styling (Borders and Colors) ---
st.markdown(
    """
    <style>
        /* Dark gray inputs with red borders */
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div.stNumberInput > div[data-baseweb="input"] {
            background-color: #1F2937 !important;
            border: 2px solid #EE312A !important;
            border-radius: 6px;
        }
        /* Action Button Setup */
        div.stButton > button[kind="primary"] {
            background-color: #1F2937 !important;
            border: 2px solid #EE312A !important;
            color: #EE312A !important;
            font-weight: bold;
            width: 100%;
            height: 50px;
            border-radius: 6px;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #EE312A !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Two-Column Layout ---
left_panel, right_panel = st.columns([2, 1], gap="large")

with left_panel:
    st.subheader("Patient & Diagnosis Info")

    col1, col2 = st.columns(2)
    with col1:
        
        policy_id = st.text_input("📋 Enter Policy ID / Claim Number", value="POL-1001")
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=20)
        gender = st.selectbox("Patient Gender", ["Male", "Female"], index=0)
        selected_illness = st.selectbox("Diagnosis", list(ILLNESS_MAPPING.keys()))

    with col2:
        raw_bill_amount = st.number_input("Total Claim Amount ($)", min_value=0.0, value=1000000.0, step=50.0)
        days_hospitalized = st.number_input("Days Spent in Hospital", min_value=0, value=1)

        # Spacing block to keep layout clean
        st.markdown("<br><div style='height: 48px;'></div>", unsafe_allow_html=True)

with right_panel:
    st.subheader("Actions")
    execute = st.button("Check Claim for Fraud Risk", type="primary")

    st.markdown("---")

    # Run the scanner when the button is clicked
    if execute:
        derived_stay = calculate_stay_category(days_hospitalized)
        derived_tier = calculate_bill_tier(raw_bill_amount)
        derived_diagnosis = ILLNESS_MAPPING[selected_illness]

        is_mismatch, mismatch_reason = check_biological_mismatch(age, gender, selected_illness)
        bio_mismatch_numeric = 1 if is_mismatch else 0

        if is_mismatch:
            # 1. Shows ONLY when there is a flat-out data mismatch (e.g., Male + Pregnancy)
            st.markdown(
                """
                <div style='background-color: #EE312A; color: white; padding: 1rem; border-radius: 6px; text-align: center; margin-bottom: 1rem;'>
                    <b style='letter-spacing: 1px;'>WARNING: FRAUD DETECTED</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style='background-color: #1F2937; padding: 1.5rem; border-radius: 6px; border: 2px solid #EE312A; text-align: center;'>
                    <p style='color: #9CA3AF; margin: 0; font-size: 0.85rem; text-transform: uppercase;'>Risk Level</p>
                    <h1 style='color: #EE312A; margin: 0; font-size: 3.5rem; font-weight: bold;'>100.0%</h1>
                    <p style='color: #EF4444; margin-top: 0.5rem; font-size: 0.9rem;'><b>Reason:</b> {mismatch_reason}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Process calculations if the data checks out
            input_df = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "Diagnosis_Group": derived_diagnosis,
                "Stay_Category": derived_stay,
                "Bill_Tier": derived_tier,
                "Days_Hospitalized": days_hospitalized,
                "Bio_Mismatch": bio_mismatch_numeric
            }])
            input_encoded = preprocessor.transform(input_df)
            risk_score = model.predict_proba(input_encoded)[0][1]
            score_percentage = risk_score * 100

            color_code = "#EE312A" if risk_score > 0.5 else "#10B981"
            status_text = "HIGH RISK CLAIM" if risk_score > 0.5 else "SAFE CLAIM"

            # Dynamic banners depending on the risk score outcome
            if risk_score > 0.5:
                # RED banner if the model finds high risk
                st.markdown(
                    """
                    <div style='background-color: #EE312A; color: white; padding: 1rem; border-radius: 6px; text-align: center; margin-bottom: 1rem;'>
                        <b style='letter-spacing: 1px;'>CRITICAL AUDIT EXCEPTION: HIGH RISK DETECTED</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # GREEN banner if no fraud or risk was detected
                st.markdown(
                    """
                    <div style='background-color: #10B981; color: white; padding: 1rem; border-radius: 6px; text-align: center; margin-bottom: 1rem;'>
                        <b style='letter-spacing: 1px;'>CLAIM PASSED: NO FRAUD DETECTED</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                <div style='background-color: #1F2937; padding: 1.5rem; border-radius: 6px; border: 2px solid {color_code}; text-align: center;'>
                    <p style='color: #9CA3AF; margin: 0; font-size: 0.85rem; text-transform: uppercase;'>Risk Level</p>
                    <h1 style='color: {color_code}; margin: 0; font-size: 3.5rem; font-weight: bold;'>{score_percentage:.1f}%</h1>
                    <p style='color: {color_code}; margin-top: 0.5rem; font-size: 0.9rem;'><b>Status:</b> {status_text}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        # Standard display before running the scanner (Completely neutral state)
        st.markdown(
            """
            <div style='background-color: #1F2937; padding: 2rem; border-radius: 6px; border: 1px dashed #4B5563; text-align: center; color: #9CA3AF;'>
                Ready. Fill in the data and click the button to scan.
            </div>
            """,
            unsafe_allow_html=True
        )
# Create a dictionary of the current claim data including the Policy ID
report_data = {
    "Policy ID": [policy_id],
    "Risk Score (%)": [f"{score_percentage:.1f}%"],
    "Age": [age],
    "Gender": [gender],
    "Bill Amount": [bill_amount],
    "Length of Stay": [length_of_stay],
    "Status": ["REJECTED / HIGH RISK" if score_percentage > 70 or is_mismatch else "APPROVED / LOW RISK"]
}

# Convert it into a downloadable CSV format
import pandas as pd
df_report = pd.DataFrame(report_data)
csv_data = df_report.to_csv(index=False).encode('utf-8')

# Add the download button to the UI
st.download_button(
    label=f"📥 Download Audit Report for {policy_id} (CSV)",
    data=csv_data,
    file_name=f"audit_report_{policy_id}.csv",
    mime="text/csv",
)
