import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import tenseal as ts
from google.colab import files
from sklearn.preprocessing import StandardScaler
import io
import json

# --- CONFIGURATION (Same as before) ---
MENTAL_CONDITIONS = [
    "Major Depressive Disorder", "Generalized Anxiety Disorder", "Kleptomania",
    "Schizophrenia", "Anorexia Nervosa", "Bipolar I Disorder",
    "Attention-Deficit/Hyperactivity Disorder (ADHD)", "Obsessive-Compulsive Disorder (OCD)",
    "Social Anxiety Disorder", "Substance Use Disorder - Alcohol"
]

BIOMARKER_COLUMNS = [
    'PHQ_9_total_score', 'GAD_7_total_score', 'YBOCS_total_score', 'AUDIT_total_score',
    'BMI', 'Weight_change_pct_6m', 'YMRS_total', 'PANSS_total', 'ASRS_total',
    'Impulsivity_score_BIS11', 'Number_of_theft_incidents', 'Pre_theft_tension_relief',
    'Suicidal_ideation_flag', 'Sleep_avg_nightly_duration', 'Sleep_fragmentation_count',
    'Resting_heart_rate_RHR', 'HRV_RMSSD_ms', 'Stroop_test_error_rate',
    'Psychomotor_activity_score', 'Hallucination_frequency_per_week',
    'Delusion_severity_score', 'LSAS_total_score', 'Substance_use_days_past_30',
    'Avg_alcohol_units_per_week', 'Medication_adherence_pct', 'Age_of_onset',
    'Family_history_flag', 'Functional_impairment_days', 'Anxiety_avoidance_behaviors',
    'Compulsions_minutes_per_day'
]

FINAL_FEATURES = MENTAL_CONDITIONS + BIOMARKER_COLUMNS
TARGET_LABEL = 'mental_health_index'

class MentalHealthModel(nn.Module):
    def __init__(self, input_dim):
        super(MentalHealthModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x): return self.fc(x)

def create_ckks_context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

def train_local_hospital(df, feature_names):
    # Clean up non-breaking spaces in the condition column
    df['mental_condition'] = df['mental_condition'].astype(str).str.replace('\xa0', ' ').str.strip()

    for cond in MENTAL_CONDITIONS:
        df[cond] = (df['mental_condition'] == cond).astype(int)

    X = df.reindex(columns=feature_names, fill_value=0).astype(float).values
    y = df[TARGET_LABEL].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MentalHealthModel(len(feature_names))
    criterion, optimizer = nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.01)

    X_t, y_t = torch.FloatTensor(X_scaled), torch.FloatTensor(y)
    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    flat_weights = model.fc.weight.detach().numpy().flatten().tolist()
    flat_weights.append(model.fc.bias.detach().item())
    return flat_weights

# --- MAIN EXECUTION WITH ENCODING FIX ---
print("Step 1: Upload your Mental Health CSV files.")
uploaded = files.upload()

if uploaded:
    context = create_ckks_context()
    hospital_ciphertexts = []

    print(f"\nStep 2: Processing {len(uploaded)} hospitals...")
    for filename, content in uploaded.items():
        try:
            # FIXED LINE: Added encoding='latin1' to handle 0xa0 error
            df_hospital = pd.read_csv(io.BytesIO(content), encoding='latin1')

            weights = train_local_hospital(df_hospital, FINAL_FEATURES)
            hospital_ciphertexts.append(ts.ckks_vector(context, weights))
            print(f"  [OK] {filename} processed.")
        except Exception as e:
            print(f"  [X] Error processing {filename}: {e}")

    if hospital_ciphertexts:
        print("\nStep 3: Secure Aggregation...")
        enc_sum = hospital_ciphertexts[0]
        for i in range(1, len(hospital_ciphertexts)):
            enc_sum += hospital_ciphertexts[i]

        print("Step 4: Decryption & Finalization...")
        decrypted = enc_sum.decrypt()
        global_w = [val / len(hospital_ciphertexts) for val in decrypted]

        output = {"model": "Global_Mental_Model", "features": FINAL_FEATURES, "weights": global_w[:-1], "bias": global_w[-1]}
        with open("global_mental_model.json", "w") as f:
            json.dump(output, f, indent=4)

        print("\nSUCCESS: Saved to 'global_mental_model.json'")
        files.download("global_mental_model.json")
