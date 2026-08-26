import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import tenseal as ts
from google.colab import files
from sklearn.preprocessing import StandardScaler
import io
import json

# --- 1. CONFIGURATION: EXACT NAMES FROM YOUR DATA ---
CONDITIONS_LIST = [
    "Polycystic Ovary Syndrome (PCOS)", "Endometriosis", "Erectile Dysfunction",
    "Prostate Cancer", "Cervical Cancer", "Chlamydia", "Syphilis", "HIV",
    "Pelvic Inflammatory Disease (PID)", "Infertility"
]

CYTOLOGY_RESULTS = ["NILM", "HSIL", "CIN2", "CIN3"]

# The 29 numerical biomarkers (excluding the categorical 'cervical_cytology_result')
NUMERICAL_BIOMARKERS = [
    'age_at_first_symptom_yrs', 'pelvic_pain_score', 'sexual_dysfunction_score',
    'testosterone_total_ng_dl', 'estrogen_total_pg_ml', 'progesterone_ng_ml',
    'LH_level_mIU_ml', 'FSH_level_mIU_ml', 'Semen_volume(mL)', 'prolactin_ng_ml',
    'AMH_ng_ml', 'PSA_ng_ml', 'sperm_count_million_ml', 'motility_pct',
    'morphology_pct', 'CD4_count_cells_ul', 'viral_load_copies_ml',
    'STD_test_positive', 'pelvic_ultrasound_cysts_count', 'endometrial_thickness_mm',
    'dyspareunia_score', 'urinary_difficulty_score', 'erection_rigidity_pct',
    'Contraceptive_use', 'infertility_duration_months', 'antibiotic_treatment_received',
    'hpv_test_positive', 'partner_infection_confirmed', 'Menstrual_irregularity_score'
]

# Total Feature Set for the Model
FINAL_FEATURES = CONDITIONS_LIST + CYTOLOGY_RESULTS + NUMERICAL_BIOMARKERS
TARGET_LABEL = 'sexual_health_index'

class SexualHealthModel(nn.Module):
    def __init__(self, input_dim):
        super(SexualHealthModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.fc(x)

def create_ckks_context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

# --- 2. LOCAL TRAINING WITH ONE-HOT ENCODING ---
def train_local_hospital(df, features):
    # One-hot encode 'reproductive_health_issue'
    for cond in CONDITIONS_LIST:
        df[cond] = (df['reproductive_health_issue'] == cond).astype(int)

    # One-hot encode 'cervical_cytology_result'
    for res in CYTOLOGY_RESULTS:
        df[res] = (df['cervical_cytology_result'] == res).astype(int)

    # Reindex and clean
    X = df.reindex(columns=features, fill_value=0).astype(float).values
    y = df[TARGET_LABEL].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SexualHealthModel(len(features))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)

    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    flat_weights = model.fc.weight.detach().numpy().flatten().tolist()
    flat_weights.append(model.fc.bias.detach().item())
    return flat_weights

# --- 3. MAIN PIPELINE ---
print("Step 1: Upload your 10 hospital CSV files (e.g., 3.csv, 4.csv, etc.)")
uploaded = files.upload()

if uploaded:
    context = create_ckks_context()
    hospital_ciphertexts = []

    print(f"\nStep 2: Training on {len(uploaded)} hospital datasets...")
    for filename, content in uploaded.items():
        df_hospital = pd.read_csv(io.BytesIO(content))
        weights = train_local_hospital(df_hospital, FINAL_FEATURES)
        hospital_ciphertexts.append(ts.ckks_vector(context, weights))
        print(f"  [OK] {filename} encrypted.")

    print("\nStep 3: Secure Aggregation...")
    encrypted_sum = hospital_ciphertexts[0]
    for i in range(1, len(hospital_ciphertexts)):
        encrypted_sum += hospital_ciphertexts[i]

    print("Step 4: Trusted Authority Decryption...")
    decrypted_sum = encrypted_sum.decrypt()
    global_weights = [val / len(hospital_ciphertexts) for val in decrypted_sum]

    # Export
    output = {"features": FINAL_FEATURES, "weights": global_weights[:-1], "bias": global_weights[-1]}
    with open("global_sexual_model.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\nSUCCESS: Global model weights saved as 'global_sexual_model.json'")
    files.download("global_sexual_model.json")

