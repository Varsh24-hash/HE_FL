import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import tenseal as ts
from google.colab import files
from sklearn.preprocessing import StandardScaler
import io
import json

# --- 1. CONFIGURATION: DIAGNOSTIC CONDITIONS & 30 BIOMARKERS ---
# The 10 specific genetic conditions for alignment
GENETIC_CONDITIONS = [
    "Cystic Fibrosis",
    "Becker Muscular Dystrophy",
    "Duchenne Muscular Dystrophy",
    "Gaucher Disease",
    "Hemophilia",
    "Porphyria",
    "Smith-Lemli-Opitz Syndrome",
    "Klinefelter Syndrome",
    "Turner Syndrome",
    "Down Syndrome"
]

# The 30 numeric biomarkers extracted from your Genetic Health CSVs
BIOMARKER_COLUMNS = [
    'onset_age_yrs', 'disease_severity_score', 'enzyme_activity_pct',
    'sweat_chloride_mmol_L', 'FEV1_pct_predicted', 'hemoglobin_g_dL',
    'platelet_count_x10_9_L', 'bilirubin_mg_dL', 'LDH_U_L', 'GBA_activity_pct',
    'factor_VIII_activity_pct', 'testosterone_nm_dl', 'estrogen_pg_ml',
    'porphyrin_level_umol_L', 'cholesterol_total_mg_dL', '7DHC_pct_of_cholesterol',
    'karyotype_X_count', 'congenital_anomalies_count', 'developmental_delay_months',
    'growth_percentile', 'family_history_degree', 'bleeding_events_per_year',
    'resp_infection_rate_per_year', 'hospitalizations_per_year', 'medication_count',
    'carrier_status', 'ERT_received', 'sweat_test_performed', 'cognitive_score',
    'genetic_mutation_count'
]

TARGET_LABEL = 'genetic_risk_score'
FINAL_FEATURES = GENETIC_CONDITIONS + BIOMARKER_COLUMNS

# --- 2. MODEL ARCHITECTURE ---
class GeneticHealthModel(nn.Module):
    def __init__(self, input_dim):
        super(GeneticHealthModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1) # Predicts Genetic Risk Score

    def forward(self, x):
        return self.fc(x)

# --- 3. ENCRYPTION SETUP (CKKS) ---
def create_ckks_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

# --- 4. PREPROCESSING & LOCAL TRAINING ---
def train_local_hospital(df, feature_names):
    # Clean up condition strings and remove non-breaking spaces (\xa0)
    df['genetic_condition'] = df['genetic_condition'].astype(str).str.replace('\xa0', ' ').str.strip()

    # One-Hot Encode conditions for consistent feature alignment
    for cond in GENETIC_CONDITIONS:
        df[cond] = (df['genetic_condition'] == cond).astype(int)

    # Select features and align (fill missing with 0)
    X = df.reindex(columns=feature_names, fill_value=0).astype(float).values
    y = df[TARGET_LABEL].values.reshape(-1, 1)

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Training Loop
    model = GeneticHealthModel(len(feature_names))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)

    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    # Extract weights + bias
    flat_weights = model.fc.weight.detach().numpy().flatten().tolist()
    flat_weights.append(model.fc.bias.detach().item())
    return flat_weights

# --- 5. EXECUTION FLOW WITH ENCODING FIX ---
print("Step 1: Upload your Genetic Health CSV files.")
uploaded = files.upload()

if len(uploaded) == 0:
    print("No files detected.")
else:
    context = create_ckks_context()
    encrypted_hospital_weights = []

    print(f"\nStep 2: Training and Encrypting models for {len(uploaded)} hospitals...")

    for filename, content in uploaded.items():
        try:
            # Using 'latin1' encoding to avoid UnicodeDecodeErrors
            df_hospital = pd.read_csv(io.BytesIO(content), encoding='latin1')

            # Train and get weights
            plaintext_weights = train_local_hospital(df_hospital, FINAL_FEATURES)

            # Encrypt weights
            enc_v = ts.ckks_vector(context, plaintext_weights)
            encrypted_hospital_weights.append(enc_v)
            print(f"  [OK] Processed and Encrypted: {filename}")
        except Exception as e:
            print(f"  [X] Error processing {filename}: {e}")

    if encrypted_hospital_weights:
        # Step 3: Secure Aggregation
        print("\nStep 3: Performing Secure Aggregation (HE)...")
        global_encrypted_sum = encrypted_hospital_weights[0]
        for i in range(1, len(encrypted_hospital_weights)):
            global_encrypted_sum += encrypted_hospital_weights[i]

        # Step 4: Decryption
        print("Step 4: Decrypting aggregated global model...")
        decrypted_sum = global_encrypted_sum.decrypt()
        num_hospitals = len(encrypted_hospital_weights)
        global_weights_final = [val / num_hospitals for val in decrypted_sum]

        # Step 5: Save Results
        model_output = {
            "model_name": "Global_Genetic_Health_Model",
            "features_ordered": FINAL_FEATURES,
            "weights": global_weights_final[:-1],
            "bias": global_weights_final[-1]
        }

        file_name = "global_genetic_model_weights.json"
        with open(file_name, "w") as f:
            json.dump(model_output, f, indent=4)

        print("\n" + "="*40)
        print("SUCCESS: Genetic Health Global Model Generated")
        print("="*40)
        files.download(file_name)
