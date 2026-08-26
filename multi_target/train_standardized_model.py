import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import os
from sklearn.preprocessing import StandardScaler
from google.colab import files

# --- STEP 1: LOAD GLOBAL DICTIONARY ---
def load_global_dictionary(dict_path):
    """
    Loads your .txt global dictionary.
    Format assumed: variant_name : standard_index
    """
    mapping = {}
    if os.path.exists(dict_path):
        with open(dict_path, 'r') as f:
            for line in f:
                if ':' in line:
                    variant, idx = line.strip().split(':')
                    mapping[variant.strip()] = int(idx)
    else:
        print(f"Warning: {dict_path} not found. Please upload it.")
    return mapping

# --- STEP 2: TRAINING FUNCTION ---
def train_local_model(csv_path, hospital_id, global_map, target_names):
    print(f"\nTraining {hospital_id} from {csv_path}...")

    # Load Data
    df = pd.read_csv(csv_path)

    # Identify Local Targets and Parameters
    local_targets = [col for col in target_names if col in df.columns]
    local_params = [col for col in df.columns if col in global_map and col not in target_names]

    if not local_targets or not local_params:
        print(f"Skipping {hospital_id}: Missing targets or parameters.")
        return None, None

    # Prepare Tensors
    X = df[local_params].values
    Y = df[local_targets].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.FloatTensor(X_scaled)
    Y_tensor = torch.FloatTensor(Y)

    # n parameters, m diseases
    n, m = len(local_params), len(local_targets)

    # Multi-Target Linear Model (Mapping P -> D)
    model = nn.Linear(n, m, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Short Training Loop
    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), Y_tensor)
        loss.backward()
        optimizer.step()

    # --- FORMAT WEIGHTS: p1d1, p1d2, ..., pndm ---
    # Weight matrix is [m, n]. Transpose to [n, m] so flattening gives the desired sequence.
    weight_matrix = model.weight.detach().numpy().T
    flattened_weights = weight_matrix.flatten()

    # --- SAVE WEIGHTS (.npy) ---
    weights_filename = f"{hospital_id}_weights.npy"
    weights_package = {
        "weights": flattened_weights,
        "global_indices": [global_map[p] for p in local_params],
        "local_targets": local_targets,
        "n_params": n,
        "m_diseases": m
    }
    np.save(weights_filename, weights_package)

    # --- SAVE MODEL (.json) ---
    model_filename = f"{hospital_id}_model.json"
    model_metadata = {
        "hospital_id": hospital_id,
        "input_features": local_params,
        "target_diseases": local_targets,
        "vector_length": len(flattened_weights),
        "sequence_format": "p[i]d[j]"
    }
    with open(model_filename, "w") as f:
        json.dump(model_metadata, f, indent=4)

    print(f"Generated: {weights_filename} and {model_filename}")
    return weights_filename, model_filename

# --- EXECUTION ---

# 1. Upload your files
print("Please upload hospital_1.csv, hospital_2.csv, hospital_3.csv, and global_dictionary.txt")
uploaded = files.upload()

# 2. Configuration
GLOBAL_TARGETS = [
    'genetic_risk_score', 'genetic_risk_scr', # Genetic
    'mental_health_index',                    # Mental
    'sexual_health_index'                     # Sexual
]
dict_file = "global_dictionary.txt"
hospital_files = ["hospital_1.csv", "hospital_2.csv", "hospital_3.csv"]

# 3. Process
global_mapping = load_global_dictionary(dict_file)
output_files = []

for i, csv_file in enumerate(hospital_files, 1):
    if csv_file in uploaded:
        w_file, m_file = train_local_model(csv_file, f"Hospital_0{i}", global_mapping, GLOBAL_TARGETS)
        if w_file: output_files.extend([w_file, m_file])

# 4. Final summary
print("\n--- Process Complete ---")
print(f"Total files generated: {len(output_files)}")
for f in output_files:
    print(f"- {f}")
