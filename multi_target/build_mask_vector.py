from google.colab import files
print("Upload hospital CSV")
uploaded_csv = files.upload()
print("Upload weights.npy")
uploaded_weights = files.upload()
print("Upload dictionary.txt")
uploaded_dict = files.upload()

csv_path = list(uploaded_csv.keys())[0]
weights_path = list(uploaded_weights.keys())[0]
dict_path = list(uploaded_dict.keys())[0]

import numpy as np
import pandas as pd
import ast

df = pd.read_csv(csv_path)
weights = np.load(weights_path).flatten()
TARGET_COL = "genetic_condition"
columns = [col.strip() for col in df.columns if col != TARGET_COL]

global_clusters = {}
with open(dict_path, "r") as f:
    for line in f:
        if line.strip():
            idx_part, list_part = line.split(":")
            idx = int(idx_part.replace("Index", "").strip())
            values = ast.literal_eval(list_part.strip())
            global_clusters[idx] = values
num_clusters = len(global_clusters)

def normalize(text):
    return text.lower().replace(" ", "").replace("_", "").replace("-", "")

mask = np.zeros(num_clusters, dtype=int)
col_to_cluster = {}
for col in columns:
    norm_col = normalize(col)
    for idx, variants in global_clusters.items():
        if any(norm_col == normalize(v) for v in variants):
            mask[idx] = 1
            col_to_cluster[col] = idx
            break
print("Mask:", mask)
# Debug (optional)
unmapped = set(columns) - set(col_to_cluster.keys())
if unmapped:
    print("Unmapped columns:", unmapped)

m = 5  # number of diseases
n_local = len(columns)
if weights.size != n_local * m:
    raise ValueError(f"Expected {n_local*m}, got {weights.size}")
W_local = weights.reshape(n_local, m)

global_vector = []
for cluster_idx in range(num_clusters):
    if mask[cluster_idx] == 1:
        found = False
        for i, col in enumerate(columns):
            if col_to_cluster.get(col) == cluster_idx:
                global_vector.extend(W_local[i])
                found = True
                break
        if not found:
            global_vector.extend([0]*m)
    else:
        global_vector.extend([0]*m)
global_vector = np.array(global_vector)

with open("final_weights.txt", "w") as f:
    f.write(f"Shape: ({num_clusters}, {m})\n")
    f.write(" ".join(map(str, global_vector)))
with open("mask.txt", "w") as f:
    f.write(" ".join(map(str, mask)))
np.save("mask.npy", mask)
np.save("final_weights.npy", global_vector)
print("Files created!")

files.download("final_weights.txt")
files.download("mask.txt")
files.download("mask.npy")
files.download("final_weights.npy")
