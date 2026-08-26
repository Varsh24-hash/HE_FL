import numpy as np
import pandas as pd
import joblib
from google.colab import files
print("Upload:")
print("1) local_model.joblib")
print("2) global_model_weights.npy")
print("3) your original training CSV")
uploaded = files.upload()
joblib_file = [f for f in uploaded if f.endswith(".joblib")][0]
npy_file = [f for f in uploaded if f.endswith(".npy")][0]
csv_file = [f for f in uploaded if f.endswith(".csv")][0]
# Load model and weights
model = joblib.load(joblib_file)
global_weights = np.load(npy_file)
# Load CSV only to get feature structure
df = pd.read_csv(csv_file, encoding="latin1")
mlp = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]
# -------------------------------
# Rebuild MLP using global weights
# -------------------------------
coefs_shapes = [w.shape for w in mlp.coefs_]
intercepts_shapes = [b.shape for b in mlp.intercepts_]
idx = 0
new_coefs = []
new_intercepts = []
for shape in coefs_shapes:
    size = np.prod(shape)
    new_coefs.append(global_weights[idx:idx+size].reshape(shape))
    idx += size
for shape in intercepts_shapes:
    size = np.prod(shape)
    new_intercepts.append(global_weights[idx:idx+size].reshape(shape))
    idx += size
mlp.coefs_ = new_coefs
mlp.intercepts_ = new_intercepts
print("Global weights injected")
# -------------------------------
# Build feature names
# -------------------------------
num_features = preprocessor.transformers_[0][2]
cat_features = (
    preprocessor.transformers_[1][1]
    .named_steps["onehot"]
    .get_feature_names_out(preprocessor.transformers_[1][2])
)
feature_names = np.concatenate([num_features, cat_features])
# -------------------------------
# Collapse network -> feature -> class
# -------------------------------
W_total = mlp.coefs_[0]
for W in mlp.coefs_[1:]:
    W_total = W_total @ W
# Absolute influence
feature_to_class = np.abs(W_total)
classes = mlp.classes_
# -------------------------------
# NORMALIZE FOR INTERPRETABLE SCALE (0-1 PER DISEASE)
# -------------------------------
# 1.0 = strongest parameter for that disease
# 0.5 = medium influence
# 0.1 = weak influence
feature_to_class_norm = feature_to_class / feature_to_class.max(axis=0)
# -------------------------------
# Compute PARALLELS (only disease-wise, not patient-wise)
# -------------------------------
rows = []
for i, disease in enumerate(classes):
    influences = feature_to_class_norm[:, i]
    top_idx = np.argsort(influences)[-2:][::-1]
    rows.append({
        "Disease": disease,
        "Top_Parameter_1": feature_names[top_idx[0]],
        "Normalized_Impact_1 (0-1)": round(float(influences[top_idx[0]]), 3),
        "Top_Parameter_2": feature_names[top_idx[1]],
        "Normalized_Impact_2 (0-1)": round(float(influences[top_idx[1]]), 3)
    })
parallels_df = pd.DataFrame(rows)
# -------------------------------
# Save only parallels
# -------------------------------
outfile = "disease_parallels_interpretable.csv"
parallels_df.to_csv(outfile, index=False)
files.download(outfile)
print("\nPARALLELS WITH INTERPRETABLE SCALE GENERATED")
display(parallels_df)
