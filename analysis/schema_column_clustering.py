import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import json
# 1. Configuration & Model Loading
MODEL_NAME = 'all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.75  # Higher is stricter
OUTPUT_FILE = "cluster_output.txt"
FILE_PATHS = ["hospital1.txt", "hospital2.txt", "hospital3.txt"]
print(f"Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
# 2. Enhanced Normalization
# Medical data often uses abbreviations; mapping these helps the model understand intent.
MEDICAL_ABBR = {
    "dob": "date of birth",
    "pid": "patient identification",
    "px": "patient",
    "dx": "diagnosis",
    "hx": "history",
    "tx": "treatment"
}
def normalize(text):
    text = text.lower().strip()
    # Replace common delimiters with spaces
    for sep in ['_', '-', '.', '/', ':']:
        text = text.replace(sep, ' ')

    # Expand abbreviations
    words = text.split()
    expanded_words = [MEDICAL_ABBR.get(w, w) for w in words]

    return ' '.join(expanded_words)
# 3. Data Loading
def load_params(file_paths):
    params = set()
    for path in file_paths:
        try:
            with open(path, 'r') as f:
                header = f.readline().strip()
                if header:
                    cols = [c.strip() for c in header.split(',')]
                    params.update(cols)
        except FileNotFoundError:
            print(f"Warning: {path} not found. Skipping.")
    return list(params)
# 4. Processing Pipeline
print("Extracting and normalizing headers...")
raw_params = load_params(FILE_PATHS)
# Create a mapping of Original -> Normalized (excluding "disease" if needed)
# Note: I removed the 'disease' filter to let the model cluster it with 'diagnosis'
param_data = [{"original": p, "normalized": normalize(p)} for p in raw_params]
normalized_list = [item["normalized"] for item in param_data]
print(f"Generating embeddings for {len(normalized_list)} unique headers...")
embeddings = model.encode(normalized_list)
# 5. Robust Clustering
# We use AgglomerativeClustering with 'cosine' metric.
# distance_threshold = 1 - similarity_threshold
dist_threshold = 1 - SIMILARITY_THRESHOLD
cluster_model = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=dist_threshold,
    metric='cosine',
    linkage='average' # Ensures the whole group is similar, not just one pair
)
cluster_labels = cluster_model.fit_predict(embeddings)
# 6. Organize and Save Results
clusters = {}
for idx, label in enumerate(cluster_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(param_data[idx]["original"])
# Print to console
print("\n--- Final Clusters ---")
for cluster_id, members in clusters.items():
    print(f"Group {cluster_id}: {members}")
# Save to file
with open(OUTPUT_FILE, "w") as f:
    f.write("Schema Mapping Results\n")
    f.write("======================\n")
    for cluster_id, members in clusters.items():
        f.write(f"Cluster {cluster_id}: {', '.join(members)}\n")
print(f"\nSuccess! Results saved to {OUTPUT_FILE}")

