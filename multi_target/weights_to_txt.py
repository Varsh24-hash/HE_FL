import numpy as np
# Load weights
weights = np.load("weights.npy")
# Flatten (ensure 1D)
weights = weights.flatten()
# IMPORTANT: set these correctly
n = 9   # number of features
m = 5    # number of diseases
# Save to txt
with open("weights.txt", "w") as f:
    # Write shape
    f.write(f"Shape: ({n}, {m})\n")

    # Write all values in one line (space separated)
    f.write(" ".join(map(str, weights)))
print("weights.txt created")
