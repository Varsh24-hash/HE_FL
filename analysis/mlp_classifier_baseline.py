import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
# ================================
# LOAD DATA (YOUR FILES)
# ================================
df1 = pd.read_csv('/mnt/data/hospital_1.csv')
df2 = pd.read_csv('/mnt/data/hospital_2.csv')
df3 = pd.read_csv('/mnt/data/hospital_3.csv')
datasets = [
    ("hospital_1", df1),
    ("hospital_2", df2),
    ("hospital_3", df3)
]
TARGET_COLUMN = 'mental_condition'   # change if needed
# ================================
# FUNCTION: TRAIN + SAVE WEIGHTS
# ================================
def train_and_save_weights(name, df, target_column):
    df = df.copy()
    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # MLP Pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42,
            early_stopping=True
        ))
    ])
    # Train model
    print(f"Training model for {name}...")
    model.fit(X_train, y_train)
    # Extract weights
    mlp = model.named_steps['mlp']
    weights = mlp.coefs_
    biases = mlp.intercepts_
    # Flatten all weights + biases into one vector
    weights_vector = np.concatenate(
        [w.flatten() for w in weights] +
        [b.flatten() for b in biases]
    )
    # Save as .npy
    filename = f"{name}_weights.npy"
    np.save(filename, weights_vector)
    print(f"Saved: {filename} ({len(weights_vector)} parameters)")
# ================================
# RUN FOR ALL HOSPITALS
# ================================
for name, df in datasets:
    train_and_save_weights(name, df, TARGET_COLUMN)
