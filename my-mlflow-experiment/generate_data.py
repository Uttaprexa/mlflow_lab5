"""
Generate Heart Disease Dataset
"""
import numpy as np
import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

print("Generating heart disease dataset...")

np.random.seed(42)
n_samples = 500

# Create features
data = {
    'age': np.random.randint(30, 80, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'cp': np.random.randint(0, 4, n_samples),
    'trestbps': np.random.randint(90, 200, n_samples),
    'chol': np.random.randint(120, 400, n_samples),
    'fbs': np.random.randint(0, 2, n_samples),
    'restecg': np.random.randint(0, 3, n_samples),
    'thalach': np.random.randint(70, 200, n_samples),
    'exang': np.random.randint(0, 2, n_samples),
    'oldpeak': np.random.uniform(0, 6, n_samples),
    'slope': np.random.randint(0, 3, n_samples),
    'ca': np.random.randint(0, 4, n_samples),
    'thal': np.random.randint(0, 4, n_samples)
}

df = pd.DataFrame(data)

# Create target with some logic
df['target'] = (
    ((df['age'] > 55) & (df['chol'] > 250)) | 
    (df['cp'] > 2) | 
    (df['thalach'] < 120)
).astype(int)

# Save to CSV
output_path = 'data/heart_disease.csv'
df.to_csv(output_path, index=False)

print(f" Dataset created successfully!")
print(f"  Location: {output_path}")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget distribution:")
print(df['target'].value_counts())