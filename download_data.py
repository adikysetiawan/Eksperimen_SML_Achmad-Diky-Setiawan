import pandas as pd
import os
from sklearn.datasets import fetch_california_housing

print("Mengambil dataset Harga Rumah (California Housing)...")

# Mengambil dataset bawaan scikit-learn (dijamin 100% tidak akan error 404)
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Memastikan folder dataset_raw ada
os.makedirs('dataset_raw', exist_ok=True)

# Menyimpan ke folder dataset_raw
save_path = os.path.join('dataset_raw', 'housing_raw.csv')
df.to_csv(save_path, index=False)

print(f"Dataset berhasil disimpan di: {save_path}")
print(f"Total data: {df.shape[0]} baris.")