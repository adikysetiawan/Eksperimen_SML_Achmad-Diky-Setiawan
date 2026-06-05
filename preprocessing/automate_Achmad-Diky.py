import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    print("Memulai proses otomatisasi data preprocessing...")

    # Membaca posisi folder saat ini agar path tidak error di GitHub Actions
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)

    # 1. Load Data dari folder dataset_raw
    raw_path = os.path.join(root_dir, 'dataset_raw', 'housing_raw.csv')
    df = pd.read_csv(raw_path)
    print(f"[OK] Data raw berhasil dimuat: {df.shape[0]} baris.")

    # 2. Preprocessing (Pisahkan fitur dan target)
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    # 3. Scaling Data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 4. Gabungkan kembali
    df_processed = X_scaled.copy()
    df_processed['MedHouseVal'] = y

    # 5. Simpan Data
    save_path = os.path.join(base_dir, 'housing_processed.csv')
    df_processed.to_csv(save_path, index=False)
    print(f"[OK] Preprocesing SELESAI! Data bersih disimpan di: {save_path}")

if __name__ == "__main__":
    run_preprocessing()