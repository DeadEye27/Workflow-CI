import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import os

# 1. Setup Path
# Menggunakan relative path agar lebih fleksibel dan tidak error saat pindah folder
BASE_DIR = Path.cwd() # Mengambil folder saat ini secara otomatis
MLRUNS_DIR = BASE_DIR / "mlruns"

# Pastikan folder mlruns ada (opsional, mlflow biasanya otomatis membuatnya)
if not MLRUNS_DIR.exists():
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# Set tracking URI ke folder lokal dengan format yang benar
# Kita gunakan path absolut dengan prefix file:/// agar konsisten
tracking_uri = "file:///" + str(MLRUNS_DIR).replace("\\", "/")
mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("Churn_Prediction_Local")
mlflow.autolog()

# 2. Load Data
# Menggunakan nama file langsung (asumsi file ada di folder yang sama dengan script)
csv_filename = "data_processed.csv"
csv_path = BASE_DIR / csv_filename

if not csv_path.exists():
    # Coba cari path manual jika tidak ketemu di current directory
    # Fallback ke path hardcoded jika script dijalankan dari tempat berbeda
    csv_path = Path(r"D:\Abid\Kuliah\Dicoding Asah\Submission_SIstem_Machine_Learning\Membangun_model\data_processed.csv")

if not csv_path.exists():
    print(f"Error: File tidak ditemukan di {csv_path}")
else:
    print(f"Memuat data dari: {csv_path}")
    print(f"Tracking URI MLflow: {mlflow.get_tracking_uri()}")
    
    data = pd.read_csv(csv_path)

    # 3. Preprocessing & Split
    # Pastikan kolom Churn ada sebelum drop
    if "Churn" in data.columns:
        X = data.drop(columns=["Churn"])
        y = data["Churn"]
    else:
        # Fallback jika target kolom berbeda namanya atau sudah dipisah
        print("Kolom 'Churn' tidak ditemukan, cek nama kolom data Anda.")
        # Contoh safety (sesuaikan jika perlu):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Training & Tracking dengan MLflow
    with mlflow.start_run() as run:
        print(f"Memulai training... (Run ID: {run.info.run_id})")
        
        # Inisialisasi Model
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.5,
            max_depth=4,
            random_state=42
        )

        # Fit Model
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Log Metrics Manual (Opsional)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", f1)

        # Log Model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_train.head()
        )

        print("Training selesai!")
        print(f"Data tersimpan di: {MLRUNS_DIR}")
        print("\nUntuk melihat dashboard, jalankan perintah berikut di terminal:")
        print("mlflow ui")