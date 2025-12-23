import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import os
import sys

# 1. Setup Path
BASE_DIR = Path.cwd()
csv_filename = "data_processed.csv"
csv_path = BASE_DIR / csv_filename


if os.environ.get("MLFLOW_RUN_ID") is None:
    mlflow.set_experiment("Churn_Prediction_Local")

mlflow.autolog()

# 2. Load Data
if not csv_path.exists():
    fallback_path = Path(r"D:\Abid\Kuliah\Dicoding Asah\Submission_SIstem_Machine_Learning\Membangun_model\data_processed.csv")
    if fallback_path.exists():
        csv_path = fallback_path
    else:
        local_csv = Path("data_processed.csv")
        if local_csv.exists():
            csv_path = local_csv

if not csv_path.exists():
    print(f"Error: File '{csv_filename}' tidak ditemukan di: {BASE_DIR}")
    print("Isi folder saat ini:", os.listdir(BASE_DIR))
    sys.exit(1) 
else:
    print(f"Memuat data dari: {csv_path}")
    print(f"Tracking URI MLflow: {mlflow.get_tracking_uri()}")
    
    data = pd.read_csv(csv_path)

    # 3. Preprocessing & Split
    if "Churn" in data.columns:
        X = data.drop(columns=["Churn"])
        y = data["Churn"]
    else:
        print("Kolom 'Churn' tidak ditemukan, menggunakan kolom terakhir sebagai target.")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

with mlflow.start_run() as run:
    print(f"Memulai training... (Run ID: {run.info.run_id})")
    print(f"Eksperimen ID: {run.info.experiment_id}")
    
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

    # Log Metrics Manual 
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_score", f1)

    # Log Model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=X_train.head()
    )

    print("Training selesai!")