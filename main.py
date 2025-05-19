import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from feature_extraction import extract_features_from_data_dict

def load_vibration_dataset(base_path):
    """
    Load CSV files from subfolders under base_path.
    Returns:
        data: list of DataFrames
        labels: list of class labels
    """
    data = []
    labels = []

    for label in os.listdir(base_path):
        folder = os.path.join(base_path, label)
        if not os.path.isdir(folder):
            continue

        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                filepath = os.path.join(folder, filename)
                try:
                    df = pd.read_csv(filepath)
                    df.dropna(inplace=True)
                    if df.shape[1] >= 3:
                        df = df.iloc[:, :3]
                        df.columns = ['X', 'Y', 'Z']
                        data.append(df)
                        labels.append(label)
                    else:
                        print(f"Skipping {filename}: not enough columns")
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    return data, labels

def train_model(features_df, model_path="data/models/random_forest_model.joblib"):
    X = features_df.drop("Label", axis=1)
    y = features_df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)

    os.makedirs("data/models", exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns)}, model_path)
    print(f"Model saved to {model_path}")

    # Plot and save confusion matrix
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    print("Confusion matrix saved to reports/confusion_matrix.png")

def main():
    print("📂 Loading vibration data from CSV Files/")
    data_path = "CSV Files"
    if not os.path.exists(data_path):
        raise FileNotFoundError("CSV Files/ folder not found")

    data, labels = load_vibration_dataset(data_path)
    print(f"✅ Loaded {len(data)} samples from {len(set(labels))} classes")

    print("⚙️ Extracting features...")
    features_df = extract_features_from_data_dict(data_dict={label: [df for df, l in zip(data, labels) if l == label] for label in set(labels)})

    os.makedirs("data/processed", exist_ok=True)
    features_csv = "data/processed/features.csv"
    features_df.to_csv(features_csv, index=False)
    print(f"📄 Features saved to {features_csv}")

    print("🧠 Training model...")
    train_model(features_df)

    print("✅ Done.")

if __name__ == "__main__":
    main()
