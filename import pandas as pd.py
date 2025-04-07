import pandas as pd
import numpy as np
import os
import json
from scipy.stats import kurtosis, skew

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les diagnostics depuis le fichier JSON
with open("alarmes_vibrations_diagnostics.json", "r", encoding="utf-8") as f:
    diagnostics_data = json.load(f)

# Fonction pour afficher les diagnostics basés sur la prédiction
def afficher_diagnostics(prediction):
    if prediction in diagnostics_data:
        infos = diagnostics_data[prediction]
        print("\n--- Diagnostic ---")
        print(f"Type de défaut: {prediction}")
        print("Causes possibles:")
        for cause in infos["causes"]:
            print(f"- {cause}")
        print("\nConséquences:")
        for consequence in infos["consequences"]:
            print(f"- {consequence}")
        print("\nActions initiales de l’opérateur:")
        for action in infos["actions"]:
            print(f"- {action}")
    else:
        print("Aucun diagnostic disponible pour cette classe.")

folder_path = r"C:\\Users\\hough\\Downloads\\CSV Files\\Inner Race Fault"
file_path = os.path.join(folder_path, os.listdir(folder_path)[0])
df = pd.read_csv(file_path, header=None)

print("Shape:", df.shape)
print(df.head())

root_path = "C:\\Users\\hough\\Downloads\\CSV Files"

label_map = {
    'Normal': 'Normal',
    'Inner Race Fault': 'Inner Race Fault',
    'Outer Race Fault': 'Outer Race Fault'
}

def extract_features(df):
    features = {}
    for axis in df.columns:
        series = df[axis]
        features[f'{axis}_mean'] = series.mean()
        features[f'{axis}_std'] = series.std()
        features[f'{axis}_rms'] = np.sqrt(np.mean(series**2))
        features[f'{axis}_peak2peak'] = series.max() - series.min()
        features[f'{axis}_kurtosis'] = kurtosis(series)
        features[f'{axis}_skew'] = skew(series)
    return features

dataset = []
for folder_name, label in label_map.items():
    folder_path = os.path.join(root_path, folder_name)
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, header=None)
            features = extract_features(df)
            features['label'] = label
            dataset.append(features)

features_df = pd.DataFrame(dataset)
print(features_df.head())
print(features_df['label'].value_counts())

X = features_df.drop('label', axis=1)
y = features_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

importances = clf.feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# Afficher le diagnostic pour chaque prédiction de test
print("\nDiagnostics pour les données test:")
for pred in np.unique(y_pred):
    afficher_diagnostics(pred)


    # The redundant code block has been removed.




features_df.to_csv("vibration_features.csv", index=False)