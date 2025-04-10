from data_load import DataLoader
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from fault_interpreter import FaultInterpreter
from config import root_path, label_map, json_rules_path

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load raw CSV data from folders
loader = DataLoader(root_path, label_map)
raw_data = loader.load_data()

# 2. Extract features from each file
extractor = FeatureExtractor()
dataset = []
for df, label in raw_data:
    features = extractor.extract(df)
    features["label"] = label
    dataset.append(features)

features_df = pd.DataFrame(dataset)
print("Feature DataFrame preview:")
print(features_df.head())
print("Label distribution:")
print(features_df["label"].value_counts())

# 3. Split dataset
X = features_df.drop("label", axis=1)
y = features_df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


# 6. Optional: save features
features_df.to_csv("vibration_features.csv", index=False)

raw_data = loader.load_data()
print(f" Sample files: {len(raw_data)}")
features_df = pd.DataFrame(dataset)
print(" Dataset size:", features_df.shape)
print(" Class counts:\n", features_df['label'].value_counts())