import streamlit as st
import pandas as pd
import os
from data_load import DataLoader
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from fault_interpreter import FaultInterpreter
from config import root_path, label_map, json_rules_path

st.set_page_config(page_title="Vibration Fault Diagnosis", layout="wide")
st.title("🧠 Vibration Fault Detection & Diagnostics")

# Load and process data
@st.cache_data(show_spinner=True)
def load_features():
    loader = DataLoader(root_path, label_map)
    raw_data = loader.load_data()
    extractor = FeatureExtractor()
    features, labels, names = [], [], []
    for i, (df, label) in enumerate(raw_data):
        feats = extractor.extract(df)
        feats['label'] = label
        features.append(feats)
        names.append(f"Sample #{i+1} - {label}")
    return pd.DataFrame(features), names

features_df, sample_names = load_features()
X = features_df.drop("label", axis=1)
y = features_df["label"]

# Train the model
@st.cache_resource(show_spinner=True)
def train_model():
    from sklearn.model_selection import train_test_split
    model = ModelTrainer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    model.train(X_train, y_train)
    return model

model = train_model()
interpreter = FaultInterpreter(json_rules_path)

# Sidebar controls
st.sidebar.header("📁 Select Vibration Sample")
sample_idx = st.sidebar.slider("Choose Sample Index", 0, len(sample_names) - 1, 0)
selected_sample = X.iloc[[sample_idx]]
prediction = model.predict(selected_sample)[0]

st.sidebar.markdown(f"**Real Label:** {features_df.iloc[sample_idx]['label']}")
st.sidebar.markdown(f"**Predicted:** {prediction}")

# Main content
st.subheader("📊 Prediction Result")
st.success(f"Predicted Fault: **{prediction}**")

st.subheader("🩺 Diagnostic Information")
if prediction in interpreter.knowledge:
    st.markdown("#### ⚠️ Causes")
    for cause in interpreter.knowledge[prediction].get("causes", []):
        st.markdown(f"- {cause}")

    st.markdown("#### ❗ Conséquences")
    for consequence in interpreter.knowledge[prediction].get("consequences", []):
        st.markdown(f"- {consequence}")

    st.markdown("#### 🛠️ Actions recommandées")
    for action in interpreter.knowledge[prediction].get("actions", []):
        st.markdown(f"- {action}")
else:
    st.warning("Aucune information disponible pour cette prédiction.")

with st.expander("🔍 View Full Feature Vector"):
    st.dataframe(selected_sample.T)

with st.expander("📈 Dataset Overview"):
    st.dataframe(features_df.head(10))
    st.markdown("### 🧮 Label Distribution")
    st.bar_chart(features_df['label'].value_counts())
