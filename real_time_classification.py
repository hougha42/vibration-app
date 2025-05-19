# real_time_classification.py
import time
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import joblib  # efficient model persistence
# import paho.mqtt.client as mqtt  # Placeholder for MQTT integration

def load_model(model_path: str):
    """
    Load a pre-trained classification model from a joblib file.
    Joblib is preferred for scikit-learn models for efficiency:contentReference[oaicite:7]{index=7}.
    """
    model = joblib.load(model_path)
    return model

def simulate_stream_and_classify(model, data_folder: str, poll_interval: float = 1.0):
    """
    Simulate a live data stream by reading CSV files from a folder.
    For each new file, read features, predict class and confidence, and output results with timestamp.
    """
    processed = set()
    while True:
        # List CSV files in folder
        files = glob.glob(f"{data_folder}/*.csv")
        for file in sorted(files):
            if file not in processed:
                # Read incoming sample (expects one row of features per file)
                sample_df = pd.read_csv(file)
                # Preprocess or extract features if needed (placeholder)
                
                # Prediction
                proba = model.predict_proba(sample_df.values)[0]
                class_idx = np.argmax(proba)
                class_label = model.classes_[class_idx]
                confidence = proba[class_idx]
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Detected '{class_label}' with confidence {confidence:.2f}")
                
                # Placeholder: publish to MQTT or socket here
                # mqtt_client.publish("machine/classification", payload=...)
                
                processed.add(file)
        time.sleep(poll_interval)

if __name__ == "__main__":
    # Example usage
    model = load_model("bearing_classifier.joblib")
    simulate_stream_and_classify(model, data_folder="live_data")
