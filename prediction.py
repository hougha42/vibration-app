import joblib
import pandas as pd
import numpy as np
from feature_extraction import extract_all_features_from_sample

def load_model(path='data/models/random_forest_model.joblib'):
    bundle = joblib.load(path)
    return bundle['model'], bundle['features']

def predict_dataframe(df):
    """
    df: pd.DataFrame with X,Y,Z columns
    Returns: (label, {class:prob,...})
    """
    model, feats = load_model()
    feat_dict = extract_all_features_from_sample(df)
    x = np.array([feat_dict.get(f,0) for f in feats]).reshape(1,-1)
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    return pred, dict(zip(model.classes_,proba))

def batch_predict(csv_paths):
    results = []
    for path in csv_paths:
        df = pd.read_csv(path)
        lab,proba = predict_dataframe(df)
        results.append({'file':path,'pred':lab,'proba':proba})
    return results
