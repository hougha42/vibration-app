# rul_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def estimate_rul(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate synthetic RUL (Remaining Useful Life) for each sample in features_df.
    Adds two new columns: 'health_index' and 'estimated_RUL'.

    The health index is computed from features (e.g. normalized RMS and kurtosis),
    then RUL = RUL_max * (1 - health_index). RUL_max is set to 100 by default.
    """
    df = features_df.copy()
    # Example: use RMS and kurtosis if available, else use any feature trends.
    # Compute normalized scores (0 to 1) for RMS and kurtosis.
    for col in ['RMS', 'kurtosis']:
        if col in df.columns:
            df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    # Combine metrics into a simple health index (clipped 0-1).
    if 'RMS_norm' in df.columns and 'kurtosis_norm' in df.columns:
        df['health_index'] = (df['RMS_norm'] + df['kurtosis_norm']) / 2
    elif 'RMS_norm' in df.columns:
        df['health_index'] = df['RMS_norm']
    else:
        # Fallback: use a single feature if available
        first_feat = df.columns[0]
        df['health_index'] = (df[first_feat] - df[first_feat].min()) / (df[first_feat].max() - df[first_feat].min())

    df['health_index'] = df['health_index'].clip(0, 1)
    # Synthetic RUL (e.g., max life 100 units)
    RUL_max = 100
    df['estimated_RUL'] = (1 - df['health_index']) * RUL_max
    df['estimated_RUL'] = df['estimated_RUL'].round(2)
    return df

def plot_rul(df: pd.DataFrame, time_col: str = None):
    """
    Plot the synthetic RUL curve over time or index.
    If time_col is provided, it is used as the x-axis; otherwise the DataFrame index is used.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(6,4))
    x = df[time_col] if time_col in df.columns else df.index
    plt.plot(x, df['estimated_RUL'], color='C1', label='Estimated RUL')
    plt.xlabel(time_col if time_col else 'Sample')
    plt.ylabel('Synthetic RUL')
    plt.title('Estimated Remaining Useful Life (Synthetic)')
    plt.legend()
    plt.tight_layout()
    plt.show()
