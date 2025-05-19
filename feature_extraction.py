import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

def extract_time_domain_features(signal):
    features = {}
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))
    features['peak'] = np.max(np.abs(signal))
    features['p2p'] = np.max(signal) - np.min(signal)
    features['variance'] = np.var(signal)
    features['skewness'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)
    features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] != 0 else 0
    features['energy'] = np.sum(signal**2)
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features['zero_crossings'] = len(zero_crossings)
    return features

def extract_frequency_domain_features(signal, sampling_rate=10000):
    features = {}
    fft_vals = np.abs(np.fft.fft(signal))
    fft_freq = np.fft.fftfreq(len(signal), 1/sampling_rate)
    idx = fft_freq > 0
    fft_vals = fft_vals[idx]
    fft_freq = fft_freq[idx]

    if len(fft_vals) > 0:
        dom = np.argmax(fft_vals)
        features['dom_freq'] = fft_freq[dom]
        features['dom_amp'] = fft_vals[dom]
        features['low_freq_energy'] = np.sum(fft_vals[fft_freq <= 200]**2)
        features['med_freq_energy'] = np.sum(fft_vals[(fft_freq > 200)&(fft_freq<=800)]**2)
        features['high_freq_energy'] = np.sum(fft_vals[fft_freq > 800]**2)
        features['spectral_centroid'] = (fft_freq*fft_vals).sum()/fft_vals.sum() if fft_vals.sum()!=0 else 0
    return features

def extract_all_features_from_sample(df):
    """
    Extract time‐ & frequency‐domain features from a DataFrame with X,Y,Z columns.
    Returns a dict of feature_name:value.
    """
    features = {}
    for axis in ['X','Y','Z']:
        if axis in df:
            sig = df[axis].values
            t_feats = extract_time_domain_features(sig)
            f_feats = extract_frequency_domain_features(sig)
            for k,v in {**t_feats, **f_feats}.items():
                features[f"{axis}_{k}"] = v
    return features

def extract_features_from_data_dict(data_dict):
    """
    data_dict: {label: [df1,df2,...], ...}
    Returns pd.DataFrame with all features + 'Label' column.
    """
    import pandas as pd
    feats, labs = [], []
    for label, samples in data_dict.items():
        for df in samples:
            feats.append(extract_all_features_from_sample(df))
            labs.append(label)
    df = pd.DataFrame(feats)
    df['Label'] = labs
    return df
