import pandas as pd

class FeatureExtractor:
    def extract(self, df):
        import numpy as np
        from scipy.stats import kurtosis, skew
        features = {}
        for axis in df.columns:
            s = df[axis]
            features[f'{axis}_mean'] = s.mean()
            features[f'{axis}_std'] = s.std()
            features[f'{axis}_rms'] = np.sqrt(np.mean(s**2))
            features[f'{axis}_peak2peak'] = s.max() - s.min()
            features[f'{axis}_kurtosis'] = kurtosis(s)
            features[f'{axis}_skew'] = skew(s)
        return features
    
# Extract features
extractor = FeatureExtractor()
data = []
# Define raw_data as a placeholder list of tuples (replace with actual data source)
raw_data = [(pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}), 'label1'),
            (pd.DataFrame({'x': [7, 8, 9], 'y': [10, 11, 12]}), 'label2')]

for df, label in raw_data:
    feats = extractor.extract(df)
    feats['label'] = label
    data.append(feats)

features_df = pd.DataFrame(data)
X = features_df.drop('label', axis=1)
y = features_df['label']