import os
import pandas as pd

def load_vibration_dataset(base_path):
    """
    Load all vibration CSV files organized in subfolders by class.

    Each subfolder (e.g., 'Normal', 'Inner Race Fault', 'Outer Race Fault') should contain CSV files
    with columns ['X', 'Y', 'Z'] representing the vibration signals on each axis.

    Parameters:
    ----------
    base_path : str
        Path to the base directory containing class-named folders.

    Returns:
    -------
    data : list of pd.DataFrame
        List of cleaned DataFrames for each CSV file.
    
    labels : list of str
        List of labels corresponding to each DataFrame.
    """
    data = []
    labels = []

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path '{base_path}' does not exist.")

    for label in os.listdir(base_path):
        folder_path = os.path.join(base_path, label)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)

                try:
                    df = pd.read_csv(file_path)
                    df.dropna(inplace=True)

                    # Ensure column names are X, Y, Z (ignore other columns)
                    if df.shape[1] >= 3:
                        df = df.iloc[:, :3]
                        df.columns = ['X', 'Y', 'Z']
                        data.append(df)
                        labels.append(label)
                    else:
                        print(f"⚠️ Skipping {filename}: not enough columns")

                except Exception as e:
                    print(f"⚠️ Error reading {file_path}: {e}")
    
    return data, labels
