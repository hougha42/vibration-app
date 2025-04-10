# data_load.py
import os
import pandas as pd

class DataLoader:
    def __init__(self, root_path, label_map):
        self.root_path = root_path
        self.label_map = label_map

    def load_data(self):
        data = []
        total_files = 0
        for folder_name, label in self.label_map.items():
            folder_path = os.path.join(self.root_path, folder_name)
            if not os.path.exists(folder_path):
                print(f" Folder not found: {folder_path}")
                continue
            class_files = 0
            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    path = os.path.join(folder_path, file)
                    try:
                        df = pd.read_csv(path, header=None)
                        data.append((df, label))
                        class_files += 1
                    except Exception as e:
                        print(f" Error reading {path}: {e}")
            print(f" Loaded {class_files} files from {folder_name}")
            total_files += class_files
        print(f"Total files loaded: {total_files}")
        return data
