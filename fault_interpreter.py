class FaultInterpreter:
    def __init__(self, json_path):
        import json
        import os
        if os.path.isdir(json_path):
            raise ValueError(f" Error: The given path '{json_path}' is a directory. Please provide a JSON file path.")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.knowledge = json.load(f)

    def explain(self, label):
        if label in self.knowledge:
            print(f"\n Diagnostic pour {label}:")
            print("Causes:")
            for cause in self.knowledge[label].get("causes", []):
                print(f"- {cause}")
            print("\nConséquences:")
            for cons in self.knowledge[label].get("consequences", []):
                print(f"- {cons}")
            print("\nActions:")
            for action in self.knowledge[label].get("actions", []):
                print(f"- {action}")
        else:
            print(f"Aucune information pour '{label}'")


# Fault interpretation
json_rules_path = r"C:\Users\hough\Desktop\pandasApp\myenv\vibration_fault_detection\alarmes_vibrations_diagnostics.json"
interpreter = FaultInterpreter(json_rules_path)

# Example predictions (replace with actual predictions as needed)
y_pred = ["fault1", "fault2", "fault3"]

for pred in set(y_pred):
    interpreter.explain(pred)