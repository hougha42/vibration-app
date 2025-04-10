class ModelTrainer:
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    def feature_importance(self, X):
        import matplotlib.pyplot as plt
        import numpy as np

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-10:]
        plt.figure(figsize=(8,5))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.title("Top 10 Important Features")
        plt.tight_layout()
        plt.show()
