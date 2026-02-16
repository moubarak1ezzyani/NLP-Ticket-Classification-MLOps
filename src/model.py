import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class TicketClassifier:
    def __init__(self, model_type="logistic_regression"):
        """
        Initialize the classifier.
        """
        if model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000)
        else:
            raise ValueError(f"Model type {model_type} not supported yet.")

    def train(self, X, y):
        """
        Split data, train the model, and evaluate performance.
        """
        print(f"--- Training Model ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        self.model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = self.model.predict(X_test)
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_test, y_test

    def save_model(self, path="data/processed/ticket_model.joblib"):
        """
        Save the trained model to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"✅ Model saved to {path}")

    def load_model(self, path="data/processed/ticket_model.joblib"):
        """
        Load a trained model from disk.
        """
        self.model = joblib.load(path)
        print(f"✅ Model loaded from {path}")

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=384, n_classes=2, random_state=42)
    clf = TicketClassifier()
    clf.train(X, y)
