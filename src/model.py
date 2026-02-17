import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_ticket_classifier(X, y, model_type="logistic_regression"):
    """
    Train and evaluate a ticket classifier (Functional approach).
    """
    print(f"--- Training Model ({model_type}) ---")
    
    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Model type {model_type} not supported yet.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test

def save_model(model, path="data/processed/ticket_model.joblib"):
    """
    Save the trained model to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def load_model(path="data/processed/ticket_model.joblib"):
    """
    Load a trained model from disk.
    """
    model = joblib.load(path)
    print(f"✅ Model loaded from {path}")
    return model

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    # Test with dummy data
    X, y = make_classification(n_samples=100, n_features=384, n_classes=2, random_state=42)
    model, _, _ = train_ticket_classifier(X, y)
    save_model(model, "data/processed/test_model.joblib")
