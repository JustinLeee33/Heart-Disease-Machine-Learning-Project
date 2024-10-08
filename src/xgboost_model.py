# xgboost_model.py

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
import os

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model."""
    # Train the model, enabling multiclass support
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # Use 'mlogloss' for multiclass
    model.fit(X_train, y_train)

    # Predict class labels
    y_pred = model.predict(X_test)

    # Get predicted probabilities for all classes (n_samples, n_classes)
    y_scores = model.predict_proba(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print results
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Return the trained model, predicted labels, and predicted probabilities for each class
    return model, y_pred, y_scores
